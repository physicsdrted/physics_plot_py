import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import re
import inspect
import io # For SVG download
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# --- Configuration (Optional: Set page title etc.) ---
st.set_page_config(page_title="Physics Plot Fitter", layout="wide")

# --- Allowed characters and functions for user equation ---
ALLOWED_CHARS = r"^[A-Za-z0-9\s\.\+\-\*\/\(\)\,\_\^]+$"
ALLOWED_NP_FUNCTIONS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan, 'atan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'ln':np.log, 'log10': np.log10, 'sqrt': np.sqrt,
    'pi': np.pi, 'abs': np.abs, 'absolute': np.abs,
}
SAFE_GLOBALS = {'__builtins__': {}}; SAFE_GLOBALS['np'] = np; SAFE_GLOBALS.update(ALLOWED_NP_FUNCTIONS)


# --- Helper Function Definitions ---

def validate_and_parse_equation(eq_string):
    """Validates equation, finds 'x' and parameters (A-Z)."""
    eq_string = eq_string.strip()
    if not eq_string: raise ValueError("Equation cannot be empty.")
    eq_string = eq_string.replace('^', '**')
    if not re.match(ALLOWED_CHARS, eq_string):
        invalid_chars = "".join(sorted(list(set(re.sub(ALLOWED_CHARS, '', eq_string)))))
        raise ValueError(f"Invalid chars: '{invalid_chars}'. Allowed: A-Z, a-z ('x'), 0-9, _ . + - * / ( ) , space, ^")
    if not re.search(r'\bx\b', eq_string): raise ValueError("Equation must contain 'x'.")
    params = sorted(list(set(re.findall(r'\b([A-Z])\b', eq_string))))
    all_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', eq_string))
    allowed_words = set(['x']) | set(params) | set(ALLOWED_NP_FUNCTIONS.keys()) | set(['np'])
    unknown_words = all_words - allowed_words
    if unknown_words: raise ValueError(f"Unknown/disallowed items: {', '.join(unknown_words)}. Use 'x', A-Z params, allowed funcs.")
    if not params: raise ValueError("No fit parameters (A-Z) found.")
    # Use st.info for console output in Streamlit context if needed, but print is fine for debugging
    # st.info(f"Identified parameters: {params}")
    return eq_string, params

def create_fit_function(eq_string, params):
    """Dynamically creates Python function from validated equation string."""
    func_name = "dynamic_fit_func"; param_str = ', '.join(params)
    # Construct the assignments for the locals dictionary used by eval
    eval_locals_assignments = [f"'{p}': {p}" for p in params]
    eval_locals_str = f"{{'x': x, {', '.join(eval_locals_assignments)}}}"

    # Capture necessary variables in the global scope for exec
    exec_globals = {'np': np, 'SAFE_GLOBALS': SAFE_GLOBALS, 'eq_string': eq_string, 'params': params}

    # Code for the function to be created by exec
    func_code = f"""
import numpy as np
# Need access to SAFE_GLOBALS from the exec_globals dict
_SAFE_GLOBALS = SAFE_GLOBALS
_EQ_STRING = eq_string # Capture eq_string
_PARAMS = params # Capture params list

def {func_name}(x, {param_str}):
    try:
        # Construct eval locals mapping param names to their current values
        eval_locals = {{'x': x}}
        # Directly use the arguments passed (A, B, C...) by curve_fit
        local_args = locals()
        for p_name in _PARAMS:
            eval_locals[p_name] = local_args[p_name]

        # Use captured _SAFE_GLOBALS and _EQ_STRING
        result = eval(_EQ_STRING, _SAFE_GLOBALS, eval_locals)

        if isinstance(result, (np.ndarray, list, tuple)):
            result = np.asarray(result)
            if np.iscomplexobj(result): result = np.real(result) # Simplified warning
            result = result.astype(float)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)): pass
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result)
        else: raise TypeError(f"Equation returned non-numeric type: {{type(result)}}")
        return result
    except ZeroDivisionError: return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
    except Exception as e: return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
"""
    local_namespace = {}
    try: exec(func_code, exec_globals, local_namespace)
    except Exception as e: raise SyntaxError(f"Failed to compile function: {e} ({type(e).__name__})") from e
    if func_name not in local_namespace: raise RuntimeError(f"Failed to create function '{func_name}' via exec.")
    return local_namespace[func_name]

def numerical_derivative(func, x, params, h=1e-7):
    """Calculates numerical derivative using central difference."""
    try:
        if params is None or not all(np.isfinite(p) for p in params):
             st.warning("Invalid parameters passed to numerical_derivative. Returning slope=0.")
             return np.zeros_like(x) if isinstance(x, np.ndarray) else 0
        y_plus_h = func(x + h, *params); y_minus_h = func(x - h, *params)
        deriv = (y_plus_h - y_minus_h) / (2 * h)
        if isinstance(deriv, np.ndarray): deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv): deriv = 0
        return deriv
    except Exception as e:
        st.warning(f"Error during num derivative: {e}. Returning slope=0.")
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def safeguard_errors(err_array, min_err=1e-12):
     """Replaces non-positive or NaN/Inf errors with a small positive number."""
     safe_err = np.array(err_array, dtype=float) # Ensure float copy
     invalid_mask = ~np.isfinite(safe_err) | (safe_err <= 0)
     num_invalid = np.sum(invalid_mask)
     if num_invalid > 0:
         st.warning(f"Found {num_invalid} invalid values (<=0, NaN, Inf) in error array. Replacing with {min_err}.")
         safe_err[invalid_mask] = min_err
     return safe_err

# --- Main App Logic ---
st.title("Physics Data Plotter and Fitter")
st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")

# --- Session State Initialization ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.x_data = None
    st.session_state.y_data = None
    st.session_state.x_err_safe = None
    st.session_state.y_err_safe = None
    st.session_state.x_axis_label = "X"
    st.session_state.y_axis_label = "Y"
    st.session_state.fit_results = None
    st.session_state.final_fig = None
    st.session_state.processed_file_key = None # Use key based on name/size

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

if uploaded_file is not None:
    # <<< Create a unique key based on file name and size >>>
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    # Process file only if it's a new file (different key)
    if current_file_key != st.session_state.get('processed_file_key', None):
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        # Reset flags and data before processing new file
        st.session_state.data_loaded = False
        st.session_state.fit_results = None
        st.session_state.final_fig = None

        try:
            # Read and validate data
            raw_df = pd.read_csv(uploaded_file, header=None)
            if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
                st.error("Invalid file structure: Needs labels (row 1), 4 data cols, >=1 data row.")
                st.stop()
            # Extract Labels safely
            try: x_label = str(raw_df.iloc[0, 0]); y_label = str(raw_df.iloc[0, 2])
            except Exception: x_label = "X (Col 1)"; y_label = "Y (Col 3)"; st.warning("Could not read labels.")
            # Extract/validate data
            df = raw_df.iloc[1:].copy()
            if df.empty or df.shape[1] != 4: st.error("No data rows or wrong number of columns."); st.stop()
            df.columns = ['x', 'x_err', 'y', 'y_err']; all_numeric = True
            for col in df.columns:
                try: df[col] = pd.to_numeric(df[col])
                except ValueError: st.error(f"Column '{col}' non-numeric."); all_numeric = False; break
            if not all_numeric: st.stop()
            if df.isnull().values.any(): st.error("NaN/empty cells detected."); st.stop()

            # Store processed data in session state
            st.session_state.x_data = df['x'].to_numpy()
            st.session_state.y_data = df['y'].to_numpy()
            st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy()))
            st.session_state.y_err_safe = safeguard_errors(df['y_err'].to_numpy())
            st.session_state.x_axis_label = x_label
            st.session_state.y_axis_label = y_label
            st.session_state.data_loaded = True
            st.session_state.processed_file_key = current_file_key # Mark file as processed
            st.success("Data loaded and validated successfully!")
            st.rerun() # Rerun to update UI state immediately after load

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.data_loaded = False
            st.session_state.processed_file_key = None # Clear processed key on error
            st.stop()

# --- Show Input Section only if data is loaded ---
if st.session_state.data_loaded:
    st.markdown("---")
    st.subheader("Enter Fit Equation")
    st.markdown("Use `x`, parameters (A-Z), functions (e.g., `sin`, `exp`, `log`, `np.cos`). Example: `A * exp(-B * x) + C`")

    # Equation Input
    eq_string_input = st.text_input("Equation:", key="equation_input")

    # Fit Button
    fit_button = st.button("Perform Fit", key="fit_button")

    if fit_button and eq_string_input:
        # Clear previous results only when button is clicked
        st.session_state.fit_results = None
        st.session_state.final_fig = None

        with st.spinner("Performing iterative fit... Please wait."):
            try:
                # --- Get Equation and Create Function ---
                processed_eq_string, params = validate_and_parse_equation(eq_string_input)
                fit_func = create_fit_function(processed_eq_string, params)

                # --- Access data from session state ---
                x_data = st.session_state.x_data; y_data = st.session_state.y_data
                x_err_safe = st.session_state.x_err_safe; y_err_safe = st.session_state.y_err_safe

                # --- Iterative Fitting ---
                popt_current = None; pcov_current = None; total_err_current = y_err_safe; fit_successful = True
                fit_progress_area = st.empty()
                for i in range(4):
                    fit_num = i + 1; fit_progress_area.info(f"Running Fit {fit_num}/4...")
                    p0 = popt_current if i > 0 else None
                    try:
                        popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=total_err_current, absolute_sigma=True, p0=p0, maxfev=5000 + i*2000, check_finite=False)
                        if pcov_current is None or not np.all(np.isfinite(pcov_current)): st.warning(f"Fit {fit_num} cov matrix non-finite."); fit_successful = False; break
                    except RuntimeError as fit_error: st.error(f"RuntimeError during fit {fit_num}: {fit_error}"); fit_successful = False; break
                    except Exception as fit_error: st.error(f"Unexpected error during fit {fit_num}: {fit_error}"); fit_successful = False; break
                    if i < 3:
                        slopes = numerical_derivative(fit_func, x_data, popt_current)
                        total_err_sq = y_err_safe**2 + (slopes * x_err_safe)**2
                        total_err_current = safeguard_errors(np.sqrt(total_err_sq)) # Use defined function

                fit_progress_area.empty()
                if not fit_successful: st.error("Fit failed during iterations."); st.stop()

                # --- Process Final Results ---
                popt_final = popt_current; pcov_final = pcov_current; total_err_final = total_err_current
                perr_final = np.sqrt(np.diag(pcov_final)); residuals_final = y_data - fit_func(x_data, *popt_final)
                chi_squared = np.sum((residuals_final / total_err_final)**2); dof = len(y_data) - len(popt_final)
                chi_squared_err = np.nan; chi_squared_red = np.nan; red_chi_squared_err = np.nan
                if dof > 0: chi_squared_err = np.sqrt(2.0 * dof); chi_squared_red = chi_squared / dof; red_chi_squared_err = np.sqrt(2.0 / dof)

                # --- Store Results for Display ---
                st.session_state.fit_results = {
                    "eq_string": processed_eq_string, "params": params, "popt": popt_final,
                    "perr": perr_final, "chi2": chi_squared, "chi2_err": chi_squared_err,
                    "dof": dof, "red_chi2": chi_squared_red, "red_chi2_err": red_chi_squared_err,
                    "total_err_final": total_err_final, "residuals_final": residuals_final,
                }
                st.success("Fit completed successfully!")

                # --- Generate Plot Figure ---
                fig = plt.figure(figsize=(10, 9.8)); gs = GridSpec(3, 1, height_ratios=[6, 2, 0.1], hspace=0.08) # Minimal height for row 3
                ax0 = fig.add_subplot(gs[0]); ax0.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
                x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_fit_curve = fit_func(x_fit_curve, *popt_final)
                ax0.plot(x_fit_curve, y_fit_curve, '-', label='Fit Line (Final Iteration)', zorder=10, linewidth=1.5); ax0.set_ylabel(st.session_state.y_axis_label)
                plot_title = f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} with Final Fit"; ax0.set_title(plot_title); ax0.legend(loc='best'); ax0.grid(True, linestyle=':', alpha=0.6); ax0.tick_params(axis='x', labelbottom=False)
                ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)
                ax1 = fig.add_subplot(gs[1], sharex=ax0); ax1.errorbar(x_data, residuals_final, yerr=total_err_final, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
                ax1.axhline(0, color='grey', linestyle='--', linewidth=1); ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals\n(Data - Final Fit)"); ax1.grid(True, linestyle=':', alpha=0.6)
                st.session_state.final_fig = fig # Store figure

            except ValueError as e: st.error(f"Input Error: {e}")
            except SyntaxError as e: st.error(f"Syntax Error: {e}")
            except RuntimeError as e: st.error(f"Fit Error during setup: {e}")
            except TypeError as e: st.error(f"Eval Error: {e}")
            except Exception as e: st.error(f"An unexpected error occurred: {e} ({type(e).__name__})")


# --- Display Results Section ---
if st.session_state.get('fit_results', None):
    st.markdown("---")
    st.subheader("Fit Results")

    # Display Plot
    if st.session_state.final_fig:
        st.pyplot(st.session_state.final_fig)

    # Prepare and Display Results Table Data using st.dataframe
    res = st.session_state.fit_results; table_rows = []
    table_rows.append({"Category": "Equation", "Value": f"y = {res['eq_string']}", "Uncertainty": ""})
    for i, p_name in enumerate(res['params']): table_rows.append({"Category": f"Parameter: {p_name}", "Value": f"{res['popt'][i]:.5g}", "Uncertainty": f"{res['perr'][i]:.3g}"})
    table_rows.append({"Category": "Chi-squared (χ²)", "Value": f"{res['chi2']:.4f}", "Uncertainty": f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ""})
    table_rows.append({"Category": "Degrees of Freedom (DoF)", "Value": f"{res['dof']}", "Uncertainty": ""})
    table_rows.append({"Category": "Reduced χ²/DoF", "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 else "N/A", "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ""})
    results_df = pd.DataFrame(table_rows)
    st.dataframe(results_df.set_index('Category'), use_container_width=True) # Display as a Streamlit DataFrame

    # SVG Download Button
    if st.session_state.final_fig:
        fn = f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit.svg"
        fn = re.sub(r'[^\w\.\-]+', '_', fn).strip('_').lower() or "fit_plot.svg"
        img_buffer = io.BytesIO()
        # Since table is now separate (st.dataframe), bbox_inches might not be needed,
        # but doesn't hurt if future elements outside axes are added.
        st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
        img_buffer.seek(0)
        st.download_button(label="Download Plot as SVG", data=img_buffer, file_name=fn, mime="image/svg+xml")

elif st.session_state.data_loaded and not fit_button:
     pass # Wait for user input

# --- Footer or additional info ---
st.markdown("---")
st.caption("Watermark 'physicsplot.com' added to the main plot.")
