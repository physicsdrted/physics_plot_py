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

# --- Configuration ---
st.set_page_config(page_title="Physics Plot Fitter", layout="wide")

# --- Allowed characters and functions ---
ALLOWED_CHARS = r"^[A-Za-z0-9\s\.\+\-\*\/\(\)\,\_\^]+$"
ALLOWED_NP_FUNCTIONS = { 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan, 'atan': np.arctan, 'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh, 'exp': np.exp, 'log': np.log, 'ln':np.log, 'log10': np.log10, 'sqrt': np.sqrt, 'pi': np.pi, 'abs': np.abs, 'absolute': np.abs, }
SAFE_GLOBALS = {'__builtins__': {}}; SAFE_GLOBALS['np'] = np; SAFE_GLOBALS.update(ALLOWED_NP_FUNCTIONS)

# --- Helper Function Definitions ---

def validate_and_parse_equation(eq_string):
    """Validates equation, finds 'x' and parameters (A-Z)."""
    # (Code remains the same)
    eq_string = eq_string.strip();
    if not eq_string: raise ValueError("Equation cannot be empty.")
    eq_string = eq_string.replace('^', '**')
    if not re.match(ALLOWED_CHARS, eq_string): invalid_chars = "".join(sorted(list(set(re.sub(ALLOWED_CHARS, '', eq_string))))); raise ValueError(f"Invalid chars: '{invalid_chars}'.")
    if not re.search(r'\bx\b', eq_string): raise ValueError("Equation must contain 'x'.")
    params = sorted(list(set(re.findall(r'\b([A-Z])\b', eq_string))))
    all_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', eq_string)); allowed_words = set(['x']) | set(params) | set(ALLOWED_NP_FUNCTIONS.keys()) | set(['np']); unknown_words = all_words - allowed_words
    if unknown_words: raise ValueError(f"Unknown items: {', '.join(unknown_words)}.")
    if not params: raise ValueError("No fit parameters (A-Z) found.")
    return eq_string, params

def create_fit_function(eq_string, params):
    """Dynamically creates Python function from validated equation string."""
    # (Code remains largely the same - ensure float output)
    func_name = "dynamic_fit_func"; param_str = ', '.join(params)
    exec_globals = {'np': np, 'SAFE_GLOBALS': SAFE_GLOBALS, 'eq_string': eq_string, 'params': params}
    func_code = f"""
import numpy as np
_SAFE_GLOBALS = SAFE_GLOBALS; _EQ_STRING = eq_string; _PARAMS = params
def {func_name}(x, {param_str}):
    try:
        eval_locals = {{'x': x}}; local_args = locals()
        for p_name in _PARAMS: eval_locals[p_name] = local_args[p_name]
        result = eval(_EQ_STRING, _SAFE_GLOBALS, eval_locals)
        # Ensure result is always float or array of floats for curve_fit
        if isinstance(result, (np.ndarray, list, tuple)):
            result = np.asarray(result)
            if np.iscomplexobj(result): result = np.real(result)
            result = result.astype(float) # Force float
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result) # Force float
        else: raise TypeError(f"Equation returned non-numeric type: {{type(result)}}")
        # Replace any NaNs/Infs from calculation with np.nan before returning
        if isinstance(result, np.ndarray): result[~np.isfinite(result)] = np.nan
        elif not np.isfinite(result): result = np.nan
        return result
    except Exception as e: return np.nan # Return NaN on any error during eval
"""
    local_namespace = {};
    try: exec(func_code, exec_globals, local_namespace)
    except Exception as e: raise SyntaxError(f"Failed to compile function: {e} ({type(e).__name__})") from e
    if func_name not in local_namespace: raise RuntimeError(f"Failed to create function '{func_name}' via exec.")
    return local_namespace[func_name]


def numerical_derivative(func, x, params, h=1e-7):
    """Calculates numerical derivative using central difference."""
    # (Code remains the same)
    try:
        if params is None or not all(np.isfinite(p) for p in params): st.warning("Invalid params to num_deriv."); return np.zeros_like(x) if isinstance(x, np.ndarray) else 0
        y_plus_h = func(x + h, *params); y_minus_h = func(x - h, *params); deriv = (y_plus_h - y_minus_h) / (2 * h)
        if isinstance(deriv, np.ndarray): deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv): deriv = 0
        return deriv
    except Exception as e: st.warning(f"Error during num derivative: {e}. Return slope=0."); return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def safeguard_errors(err_array, min_err=1e-9): # <<< INCREASED min_err
     """Replaces non-positive or NaN/Inf errors with a small positive number."""
     safe_err = np.array(err_array, dtype=float); invalid_mask = ~np.isfinite(safe_err) | (safe_err <= 0)
     num_invalid = np.sum(invalid_mask)
     if num_invalid > 0: st.warning(f"Found {num_invalid} invalid values in error array. Replacing with {min_err}."); safe_err[invalid_mask] = min_err
     return safe_err

# --- Main App Logic ---
st.title("Physics Data Plotter and Fitter")
st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")

# --- Session State Initialization --- (Same as before)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False; st.session_state.x_data = None; st.session_state.y_data = None
    st.session_state.x_err_safe = None; st.session_state.y_err_safe = None; st.session_state.x_axis_label = "X"
    st.session_state.y_axis_label = "Y"; st.session_state.fit_results = None; st.session_state.final_fig = None
    st.session_state.processed_file_key = None; st.session_state.df_head = None

# --- File Uploader --- (Same as before)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

if uploaded_file is not None:
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if current_file_key != st.session_state.get('processed_file_key', None):
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        st.session_state.data_loaded = False; st.session_state.fit_results = None; st.session_state.final_fig = None; st.session_state.df_head = None
        try:
            # Read as string first
            raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
            if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4: st.error("Invalid file structure."); st.stop()
            try: x_label = str(raw_df.iloc[0, 0]); y_label = str(raw_df.iloc[0, 2])
            except Exception: x_label = "X (Col 1)"; y_label = "Y (Col 3)"; st.warning("Could not read labels.")
            df = raw_df.iloc[1:].copy()
            if df.empty or df.shape[1] != 4: st.error("No data rows or wrong cols."); st.stop()
            df.columns = ['x', 'x_err', 'y', 'y_err']
            # Convert to numeric carefully
            converted_cols = {}; conversion_failed = False
            for col in df.columns:
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.isnull().any():
                        first_bad_index = numeric_col.index[numeric_col.isnull()][0] + 2
                        st.error(f"Col '{col}' has non-numeric/empty near row {first_bad_index}."); conversion_failed = True; break
                    else: converted_cols[col] = pd.to_numeric(df[col])
                except Exception as e: st.error(f"Error converting col '{col}': {e}"); conversion_failed = True; break
            if conversion_failed: st.stop()
            df = pd.DataFrame(converted_cols)

            # --- Store processed data and preview ---
            st.session_state.x_data = df['x'].to_numpy(); st.session_state.y_data = df['y'].to_numpy()
            st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy()))
            st.session_state.y_err_safe = safeguard_errors(df['y_err'].to_numpy()) # <<< Uses updated min_err
            st.session_state.x_axis_label = x_label; st.session_state.y_axis_label = y_label
            st.session_state.df_head = df.head(10)
            st.session_state.data_loaded = True; st.session_state.processed_file_key = current_file_key
            st.success("Data loaded and validated successfully!")
            # No st.rerun() here

        except pd.errors.ParserError as pe: st.error(f"CSV Parsing Error: {pe}."); st.stop()
        except Exception as e: st.error(f"Error processing file: {e}"); st.stop()

# --- Display Data Preview and Initial Plot if data loaded ---
if st.session_state.data_loaded:
    # Display Table Preview
    if st.session_state.df_head is not None:
        st.markdown("---"); st.subheader("Loaded Data Preview (First 10 Rows)")
        st.dataframe(st.session_state.df_head, use_container_width=True)
        st.markdown("---")
    # Display Initial Plot
    st.subheader("Initial Data Plot")
    try:
        fig_initial, ax_initial = plt.subplots(figsize=(10, 6))
        ax_initial.errorbar(st.session_state.x_data, st.session_state.y_data, yerr=st.session_state.y_err_safe, xerr=st.session_state.x_err_safe, fmt='o', linestyle='None', capsize=5, label='Data', zorder=5)
        ax_initial.set_xlabel(st.session_state.x_axis_label); ax_initial.set_ylabel(st.session_state.y_axis_label); ax_initial.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)")
        ax_initial.grid(True, linestyle=':', alpha=0.7); ax_initial.legend(); plt.tight_layout(); st.pyplot(fig_initial)
    except Exception as plot_err: st.error(f"Error generating initial plot: {plot_err}")

    # --- Show Fitting Controls OR Results ---
    if not st.session_state.get('fit_results', None):
        # --- Fitting Controls ---
        st.markdown("---"); st.subheader("Enter Fit Equation"); st.markdown("Use `x`, params (A-Z), funcs (e.g., `sin`, `exp`). Ex: `A * exp(-B * x) + C`")
        eq_string_input = st.text_input("Equation:", key="equation_input")
        fit_button = st.button("Perform Fit", key="fit_button")

        if fit_button and eq_string_input:
            st.session_state.final_fig = None
            with st.spinner("Performing iterative fit... Please wait."):
                try:
                    processed_eq_string, params = validate_and_parse_equation(eq_string_input)
                    fit_func = create_fit_function(processed_eq_string, params)
                    x_data = st.session_state.x_data; y_data = st.session_state.y_data; x_err_safe = st.session_state.x_err_safe; y_err_safe = st.session_state.y_err_safe
                    popt_current = None; pcov_current = None; total_err_current = y_err_safe; fit_successful = True
                    fit_progress_area = st.empty()
                    for i in range(4): # Fit loop
                        fit_num = i + 1; fit_progress_area.info(f"Running Fit {fit_num}/4...")
                        p0 = popt_current if i > 0 else None

                        # <<< Add Debug Print Before curve_fit >>>
                        st.write(f"Fit {fit_num} Inputs:")
                        st.write(f"  x_data[:5]: {x_data[:5]}")
                        st.write(f"  y_data[:5]: {y_data[:5]}")
                        st.write(f"  sigma[:5]: {total_err_current[:5]}")
                        st.write(f"  sigma min/max: {np.min(total_err_current)}, {np.max(total_err_current)}")
                        st.write(f"  p0: {p0}")

                        try:
                            popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=total_err_current, absolute_sigma=True, p0=p0, maxfev=5000 + i*2000) # Removed check_finite
                            # Check covariance after successful fit
                            if pcov_current is None or not np.all(np.isfinite(pcov_current)):
                                st.warning(f"Fit {fit_num} succeeded (popt={popt_current}) but cov matrix non-finite. Stopping iteration.")
                                fit_successful = False; break
                        except ValueError as fit_error: st.error(f"ValueError during fit {fit_num}: {fit_error}"); fit_successful = False; break
                        except RuntimeError as fit_error: st.error(f"RuntimeError during fit {fit_num} (failed convergence?): {fit_error}"); fit_successful = False; break
                        except Exception as fit_error: st.error(f"Unexpected error during fit {fit_num}: {fit_error}"); fit_successful = False; break

                        # Recalculate error if not last fit
                        if i < 3 and fit_successful: # <<< Check fit_successful here
                            slopes = numerical_derivative(fit_func, x_data, popt_current)
                            total_err_sq = y_err_safe**2 + (slopes * x_err_safe)**2
                            total_err_current = safeguard_errors(np.sqrt(total_err_sq))

                    fit_progress_area.empty()
                    if not fit_successful: st.error("Fit failed during iterations."); st.stop()

                    # --- Process Final Results --- (Same as before)
                    popt_final = popt_current; pcov_final = pcov_current; total_err_final = total_err_current
                    perr_final = np.sqrt(np.diag(pcov_final)); residuals_final = y_data - fit_func(x_data, *popt_final)
                    chi_squared = np.sum((residuals_final / total_err_final)**2); dof = len(y_data) - len(popt_final)
                    chi_squared_err = np.nan; chi_squared_red = np.nan; red_chi_squared_err = np.nan
                    if dof > 0: chi_squared_err = np.sqrt(2.0 * dof); chi_squared_red = chi_squared / dof; red_chi_squared_err = np.sqrt(2.0 / dof)
                    st.session_state.fit_results = { "eq_string": processed_eq_string, "params": params, "popt": popt_final, "perr": perr_final, "chi2": chi_squared, "chi2_err": chi_squared_err, "dof": dof, "red_chi2": chi_squared_red, "red_chi2_err": red_chi_squared_err, "total_err_final": total_err_final, "residuals_final": residuals_final, }
                    st.success("Fit completed successfully!")

                    # --- Generate Final Plot Figure --- (Same as before)
                    fig = plt.figure(figsize=(10, 9.8)); gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
                    ax0 = fig.add_subplot(gs[0]); ax0.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
                    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_fit_curve = fit_func(x_fit_curve, *popt_final)
                    ax0.plot(x_fit_curve, y_fit_curve, '-', label='Fit Line (Final Iteration)', zorder=10, linewidth=1.5); ax0.set_ylabel(st.session_state.y_axis_label); plot_title = f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} with Final Fit"; ax0.set_title(plot_title); ax0.legend(loc='best'); ax0.grid(True, linestyle=':', alpha=0.6); ax0.tick_params(axis='x', labelbottom=False)
                    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)
                    ax1 = fig.add_subplot(gs[1], sharex=ax0); ax1.errorbar(x_data, residuals_final, yerr=total_err_final, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
                    ax1.axhline(0, color='grey', linestyle='--', linewidth=1); ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals\n(Data - Final Fit)"); ax1.grid(True, linestyle=':', alpha=0.6)
                    fig.tight_layout(pad=1.0); st.session_state.final_fig = fig
                    st.rerun() # Rerun to display results

                # --- Error Handling for outer try block --- (Same as before)
                except ValueError as e: st.error(f"Input Error: {e}")
                except SyntaxError as e: st.error(f"Syntax Error: {e}")
                except RuntimeError as e: st.error(f"Fit Error during setup: {e}")
                except TypeError as e: st.error(f"Eval Error: {e}")
                except Exception as e: st.error(f"An unexpected error occurred: {e} ({type(e).__name__})")

    else: # If data is loaded AND results exist, display results
        # --- Display Results Section --- (Same as before)
        st.markdown("---"); st.subheader("Fit Results")
        if st.session_state.final_fig: st.pyplot(st.session_state.final_fig)
        res = st.session_state.fit_results; table_rows = []
        table_rows.append({"Category": "Equation", "Value": f"y = {res['eq_string']}", "Uncertainty": ""})
        for i, p_name in enumerate(res['params']): table_rows.append({"Category": f"Parameter: {p_name}", "Value": f"{res['popt'][i]:.5g}", "Uncertainty": f"{res['perr'][i]:.3g}"})
        table_rows.append({"Category": "Chi-squared (χ²)", "Value": f"{res['chi2']:.4f}", "Uncertainty": f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ""})
        table_rows.append({"Category": "Degrees of Freedom (DoF)", "Value": f"{res['dof']}", "Uncertainty": ""})
        table_rows.append({"Category": "Reduced χ²/DoF", "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 else "N/A", "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ""})
        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df.set_index('Category'), use_container_width=True)
        if st.session_state.final_fig: # Download button
            fn = f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit.svg"; fn = re.sub(r'[^\w\.\-]+', '_', fn).strip('_').lower() or "fit_plot.svg"
            img_buffer = io.BytesIO(); st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1); img_buffer.seek(0)
            st.download_button(label="Download Plot as SVG", data=img_buffer, file_name=fn, mime="image/svg+xml")

# --- Footer --- (Same as before)
st.markdown("---")
st.caption("Watermark 'physicsplot.com' added to the main plot.")
