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
# <<< Tell Matplotlib to use STIX fonts for better mathtext rendering >>>
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Allowed characters and functions ---
ALLOWED_CHARS = r"^[A-Za-z0-9\s\.\+\-\*\/\(\)\,\_\^]+$"
ALLOWED_NP_FUNCTIONS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan, 'atan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'ln':np.log, 'log10': np.log10, 'sqrt': np.sqrt,
    'pi': np.pi, 'abs': np.abs, 'absolute': np.abs,
}
# Define SAFE_GLOBALS correctly at the top level
SAFE_GLOBALS = {'__builtins__': {}}
SAFE_GLOBALS['np'] = np
SAFE_GLOBALS.update(ALLOWED_NP_FUNCTIONS)

# --- Helper Function Definitions ---

def validate_and_parse_equation(eq_string):
    """Validates equation, finds 'x' and parameters (A-Z)."""
    # (Same as previous working version)
    eq_string = eq_string.strip();
    if not eq_string: raise ValueError("Equation cannot be empty.")
    eq_string = eq_string.replace('^', '**') # Keep internal representation with **
    if not re.match(ALLOWED_CHARS, eq_string): invalid_chars = "".join(sorted(list(set(re.sub(ALLOWED_CHARS, '', eq_string))))); raise ValueError(f"Invalid chars: '{invalid_chars}'.")
    if not re.search(r'\bx\b', eq_string): raise ValueError("Equation must contain 'x'.")
    params = sorted(list(set(re.findall(r'\b([A-Z])\b', eq_string))))
    all_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', eq_string)); allowed_words = set(['x']) | set(params) | set(ALLOWED_NP_FUNCTIONS.keys()) | set(['np']); unknown_words = all_words - allowed_words
    if unknown_words: raise ValueError(f"Unknown/disallowed items: {', '.join(unknown_words)}.")
    if not params: raise ValueError("No fit parameters (A-Z) found.")
    return eq_string, params # Return original string with potential '**'

# Using the create_fit_function version confirmed to work previously
def create_fit_function(eq_string, params):
    """Dynamically creates Python function from validated equation string."""
    func_name = "dynamic_fit_func"; param_str = ', '.join(params)
    eval_locals_assignments = [f"'{p}': {{p}}" for p in params] # Correct f-string
    eval_locals_str = f"{{'x': x, {', '.join(eval_locals_assignments)}}}"
    func_code = f"""
import numpy as np
# SAFE_GLOBALS and eq_string are implicitly captured from exec_globals
def {func_name}(x, {param_str}):
    result = np.nan # Initialize result BEFORE the main try block
    try:
        eval_locals = {eval_locals_str}
        # Use _EQ_STRING and _SAFE_GLOBALS passed via exec_globals
        try:
            result = eval(_EQ_STRING, _SAFE_GLOBALS, eval_locals)
        except Exception as e_eval:
            # print(f"DEBUG: !!! ERROR during eval: {{repr(e_eval)}} ({{type(e_eval).__name__}})") # Safe debug print
            result = np.nan # Assign NaN if eval fails

        # --- Validation/Conversion ---
        if isinstance(result, (np.ndarray, list, tuple)):
            result = np.asarray(result);
            if np.iscomplexobj(result): result = np.real(result)
            result = result.astype(float);
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result)
        elif not isinstance(result, (np.ndarray, float)):
             # print(f"DEBUG: Result type not ndarray/float after checks. Val: {{repr(result)}}") # Safe debug print
             result = np.nan

        if isinstance(result, np.ndarray): result[~np.isfinite(result)] = np.nan
        elif not np.isfinite(result): result = np.nan
        return result
    # --- Error Handling for outer logic ---
    except Exception as e_outer:
        # print(f"DEBUG: !!! ERROR in outer try block of {func_name}: {{repr(e_outer)}}") # Safe debug print
        try: return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
        except: return np.nan
"""
    # Globals needed when compiling the function via exec
    exec_globals = {'np': np, '_SAFE_GLOBALS': SAFE_GLOBALS, '_EQ_STRING': eq_string}
    local_namespace = {}
    try: exec(func_code, exec_globals, local_namespace)
    except Exception as e_compile: raise SyntaxError(f"Failed to compile function: {e_compile} ({type(e_compile).__name__})") from e_compile
    if func_name not in local_namespace: raise RuntimeError(f"Function '{func_name}' not found after exec.")
    return local_namespace[func_name]


def numerical_derivative(func, x, params, h=1e-7):
    """Calculates numerical derivative using central difference."""
    # (Same as before)
    try:
        if params is None or not all(np.isfinite(p) for p in params): st.warning("Invalid params to num_deriv."); return np.zeros_like(x) if isinstance(x, np.ndarray) else 0
        y_plus_h = func(x + h, *params); y_minus_h = func(x - h, *params); deriv = (y_plus_h - y_minus_h) / (2 * h)
        if isinstance(deriv, np.ndarray): deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv): deriv = 0
        return deriv
    except Exception as e: st.warning(f"Error during num derivative: {e}. Return slope=0."); return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def safeguard_errors(err_array, min_err=1e-9):
     """Replaces non-positive or NaN/Inf errors with a small positive number."""
     # (Same as before)
     safe_err = np.array(err_array, dtype=float); invalid_mask = ~np.isfinite(safe_err) | (safe_err <= 0)
     num_invalid = np.sum(invalid_mask)
     if num_invalid > 0: st.warning(f"Found {num_invalid} invalid values in error array. Replacing with {min_err}."); safe_err[invalid_mask] = min_err
     return safe_err

# <<< Function to Format Equation for Mathtext >>>
def format_equation_mathtext(eq_string_orig):
    """Attempts to format the equation string for Matplotlib's mathtext."""
    # Work on a copy
    formatted = eq_string_orig
    # Basic replacements - order can matter
    formatted = re.sub(r'np\.exp\((.*?)\)', r'e^{\\1}', formatted); formatted = re.sub(r'\bexp\((.*?)\)', r'e^{\\1}', formatted)
    formatted = re.sub(r'np\.sqrt\((.*?)\)', r'\\sqrt{\\1}', formatted); formatted = re.sub(r'\bsqrt\((.*?)\)', r'\\sqrt{\\1}', formatted)
    formatted = re.sub(r'np\.sin\((.*?)\)', r'\\mathrm{sin}(\\1)', formatted); formatted = re.sub(r'\bsin\((.*?)\)', r'\\mathrm{sin}(\\1)', formatted)
    formatted = re.sub(r'np\.cos\((.*?)\)', r'\\mathrm{cos}(\\1)', formatted); formatted = re.sub(r'\bcos\((.*?)\)', r'\\mathrm{cos}(\\1)', formatted)
    formatted = re.sub(r'np\.tan\((.*?)\)', r'\\mathrm{tan}(\\1)', formatted); formatted = re.sub(r'\btan\((.*?)\)', r'\\mathrm{tan}(\\1)', formatted)
    formatted = re.sub(r'np\.log10\((.*?)\)', r'\\log_{10}(\\1)', formatted); formatted = re.sub(r'\blog10\((.*?)\)', r'\\log_{10}(\\1)', formatted)
    formatted = re.sub(r'np\.ln\((.*?)\)', r'\\ln(\\1)', formatted); formatted = re.sub(r'\bln\((.*?)\)', r'\\ln(\\1)', formatted)
    formatted = re.sub(r'np\.log\((.*?)\)', r'\\ln(\\1)', formatted); formatted = re.sub(r'\blog\((.*?)\)', r'\\ln(\\1)', formatted)
    formatted = formatted.replace('**', '^'); formatted = formatted.replace('*', r'\cdot '); formatted = formatted.replace('/', r'/')
    formatted = formatted.replace('np.pi', r'\pi'); formatted = formatted.replace('pi', r'\pi')
    formatted = re.sub(r'\b([A-Z])\b', r'{\1}', formatted); formatted = formatted.replace('x', '{x}')
    formatted = f'${formatted}$'; formatted = f'$y = {formatted[1:]}';
    formatted = re.sub(r'\s*\^\s*', '^', formatted); formatted = re.sub(r'\s*\\cdot\s*', r'\\cdot ', formatted); formatted = re.sub(r'\s*([\+\-\/])\s*', r' \1 ', formatted); formatted = formatted.replace('$$', '$')
    return formatted

# <<< NEW Plot Generation Functions >>>
def generate_plot_figure(x_data, y_data, x_err, y_err, fit_func, popt, results):
    """Generates the Matplotlib figure with Data, Fit, and Residuals."""
    fig = plt.figure(figsize=(10, 7)); gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax0 = fig.add_subplot(gs[0]); ax0.errorbar(x_data, y_data, yerr=y_err, xerr=x_err, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_fit_curve = fit_func(x_fit_curve, *popt)
    # <<< Use legend_label from results dict >>>
    ax0.plot(x_fit_curve, y_fit_curve, '-', label=results.get('legend_label', 'Fit Line'), zorder=10, linewidth=1.5) # Use .get for fallback
    ax0.set_ylabel(st.session_state.y_axis_label); ax0.set_title(results['plot_title'])
    ax0.legend(loc='best', fontsize='large'); ax0.grid(True, linestyle=':', alpha=0.6); ax0.tick_params(axis='x', labelbottom=False)
    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)
    ax1 = fig.add_subplot(gs[1], sharex=ax0); ax1.errorbar(x_data, results['residuals_final'], yerr=results['total_err_final'], fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=1); ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals\n(Data - Final Fit)"); ax1.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout(pad=1.0)
    return fig

def generate_combined_figure(x_data, y_data, x_err, y_err, fit_func, popt, results):
    """Generates a new figure including the plot AND the results table using Matplotlib."""
    # This function remains necessary for the combined SVG download.
    fig_comb = plt.figure(figsize=(10, 9.8)); gs_comb = GridSpec(3, 1, height_ratios=[6, 2, 3.5], hspace=0.08, figure=fig_comb)
    # Re-draw Top Plot
    ax0_comb = fig_comb.add_subplot(gs_comb[0]); ax0_comb.errorbar(x_data, y_data, yerr=y_err, xerr=x_err, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_fit_curve = fit_func(x_fit_curve, *popt)
    ax0_comb.plot(x_fit_curve, y_fit_curve, '-', label=results.get('legend_label', 'Fit Line'), zorder=10, linewidth=1.5); ax0_comb.set_ylabel(st.session_state.y_axis_label); ax0_comb.set_title(results['plot_title'])
    ax0_comb.legend(loc='best', fontsize='large'); ax0_comb.grid(True, linestyle=':', alpha=0.6); ax0_comb.tick_params(axis='x', labelbottom=False); ax0_comb.text(0.5, 0.5, 'physicsplot.com', transform=ax0_comb.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)
    # Re-draw Middle Plot (Residuals)
    ax1_comb = fig_comb.add_subplot(gs_comb[1], sharex=ax0_comb); ax1_comb.errorbar(x_data, results['residuals_final'], yerr=results['total_err_final'], fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
    ax1_comb.axhline(0, color='grey', linestyle='--', linewidth=1); ax1_comb.set_xlabel(st.session_state.x_axis_label); ax1_comb.set_ylabel("Residuals\n(Data - Final Fit)"); ax1_comb.grid(True, linestyle=':', alpha=0.6)
    # Add Table to Bottom Axes
    ax_table_comb = fig_comb.add_subplot(gs_comb[2]); ax_table_comb.axis('off')
    res = results; eq_row = ['Equation:', f"y = {res['eq_string'].replace('**','^')}", '']; header_row = ['Parameter', 'Value', 'Uncertainty']
    param_results_rows = [[f"{p_name}", f"{res['popt'][i]:.5g}", f"{res['perr'][i]:.3g}"] for i, p_name in enumerate(res['params'])]
    chi2_err_str = f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ''; redchi2_err_str = f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ''
    chi2_row = ['Chi-squared (χ²):', f"{res['chi2']:.4f}", chi2_err_str]; dof_row = ['Degrees of Freedom (DoF):', f"{res['dof']}", '']; redchi2_row = ['Reduced χ²/DoF:', f"{res['red_chi2']:.4f}" if res['dof'] > 0 else 'N/A', redchi2_err_str]
    table_data = [eq_row] + [header_row] + param_results_rows + [chi2_row, dof_row, redchi2_row]; col_widths = [0.25, 0.45, 0.30]
    the_table = ax_table_comb.table(cellText=table_data, colLabels=None, colWidths=col_widths, cellLoc='center', loc='center')
    the_table.auto_set_font_size(False); the_table.set_fontsize(9.5); the_table.scale(1, 1.4); cells = the_table.get_celld(); num_rows = len(table_data); num_cols = len(col_widths)
    for r in range(num_rows): # Styling loop
        for c in range(num_cols):
            cell = cells[(r, c)]; cell.set_visible(True); cell.set_text_props(ha='center', weight='normal')
            if r == 0: # Eq row
                if c == 0: cell.set_text_props(ha='center', weight='bold')
                if c == 1: cell.set_text_props(ha='left')
                if c == 2: cell.set_visible(False)
            elif r == 1: # Header row
                cell.set_text_props(weight='bold'); cell.set_facecolor('#E0E0E0')
            elif 1 < r <= (1 + len(res['params'])): # Param rows
                if c == 0: cell.set_text_props(ha='right')
            else: # Stat rows
                if c == 0: cell.set_text_props(ha='right')
                is_dof_row = (table_data[r][0].startswith('Degrees of Freedom'))
                if is_dof_row and c == 2: cell.set_visible(False)
    fig_comb.tight_layout(pad=1.0, h_pad=2.0) # Adjust combined layout
    return fig_comb

# --- Main App Logic ---
st.title("Physics Data Plotter and Fitter")
st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")
if 'data_loaded' not in st.session_state: st.session_state.update({'data_loaded':False, 'fit_results':None, 'final_plot_fig':None, 'processed_file_key':None, 'df_head':None, 'x_axis_label':'X', 'y_axis_label':'Y'})
uploaded_file = st.file_uploader("Choose CSV", type="csv", key="fup")
if uploaded_file:
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if current_file_key != st.session_state.get('processed_file_key', None):
        st.info(f"Processing {uploaded_file.name}")
        st.session_state.update({'data_loaded':False,'fit_results':None,'final_plot_fig':None,'df_head':None})
        try:
            raw_df=pd.read_csv(uploaded_file, header=None, dtype=str); df=raw_df.iloc[1:].copy(); df.columns=['x','x_err','y','y_err']
            if df.empty or df.shape[1]!=4: st.error("Bad data structure"); st.stop()
            try: x_label=str(raw_df.iloc[0,0]); y_label=str(raw_df.iloc[0,2])
            except: x_label="X"; y_label="Y"; st.warning("Bad labels")
            conv = {}; fail=False
            for col in df.columns:
                try: num_col=pd.to_numeric(df[col],errors='coerce');
                except: st.error(f"Conv err {col}"); fail=True; break
                if num_col.isnull().any(): idx=num_col.index[num_col.isnull()][0]+2; st.error(f"NaN in {col} near row {idx}"); fail=True; break
                else: conv[col]=pd.to_numeric(df[col])
            if fail: st.stop()
            df=pd.DataFrame(conv)
            st.session_state.update({'x_data':df['x'].to_numpy(),'y_data':df['y'].to_numpy(),'x_err_safe':safeguard_errors(np.abs(df['x_err'].to_numpy())), 'y_err_safe':safeguard_errors(df['y_err'].to_numpy()),'x_axis_label':x_label,'y_axis_label':y_label,'df_head':df.head(10),'data_loaded':True,'processed_file_key':current_file_key})
            st.success("Loaded!")
        except Exception as e: st.error(f"File processing error: {e}"); st.stop()
if st.session_state.data_loaded:
    if st.session_state.df_head is not None: st.markdown("---"); st.subheader("Data Preview"); st.dataframe(st.session_state.df_head,use_container_width=True); st.markdown("---")
    st.subheader("Initial Plot"); fig_i, ax_i = plt.subplots(figsize=(10,6)); ax_i.errorbar(st.session_state.x_data, st.session_state.y_data, yerr=st.session_state.y_err_safe, xerr=st.session_state.x_err_safe, fmt='o', ls='None', capsize=5, label='Data'); ax_i.set_xlabel(st.session_state.x_axis_label); ax_i.set_ylabel(st.session_state.y_axis_label); ax_i.set_title("Raw Data"); ax_i.grid(ls=':'); ax_i.legend(); plt.tight_layout(); st.pyplot(fig_i)

    # --- Show Fitting Controls OR Results ---
    if not st.session_state.get('fit_results', None):
        # --- Fitting Controls ---
        st.markdown("---"); st.subheader("Enter Fit Details")
        # <<< ADDED INSTRUCTIONS AND EXAMPLES >>>
        st.markdown("""
        **Instructions:**
        *   Use `x` for the independent variable.
        *   Use single uppercase letters (A-Z) for fit parameters.
        *   Use standard Python math operators: `+`, `-`, `*`, `/`, `**` (power).
        *   Allowed functions (use directly or with `np.` prefix):
            `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `atan`,
            `sinh`, `cosh`, `tanh`, `exp`, `log` (natural), `ln` (natural), `log10`,
            `sqrt`, `abs`, `absolute`.
        *   Use `pi` for the constant π.

        **Examples:**
        *   Linear: `A * x + B`
        *   Quadratic: `A * x**2 + B * x + C`
        *   Power Law: `A * x**B`
        *   Exponential Decay: `A * exp(-B * x) + C`
        *   Sine Wave: `A * sin(B * x + C) + D`
        *   Square Root: `A * sqrt(x) + B`
        """)
        eq_string_input = st.text_input("Equation:", key="eq_in")
        title_input = st.text_input("Optional Plot Title:", key="title_in")
        fit_button = st.button("Perform Fit", key="fit_b")

        if fit_button and eq_string_input:
            st.session_state.final_plot_fig = None
            with st.spinner("Performing iterative fit..."):
                try: # Outer try
                    processed_eq_string, params = validate_and_parse_equation(eq_string_input)
                    try: legend_label_str = format_equation_mathtext(processed_eq_string)
                    except Exception as fmt_err: st.warning(f"Legend format failed: {fmt_err}."); legend_label_str = f"Fit: y = {processed_eq_string.replace('**','^')}"
                    st.write("Creating fit function...")
                    try: fit_func = create_fit_function(processed_eq_string, params); st.success("Function created.")
                    except Exception as create_err: st.error(f"Function creation fail: {create_err}"); import traceback; st.error(traceback.format_exc()); st.stop()

                    # --- Iterative Fitting Loop --- (Simplified Fit 1 still active)
                    x_data = st.session_state.x_data; y_data = st.session_state.y_data; x_err_safe = st.session_state.x_err_safe; y_err_safe = st.session_state.y_err_safe
                    popt_current = None; pcov_current = None; total_err_current = y_err_safe; fit_successful = True; fit_progress_area = st.empty()
                    for i in range(4):
                        fit_num = i + 1; fit_progress_area.info(f"Running Fit {fit_num}/4...")
                        p0 = popt_current if i > 0 else np.ones(len(params)); sigma_to_use = total_err_current.copy()
                        try: # Inner curve_fit try
                            if i == 0: popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=sigma_to_use, p0=p0, maxfev=5000 + i*2000)
                            else: popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=sigma_to_use, absolute_sigma=True, p0=p0, maxfev=5000 + i*2000, bounds=(-np.inf, np.inf))
                            if pcov_current is None or not np.all(np.isfinite(pcov_current)): st.warning(f"Fit {fit_num} cov matrix non-finite."); # Allow continue
                        except Exception as fit_error: st.error(f"Error fit {fit_num}: {fit_error}"); fit_successful = False; break
                        if i < 3 and fit_successful: slopes = numerical_derivative(fit_func, x_data, popt_current); total_err_sq = y_err_safe**2 + (slopes * x_err_safe)**2; total_err_current = safeguard_errors(np.sqrt(total_err_sq))
                        elif not fit_successful: break
                    fit_progress_area.empty()
                    if not fit_successful or popt_current is None or pcov_current is None or not np.all(np.isfinite(pcov_current)): st.error("Fit failed or invalid covariance."); st.stop()

                    # --- Process Final Results ---
                    popt_final = popt_current; pcov_final = pcov_current; total_err_final = sigma_to_use
                    perr_final = np.sqrt(np.diag(pcov_final)); residuals_final = y_data - fit_func(x_data, *popt_final)
                    chi_squared = np.sum((residuals_final / total_err_final)**2); dof = len(y_data) - len(popt_final)
                    chi_squared_err = np.nan; chi_squared_red = np.nan; red_chi_squared_err = np.nan
                    if dof > 0: chi_squared_err = np.sqrt(2.0 * dof); chi_squared_red = chi_squared / dof; red_chi_squared_err = np.sqrt(2.0 / dof)
                    user_title_str = title_input.strip(); final_plot_title = user_title_str if user_title_str else f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} with Final Fit"
                    st.session_state.fit_results = { "eq_string": processed_eq_string, "params": params, "popt": popt_final, "perr": perr_final, "chi2": chi_squared, "chi2_err": chi_squared_err, "dof": dof, "red_chi2": chi_squared_red, "red_chi2_err": red_chi_squared_err, "total_err_final": total_err_final, "residuals_final": residuals_final, "plot_title": final_plot_title, "legend_label": legend_label_str }
                    st.success("Fit completed!")

                    # --- Generate and Store PLOT-ONLY figure ---
                    st.session_state.final_plot_fig = generate_plot_figure(x_data, y_data, x_err_safe, y_err_safe, fit_func, popt_final, st.session_state.fit_results)
                    st.rerun()

                # --- Outer error handling ---
                except Exception as e_setup: st.error(f"Unexpected error: {e_setup}"); import traceback; st.error(traceback.format_exc())

    else: # Display results
        # --- Display Results Section ---
        st.markdown("---"); st.subheader("Fit Results")
        if st.session_state.final_plot_fig: st.pyplot(st.session_state.final_plot_fig)
        res = st.session_state.fit_results; table_rows = []
        table_rows.append({"Category": "Equation", "Value": f"y = {res['eq_string'].replace('**','^')}", "Uncertainty": ""})
        for i, p_name in enumerate(res['params']): table_rows.append({"Category": f"Parameter: {p_name}", "Value": f"{res['popt'][i]:.5g}", "Uncertainty": f"{res['perr'][i]:.3g}"})
        table_rows.append({"Category": "Chi-squared (χ²)", "Value": f"{res['chi2']:.4f}", "Uncertainty": f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ""})
        table_rows.append({"Category": "Degrees of Freedom (DoF)", "Value": f"{res['dof']}", "Uncertainty": ""})
        table_rows.append({"Category": "Reduced χ²/DoF", "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 else "N/A", "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ""})
        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df.set_index('Category'), use_container_width=True)

        # --- Download Buttons --- *MODIFIED - ONLY ONE BUTTON NOW*
        if st.session_state.final_plot_fig:
            plot_title_for_filename = res.get('plot_title', f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit")
            fn_plot = re.sub(r'[^\w\.\-]+', '_', plot_title_for_filename).strip('_').lower() or "fit_plot"
            fn_plot += "_plot.svg" # Changed filename slightly
            img_buffer_plot = io.BytesIO()
            st.session_state.final_plot_fig.savefig(img_buffer_plot, format='svg', bbox_inches='tight', pad_inches=0.1)
            img_buffer_plot.seek(0)
            st.download_button(
                label="Download Plot as SVG", # Changed label
                data=img_buffer_plot,
                file_name=fn_plot,
                mime="image/svg+xml",
                key="download_plot" # Changed key
            )
        # Removed the second download button and its logic


# --- Footer ---
st.markdown("---")
st.caption("Watermark 'physicsplot.com' added to the main plot.")
