import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import re
import inspect
import io # For SVG download
import copy # For potentially copying figure later if needed (though redraw is cleaner)
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# --- Configuration ---
st.set_page_config(page_title="Physics Plot Fitter", layout="wide")
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# --- Allowed characters and functions ---
ALLOWED_CHARS = r"^[A-Za-z0-9\s\.\+\-\*\/\(\)\,\_\^]+$"
ALLOWED_NP_FUNCTIONS = { 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan, 'atan': np.arctan, 'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh, 'exp': np.exp, 'log': np.log, 'ln':np.log, 'log10': np.log10, 'sqrt': np.sqrt, 'pi': np.pi, 'abs': np.abs, 'absolute': np.abs, }
SAFE_GLOBALS = {'__builtins__': {}}; SAFE_GLOBALS['np'] = np; SAFE_GLOBALS.update(ALLOWED_NP_FUNCTIONS)

# --- Helper Function Definitions ---
# (validate_and_parse_equation, create_fit_function, numerical_derivative, safeguard_errors, format_equation_mathtext)
# ... (Keep ALL these helper functions exactly as defined in the previous working version) ...
def validate_and_parse_equation(eq_string):
    """Validates equation, finds 'x' and parameters (A-Z)."""
    eq_string = eq_string.strip();
    if not eq_string: raise ValueError("Equation cannot be empty.")
    eq_string = eq_string.replace('^', '**')
    if not re.match(ALLOWED_CHARS, eq_string): invalid_chars = "".join(sorted(list(set(re.sub(ALLOWED_CHARS, '', eq_string))))); raise ValueError(f"Invalid chars: '{invalid_chars}'.")
    if not re.search(r'\bx\b', eq_string): raise ValueError("Equation must contain 'x'.")
    params = sorted(list(set(re.findall(r'\b([A-Z])\b', eq_string))))
    all_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', eq_string)); allowed_words = set(['x']) | set(params) | set(ALLOWED_NP_FUNCTIONS.keys()) | set(['np']); unknown_words = all_words - allowed_words
    if unknown_words: raise ValueError(f"Unknown/disallowed items: {', '.join(unknown_words)}.")
    if not params: raise ValueError("No fit parameters (A-Z) found.")
    return eq_string, params

def create_fit_function(eq_string, params):
    """Dynamically creates Python function from validated equation string."""
    func_name = "dynamic_fit_func"; param_str = ', '.join(params)
    eval_locals_assignments = [f("'{p}': {p}") for p in params]; eval_locals_str = f"{{'x': x, {', '.join(eval_locals_assignments)}}}"
    func_code = f"""
import numpy as np
def {func_name}(x, {param_str}):
    result = np.nan
    try:
        eval_locals = {eval_locals_str}
        try: result = eval(_EQ_STRING, _SAFE_GLOBALS, eval_locals)
        except Exception as e_eval: result = np.nan
        if isinstance(result, (np.ndarray, list, tuple)): result = np.asarray(result);
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result)
        elif not isinstance(result, (np.ndarray, float)): result = np.nan
        if isinstance(result, np.ndarray): result[~np.isfinite(result)] = np.nan
        elif not np.isfinite(result): result = np.nan
        return result
    except Exception as e_outer:
        try: return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
        except: return np.nan
"""
    exec_globals = {'np': np, '_SAFE_GLOBALS': SAFE_GLOBALS, '_EQ_STRING': eq_string}
    local_namespace = {}
    try: exec(func_code, exec_globals, local_namespace)
    except Exception as e_compile: raise SyntaxError(f"Compile failed: {e_compile}") from e_compile
    if func_name not in local_namespace: raise RuntimeError(f"Function creation failed.")
    return local_namespace[func_name]

def numerical_derivative(func, x, params, h=1e-7):
    """Calculates numerical derivative using central difference."""
    try:
        if params is None or not all(np.isfinite(p) for p in params): return np.zeros_like(x) if isinstance(x, np.ndarray) else 0
        y_plus_h = func(x + h, *params); y_minus_h = func(x - h, *params); deriv = (y_plus_h - y_minus_h) / (2 * h)
        if isinstance(deriv, np.ndarray): deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv): deriv = 0
        return deriv
    except Exception as e: return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def safeguard_errors(err_array, min_err=1e-9):
     """Replaces non-positive or NaN/Inf errors with a small positive number."""
     safe_err = np.array(err_array, dtype=float); invalid_mask = ~np.isfinite(safe_err) | (safe_err <= 0)
     num_invalid = np.sum(invalid_mask)
     if num_invalid > 0: st.warning(f"Invalid errors replaced ({num_invalid})."); safe_err[invalid_mask] = min_err
     return safe_err

def format_equation_mathtext(eq_string_orig):
    """Attempts to format the equation string for Matplotlib's mathtext."""
    formatted = eq_string_orig;
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

# --- <<< NEW Plot Generation Functions >>> ---

def generate_plot_figure(x_data, y_data, x_err, y_err, fit_func, popt, results):
    """Generates the Matplotlib figure with Data, Fit, and Residuals."""
    fig = plt.figure(figsize=(10, 7)) # Adjusted figsize slightly smaller if table not included
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08) # 2 rows for plot + residuals

    # Top Plot
    ax0 = fig.add_subplot(gs[0])
    ax0.errorbar(x_data, y_data, yerr=y_err, xerr=x_err, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200)
    y_fit_curve = fit_func(x_fit_curve, *popt)
    ax0.plot(x_fit_curve, y_fit_curve, '-', label=results['legend_label'], zorder=10, linewidth=1.5)
    ax0.set_ylabel(st.session_state.y_axis_label) # Get from session state
    ax0.set_title(results['plot_title'])
    ax0.legend(loc='best', fontsize='large')
    ax0.grid(True, linestyle=':', alpha=0.6)
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)

    # Middle Plot (Residuals)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.errorbar(x_data, results['residuals_final'], yerr=results['total_err_final'], fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax1.set_xlabel(st.session_state.x_axis_label) # Get from session state
    ax1.set_ylabel("Residuals\n(Data - Final Fit)")
    ax1.grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout(pad=1.0)
    return fig

def generate_combined_figure(x_data, y_data, x_err, y_err, fit_func, popt, results):
    """Generates a new figure including the plot AND the results table using Matplotlib."""
    fig_comb = plt.figure(figsize=(10, 9.8)) # Taller figure
    # GridSpec with 3 rows: plot, residuals, table space
    gs_comb = GridSpec(3, 1, height_ratios=[6, 2, 3.5], hspace=0.08, figure=fig_comb)

    # --- Re-draw Top Plot ---
    ax0_comb = fig_comb.add_subplot(gs_comb[0])
    ax0_comb.errorbar(x_data, y_data, yerr=y_err, xerr=x_err, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200)
    y_fit_curve = fit_func(x_fit_curve, *popt)
    ax0_comb.plot(x_fit_curve, y_fit_curve, '-', label=results['legend_label'], zorder=10, linewidth=1.5)
    ax0_comb.set_ylabel(st.session_state.y_axis_label)
    ax0_comb.set_title(results['plot_title'])
    ax0_comb.legend(loc='best', fontsize='large')
    ax0_comb.grid(True, linestyle=':', alpha=0.6)
    ax0_comb.tick_params(axis='x', labelbottom=False)
    ax0_comb.text(0.5, 0.5, 'physicsplot.com', transform=ax0_comb.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)

    # --- Re-draw Middle Plot (Residuals) ---
    ax1_comb = fig_comb.add_subplot(gs_comb[1], sharex=ax0_comb)
    ax1_comb.errorbar(x_data, results['residuals_final'], yerr=results['total_err_final'], fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
    ax1_comb.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax1_comb.set_xlabel(st.session_state.x_axis_label)
    ax1_comb.set_ylabel("Residuals\n(Data - Final Fit)")
    ax1_comb.grid(True, linestyle=':', alpha=0.6)

    # --- Add Table to Bottom Axes ---
    ax_table_comb = fig_comb.add_subplot(gs_comb[2])
    ax_table_comb.axis('off') # Hide table axes

    # Prepare table data from results dictionary
    res = results
    eq_row = ['Equation:', f"y = {res['eq_string'].replace('**','^')}", ''] # Show ^ in table eq
    header_row = ['Parameter', 'Value', 'Uncertainty']
    param_results_rows = [[f"{p_name}", f"{res['popt'][i]:.5g}", f"{res['perr'][i]:.3g}"] for i, p_name in enumerate(res['params'])]
    chi2_err_str = f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ''
    redchi2_err_str = f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ''
    chi2_row = ['Chi-squared (χ²):', f"{res['chi2']:.4f}", chi2_err_str]
    dof_row = ['Degrees of Freedom (DoF):', f"{res['dof']}", '']
    redchi2_row = ['Reduced χ²/DoF:', f"{res['red_chi2']:.4f}" if res['dof'] > 0 else 'N/A', redchi2_err_str]
    table_data = [eq_row] + [header_row] + param_results_rows + [chi2_row, dof_row, redchi2_row]

    col_widths = [0.25, 0.45, 0.30]

    # Create and style the table
    the_table = ax_table_comb.table(cellText=table_data, colLabels=None, colWidths=col_widths, cellLoc='center', loc='center')
    the_table.auto_set_font_size(False); the_table.set_fontsize(9.5); the_table.scale(1, 1.4)
    cells = the_table.get_celld(); num_rows = len(table_data); num_cols = len(col_widths)
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
                # Keep 3rd cell visible (except if hidden below)
                is_dof_row = (table_data[r][0].startswith('Degrees of Freedom'))
                if is_dof_row and c == 2: cell.set_visible(False) # Hide only DoF uncertainty cell

    # Adjust layout slightly for the combined figure
    fig_comb.tight_layout(pad=1.0, h_pad=2.0) # Add vertical padding maybe

    return fig_comb

# --- Main App Logic ---
# ... (Title, Description, Session State Init remain the same) ...
st.title("Physics Data Plotter and Fitter")
st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False; st.session_state.x_data = None; st.session_state.y_data = None; st.session_state.x_err_safe = None; st.session_state.y_err_safe = None; st.session_state.x_axis_label = "X"; st.session_state.y_axis_label = "Y"; st.session_state.fit_results = None; st.session_state.final_plot_fig = None; st.session_state.processed_file_key = None; st.session_state.df_head = None # Renamed final_fig

# --- File Uploader & Processing --- (Same as before)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
if uploaded_file is not None:
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if current_file_key != st.session_state.get('processed_file_key', None):
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        st.session_state.data_loaded = False; st.session_state.fit_results = None; st.session_state.final_plot_fig = None; st.session_state.df_head = None
        try:
            raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
            if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4: st.error("Invalid file structure."); st.stop()
            try: x_label = str(raw_df.iloc[0, 0]); y_label = str(raw_df.iloc[0, 2])
            except Exception: x_label = "X (Col 1)"; y_label = "Y (Col 3)"; st.warning("Could not read labels.")
            df = raw_df.iloc[1:].copy()
            if df.empty or df.shape[1] != 4: st.error("No data rows or wrong cols."); st.stop()
            df.columns = ['x', 'x_err', 'y', 'y_err']; converted_cols = {}; conversion_failed = False
            for col in df.columns:
                try: numeric_col = pd.to_numeric(df[col], errors='coerce');
                except Exception as e: st.error(f"Error converting col '{col}': {e}"); conversion_failed = True; break
                if numeric_col.isnull().any(): first_bad_index = numeric_col.index[numeric_col.isnull()][0] + 2; st.error(f"Col '{col}' non-numeric near row {first_bad_index}."); conversion_failed = True; break
                else: converted_cols[col] = pd.to_numeric(df[col])
            if conversion_failed: st.stop()
            df = pd.DataFrame(converted_cols)
            st.session_state.x_data = df['x'].to_numpy(); st.session_state.y_data = df['y'].to_numpy()
            st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy()))
            st.session_state.y_err_safe = safeguard_errors(df['y_err'].to_numpy())
            st.session_state.x_axis_label = x_label; st.session_state.y_axis_label = y_label
            st.session_state.df_head = df.head(10); st.session_state.data_loaded = True; st.session_state.processed_file_key = current_file_key
            st.success("Data loaded!")
        except pd.errors.ParserError as pe: st.error(f"CSV Parsing Error: {pe}."); st.stop()
        except Exception as e: st.error(f"Error processing file: {e}"); st.stop()

# --- Display Data Preview and Initial Plot if data loaded --- (Same as before)
if st.session_state.data_loaded:
    if st.session_state.df_head is not None: st.markdown("---"); st.subheader("Loaded Data Preview"); st.dataframe(st.session_state.df_head, use_container_width=True); st.markdown("---")
    st.subheader("Initial Data Plot");
    try:
        fig_initial, ax_initial = plt.subplots(figsize=(10, 6)); ax_initial.errorbar(st.session_state.x_data, st.session_state.y_data, yerr=st.session_state.y_err_safe, xerr=st.session_state.x_err_safe, fmt='o', linestyle='None', capsize=5, label='Data', zorder=5)
        ax_initial.set_xlabel(st.session_state.x_axis_label); ax_initial.set_ylabel(st.session_state.y_axis_label); ax_initial.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)"); ax_initial.grid(True, linestyle=':', alpha=0.7); ax_initial.legend(); plt.tight_layout(); st.pyplot(fig_initial)
    except Exception as plot_err: st.error(f"Error generating initial plot: {plot_err}")

    # --- Show Fitting Controls OR Results ---
    if not st.session_state.get('fit_results', None):
        # --- Fitting Controls ---
        st.markdown("---"); st.subheader("Enter Fit Details")
        eq_string_input = st.text_input("Equation:", help="Use x, params A-Z, funcs (e.g., sin, exp, log, np.cos). Ex: A * exp(-B * x) + C", key="equation_input")
        title_input = st.text_input("Optional Plot Title:", help="Leave blank for default title.", key="plot_title_input")
        fit_button = st.button("Perform Fit", key="fit_button")

        if fit_button and eq_string_input:
            st.session_state.final_plot_fig = None # Clear previous plot figure
            with st.spinner("Performing iterative fit... Please wait."):
                try:
                    processed_eq_string, params = validate_and_parse_equation(eq_string_input)
                    try: legend_label_str = format_equation_mathtext(processed_eq_string)
                    except Exception as fmt_err: st.warning(f"Legend format failed: {fmt_err}."); legend_label_str = f"Fit: y = {processed_eq_string.replace('**','^')}"
                    st.write("Attempting to create fit function...")
                    try: fit_func = create_fit_function(processed_eq_string, params); st.success("Fit function created.")
                    except Exception as create_err: st.error(f"Function creation failed: {create_err}"); import traceback; st.error(traceback.format_exc()); st.stop()

                    # --- Data Setup & Iterative Fitting Loop --- (Same as before)
                    x_data = st.session_state.x_data; y_data = st.session_state.y_data; x_err_safe = st.session_state.x_err_safe; y_err_safe = st.session_state.y_err_safe
                    popt_current = None; pcov_current = None; total_err_current = y_err_safe; fit_successful = True
                    fit_progress_area = st.empty()
                    for i in range(4):
                        fit_num = i + 1; fit_progress_area.info(f"Running Fit {fit_num}/4...")
                        p0 = popt_current if i > 0 else None; sigma_to_use = total_err_current.copy()
                        try: # Inner curve_fit try
                            if i == 0: popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=sigma_to_use, p0=p0, maxfev=5000 + i*2000)
                            else: popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=sigma_to_use, absolute_sigma=True, p0=p0, maxfev=5000 + i*2000, bounds=(-np.inf, np.inf))
                            if pcov_current is None or not np.all(np.isfinite(pcov_current)): st.warning(f"Fit {fit_num} cov matrix non-finite."); # Allow continue
                        except Exception as fit_error: st.error(f"Error during fit {fit_num}: {fit_error}"); fit_successful = False; break
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
                    st.success("Fit completed successfully!")

                    # --- Generate and Store the PLOT-ONLY figure --- *MODIFIED*
                    st.session_state.final_plot_fig = generate_plot_figure(
                        x_data, y_data, x_err_safe, y_err_safe, fit_func, popt_final, st.session_state.fit_results
                    )

                    st.rerun() # Rerun to display results

                # --- Outer error handling ---
                except Exception as e_setup: st.error(f"Unexpected error: {e_setup}"); import traceback; st.error(traceback.format_exc())

    else: # If data is loaded AND results exist, display them
        # --- Display Results Section --- *MODIFIED*
        st.markdown("---"); st.subheader("Fit Results")

        # Display Plot (Plot Only Figure)
        if st.session_state.final_plot_fig:
            st.pyplot(st.session_state.final_plot_fig) # Display the plot-only figure

        # Display Results Table (st.dataframe)
        res = st.session_state.fit_results; table_rows = []
        table_rows.append({"Category": "Equation", "Value": f"y = {res['eq_string'].replace('**','^')}", "Uncertainty": ""}) # Show ^ in table
        for i, p_name in enumerate(res['params']): table_rows.append({"Category": f"Parameter: {p_name}", "Value": f"{res['popt'][i]:.5g}", "Uncertainty": f"{res['perr'][i]:.3g}"})
        table_rows.append({"Category": "Chi-squared (χ²)", "Value": f"{res['chi2']:.4f}", "Uncertainty": f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ""})
        table_rows.append({"Category": "Degrees of Freedom (DoF)", "Value": f"{res['dof']}", "Uncertainty": ""})
        table_rows.append({"Category": "Reduced χ²/DoF", "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 else "N/A", "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ""})
        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df.set_index('Category'), use_container_width=True)

        # --- Download Buttons --- *MODIFIED*
        col1, col2 = st.columns(2) # Create columns for buttons

        # Button 1: Download Plot Only
        with col1:
            if st.session_state.final_plot_fig:
                plot_title_for_filename = res.get('plot_title', f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit")
                fn_plot = re.sub(r'[^\w\.\-]+', '_', plot_title_for_filename).strip('_').lower() or "fit_plot"
                fn_plot += "_plot_only.svg"
                img_buffer_plot = io.BytesIO()
                st.session_state.final_plot_fig.savefig(img_buffer_plot, format='svg', bbox_inches='tight', pad_inches=0.1)
                img_buffer_plot.seek(0)
                st.download_button(
                    label="Download Plot Only (SVG)",
                    data=img_buffer_plot,
                    file_name=fn_plot,
                    mime="image/svg+xml",
                    key="download_plot_only"
                )

        # Button 2: Download Plot + Table
        with col2:
            # Generate the combined figure on demand when button is clicked
            plot_title_for_filename = res.get('plot_title', f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit")
            fn_combined = re.sub(r'[^\w\.\-]+', '_', plot_title_for_filename).strip('_').lower() or "fit_plot"
            fn_combined += "_with_table.svg"
            try:
                # Need data for re-plotting in combined figure
                x_data = st.session_state.x_data; y_data = st.session_state.y_data
                x_err_safe = st.session_state.x_err_safe; y_err_safe = st.session_state.y_err_safe
                # Need function and parameters
                fit_func = create_fit_function(res['eq_string'], res['params']) # Recreate function if needed
                popt = res['popt']

                fig_combined = generate_combined_figure(x_data, y_data, x_err_safe, y_err_safe, fit_func, popt, res)
                img_buffer_combined = io.BytesIO()
                fig_combined.savefig(img_buffer_combined, format='svg', bbox_inches='tight', pad_inches=0.1)
                img_buffer_combined.seek(0)
                plt.close(fig_combined) # Close the combined figure after saving to buffer

                st.download_button(
                    label="Download Plot + Table (SVG)",
                    data=img_buffer_combined,
                    file_name=fn_combined,
                    mime="image/svg+xml",
                    key="download_combined"
                )
            except Exception as e_comb:
                st.error(f"Error generating combined plot/table SVG: {e_comb}")


# --- Footer ---
st.markdown("---")
st.caption("Watermark 'physicsplot.com' added to the main plot.")
