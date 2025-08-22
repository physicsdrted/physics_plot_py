import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import re
import inspect
import io # For StringIO and SVG download
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# --- Configuration ---
st.set_page_config(page_title="Physics Plot", layout="wide")
# <<< Tell Matplotlib to use STIX fonts for better mathtext rendering >>>
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# tell MatPlotLib to use larger fonts for the title and axis labels
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20


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
def format_value_uncertainty(value, uncertainty):
    """
    Formats a value and its uncertainty into a standard scientific string:
    (aaa.a ± bbb.b) × 10^nnn, following specific formatting rules.
    """
    if not np.isfinite(value) or not np.isfinite(uncertainty) or uncertainty <= 0:
        val_str = f"{value:.5g}" if np.isfinite(value) else "N/A"
        unc_str = f"{uncertainty:.3g}" if np.isfinite(uncertainty) else "N/A"
        return f"{val_str} ± {unc_str}"

    exponent_of_unc = np.floor(np.log10(abs(uncertainty)))
    eng_exponent = int(3 * np.floor(exponent_of_unc / 3))+3   
    scaler = 10**(-eng_exponent)
    scaled_value = value * scaler
    scaled_uncertainty = uncertainty * scaler
    log10_scaled_unc = np.floor(np.log10(abs(scaled_uncertainty)))
    decimal_places = max(0, 2 - int(log10_scaled_unc))
    val_fmt = f"{scaled_value:.{decimal_places}f}"
    unc_fmt = f"{scaled_uncertainty:.{decimal_places}f}"

    if eng_exponent != 0:
        return f"$({val_fmt} \\pm {unc_fmt}) \\times 10^{{{eng_exponent}}}$"
    else:
        return f"$({val_fmt} \\pm {unc_fmt})$"

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
    eval_locals_assignments = [f"'{p}': {p}" for p in params]
    eval_locals_str = f"{{'x': x, {', '.join(eval_locals_assignments)}}}"
    func_code = f"""
import numpy as np
def {func_name}(x, {param_str}):
    result = np.nan
    try:
        eval_locals = {eval_locals_str}
        try:
            result = eval('{eq_string}', SAFE_GLOBALS, eval_locals)
        except Exception as e_eval:
            result = np.nan
        if isinstance(result, (np.ndarray, list, tuple)):
            result = np.asarray(result)
            if np.iscomplexobj(result): result = np.real(result)
            result = result.astype(float)
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result)
        else:
             result = np.nan
        if isinstance(result, np.ndarray): result[~np.isfinite(result)] = np.nan
        elif not np.isfinite(result): result = np.nan
        return result
    except Exception as e_outer:
        try: return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
        except: return np.nan
"""
    exec_globals = {'np': np, 'SAFE_GLOBALS': SAFE_GLOBALS}
    local_namespace = {}
    try: exec(func_code, exec_globals, local_namespace)
    except Exception as e_compile: raise SyntaxError(f"Failed to compile function: {e_compile}") from e_compile
    if func_name not in local_namespace: raise RuntimeError(f"Function '{func_name}' not found after exec.")
    return local_namespace[func_name]

def numerical_derivative(func, x, params, h=1e-7):
    """Calculates numerical derivative using central difference."""
    try:
        if params is None or not all(np.isfinite(p) for p in params): return np.zeros_like(x) if isinstance(x, np.ndarray) else 0
        y_plus_h = func(x + h, *params); y_minus_h = func(x - h, *params); deriv = (y_plus_h - y_minus_h) / (2 * h)
        if isinstance(deriv, np.ndarray): deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv): deriv = 0
        return deriv
    except Exception: return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def safeguard_errors(err_array, min_err=1e-9):
     """Replaces non-positive or NaN/Inf errors with a small positive number."""
     safe_err = np.array(err_array, dtype=float); invalid_mask = ~np.isfinite(safe_err) | (safe_err <= 0)
     num_invalid = np.sum(invalid_mask)
     if num_invalid > 0: st.warning(f"Found {num_invalid} invalid values in error array. Replacing with {min_err}."); safe_err[invalid_mask] = min_err
     return safe_err

def format_equation_mathtext(eq_string_orig):
    """Attempts to format the equation string for Matplotlib's mathtext."""
    formatted = eq_string_orig
    formatted = re.sub(r'np\.exp\((.*?)\)', r'e^{\1}', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\bexp\((.*?)\)', r'e^{\1}', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.sqrt\((.*?)\)', r'\\sqrt{\1}', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\bsqrt\((.*?)\)', r'\\sqrt{\1}', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.sin\((.*?)\)', r'\\mathrm{sin}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\bsin\((.*?)\)', r'\\mathrm{sin}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.cos\((.*?)\)', r'\\mathrm{cos}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\bcos\((.*?)\)', r'\\mathrm{cos}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.tan\((.*?)\)', r'\\mathrm{tan}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\btan\((.*?)\)', r'\\mathrm{tan}(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.log10\((.*?)\)', r'\\log_{10}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\blog10\((.*?)\)', r'\\log_{10}(\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.ln\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\bln\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.log\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\blog\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.abs\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\babs\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.absolute\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\babsolute\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = formatted.replace('np.pi', r'\pi').replace('pi', r'\pi')
    formatted = formatted.replace('**', '^').replace('*', r'\cdot ')
    formatted_final = formatted
    try:
        formatted_final = re.sub(r'(?<![a-zA-Z0-9_{])([A-Z])(?![a-zA-Z0-9_}])', r'{\1}', formatted_final)
        formatted_final = re.sub(r'(?<![a-zA-Z0-9_{])(x)(?![a-zA-Z0-9_}])', r'{x}', formatted_final)
    except Exception as e_re:
        st.warning(f"Regex warning during brace insertion: {e_re}. Using previous format.")
    return f'$y = {re.sub(r"s{2,}", " ", formatted_final.replace("$$", "$"))}$'

def perform_the_autofit(initial_guesses):
    """Takes a list of initial guesses, performs the iterative curve fit, and stores results. Returns True on success, False on failure."""
    try:
        with st.spinner("Performing iterative fit..."):
            fit_func, params = st.session_state.fit_func, st.session_state.params
            x_data, y_data = st.session_state.x_data, st.session_state.y_data
            x_err_safe, y_err_safe = st.session_state.x_err_safe, st.session_state.y_err_safe
            popt_current, total_err_current = list(initial_guesses), y_err_safe.copy()

            for i in range(4):
                sigma_to_use = total_err_current.copy()
                if not np.all(np.isfinite(sigma_to_use)) or np.any(sigma_to_use <= 0): sigma_to_use = None
                if not all(np.isfinite(p) for p in popt_current): raise RuntimeError(f"Invalid guess: {popt_current}")
                try:
                    popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=sigma_to_use, p0=popt_current, absolute_sigma=True, maxfev=8000, check_finite=(True, True))
                    if not np.all(np.isfinite(popt_current)): raise RuntimeError("Fit resulted in non-finite parameters.")
                    if pcov_current is None or not np.all(np.isfinite(np.diag(pcov_current))) or np.any(np.diag(pcov_current) < 0):
                        if pcov_current is None: pcov_current = np.full((len(params), len(params)), np.inf)
                except RuntimeError as fit_error: raise RuntimeError(f"Fit failed to converge: {fit_error}") from fit_error
                if i < 3:
                    slopes = numerical_derivative(fit_func, x_data, popt_current)
                    if np.all(np.isfinite(slopes)): total_err_current = safeguard_errors(np.sqrt(y_err_safe**2 + (slopes * x_err_safe)**2))
                    else: total_err_current = y_err_safe.copy()

            popt_final, pcov_final, total_err_final = popt_current, pcov_current, sigma_to_use if sigma_to_use is not None else np.ones_like(y_data)
            diag_pcov = np.diag(pcov_final)
            perr_final = np.sqrt(diag_pcov) if not np.any(diag_pcov < 0) else np.full(len(popt_final), np.nan)
            residuals_final = y_data - fit_func(x_data, *popt_final)
            dof = len(y_data) - len(popt_final)
            chi_squared, chi_squared_err, chi_squared_red, red_chi_squared_err = (np.sum((residuals_final / total_err_final)**2), np.sqrt(2.0*dof), np.sum((residuals_final / total_err_final)**2)/dof, np.sqrt(2.0/dof)) if dof > 0 else (np.nan,)*4
            
            st.session_state.fit_results = {"eq_string": st.session_state.processed_eq_string, "params": params, "popt": popt_final, "perr": perr_final, "chi2": chi_squared, "chi2_err": chi_squared_err, "dof": dof, "red_chi2": chi_squared_red, "red_chi2_err": red_chi_squared_err, "residuals_final": residuals_final, "total_err_final": total_err_final}
            
            equation_label = st.session_state.legend_label_str
            stats_parts = [f"${p_name} = {format_value_uncertainty(popt_final[i], perr_final[i]).replace('$', '')}$" for i, p_name in enumerate(params)]
            stats_parts.append(f"$\\chi^2/DoF = {format_value_uncertainty(chi_squared_red, red_chi_squared_err).replace('$', '')}$")
            stats_label = "\n".join(stats_parts)

            fig = plt.figure(figsize=(10, 9.8)); gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08); ax0 = fig.add_subplot(gs[0])
            ax0.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
            x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_fit_curve = fit_func(x_fit_curve, *popt_final)
            ax0.plot(x_fit_curve, y_fit_curve, '-', label=equation_label, zorder=10, linewidth=1.5); ax0.plot([], [], ' ', label=stats_label)
            
            user_title_str = st.session_state.plot_title_input.strip()
            final_plot_title = user_title_str if user_title_str else f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
            ax0.set_ylabel(st.session_state.y_axis_label); ax0.set_title(final_plot_title); ax0.legend(loc='best', fontsize='large'); ax0.grid(True, linestyle=':', alpha=0.6); ax0.tick_params(axis='x', labelbottom=False)
            ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)

            ax1 = fig.add_subplot(gs[1], sharex=ax0)
            ax1.errorbar(residuals_final, yerr=total_err_final, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
            ax1.axhline(0, color='grey', linestyle='--', linewidth=1); ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals"); ax1.grid(True, linestyle=':', alpha=0.6)
            fig.tight_layout(pad=1.0); st.session_state.final_fig = fig; return True
    except Exception as e:
        st.error(f"Error during fitting process: {e}"); st.session_state.fit_results = None; st.session_state.final_fig = None; return False

# ##################################################################
# ############# BEGINNING OF MODIFIED CODE BLOCK ###################
# ##################################################################
def reset_all_fit_state():
    """Clears all session state related to a specific fit, preparing for new data."""
    st.session_state.fit_results = None
    st.session_state.final_fig = None
    st.session_state.show_guess_stage = False
    st.session_state.processed_eq_string = None
    st.session_state.params = []
    st.session_state.fit_func = None
    st.session_state.legend_label_str = ""
    st.session_state.plot_title_input = ""
    st.session_state.last_eq_input = ""
    keys_to_remove = [k for k in st.session_state if k.startswith("init_guess_")]
    for key in keys_to_remove:
        del st.session_state[key]

def process_and_store_data(data_source):
    """
    Reads data from a source (file or string), validates it, converts to numeric,
    and stores it in the session state. Returns True on success, False on failure.
    """
    try:
        raw_df = pd.read_csv(data_source, header=None, dtype=str)
        if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
            st.error("Invalid data structure: Ensure a header row and at least one data row with 4 columns.")
            return False

        x_label, y_label = str(raw_df.iloc[0, 0]), str(raw_df.iloc[0, 2])
        df = raw_df.iloc[1:].copy()
        df.columns = ['x', 'x_err', 'y', 'y_err']
        
        for col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.isnull().any():
                first_bad_index = numeric_col.index[numeric_col.isnull()][0] + 2
                st.error(f"Column '{col}' contains non-numeric data near row {first_bad_index}. Please check your data.")
                return False
            df[col] = numeric_col
        
        st.session_state.x_data = df['x'].to_numpy()
        st.session_state.y_data = df['y'].to_numpy()
        st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy()))
        st.session_state.y_err_safe = safeguard_errors(np.abs(df['y_err'].to_numpy()))
        st.session_state.x_axis_label, st.session_state.y_axis_label = x_label, y_label
        st.session_state.df_head = df.head(10)
        st.session_state.data_loaded = True
        st.success("Data loaded successfully!")
        return True

    except Exception as e:
        st.error(f"An unexpected error occurred while processing the data: {e}")
        return False
# ##################################################################
# ############### END OF MODIFIED CODE BLOCK #######################
# ##################################################################

# --- Main App Logic ---
st.image("https://raw.githubusercontent.com/physicsdrted/physics_plot_py/refs/heads/main/logo.png", width=200)

if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'show_guess_stage' not in st.session_state: st.session_state.show_guess_stage = False
if 'processed_file_key' not in st.session_state: st.session_state.processed_file_key = None

# --- Data Input Section ---
st.subheader("Step 1: Provide Data")
st.download_button(label="Download Example CSV", data=pd.DataFrame(
    [[0.0,0.001,0.2598,0.001],[0.05,0.001,0.3521,0.001],[0.1,0.001,0.4176,0.001],[0.15,0.001,0.4593,0.001],[0.2,0.001,0.4768,0.001]],
    columns=['time (s)','err_t','height (m)','err_h']).to_csv(index=False),
    file_name="example_data.csv", mime="text/csv"
)

input_method = st.radio("Choose input method:", ("Upload CSV File", "Paste Data"), horizontal=True)

data_source = None
new_data_key = None

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")
    if uploaded_file:
        data_source = uploaded_file
        new_data_key = f"{uploaded_file.name}_{uploaded_file.size}"

elif input_method == "Paste Data":
    placeholder_data = "time (s),t_err,height (m),h_err\n0.0,0.001,0.2598,0.001\n0.05,0.001,0.3521,0.001\n0.1,0.001,0.4176,0.001"
    pasted_text = st.text_area("Paste comma-separated data:", placeholder=placeholder_data, height=150)
    if st.button("Process Pasted Data"):
        if pasted_text:
            data_source = io.StringIO(pasted_text)
            new_data_key = hash(pasted_text)
        else:
            st.warning("Text area is empty.")

if data_source and new_data_key != st.session_state.get('processed_file_key'):
    reset_all_fit_state()
    if process_and_store_data(data_source):
        st.session_state.processed_file_key = new_data_key
    else:
        st.session_state.data_loaded = False
        st.session_state.processed_file_key = None

# --- Main Application Flow (runs if data is loaded) ---
if st.session_state.data_loaded:
    st.markdown("---")
    st.subheader("Loaded Data Preview")
    st.dataframe(st.session_state.df_head, use_container_width=True)
    
    st.subheader("Initial Data Plot")
    try:
        fig_initial, ax_initial = plt.subplots(figsize=(10, 6))
        ax_initial.errorbar(st.session_state.x_data, st.session_state.y_data, yerr=st.session_state.y_err_safe, xerr=st.session_state.x_err_safe, fmt='o', linestyle='None', capsize=5, label='Data')
        ax_initial.set_xlabel(st.session_state.x_axis_label); ax_initial.set_ylabel(st.session_state.y_axis_label)
        ax_initial.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)")
        ax_initial.grid(True, linestyle=':', alpha=0.7); ax_initial.legend(); plt.tight_layout()
        st.pyplot(fig_initial); plt.close(fig_initial)
    except Exception as e: st.error(f"Error generating initial plot: {e}")

    st.markdown("---")

    if not st.session_state.show_guess_stage and not st.session_state.fit_results:
        st.subheader("Step 2: Enter Fit Details")
        st.markdown("**Instructions:** Use `x` for the independent variable, and single uppercase letters (A-Z) for fit parameters. See README for allowed functions.")
        eq_string_input = st.text_input("Equation:", value=st.session_state.get('last_eq_input', "A * sin(B * x + C) + D"), key="equation_input")
        st.session_state.plot_title_input = st.text_input("Optional Plot Title:", value=st.session_state.get('plot_title_input', ""), key="plot_title_input_widget")
        
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("Set Equation & Try a Manual Fit"):
            if eq_string_input:
                try:
                    processed_eq, params = validate_and_parse_equation(eq_string_input)
                    st.session_state.processed_eq_string, st.session_state.params = processed_eq, params
                    st.session_state.fit_func = create_fit_function(processed_eq, params)
                    st.session_state.legend_label_str = format_equation_mathtext(processed_eq)
                    st.session_state.last_eq_input = eq_string_input
                    st.session_state.show_guess_stage = True
                    st.rerun()
                except Exception as e: st.error(f"Input Error: {e}")

        if b_col2.button("Set Equation & Perform Autofit"):
            if eq_string_input:
                try:
                    processed_eq, params = validate_and_parse_equation(eq_string_input)
                    st.session_state.processed_eq_string, st.session_state.params = processed_eq, params
                    st.session_state.fit_func = create_fit_function(processed_eq, params)
                    st.session_state.legend_label_str = format_equation_mathtext(processed_eq)
                    st.session_state.last_eq_input = eq_string_input
                    if perform_the_autofit([1.0] * len(params)):
                        st.session_state.show_guess_stage = False
                        st.rerun()
                    else:
                        st.warning("Automatic fit failed. Please provide a manual starting fit.")
                        for p in params: st.session_state[f"init_guess_{p}"] = 1.0
                        st.session_state.show_guess_stage = True
                        st.rerun()
                except Exception as e: st.error(f"Input Error: {e}")

    elif st.session_state.show_guess_stage and not st.session_state.fit_results:
        st.subheader("Step 2.5: Manual Fit & Preview")
        st.info(f"Using Equation: y = {st.session_state.processed_eq_string}")
        params, fit_func = st.session_state.params, st.session_state.fit_func
        if not params or not fit_func: st.error("Fit function not available."); st.stop()

        initial_guesses = {}
        cols = st.columns(len(params))
        for i, param in enumerate(params):
            guess_key = f"init_guess_{param}"
            if guess_key not in st.session_state: st.session_state[guess_key] = 1.0
            current_value = st.session_state[guess_key]
            fmt = "%.3e" if current_value != 0 and (abs(current_value) < 0.01 or abs(current_value) > 1000) else "%.3f"
            initial_guesses[param] = cols[i].number_input(f"Param {param}", value=current_value, key=guess_key, step=None, format=fmt)

        st.markdown("---")
        st.write("**Preview with Current Parameter Values:**")
        try:
            current_guess_values = [initial_guesses[p] for p in params]
            x_data, y_data, x_err_safe, y_err_safe = st.session_state.x_data, st.session_state.y_data, st.session_state.x_err_safe, st.session_state.y_err_safe
            residuals_preview = y_data - fit_func(x_data, *current_guess_values)
            slopes_preview = numerical_derivative(fit_func, x_data, current_guess_values)
            total_err_preview_safe = safeguard_errors(np.sqrt(y_err_safe**2 + (slopes_preview * x_err_safe)**2))
            dof_preview = len(x_data) - len(params)
            chi_squared_preview, red_chi_squared_preview, red_chi_squared_err_preview = (np.sum((residuals_preview / total_err_preview_safe)**2), np.sum((residuals_preview / total_err_preview_safe)**2)/dof_preview, np.sqrt(2.0/dof_preview)) if dof_preview > 0 else (np.nan,)*3
            
            st.metric("Manual Fit Reduced χ²/DoF", f"{red_chi_squared_preview:.4f}")

            equation_label, stats_parts = st.session_state.legend_label_str, [f"${p} = {val:.4g}$" for p, val in initial_guesses.items()]
            stats_parts.append(f"$\\chi^2/DoF = {format_value_uncertainty(red_chi_squared_preview, red_chi_squared_err_preview).replace('$', '')}$")
            stats_label = "\n".join(stats_parts)
            
            fig_preview = plt.figure(figsize=(10, 8)); gs_preview = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08); ax0_preview = fig_preview.add_subplot(gs_preview[0])
            ax0_preview.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
            x_preview_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_preview_curve = fit_func(x_preview_curve, *current_guess_values)
            ax0_preview.plot(x_preview_curve, y_preview_curve, 'r--', label=equation_label, zorder=10); ax0_preview.plot([], [], ' ', label=stats_label)
            
            preview_title = st.session_state.plot_title_input.strip() or f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
            ax0_preview.set_ylabel(st.session_state.y_axis_label); ax0_preview.set_title(preview_title); ax0_preview.legend(loc='best', fontsize='large'); ax0_preview.grid(True, linestyle=':', alpha=0.7); ax0_preview.tick_params(axis='x', labelbottom=False)

            ax1_preview = fig_preview.add_subplot(gs_preview[1], sharex=ax0_preview)
            ax1_preview.errorbar(x_data, residuals_preview, yerr=total_err_preview_safe, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
            ax1_preview.axhline(0, color='grey', linestyle='--', linewidth=1); ax1_preview.set_xlabel(st.session_state.x_axis_label); ax1_preview.set_ylabel("Residuals"); ax1_preview.grid(True, linestyle=':', alpha=0.6)
            fig_preview.tight_layout(pad=1.0); st.pyplot(fig_preview); plt.close(fig_preview)
        except Exception as e: st.warning(f"Could not generate preview plot: {e}.")
            
        st.markdown("---")
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("Define New Fit"): st.session_state.show_guess_stage=False; st.rerun()
        if b_col2.button("Perform Autofit"):
            if perform_the_autofit([st.session_state[f"init_guess_{p}"] for p in params]):
                st.session_state.show_guess_stage = False
                st.rerun()

    elif st.session_state.fit_results:
        st.subheader("Step 3: Fit Results")
        res = st.session_state.fit_results
        if st.session_state.final_fig: st.pyplot(st.session_state.final_fig); plt.close(st.session_state.final_fig)
        
        st.markdown("##### Fit Statistics")
        table_rows = [("**Equation**", f"`y = {res['eq_string']}`"), ("**Chi-squared (χ²)**", format_value_uncertainty(res['chi2'], res['chi2_err'])), ("**Degrees of Freedom (DoF)**", f"{res['dof']}")]
        st.markdown("| Category | Value |\n|---:|:---| \n" + "\n".join(f"| {cat} | {val} |" for cat, val in table_rows))
        
        if st.session_state.final_fig:
            try:
                fn = re.sub(r'[^\w\.\-]+', '_', st.session_state.plot_title_input.strip() or f"fit_plot").strip('_').lower() + ".svg"
                img_buffer = io.BytesIO(); st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1); img_buffer.seek(0)
                st.download_button("Download Plot as SVG", img_buffer, file_name=fn, mime="image/svg+xml")
            except Exception as e: st.warning(f"Could not prepare plot for download: {e}")

        if st.button("Define New Fit"): reset_all_fit_state(); st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Updated 5/20/2025")
