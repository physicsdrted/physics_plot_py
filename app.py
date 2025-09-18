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
st.set_page_config(page_title="Physics Plot", layout="wide")
# <<< Tell Matplotlib to use STIX fonts for better mathtext rendering >>>
FIG_WIDTH = 8
FIG_HEIGHT = 9
plt.rcParams['figure.figsize'] = [FIG_WIDTH, FIG_HEIGHT]
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
        # Fallback for invalid data: return value and uncertainty separately.
        val_str = f"{value:.5g}" if np.isfinite(value) else "N/A"
        unc_str = f"{uncertainty:.3g}" if np.isfinite(uncertainty) else "N/A"
        return f"{val_str} ± {unc_str}"

    # 1. Determine the engineering exponent (multiple of 3) for the uncertainty
    exponent_of_unc = np.floor(np.log10(abs(uncertainty)))
    eng_exponent = int(3 * np.floor(exponent_of_unc / 3))+3

    # 2. Scale the value and uncertainty by the chosen exponent
    scaler = 10**(-eng_exponent)
    scaled_value = value * scaler
    scaled_uncertainty = uncertainty * scaler

    # 3. Determine the number of decimal places needed to show 3 significant
    #    figures for the scaled uncertainty.
    #    Formula: num_decimals = 2 - floor(log10(scaled_uncertainty))
    log10_scaled_unc = np.floor(np.log10(abs(scaled_uncertainty)))
    decimal_places = max(0, 2 - int(log10_scaled_unc))

    # 4. Format the scaled numbers to the calculated number of decimal places
    val_fmt = f"{scaled_value:.{decimal_places}f}"
    unc_fmt = f"{scaled_uncertainty:.{decimal_places}f}"

    # 5. Assemble the final mathtext string
    if eng_exponent != 0:
        return f"$({val_fmt} \\pm {unc_fmt}) \\times 10^{{{eng_exponent}}}$"
    else:
        return f"$({val_fmt} \\pm {unc_fmt})$"

def validate_and_parse_equation(eq_string):
    """Validates equation, finds 'x' and parameters (A-Z)."""
    eq_string = eq_string.strip()
    # Gracefully handle "y =" at the start of the equation
    eq_string = re.sub(r'^\s*y\s*=\s*', '', eq_string, flags=re.IGNORECASE).strip()
    
    if not eq_string: raise ValueError("Equation cannot be empty.")
    eq_string = eq_string.replace('^', '**') # Keep internal representation with **
    if not re.match(ALLOWED_CHARS, eq_string): invalid_chars = "".join(sorted(list(set(re.sub(ALLOWED_CHARS, '', eq_string))))); raise ValueError(f"Invalid chars: '{invalid_chars}'.")
    if not re.search(r'\bx\b', eq_string): raise ValueError("Equation must contain 'x'.")
    params = sorted(list(set(re.findall(r'\b([A-Z])\b', eq_string))))
    all_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', eq_string)); allowed_words = set(['x']) | set(params) | set(ALLOWED_NP_FUNCTIONS.keys()) | set(['np']); unknown_words = all_words - allowed_words
    if unknown_words: raise ValueError(f"Unknown/disallowed items: {', '.join(unknown_words)}.")
    if not params: raise ValueError("No fit parameters (A-Z) found.")
    return eq_string, params # Return original string with potential '**'

def create_fit_function(eq_string, params):
    """Dynamically creates Python function from validated equation string."""
    func_name = "dynamic_fit_func"; param_str = ', '.join(params)
    eval_locals_assignments = [f"'{p}': {p}" for p in params]
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
            # print(f"DEBUG: !!! ERROR during eval: {{repr(e_eval)}} ({{type(e_eval).__name__}})")
            result = np.nan # Assign NaN if eval fails

        # --- Validation/Conversion ---
        if isinstance(result, (np.ndarray, list, tuple)):
            result = np.asarray(result);
            if np.iscomplexobj(result): result = np.real(result)
            result = result.astype(float);
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result)
        elif not isinstance(result, (np.ndarray, float)):
             # print(f"DEBUG: Result type not ndarray/float after checks. Val: {{repr(result)}}")
             result = np.nan

        if isinstance(result, np.ndarray): result[~np.isfinite(result)] = np.nan
        elif not np.isfinite(result): result = np.nan
        return result
    # --- Error Handling for outer logic ---
    except Exception as e_outer:
        # print(f"DEBUG: !!! ERROR in outer try block of {func_name}: {{repr(e_outer)}}")
        try: return np.nan * np.ones_like(x) if isinstance(x, np.ndarray) else np.nan
        except: return np.nan
"""
    exec_globals = {'np': np, '_SAFE_GLOBALS': SAFE_GLOBALS, '_EQ_STRING': eq_string}
    local_namespace = {}
    try: exec(func_code, exec_globals, local_namespace)
    except Exception as e_compile: raise SyntaxError(f"Failed to compile function: {e_compile} ({type(e_compile).__name__})") from e_compile
    if func_name not in local_namespace: raise RuntimeError(f"Function '{func_name}' not found after exec.")
    return local_namespace[func_name]

def numerical_derivative(func, x, params, h=1e-7):
    """Calculates numerical derivative using central difference."""
    try:
        if params is None or not all(np.isfinite(p) for p in params): st.warning("Invalid params to num_deriv."); return np.zeros_like(x) if isinstance(x, np.ndarray) else 0
        y_plus_h = func(x + h, *params); y_minus_h = func(x - h, *params); deriv = (y_plus_h - y_minus_h) / (2 * h)
        if isinstance(deriv, np.ndarray): deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv): deriv = 0
        return deriv
    except Exception as e: st.warning(f"Error during num derivative: {e}. Return slope=0."); return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

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
    formatted = formatted.replace('np.pi', r'\pi')
    formatted = formatted.replace('pi', r'\pi')
    formatted = formatted.replace('**', '^')
    formatted = formatted.replace('*', r'\cdot ')
    formatted = formatted.replace('/', r'/')
    formatted_final = formatted
    try:
        formatted_final = re.sub(r'(?<![a-zA-Z0-9_{])([A-Z])(?![a-zA-Z0-9_}])', r'{\1}', formatted_final)
        formatted_final = re.sub(r'(?<![a-zA-Z0-9_{])(x)(?![a-zA-Z0-9_}])', r'{x}', formatted_final)
    except Exception as e_re:
        st.warning(f"Regex warning during brace insertion: {e_re}. Using previous format.")
    formatted_final = f'$y = {formatted_final}$'
    formatted_final = formatted_final.replace('$$', '$')
    formatted_final = re.sub(r'\s{2,}', ' ', formatted_final)
    return formatted_final

def recreate_final_figure(xlim=None, ylim=None):
    """
    Regenerates the final plot figure using data stored in session_state.
    Allows for custom axis limits to be applied.
    """
    res = st.session_state.fit_results
    fit_func = st.session_state.fit_func
    x_data = st.session_state.x_data
    y_data = st.session_state.y_data
    x_err_safe = st.session_state.x_err_safe
    y_err_safe = st.session_state.y_err_safe

    # Recreate the labels
    equation_label = st.session_state.legend_label_str
    stats_parts = []
    for i, p_name in enumerate(res['params']):
        param_str = format_value_uncertainty(res['popt'][i], res['perr'][i])
        stats_parts.append(f"${p_name} = {param_str.replace('$', '')}$")
    red_chi2_str = format_value_uncertainty(res['red_chi2'], res['red_chi2_err'])
    stats_parts.append(f"$\\chi^2/DoF = {red_chi2_str.replace('$', '')}$")
    stats_label = "\n".join(stats_parts)

    fig = plt.figure()
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax0.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
    
    # Generate fit curve based on data range or specified xlim
    x_min_plot = xlim[0] if xlim else np.min(x_data)
    x_max_plot = xlim[1] if xlim else np.max(x_data)
    x_fit_curve = np.linspace(x_min_plot, x_max_plot, 400)
    y_fit_curve = fit_func(x_fit_curve, *res['popt'])
    
    ax0.plot(x_fit_curve, y_fit_curve, '-', label=equation_label, zorder=10, linewidth=1.5)
    ax0.plot([], [], ' ', label=stats_label)

    user_title_str = st.session_state.plot_title_input.strip()
    final_plot_title = user_title_str if user_title_str else f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
    ax0.set_ylabel(st.session_state.y_axis_label)
    ax0.set_title(final_plot_title)
    ax0.legend(loc='best', fontsize='large')
    ax0.grid(True, linestyle=':', alpha=0.6)
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)

    # Set custom axis limits if provided
    if xlim: ax0.set_xlim(xlim)
    if ylim: ax0.set_ylim(ylim)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.errorbar(x_data, res['residuals_final'], yerr=res['total_err_final'], fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax1.set_xlabel(st.session_state.x_axis_label)
    ax1.set_ylabel("Residuals")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Ensure residual plot ylim is symmetrical and reasonable
    max_resid_err = np.max(np.abs(res['residuals_final']) + res['total_err_final'])
    ax1.set_ylim(-max_resid_err * 1.1, max_resid_err * 1.1)
    
    fig.tight_layout(pad=1.0)
    
    return fig, ax0.get_xlim(), ax0.get_ylim()

def perform_the_autofit(initial_guesses):
    """
    Takes a list of initial guesses, performs the iterative curve fit,
    and stores results in the session state.
    Returns True on success, False on failure.
    """
    try:
        with st.spinner("Performing iterative fit... Please wait."):
            fit_func = st.session_state.fit_func
            params = st.session_state.params
            x_data = st.session_state.x_data
            y_data = st.session_state.y_data
            x_err_safe = st.session_state.x_err_safe
            y_err_safe = st.session_state.y_err_safe

            popt_current = list(initial_guesses)
            pcov_current = None
            total_err_current = y_err_safe.copy()

            max_iterations = 4
            for i in range(max_iterations):
                sigma_to_use = total_err_current.copy()

                if not np.all(np.isfinite(sigma_to_use)) or np.any(sigma_to_use <= 0):
                    sigma_to_use = None
                if not all(np.isfinite(p) for p in popt_current):
                    raise RuntimeError(f"Invalid initial parameter guess detected ({popt_current}).")

                try:
                    popt_current, pcov_current = curve_fit(
                        fit_func, x_data, y_data, sigma=sigma_to_use, p0=popt_current,
                        absolute_sigma=True, maxfev=8000, check_finite=(True, True)
                    )
                    if not np.all(np.isfinite(popt_current)):
                        raise RuntimeError("Fit resulted in non-finite parameters.")
                    if pcov_current is None or not np.all(np.isfinite(np.diag(pcov_current))) or np.any(np.diag(pcov_current) < 0):
                        if pcov_current is None: pcov_current = np.full((len(params), len(params)), np.inf)

                except RuntimeError as fit_error:
                    raise RuntimeError(f"Fit failed to converge: {fit_error}") from fit_error

                if i < max_iterations - 1:
                    slopes = numerical_derivative(fit_func, x_data, popt_current)
                    if np.all(np.isfinite(slopes)):
                        total_err_sq = y_err_safe**2 + (slopes * x_err_safe)**2
                        total_err_current = safeguard_errors(np.sqrt(total_err_sq))
                    else:
                        total_err_current = y_err_safe.copy()

            popt_final = popt_current
            pcov_final = pcov_current
            total_err_final = sigma_to_use if sigma_to_use is not None else np.ones_like(y_data)
            diag_pcov = np.diag(pcov_final)
            perr_final = np.sqrt(diag_pcov) if not np.any(diag_pcov < 0) else np.full(len(popt_final), np.nan)
            residuals_final = y_data - fit_func(x_data, *popt_final)
            dof = len(y_data) - len(popt_final)
            if dof > 0:
                chi_squared = np.sum((residuals_final / total_err_final)**2)
                chi_squared_err = np.sqrt(2.0 * dof)
                chi_squared_red = chi_squared / dof
                red_chi_squared_err = np.sqrt(2.0 / dof)
            else:
                chi_squared, chi_squared_err, chi_squared_red, red_chi_squared_err = np.nan, np.nan, np.nan, np.nan

            st.session_state.fit_results = {
                "eq_string": st.session_state.processed_eq_string, "params": params, "popt": popt_final, "perr": perr_final,
                "chi2": chi_squared, "chi2_err": chi_squared_err, "dof": dof, "red_chi2": chi_squared_red,
                "red_chi2_err": red_chi_squared_err, "residuals_final": residuals_final, "total_err_final": total_err_final
            }

            # Generate the initial plot and store it along with auto-calculated axis limits
            fig, xlim_auto, ylim_auto = recreate_final_figure()
            st.session_state.final_fig = fig
            st.session_state.auto_limits = {'x': xlim_auto, 'y': ylim_auto}
            
            # Initialize states for the axis control UI
            st.session_state.xlim_current = xlim_auto
            st.session_state.ylim_current = ylim_auto
            
            return True

    except Exception as e:
        st.error(f"Error during fitting process: {e}")
        st.session_state.fit_results = None
        st.session_state.final_fig = None
        return False

def parse_data_string(data_str: str) -> list[float]:
    """Cleans and parses a string of numbers separated by spaces, commas, or tabs."""
    if not isinstance(data_str, str): return []
    cleaned_str = data_str.strip()
    if not cleaned_str: return []
    
    # Split by one or more occurrences of space, comma, semicolon, or tab
    items = re.split(r'[\s,;\t]+', cleaned_str)
    
    numbers = []
    for item in items:
        if item:  # Ensure item is not an empty string
            try:
                numbers.append(float(item))
            except ValueError:
                raise ValueError(f"Could not convert '{item}' to a number. Please check your input.")
    return numbers

#custom HTML for banner
custom_html = """
<div class="banner">
    <img src="https://raw.githubusercontent.com/physicsdrted/physics_plot_py/refs/heads/main/logo.png" alt="Banner Image">
</div>
<style>
    .banner {
        width: 100%;
        height: 200px;
        overflow: visible;
    }
    .banner img {
        width: auto;
        height: 70%;
        object-fit: contain;
    }
</style>
"""

# --- Main App Logic ---
st.components.v1.html(custom_html)

# Initialize session state variables
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
    st.session_state.processed_file_key = None
    st.session_state.df_head = None
if 'show_guess_stage' not in st.session_state:
    st.session_state.show_guess_stage = False
if 'processed_eq_string' not in st.session_state:
    st.session_state.processed_eq_string = None
if 'params' not in st.session_state:
    st.session_state.params = []
if 'fit_func' not in st.session_state:
    st.session_state.fit_func = None
if 'legend_label_str' not in st.session_state:
    st.session_state.legend_label_str = ""
if 'plot_title_input' not in st.session_state:
    st.session_state.plot_title_input = ""
# BUG FIX: Add a counter to create new keys for the file uploader
if 'uploader_key_counter' not in st.session_state:
    st.session_state.uploader_key_counter = 0


# --- Data Input Section ---
tab1, tab2 = st.tabs(["Upload CSV File", "Enter Data Manually"])

with tab1:
    st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")
    
    @st.cache_data
    def get_data():
        df = pd.DataFrame(np.array([[0.0, 0.001, 0.2598, 0.001], [0.05, 0.001, 0.3521, 0.001], [0.1, 0.001, 0.4176, 0.001], [0.15, 0.001, 0.4593, 0.001], [0.2, 0.001, 0.4768, 0.001], [0.25, 0.001, 0.4696, 0.001], [0.3, 0.001, 0.4380, 0.001]]),
                       columns=['time (s)', ' ', 'height (m)' ,' '])
        return df

    @st.cache_data
    def convert_for_download(df):
        return df.to_csv(index=False).encode("utf-8")

    df_example = get_data()
    csv_example = convert_for_download(df_example)

    st.download_button(
        label="Download Example CSV",
        data=csv_example,
        file_name="data.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    # Use a dynamic key for the file uploader
    uploader_key = f"file_uploader_{st.session_state.uploader_key_counter}"
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key=uploader_key)
    
    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if current_file_key != st.session_state.get('processed_file_key', None):
            st.info(f"Processing new uploaded file: {uploaded_file.name}")
            # Full reset of application state
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

            try:
                raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
                if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
                    st.error("Invalid file structure: Ensure header row and at least one data row with 4 columns.")
                    st.session_state.data_loaded = False; st.session_state.processed_file_key = None; st.stop()
                x_label = str(raw_df.iloc[0, 0]); y_label = str(raw_df.iloc[0, 2])
                df = raw_df.iloc[1:].copy()
                if df.empty:
                    st.error("No data rows found."); st.session_state.data_loaded = False; st.session_state.processed_file_key = None; st.stop()
                df.columns = ['x', 'x_err', 'y', 'y_err']
                df = df.apply(pd.to_numeric, errors='coerce')
                if df.isnull().values.any():
                    st.error("Data contains non-numeric values. Please check your CSV file."); st.session_state.data_loaded = False; st.session_state.processed_file_key = None; st.stop()

                st.session_state.x_data = df['x'].to_numpy(); st.session_state.y_data = df['y'].to_numpy()
                st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy())); st.session_state.y_err_safe = safeguard_errors(np.abs(df['y_err'].to_numpy()))
                st.session_state.x_axis_label = x_label; st.session_state.y_axis_label = y_label
                st.session_state.df_head = df.head(10)
                st.session_state.data_loaded = True; st.session_state.processed_file_key = current_file_key
                st.success("New data loaded successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"An unexpected error occurred while processing the file: {e}")
                st.session_state.data_loaded = False; st.session_state.processed_file_key = None; st.stop()

with tab2:
    st.write("Enter data and axis labels below. Separate numbers with spaces, commas, or new lines.")
    with st.form("manual_data_form"):
        # Text inputs for labels
        label_col1, label_col2 = st.columns(2)
        with label_col1:
            x_label_manual = st.text_input("X-Axis Label", "time (s)")
        with label_col2:
            y_label_manual = st.text_input("Y-Axis Label", "height (m)")

        # Text areas for data columns
        data_col1, data_col2, data_col3, data_col4 = st.columns(4)
        with data_col1:
            x_data_manual = st.text_area("X-Values", "0.0\n0.05\n0.1\n0.15\n0.2\n0.25\n0.3")
        with data_col2:
            x_err_manual = st.text_area("X-Uncertainties", "0.001")
        with data_col3:
            y_data_manual = st.text_area("Y-Values", "0.2598\n0.3521\n0.4176\n0.4593\n0.4768\n0.4696\n0.4380")
        with data_col4:
            y_err_manual = st.text_area("Y-Uncertainties", "0.001")
        
        submitted = st.form_submit_button("Load Manual Data")

    if submitted:
        try:
            # Step 1: Parse all data from text areas
            x_dat = parse_data_string(x_data_manual)
            y_dat = parse_data_string(y_data_manual)
            x_err = parse_data_string(x_err_manual)
            y_err = parse_data_string(y_err_manual)
            
            # Step 2: Validate the parsed data
            if not x_dat or not y_dat:
                st.error("X-Values and Y-Values cannot be empty."); st.stop()

            # Handle single uncertainty values by broadcasting them to the full data length
            if len(x_err) == 1: x_err = [x_err[0]] * len(x_dat)
            if len(y_err) == 1: y_err = [y_err[0]] * len(y_dat)

            # Check for consistent lengths
            if not (len(x_dat) == len(y_dat) == len(x_err) == len(y_err)):
                st.error(f"Data length mismatch! X: {len(x_dat)}, Y: {len(y_dat)}, X_Err: {len(x_err)}, Y_Err: {len(y_err)}. All columns must have the same number of entries (a single uncertainty value is applied to all points).")
                st.stop()
            
            # Step 3: Data is valid. Unconditionally reset the application state.
            st.info("Processing manual data...")
            
            # BUG FIX: Increment the uploader key to force a reset of the file_uploader widget
            st.session_state.uploader_key_counter += 1
            
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

            # Step 4: Load the new data into session_state
            df_manual = pd.DataFrame({'x': x_dat, 'x_err': x_err, 'y': y_dat, 'y_err': y_err})

            st.session_state.x_data = df_manual['x'].to_numpy()
            st.session_state.y_data = df_manual['y'].to_numpy()
            st.session_state.x_err_safe = safeguard_errors(np.abs(df_manual['x_err'].to_numpy()))
            st.session_state.y_err_safe = safeguard_errors(np.abs(df_manual['y_err'].to_numpy()))
            st.session_state.x_axis_label = x_label_manual.strip() if x_label_manual.strip() else "X"
            st.session_state.y_axis_label = y_label_manual.strip() if y_label_manual.strip() else "Y"
            st.session_state.df_head = df_manual.head(10)
            
            # Step 5: Set a new key and status, then rerun the app
            manual_data_key = f"manual_{x_label_manual}_{y_label_manual}_{x_data_manual}_{x_err_manual}_{y_data_manual}_{y_err_manual}"
            st.session_state.processed_file_key = manual_data_key
            st.session_state.data_loaded = True
            st.success("Manual data loaded successfully!")
            st.rerun()

        except ValueError as e:
            st.error(f"Input Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Main Application Flow (post-data-load) ---
if st.session_state.data_loaded:
    if st.session_state.df_head is not None:
        st.markdown("---")
        st.subheader("Loaded Data Preview")
        st.dataframe(st.session_state.df_head, use_container_width=True)
        st.markdown("---")

    # Only show the initial plot if a fit has NOT been performed yet
    if not st.session_state.fit_results:
        st.subheader("Initial Data Plot")
        try:
            fig_initial, ax_initial = plt.subplots()
            ax_initial.errorbar(st.session_state.x_data, st.session_state.y_data, yerr=st.session_state.y_err_safe, xerr=st.session_state.x_err_safe, fmt='o', linestyle='None', capsize=5, label='Data', zorder=5)
            ax_initial.set_xlabel(st.session_state.x_axis_label)
            ax_initial.set_ylabel(st.session_state.y_axis_label)
            ax_initial.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)")
            ax_initial.grid(True, linestyle=':', alpha=0.7)
            ax_initial.legend()
            plt.tight_layout()
            st.pyplot(fig_initial)
            plt.close(fig_initial)
        except Exception as plot_err:
            st.error(f"Error generating initial plot: {plot_err}")

        st.markdown("---")

    if not st.session_state.show_guess_stage and not st.session_state.fit_results:
        st.subheader("Step 1: Enter Fit Details")
        st.markdown("""**Instructions:**
*   Use `x` for the independent variable.
*   Use single uppercase letters (A-Z) for fit parameters.
*   Use standard Python math operators: `+`, `-`, `*`, `/`, `**` (power).
*   Allowed functions: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log` (natural), `ln` (natural), `log10`, `sqrt`, `abs`, `absolute`.
*   Use `pi` for the constant π.
**Examples:**
*   Linear: `A * x + B`
*   Quadratic: `A * x**2 + B * x + C`""")

        eq_string_input = st.text_input("Equation:", value=st.session_state.get('last_eq_input', ""), help="Use x, params A-Z, funcs. Ex: A * exp(-B * x) + C", key="equation_input")
        st.session_state.plot_title_input = st.text_input("Optional Plot Title:", value=st.session_state.get('plot_title_input', ""), help="Leave blank for default title.", key="plot_title_input_widget")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            manual_fit_button = st.button("Set Equation & Try a Manual Fit", key="manual_fit_button")
        with b_col2:
            direct_autofit_button = st.button("Set Equation & Perform Autofit", key="direct_autofit_button")

        if (manual_fit_button or direct_autofit_button) and eq_string_input:
            st.session_state.last_eq_input = eq_string_input
            st.session_state.fit_results = None
            st.session_state.final_fig = None
            validation_passed = False
            with st.spinner("Validating equation..."):
                try:
                    processed_eq, params_list = validate_and_parse_equation(eq_string_input)
                    legend_label = format_equation_mathtext(processed_eq)
                    fit_function = create_fit_function(processed_eq, params_list)
                    st.session_state.processed_eq_string = processed_eq
                    st.session_state.params = params_list
                    st.session_state.fit_func = fit_function
                    st.session_state.legend_label_str = legend_label
                    validation_passed = True
                except (ValueError, SyntaxError, RuntimeError) as e:
                    st.error(f"Input Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error during setup: {e}")

            if validation_passed:
                if manual_fit_button:
                    st.session_state.show_guess_stage = True
                    st.rerun()

                elif direct_autofit_button:
                    default_guesses = [1.0] * len(st.session_state.params)
                    fit_successful = perform_the_autofit(default_guesses)

                    if fit_successful:
                        st.session_state.show_guess_stage = False
                        st.rerun()
                    else:
                        st.warning("Automatic fit failed. The model may be too complex or the data too noisy for a default guess. Please provide a manual starting fit.")
                        for i, param in enumerate(st.session_state.params):
                            st.session_state[f"init_guess_{param}"] = default_guesses[i]
                        st.session_state.show_guess_stage = True
                        st.rerun()

    elif st.session_state.show_guess_stage and not st.session_state.fit_results:
        st.subheader("Step 2: Manual Fit & Preview")
        st.info(f"Using Equation: y = {st.session_state.processed_eq_string}")

        params = st.session_state.params
        fit_func = st.session_state.fit_func

        if not params or fit_func is None:
             st.error("Error: Parameters or fit function not available. Please re-enter equation.")
             st.session_state.show_guess_stage = False
             st.rerun()

        initial_guesses = {}
        cols = st.columns(len(params))
        for i, param in enumerate(params):
            with cols[i]:
                guess_key = f"init_guess_{param}"
                if guess_key not in st.session_state:
                    st.session_state[guess_key] = 1.0

                current_value = st.session_state[guess_key]
                format_specifier = "%.3f"
                if current_value != 0:
                    if abs(current_value) < 0.01 or abs(current_value) > 1000:
                        format_specifier = "%.3e"

                initial_guesses[param] = st.number_input(f"Parameter {param}", value=st.session_state[guess_key], key=guess_key, step=None, format=format_specifier)

        st.markdown("---")
        st.write("**Preview with Current Parameter Values:**")

        try:
            current_guess_values = [initial_guesses[p] for p in params]
            x_data = st.session_state.x_data
            y_data = st.session_state.y_data
            x_err_safe = st.session_state.x_err_safe
            y_err_safe = st.session_state.y_err_safe

            y_guess_at_points = fit_func(x_data, *current_guess_values)
            residuals_preview = y_data - y_guess_at_points

            slopes_preview = numerical_derivative(fit_func, x_data, current_guess_values)
            total_err_sq_preview = y_err_safe**2 + (slopes_preview * x_err_safe)**2
            total_err_preview_safe = safeguard_errors(np.sqrt(total_err_sq_preview))
            dof_preview = len(x_data) - len(params)

            if dof_preview > 0:
                chi_squared_preview = np.sum((residuals_preview / total_err_preview_safe)**2)
                red_chi_squared_preview = chi_squared_preview / dof_preview
                red_chi_squared_err_preview = np.sqrt(2.0 / dof_preview)
            else:
                chi_squared_preview, red_chi_squared_preview, red_chi_squared_err_preview = np.nan, np.nan, np.nan

            metric_cols = st.columns(2)
            metric_cols[0].metric("Manual Fit Chi-squared (χ²)", f"{chi_squared_preview:.4f}")
            metric_cols[1].metric("Manual Fit Reduced χ²/DoF", f"{red_chi_squared_preview:.4f}", help=f"Calculated with DoF = {dof_preview}")

            # --- Build the legend labels for the preview plot ---
            equation_label = st.session_state.legend_label_str
            stats_parts = []
            for i, p_name in enumerate(params):
                stats_parts.append(f"${p_name} = {current_guess_values[i]:.4g}$")
            red_chi2_str_preview = format_value_uncertainty(red_chi_squared_preview, red_chi_squared_err_preview)
            stats_parts.append(f"$\\chi^2/DoF = {red_chi2_str_preview.replace('$', '')}$")
            stats_label = "\n".join(stats_parts)

            # --- Generate points for the smooth preview curve ---
            x_min_data, x_max_data = np.min(x_data), np.max(x_data)
            x_preview_curve = np.linspace(x_min_data, x_max_data, 200)
            y_preview_curve = fit_func(x_preview_curve, *current_guess_values)

            # --- Create a two-panel plot (main + residuals) ---
            fig_preview = plt.figure()
            gs_preview = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

            ax0_preview = fig_preview.add_subplot(gs_preview[0])
            ax0_preview.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
            ax0_preview.plot(x_preview_curve, y_preview_curve, 'r--', label=equation_label, zorder=10)
            ax0_preview.plot([], [], ' ', label=stats_label)

            preview_title = st.session_state.plot_title_input.strip() or f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
            ax0_preview.set_ylabel(st.session_state.y_axis_label)
            ax0_preview.set_title(preview_title)
            ax0_preview.legend(loc='best', fontsize='large')
            ax0_preview.grid(True, linestyle=':', alpha=0.7)
            ax0_preview.tick_params(axis='x', labelbottom=False)

            ax1_preview = fig_preview.add_subplot(gs_preview[1], sharex=ax0_preview)
            ax1_preview.errorbar(x_data, residuals_preview, yerr=total_err_preview_safe, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
            ax1_preview.axhline(0, color='grey', linestyle='--', linewidth=1)
            ax1_preview.set_xlabel(st.session_state.x_axis_label)
            ax1_preview.set_ylabel("Residuals")
            ax1_preview.grid(True, linestyle=':', alpha=0.6)

            fig_preview.tight_layout(pad=1.0)
            st.pyplot(fig_preview)
            plt.close(fig_preview)

        except Exception as preview_err:
            st.warning(f"Could not generate preview plot or stats: {preview_err}. Check parameter values and equation.")

        st.markdown("---")
        st.write("If this manual fit is unsatisfactory, you can start over. If it serves as a good initial guess, proceed to the automatic fit.")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Define New Fit", key="redefine_fit_button"):
                st.session_state.show_guess_stage = False
                st.session_state.fit_results = None
                st.session_state.final_fig = None
                st.session_state.processed_eq_string = None
                st.session_state.params = []
                st.session_state.fit_func = None
                for key in list(st.session_state.keys()):
                     if key.startswith("init_guess_"):
                         del st.session_state[key]
                st.rerun()

        with b_col2:
            autofit_button = st.button("Perform Autofit", key="autofit_button")

        if autofit_button:
            final_initial_guesses = [st.session_state[f"init_guess_{p}"] for p in params]
            if perform_the_autofit(final_initial_guesses):
                st.session_state.show_guess_stage = False
                st.rerun()

    elif st.session_state.fit_results:
        st.subheader("Step 3: Fit Results")
        
        if st.session_state.final_fig:
            st.pyplot(st.session_state.final_fig)
            plt.close(st.session_state.final_fig)
        else:
            st.warning("Final plot figure not found in session state.")

        # --- Axis Control UI ---
        st.markdown("---")
        st.markdown("##### Adjust Plot Axes")

        def handle_origin_toggle():
            # This function is triggered when the checkbox state changes
            if st.session_state.include_origin_checkbox:
                # If checkbox is now ticked, calculate new limits including origin
                auto_x = st.session_state.auto_limits['x']
                auto_y = st.session_state.auto_limits['y']
                st.session_state.xlim_current = (min(0, auto_x[0]), max(0, auto_x[1]))
                st.session_state.ylim_current = (min(0, auto_y[0]), max(0, auto_y[1]))
            else:
                # If checkbox is now unticked, revert to the original auto limits
                st.session_state.xlim_current = st.session_state.auto_limits['x']
                st.session_state.ylim_current = st.session_state.auto_limits['y']
            
            # Regenerate the figure with the new limits
            new_fig, _, _ = recreate_final_figure(xlim=st.session_state.xlim_current, ylim=st.session_state.ylim_current)
            st.session_state.final_fig = new_fig

        st.checkbox("Include Origin (0,0)", key='include_origin_checkbox', on_change=handle_origin_toggle)
        
        c1, c2 = st.columns(2)
        with c1:
            xmin = st.number_input("X-Min", value=st.session_state.xlim_current[0], step=None, format="%.3g", key="num_xmin")
            ymin = st.number_input("Y-Min", value=st.session_state.ylim_current[0], step=None, format="%.3g", key="num_ymin")
        with c2:
            xmax = st.number_input("X-Max", value=st.session_state.xlim_current[1], step=None, format="%.3g", key="num_xmax")
            ymax = st.number_input("Y-Max", value=st.session_state.ylim_current[1], step=None, format="%.3g", key="num_ymax")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Update Plot with Manual Range", use_container_width=True):
                st.session_state.xlim_current = (xmin, xmax)
                st.session_state.ylim_current = (ymin, ymax)
                new_fig, _, _ = recreate_final_figure(xlim=(xmin, xmax), ylim=(ymin, ymax))
                st.session_state.final_fig = new_fig
                st.rerun() # Rerun to reflect the changes
        with b2:
            if st.button("Reset to Auto Range", use_container_width=True):
                # Reset limits in state and untick the origin checkbox
                st.session_state.xlim_current = st.session_state.auto_limits['x']
                st.session_state.ylim_current = st.session_state.auto_limits['y']
                st.session_state.include_origin_checkbox = False 
                new_fig, _, _ = recreate_final_figure()
                st.session_state.final_fig = new_fig
                st.rerun() # Rerun to apply reset

        st.markdown("---")
        
        # --- Final Actions Section ---
        f1, f2 = st.columns(2)
        with f1:
            # Prepare and show the download button
            if st.session_state.final_fig:
                try:
                    user_title = st.session_state.plot_title_input.strip()
                    default_title = f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit"
                    plot_title_for_filename = user_title or default_title
                    fn = re.sub(r'[^\w\.\-]+', '_', plot_title_for_filename).strip('_').lower() or "fit_plot"
                    fn += ".svg"
                    img_buffer = io.BytesIO()
                    st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
                    img_buffer.seek(0)
                    st.download_button(label="Download Plot as SVG", data=img_buffer, file_name=fn, mime="image/svg+xml", use_container_width=True)
                except Exception as dl_err:
                     st.warning(f"Could not prepare plot for download: {dl_err}")

        with f2:
            # Show the "Define New Fit" button
            if st.button("Define New Fit", use_container_width=True, type="primary"):
                # Clean up axis control state variables
                for key in ['auto_limits', 'xlim_current', 'ylim_current', 'include_origin_checkbox']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.session_state.show_guess_stage = False
                st.session_state.fit_results = None
                st.session_state.final_fig = None
                st.session_state.processed_eq_string = None
                st.session_state.params = []
                st.session_state.fit_func = None
                for key in list(st.session_state.keys()):
                     if key.startswith("init_guess_"):
                         del st.session_state[key]
                st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Updated 9/18/2025 | [Old Version of Physics Plot](https://physicsplot.shinyapps.io/PhysicsPlot20231011/)")
