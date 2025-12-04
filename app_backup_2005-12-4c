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

# Set your desired aspect ratio here. The CSS will respect it.
FIG_WIDTH = 7.5
FIG_HEIGHT = 8
plt.rcParams['figure.figsize'] = [FIG_WIDTH, FIG_HEIGHT]

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
    Regenerates the final plot, distinguishing between included and excluded points.
    """
    res = st.session_state.fit_results
    fit_func = st.session_state.fit_func
    
    # Get the full dataset and the inclusion mask from the session state DataFrame
    full_df = st.session_state.data_df
    include_mask = full_df['Include in Fit'].astype(bool)
    
    x_data_full = full_df['X'].to_numpy()
    y_data_full = full_df['Y'].to_numpy()
    x_err_full = safeguard_errors(np.abs(full_df['X_Err'].to_numpy()))
    y_err_full = safeguard_errors(np.abs(full_df['Y_Err'].to_numpy()))

    # Create labels for the legend
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

    # Plot INCLUDED data points (solid markers)
    ax0.errorbar(x_data_full[include_mask], y_data_full[include_mask], 
                 yerr=y_err_full[include_mask], xerr=x_err_full[include_mask], 
                 fmt='o', markersize=4, linestyle='None', capsize=3, label='Included Data', zorder=5)
    
    # Plot EXCLUDED data points (hollow markers) if any exist
    if np.sum(~include_mask) > 0:
        ax0.errorbar(x_data_full[~include_mask], y_data_full[~include_mask], 
                     yerr=y_err_full[~include_mask], xerr=x_err_full[~include_mask],
                     fmt='o', markerfacecolor='none', markeredgecolor='gray', markersize=4,
                     linestyle='None', capsize=3, label='Excluded Data', ecolor='gray', zorder=4)

    # Generate fit curve based on the full data range or specified xlim
    x_min_plot = xlim[0] if xlim else np.min(x_data_full)
    x_max_plot = xlim[1] if xlim else np.max(x_data_full)
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
    # The residuals are only calculated for INCLUDED points.
    ax1.errorbar(res['x_data_for_residuals'], res['residuals_final'], yerr=res['total_err_final'], fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax1.set_xlabel(st.session_state.x_axis_label)
    ax1.set_ylabel("Residuals")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    max_resid_err = np.max(np.abs(res['residuals_final']) + res['total_err_final'])
    ax1.set_ylim(-max_resid_err * 1.1, max_resid_err * 1.1)
    
    fig.tight_layout(pad=1.0)
    
    return fig, ax0.get_xlim(), ax0.get_ylim()

def perform_the_autofit(initial_guesses):
    """
    Performs the iterative curve fit using only the INCLUDED data points.
    """
    try:
        with st.spinner("Performing iterative fit... Please wait."):
            fit_func = st.session_state.fit_func
            params = st.session_state.params
            
            # MODIFICATION: Filter data based on the 'Include in Fit' column
            full_df = st.session_state.data_df
            include_mask = full_df['Include in Fit'].astype(bool)
            
            if np.sum(include_mask) <= len(params):
                st.error(f"Fit failed: You have selected {np.sum(include_mask)} data points, but the model has {len(params)} parameters. You must include more points than parameters.")
                return False

            x_data_fit = full_df['X'][include_mask].to_numpy()
            y_data_fit = full_df['Y'][include_mask].to_numpy()
            x_err_fit = safeguard_errors(np.abs(full_df['X_Err'][include_mask].to_numpy()))
            y_err_fit = safeguard_errors(np.abs(full_df['Y_Err'][include_mask].to_numpy()))

            popt_current = list(initial_guesses)
            pcov_current = None
            total_err_current = y_err_fit.copy()

            max_iterations = 4
            for i in range(max_iterations):
                sigma_to_use = total_err_current.copy()

                if not np.all(np.isfinite(sigma_to_use)) or np.any(sigma_to_use <= 0):
                    sigma_to_use = None
                if not all(np.isfinite(p) for p in popt_current):
                    raise RuntimeError(f"Invalid initial parameter guess detected ({popt_current}).")

                try:
                    popt_current, pcov_current = curve_fit(
                        fit_func, x_data_fit, y_data_fit, sigma=sigma_to_use, p0=popt_current,
                        absolute_sigma=True, maxfev=8000, check_finite=(True, True)
                    )
                except RuntimeError as fit_error:
                    raise RuntimeError(f"Fit failed to converge: {fit_error}") from fit_error
                
                # Further checks for fit validity
                if not np.all(np.isfinite(popt_current)) or pcov_current is None or not np.all(np.isfinite(np.diag(pcov_current))) or np.any(np.diag(pcov_current) < 0):
                     raise RuntimeError("Fit resulted in non-finite parameters or invalid covariance matrix.")

                if i < max_iterations - 1:
                    slopes = numerical_derivative(fit_func, x_data_fit, popt_current)
                    total_err_sq = y_err_fit**2 + (slopes * x_err_fit)**2
                    total_err_current = safeguard_errors(np.sqrt(total_err_sq))

            popt_final, pcov_final = popt_current, pcov_current
            total_err_final = total_err_current
            perr_final = np.sqrt(np.diag(pcov_final))
            residuals_final = y_data_fit - fit_func(x_data_fit, *popt_final)
            dof = len(y_data_fit) - len(popt_final)
            
            if dof > 0:
                chi_squared = np.sum((residuals_final / total_err_final)**2)
                red_chi_squared = chi_squared / dof
                red_chi_squared_err = np.sqrt(2.0 / dof)
            else:
                chi_squared, red_chi_squared, red_chi_squared_err = np.nan, np.nan, np.nan

            st.session_state.fit_results = {
                "params": params, "popt": popt_final, "perr": perr_final, "dof": dof,
                "red_chi2": red_chi_squared, "red_chi2_err": red_chi_squared_err,
                "residuals_final": residuals_final, "total_err_final": total_err_final,
                "x_data_for_residuals": x_data_fit # Store the x-data used for the fit
            }

            fig, xlim_auto, ylim_auto = recreate_final_figure()
            st.session_state.final_fig = fig
            st.session_state.auto_limits = {'x': xlim_auto, 'y': ylim_auto}
            st.session_state.xlim_current, st.session_state.ylim_current = xlim_auto, ylim_auto
            
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
    # MODIFICATION: Use a single DataFrame to store all data and inclusion status
    st.session_state.data_df = None
    st.session_state.x_axis_label = "X"
    st.session_state.y_axis_label = "Y"
    st.session_state.fit_results = None
    st.session_state.final_fig = None
    st.session_state.processed_file_key = None

    # State for the active data input tab and manual entry text
    st.session_state.active_data_tab = "Upload CSV File"
    st.session_state.manual_x_label = "time (s)"
    st.session_state.manual_y_label = "height (m)"
    st.session_state.manual_x_data_str = "0.0\n0.05\n0.1\n0.15\n0.2\n0.25\n0.3"
    st.session_state.manual_x_err_str = "0.001"
    st.session_state.manual_y_data_str = "0.2598\n0.3521\n0.4176\n0.4593\n0.4768\n0.4696\n0.4380"
    st.session_state.manual_y_err_str = "0.001"

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
if 'uploader_key_counter' not in st.session_state:
    st.session_state.uploader_key_counter = 0

# --- Data Input Section ---

selected_tab = st.radio(
    "Data Input Method",
    ["Upload CSV File", "Enter Data Manually"],
    key='active_data_tab',
    horizontal=True,
    label_visibility="collapsed"
)

if selected_tab == "Upload CSV File":
    def handle_file_upload():
        uploader_key = f"file_uploader_{st.session_state.uploader_key_counter}"
        uploaded_file = st.session_state.get(uploader_key)
        if not uploaded_file: return

        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if current_file_key == st.session_state.get('processed_file_key', None): return

        # Reset application state
        st.session_state.fit_results = None; st.session_state.final_fig = None
        st.session_state.show_guess_stage = False; st.session_state.processed_eq_string = None
        st.session_state.params = []; st.session_state.fit_func = None
        st.session_state.legend_label_str = ""; st.session_state.plot_title_input = ""
        st.session_state.last_eq_input = ""
        for k in [k for k in st.session_state if k.startswith("init_guess_")]: del st.session_state[k]

        try:
            raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
            if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
                st.error("Invalid file structure."); return
            
            x_label, y_label = str(raw_df.iloc[0, 0]), str(raw_df.iloc[0, 2])
            df = raw_df.iloc[1:].copy()
            df.columns = ['X', 'X_Err', 'Y', 'Y_Err']
            df = df.apply(pd.to_numeric, errors='coerce')

            if df.empty or df.isnull().values.any():
                st.error("Data contains non-numeric values or is empty."); return

            # MODIFICATION: Add the 'Include in Fit' column
            df['Include in Fit'] = True
            st.session_state.data_df = df # Store the complete DataFrame
            st.session_state.x_axis_label, st.session_state.y_axis_label = x_label, y_label
            st.session_state.data_loaded = True
            st.session_state.processed_file_key = current_file_key

            st.session_state.manual_x_label = x_label
            st.session_state.manual_y_label = y_label
            st.session_state.manual_x_data_str = "\n".join(df['X'].astype(str))
            st.session_state.manual_x_err_str = "\n".join(df['X_Err'].astype(str))
            st.session_state.manual_y_data_str = "\n".join(df['Y'].astype(str))
            st.session_state.manual_y_err_str = "\n".join(df['Y_Err'].astype(str))
            
            st.session_state.active_data_tab = "Enter Data Manually"

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.data_loaded = False

    st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")
    # ... [The get_data, convert_for_download, and download_button for example CSV remain unchanged] ...
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
    st.download_button(label="Download Example CSV", data=csv_example, file_name="data.csv", mime="text/csv", icon=":material/download:")
    
    uploader_key = f"file_uploader_{st.session_state.uploader_key_counter}"
    st.file_uploader("Choose a CSV file", type="csv", key=uploader_key, on_change=handle_file_upload)

elif selected_tab == "Enter Data Manually":
    st.write("Enter data and axis labels below. Separate numbers with spaces, commas, or new lines.")
    with st.form("manual_data_form"):
        label_col1, label_col2 = st.columns(2)
        with label_col1: x_label_manual = st.text_input("X-Axis Label", value=st.session_state.manual_x_label)
        with label_col2: y_label_manual = st.text_input("Y-Axis Label", value=st.session_state.manual_y_label)
        data_col1, data_col2, data_col3, data_col4 = st.columns(4)
        with data_col1: x_data_manual = st.text_area("X-Values", value=st.session_state.manual_x_data_str)
        with data_col2: x_err_manual = st.text_area("X-Uncertainties", value=st.session_state.manual_x_err_str)
        with data_col3: y_data_manual = st.text_area("Y-Values", value=st.session_state.manual_y_data_str)
        with data_col4: y_err_manual = st.text_area("Y-Uncertainties", value=st.session_state.manual_y_err_str)
        submitted = st.form_submit_button("Load/Update Data")

    if submitted:
        try:
            x_dat, y_dat = parse_data_string(x_data_manual), parse_data_string(y_data_manual)
            x_err, y_err = parse_data_string(x_err_manual), parse_data_string(y_err_manual)
            if not x_dat or not y_dat: st.error("X and Y values cannot be empty."); st.stop()
            if len(x_err) == 1: x_err = [x_err[0]] * len(x_dat)
            if len(y_err) == 1: y_err = [y_err[0]] * len(y_dat)
            if not (len(x_dat) == len(y_dat) == len(x_err) == len(y_err)):
                st.error(f"Data length mismatch! X:{len(x_dat)}, Y:{len(y_dat)}, X_Err:{len(x_err)}, Y_Err:{len(y_err)}."); st.stop()

            st.session_state.uploader_key_counter += 1
            st.session_state.fit_results = None; st.session_state.final_fig = None
            st.session_state.show_guess_stage = False
            
            # MODIFICATION: Create and store the new DataFrame
            df_manual = pd.DataFrame({'X': x_dat, 'X_Err': x_err, 'Y': y_dat, 'Y_Err': y_err})
            df_manual['Include in Fit'] = True
            st.session_state.data_df = df_manual
            
            st.session_state.x_axis_label = x_label_manual.strip() or "X"
            st.session_state.y_axis_label = y_label_manual.strip() or "Y"
            st.session_state.processed_file_key = f"manual_{hash(x_data_manual)}_{hash(y_data_manual)}"
            st.session_state.data_loaded = True
            
            # Persist the current text in the input boxes
            st.session_state.manual_x_label, st.session_state.manual_y_label = x_label_manual, y_label_manual
            st.session_state.manual_x_data_str, st.session_state.manual_y_data_str = x_data_manual, y_data_manual
            st.session_state.manual_x_err_str, st.session_state.manual_y_err_str = x_err_manual, y_err_manual
            
            st.success("Data loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading manual data: {e}")

    if st.session_state.data_loaded:
        st.markdown("---")
        # ... [The download button for current data remains unchanged] ...
        @st.cache_data
        def convert_current_data_to_csv(df, x_label, y_label):
            output = io.StringIO()
            header = f'"{x_label}"," ","{y_label}"," "'
            output.write(header + '\n')
            # Save only the data columns, not the 'Include' column
            df_to_save = df[['X', 'X_Err', 'Y', 'Y_Err']]
            df_to_save.to_csv(output, index=False, header=False)
            return output.getvalue().encode('utf-8')
        csv_to_download = convert_current_data_to_csv(st.session_state.data_df, st.session_state.x_axis_label, st.session_state.y_axis_label)
        st.download_button(label="Download Current Data as CSV", data=csv_to_download, file_name="current_data.csv", mime="text/csv")
        
# --- Main Application Flow (post-data-load) ---
if st.session_state.data_loaded:
    st.markdown("---")

    st.subheader("Select Data Points for Fitting")
    st.info("Uncheck the 'Include in Fit' box for any data points you wish to exclude. Excluded points will be shown as hollow gray circles on the plots but will not be used for fitting or chi-squared calculations.")
    
    edited_df = st.data_editor(
        st.session_state.data_df,
        column_config={
            "X": st.column_config.NumberColumn(format="%.4g"),
            "X_Err": st.column_config.NumberColumn(format="%.3g"),
            "Y": st.column_config.NumberColumn(format="%.4g"),
            "Y_Err": st.column_config.NumberColumn(format="%.3g"),
        },
        disabled=["X", "X_Err", "Y", "Y_Err"],
        use_container_width=True,
        key="data_editor"
    )
    st.session_state.data_df = edited_df
    
    # MODIFICATION: Add a button to refit the data after changing selections
    # This button only appears AFTER an initial fit has been performed.
    if 'fit_results' in st.session_state and st.session_state.fit_results:
        if st.button("Update Fit with New Selection", type="primary", use_container_width=True):
            # Use the previous best-fit parameters as the initial guess for the refit
            initial_guesses = st.session_state.fit_results['popt']
            if perform_the_autofit(initial_guesses):
                st.success("Fit updated successfully!")
                st.rerun()

    st.markdown("---")

    # MODIFICATION: Refactor the main logic to be more robust and fix the AttributeError
    # This safer structure checks for the existence of 'fit_results' before trying to access it.
    
    # STATE 3: A fit has been successfully performed. Show the results.
    if 'fit_results' in st.session_state and st.session_state.fit_results:
        st.subheader("Step 3: Fit Results")
        
        if st.session_state.final_fig:
            st.pyplot(st.session_state.final_fig)
            plt.close(st.session_state.final_fig)

        st.markdown("---")
        st.markdown("##### Adjust Plot Axes")

        def handle_origin_toggle():
            if st.session_state.include_origin_checkbox:
                auto_x, auto_y = st.session_state.auto_limits['x'], st.session_state.auto_limits['y']
                st.session_state.xlim_current = (min(0, auto_x[0]), max(0, auto_x[1]))
                st.session_state.ylim_current = (min(0, auto_y[0]), max(0, auto_y[1]))
            else:
                st.session_state.xlim_current = st.session_state.auto_limits['x']
                st.session_state.ylim_current = st.session_state.auto_limits['y']
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
                st.session_state.xlim_current, st.session_state.ylim_current = (xmin, xmax), (ymin, ymax)
                new_fig, _, _ = recreate_final_figure(xlim=(xmin, xmax), ylim=(ymin, ymax))
                st.session_state.final_fig = new_fig
                st.rerun()
        with b2:
            if st.button("Reset to Auto Range", use_container_width=True):
                st.session_state.xlim_current = st.session_state.auto_limits['x']
                st.session_state.ylim_current = st.session_state.auto_limits['y']
                st.session_state.include_origin_checkbox = False 
                new_fig, _, _ = recreate_final_figure()
                st.session_state.final_fig = new_fig
                st.rerun()

        st.markdown("---")
        
        f1, f2 = st.columns(2)
        with f1:
            if st.session_state.final_fig:
                try:
                    user_title = st.session_state.plot_title_input.strip()
                    default_title = f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit"
                    fn = re.sub(r'[^\w\.\-]+', '_', user_title or default_title).strip('_').lower() or "fit_plot"
                    img_buffer = io.BytesIO()
                    st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
                    img_buffer.seek(0)
                    st.download_button(label="Download Plot as SVG", data=img_buffer, file_name=f"{fn}.svg", mime="image/svg+xml", use_container_width=True)
                except Exception as dl_err:
                     st.warning(f"Could not prepare plot for download: {dl_err}")

        with f2:
            if st.button("Define New Fit", use_container_width=True):
                # Clean up state before returning to the equation definition stage
                keys_to_delete = [
                    'auto_limits', 'xlim_current', 'ylim_current', 'include_origin_checkbox', 
                    'fit_results', 'final_fig', 'processed_eq_string', 'fit_func', 
                    'show_guess_stage', 'params'
                ]
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                for key in [k for k in st.session_state if k.startswith("init_guess_")]:
                    del st.session_state[key]
                st.rerun()

    # STATE 2: The user has defined an equation and is making a manual guess.
    elif 'show_guess_stage' in st.session_state and st.session_state.show_guess_stage:
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
                if guess_key not in st.session_state: st.session_state[guess_key] = 1.0
                current_value = st.session_state[guess_key]
                format_specifier = "%.3f" if 0.01 <= abs(current_value) <= 1000 and current_value != 0 else "%.3e"
                initial_guesses[param] = st.number_input(f"Parameter {param}", value=st.session_state[guess_key], key=guess_key, step=None, format=format_specifier)

        st.markdown("---")
        st.write("**Preview with Current Parameter Values:**")

        try:
            full_df = st.session_state.data_df
            include_mask = full_df['Include in Fit'].astype(bool)
            x_full, y_full = full_df['X'].to_numpy(), full_df['Y'].to_numpy()
            x_err_full = safeguard_errors(np.abs(full_df['X_Err'].to_numpy()))
            y_err_full = safeguard_errors(np.abs(full_df['Y_Err'].to_numpy()))
            x_fit, y_fit, x_err_fit, y_err_fit = x_full[include_mask], y_full[include_mask], x_err_full[include_mask], y_err_full[include_mask]
            
            current_guess_values = [initial_guesses[p] for p in params]
            y_guess = fit_func(x_fit, *current_guess_values)
            residuals = y_fit - y_guess
            slopes = numerical_derivative(fit_func, x_fit, current_guess_values)
            total_err = safeguard_errors(np.sqrt(y_err_fit**2 + (slopes * x_err_fit)**2))
            dof = len(x_fit) - len(params)

            if dof > 0: chi2, red_chi2 = np.sum((residuals / total_err)**2), np.sum((residuals / total_err)**2) / dof
            else: chi2, red_chi2 = np.nan, np.nan

            metric_cols = st.columns(2)
            metric_cols[0].metric("Manual Fit Chi-squared (χ²)", f"{chi2:.4f}")
            metric_cols[1].metric("Manual Fit Reduced χ²/DoF", f"{red_chi2:.4f}", help=f"Calculated with DoF = {dof}")

            fig_preview = plt.figure()
            gs_preview = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax0 = fig_preview.add_subplot(gs_preview[0])

            ax0.errorbar(x_full[include_mask], y_full[include_mask], yerr=y_err_full[include_mask], xerr=x_err_full[include_mask], fmt='o', markersize=4, linestyle='None', capsize=3, label='Included Data', zorder=5)
            if np.sum(~include_mask) > 0:
                ax0.errorbar(x_full[~include_mask], y_full[~include_mask], yerr=y_err_full[~include_mask], xerr=x_err_full[~include_mask], fmt='o', markerfacecolor='none', markeredgecolor='gray', ecolor='gray', markersize=4, linestyle='None', capsize=3, label='Excluded Data', zorder=4)

            x_curve = np.linspace(np.min(x_full), np.max(x_full), 200)
            y_curve = fit_func(x_curve, *current_guess_values)
            ax0.plot(x_curve, y_curve, 'r--', label="Manual Guess", zorder=10)

            title = st.session_state.plot_title_input.strip() or f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
            ax0.set_ylabel(st.session_state.y_axis_label); ax0.set_title(title)
            ax0.legend(loc='best', fontsize='large'); ax0.grid(True, linestyle=':', alpha=0.7)
            ax0.tick_params(axis='x', labelbottom=False)

            ax1 = fig_preview.add_subplot(gs_preview[1], sharex=ax0)
            ax1.errorbar(x_fit, residuals, yerr=total_err, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
            ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
            ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals")
            ax1.grid(True, linestyle=':', alpha=0.6)
            fig_preview.tight_layout(pad=1.0)
            st.pyplot(fig_preview); plt.close(fig_preview)
        except Exception as e:
            st.warning(f"Could not generate preview plot or stats: {e}. Check parameter values and equation.")

        st.markdown("---")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Define New Fit", key="redefine_fit_button"):
                st.session_state.show_guess_stage = False
                st.session_state.fit_results = None
                for key in [k for k in st.session_state if k.startswith("init_guess_")]: del st.session_state[key]
                st.rerun()
        with b_col2:
            if st.button("Perform Autofit", key="autofit_button"):
                final_guesses = [st.session_state[f"init_guess_{p}"] for p in params]
                if perform_the_autofit(final_guesses):
                    st.session_state.show_guess_stage = False
                    st.rerun()

    # STATE 1: Data is loaded. Show the initial plot and prompt for an equation.
    else:
        st.subheader("Initial Data Plot")
        try:
            full_df = st.session_state.data_df
            include_mask = full_df['Include in Fit'].astype(bool)
            x_full, y_full = full_df['X'].to_numpy(), full_df['Y'].to_numpy()
            x_err_full = safeguard_errors(np.abs(full_df['X_Err'].to_numpy()))
            y_err_full = safeguard_errors(np.abs(full_df['Y_Err'].to_numpy()))

            fig_initial, ax_initial = plt.subplots()
            ax_initial.errorbar(x_full[include_mask], y_full[include_mask], yerr=y_err_full[include_mask], xerr=x_err_full[include_mask], fmt='o', linestyle='None', capsize=5, label='Included Data', zorder=5)
            if np.sum(~include_mask) > 0:
                 ax_initial.errorbar(x_full[~include_mask], y_full[~include_mask], yerr=y_err_full[~include_mask], xerr=x_err_full[~include_mask], fmt='o', markerfacecolor='none', markeredgecolor='gray', ecolor='gray', linestyle='None', capsize=5, label='Excluded Data', zorder=4)

            ax_initial.set_xlabel(st.session_state.x_axis_label)
            ax_initial.set_ylabel(st.session_state.y_axis_label)
            ax_initial.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)")
            ax_initial.grid(True, linestyle=':', alpha=0.7); ax_initial.legend()
            plt.tight_layout(); st.pyplot(fig_initial); plt.close(fig_initial)
        except Exception as plot_err:
            st.error(f"Error generating initial plot: {plot_err}")

        st.markdown("---")
        st.subheader("Step 1: Enter Fit Details")
        st.markdown("""**Instructions:** (etc.)""") # Instructions hidden for brevity

        eq_string_input = st.text_input("Equation:", value=st.session_state.get('last_eq_input', ""), key="equation_input")
        st.session_state.plot_title_input = st.text_input("Optional Plot Title:", value=st.session_state.get('plot_title_input', ""), key="plot_title_input_widget")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            manual_fit_button = st.button("Set Equation & Try a Manual Fit", key="manual_fit_button")
        with b_col2:
            direct_autofit_button = st.button("Set Equation & Perform Autofit", key="direct_autofit_button")

        if (manual_fit_button or direct_autofit_button) and eq_string_input:
            st.session_state.last_eq_input = eq_string_input
            if 'fit_results' in st.session_state: del st.session_state.fit_results
            if 'final_fig' in st.session_state: del st.session_state.final_fig
            validation_passed = False
            with st.spinner("Validating equation..."):
                try:
                    processed_eq, params_list = validate_and_parse_equation(eq_string_input)
                    st.session_state.processed_eq_string = processed_eq
                    st.session_state.params = params_list
                    st.session_state.fit_func = create_fit_function(processed_eq, params_list)
                    st.session_state.legend_label_str = format_equation_mathtext(processed_eq)
                    validation_passed = True
                except Exception as e:
                    st.error(f"Input Error: {e}")
            
            if validation_passed:
                if manual_fit_button:
                    st.session_state.show_guess_stage = True
                    st.rerun()
                elif direct_autofit_button:
                    guesses = [1.0] * len(st.session_state.params)
                    if perform_the_autofit(guesses):
                        st.session_state.show_guess_stage = False
                        st.rerun()
                    else:
                        st.warning("Automatic fit failed. Please provide a manual starting fit.")
                        for i, param in enumerate(st.session_state.params): st.session_state[f"init_guess_{param}"] = guesses[i]
                        st.session_state.show_guess_stage = True
                        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Updated 12/4/2025 | [Old Version of Physics Plot](https://physicsplot.shinyapps.io/PhysicsPlot20231011/)")
