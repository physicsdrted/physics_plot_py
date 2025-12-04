import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import io
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Physics Plot", layout="wide")

# --- MATPLOTLIB STYLING ---
# Set the figure size and font styles for all plots.
plt.rcParams['figure.figsize'] = [7.5, 8]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20

# --- SECURITY & DYNAMIC FUNCTION GLOBALS ---
# Define allowed characters for user-input equations to prevent malicious code.
ALLOWED_CHARS = r"^[A-Za-z0-9\s\.\+\-\*\/\(\)\,\_\^]+$"

# Define a whitelist of numpy functions that can be used in equations.
ALLOWED_NP_FUNCTIONS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan, 'atan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'ln': np.log, 'log10': np.log10, 'sqrt': np.sqrt,
    'pi': np.pi, 'abs': np.abs, 'absolute': np.abs,
}

# Create a safe global environment for the `eval()` call when creating the fit function.
# This ensures that only whitelisted functions and variables are accessible.
SAFE_GLOBALS = {'__builtins__': {}}
SAFE_GLOBALS['np'] = np
SAFE_GLOBALS.update(ALLOWED_NP_FUNCTIONS)


# --- HELPER FUNCTION DEFINITIONS ---

def format_value_uncertainty(value, uncertainty):
    """
    Formats a value and its uncertainty into a standard scientific string
    following particle data group (PDG) conventions: (aaa.a ± bbb.b) × 10^nnn.
    """
    if not np.isfinite(value) or not np.isfinite(uncertainty) or uncertainty <= 0:
        val_str = f"{value:.5g}" if np.isfinite(value) else "N/A"
        unc_str = f"{uncertainty:.3g}" if np.isfinite(uncertainty) else "N/A"
        return f"{val_str} ± {unc_str}"

    # Determine the engineering exponent (a multiple of 3) for the uncertainty.
    exponent_of_unc = np.floor(np.log10(abs(uncertainty)))
    eng_exponent = int(3 * np.floor(exponent_of_unc / 3)) + 3

    # Scale the value and uncertainty to this exponent.
    scaler = 10**(-eng_exponent)
    scaled_value = value * scaler
    scaled_uncertainty = uncertainty * scaler

    # Determine decimal places needed to show 3 significant figures for the uncertainty.
    log10_scaled_unc = np.floor(np.log10(abs(scaled_uncertainty)))
    decimal_places = max(0, 2 - int(log10_scaled_unc))

    val_fmt = f"{scaled_value:.{decimal_places}f}"
    unc_fmt = f"{scaled_uncertainty:.{decimal_places}f}"

    # Assemble the final mathtext string for plotting.
    if eng_exponent != 0:
        return f"$({val_fmt} \\pm {unc_fmt}) \\times 10^{{{eng_exponent}}}$"
    else:
        return f"$({val_fmt} \\pm {unc_fmt})$"

def validate_and_parse_equation(eq_string):
    """
    Validates a user-provided equation string for safety and correctness.
    Extracts the independent variable 'x' and fit parameters (A-Z).
    """
    eq_string = eq_string.strip()
    eq_string = re.sub(r'^\s*y\s*=\s*', '', eq_string, flags=re.IGNORECASE).strip()
    if not eq_string: raise ValueError("Equation cannot be empty.")

    eq_string = eq_string.replace('^', '**')  # Convert to Python's power operator.

    # Security checks against the whitelists.
    if not re.match(ALLOWED_CHARS, eq_string):
        invalid_chars = "".join(sorted(list(set(re.sub(ALLOWED_CHARS, '', eq_string)))))
        raise ValueError(f"Invalid characters: '{invalid_chars}'.")
    if not re.search(r'\bx\b', eq_string):
        raise ValueError("Equation must contain 'x' as the independent variable.")

    # Extract parameters and check for any unknown/disallowed words.
    params = sorted(list(set(re.findall(r'\b([A-Z])\b', eq_string))))
    all_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', eq_string))
    allowed_words = {'x'} | set(params) | set(ALLOWED_NP_FUNCTIONS.keys()) | {'np'}
    unknown_words = all_words - allowed_words
    if unknown_words:
        raise ValueError(f"Unknown or disallowed items: {', '.join(unknown_words)}.")
    if not params:
        raise ValueError("No fit parameters (A-Z) found in the equation.")

    return eq_string, params

def create_fit_function(eq_string, params):
    """
    Dynamically creates a Python function from a validated equation string using `exec`.
    The created function is safe because the input string has been sanitized.
    """
    func_name = "dynamic_fit_func"
    param_str = ', '.join(params)
    
    # Construct the body of the function as a string.
    func_code = f"""
import numpy as np
def {func_name}(x, {param_str}):
    try:
        # The `eval` call uses the safe globals and a local dictionary
        # containing only x and the fit parameters.
        result = eval(_EQ_STRING, _SAFE_GLOBALS, {{'x': x, {", ".join(f"'{p}': {p}" for p in params)}}})

        # Ensure the result is a numpy array of floats, handling various possible return types.
        if isinstance(result, (np.ndarray, list, tuple)):
            result = np.asarray(result)
            if np.iscomplexobj(result): result = np.real(result)
            result = result.astype(float)
        elif isinstance(result, complex): result = float(result.real)
        elif isinstance(result, (int, float)): result = float(result)
        
        # Handle cases where the calculation might fail (e.g., log of a negative number).
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = np.nan
        elif not np.isfinite(result):
            result = np.nan
        return result
    except Exception:
        # If any error occurs during evaluation, return NaN.
        return np.full_like(x, np.nan) if isinstance(x, np.ndarray) else np.nan
"""
    # Execute the function definition in a controlled namespace.
    exec_globals = {'np': np, '_SAFE_GLOBALS': SAFE_GLOBALS, '_EQ_STRING': eq_string}
    local_namespace = {}
    try:
        exec(func_code, exec_globals, local_namespace)
    except Exception as e_compile:
        raise SyntaxError(f"Failed to compile function from equation: {e_compile}")

    if func_name not in local_namespace:
        raise RuntimeError("Could not create fit function.")
    return local_namespace[func_name]

def numerical_derivative(func, x, params, h=1e-7):
    """
    Calculates the numerical derivative using the central difference method.
    Safely handles both scalar and array inputs for 'x'.
    """
    try:
        if params is None or not all(np.isfinite(p) for p in params):
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

        # This calculation works correctly for both scalar and array inputs.
        y_plus_h = func(x + h, *params)
        y_minus_h = func(x - h, *params)
        deriv = (y_plus_h - y_minus_h) / (2 * h)

        # Sanitize the result: check if it's an array before trying to access elements.
        if isinstance(deriv, np.ndarray):
            # If it's an array, replace any non-finite values with zero.
            deriv[~np.isfinite(deriv)] = 0
        elif not np.isfinite(deriv):
            # If it's a scalar (float), just reassign it to zero if it's not finite.
            deriv = 0
        
        return deriv
    except Exception as e:
        st.warning(f"Derivative calculation failed: {e}. Returning slope=0.")
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0

def safeguard_errors(err_array, min_err=1e-9):
    """Replaces non-positive or non-finite errors with a small positive number to avoid division by zero."""
    safe_err = np.array(err_array, dtype=float)
    invalid_mask = ~np.isfinite(safe_err) | (safe_err <= 0)
    if np.any(invalid_mask):
        st.warning(f"Found {np.sum(invalid_mask)} invalid uncertainty values. Replacing with {min_err}.")
        safe_err[invalid_mask] = min_err
    return safe_err

def format_equation_mathtext(eq_string):
    """Converts a Python-style equation string into a LaTeX-like string for Matplotlib's mathtext."""
    replacements = [
        (r'np\.exp\((.*?)\)', r'e^{\1}'), (r'\bexp\((.*?)\)', r'e^{\1}'),
        (r'np\.sqrt\((.*?)\)', r'\\sqrt{\1}'), (r'\bsqrt\((.*?)\)', r'\\sqrt{\1}'),
        (r'np\.sin\((.*?)\)', r'\\mathrm{sin}(\1)'), (r'\bsin\((.*?)\)', r'\\mathrm{sin}(\1)'),
        (r'np\.cos\((.*?)\)', r'\\mathrm{cos}(\1)'), (r'\bcos\((.*?)\)', r'\\mathrm{cos}(\1)'),
        (r'np\.tan\((.*?)\)', r'\\mathrm{tan}(\1)'), (r'\btan\((.*?)\)', r'\\mathrm{tan}(\1)'),
        (r'np\.log10\((.*?)\)', r'\\log_{10}(\1)'), (r'\blog10\((.*?)\)', r'\\log_{10}(\1)'),
        (r'np\.log\((.*?)\)', r'\\ln(\1)'), (r'\blog\((.*?)\)', r'\\ln(\1)'),
        (r'np\.ln\((.*?)\)', r'\\ln(\1)'), (r'\bln\((.*?)\)', r'\\ln(\1)'),
        (r'np\.abs\((.*?)\)', r'|\1|'), (r'\babs\((.*?)\)', r'|\1|'),
    ]
    formatted = eq_string
    for pattern, repl in replacements:
        formatted = re.sub(pattern, repl, formatted, flags=re.IGNORECASE)

    formatted = formatted.replace('np.pi', r'\pi').replace('pi', r'\pi')
    formatted = formatted.replace('**', '^').replace('*', r'\cdot ')
    
    return f'$y = {formatted}$'

def recreate_final_figure(xlim=None, ylim=None):
    """Regenerates the final plot, distinguishing between included and excluded data points."""
    res = st.session_state.fit_results
    fit_func = st.session_state.fit_func
    
    full_df = st.session_state.data_df
    include_mask = full_df['Include in Fit'].astype(bool)
    
    # Create the legend labels from the fit results.
    stats_parts = []
    for i, p_name in enumerate(res['params']):
        param_str = format_value_uncertainty(res['popt'][i], res['perr'][i])
        stats_parts.append(f"${p_name} = {param_str.replace('$', '')}$")
    red_chi2_str = format_value_uncertainty(res['red_chi2'], res['red_chi2_err'])
    stats_parts.append(f"$\\chi^2/DoF = {red_chi2_str.replace('$', '')}$")
    stats_label = "\n".join(stats_parts)

    # Create a two-panel figure: main plot and residuals.
    fig = plt.figure()
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax0 = fig.add_subplot(gs[0])

    # Conditionally plot data based on whether any points are excluded.
    if include_mask.all():
        ax0.errorbar(full_df['X'], full_df['Y'], yerr=full_df['Y_Err'], xerr=full_df['X_Err'],
                     fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
    else:
        ax0.errorbar(full_df.loc[include_mask, 'X'], full_df.loc[include_mask, 'Y'],
                     yerr=full_df.loc[include_mask, 'Y_Err'], xerr=full_df.loc[include_mask, 'X_Err'],
                     fmt='o', markersize=4, linestyle='None', capsize=3, label='Included Data', zorder=5)
        ax0.errorbar(full_df.loc[~include_mask, 'X'], full_df.loc[~include_mask, 'Y'],
                     yerr=full_df.loc[~include_mask, 'Y_Err'], xerr=full_df.loc[~include_mask, 'X_Err'],
                     fmt='o', markerfacecolor='none', markeredgecolor='gray', markersize=4,
                     linestyle='None', capsize=3, label='Excluded Data', ecolor='gray', zorder=4)

    # Generate and plot the smooth best-fit curve.
    x_min_plot = xlim[0] if xlim else full_df['X'].min()
    x_max_plot = xlim[1] if xlim else full_df['X'].max()
    x_fit_curve = np.linspace(x_min_plot, x_max_plot, 400)
    y_fit_curve = fit_func(x_fit_curve, *res['popt'])
    ax0.plot(x_fit_curve, y_fit_curve, '-', label=st.session_state.legend_label_str, zorder=10, linewidth=1.5)
    
    # Add a blank plot element to host the stats text in the legend.
    ax0.plot([], [], ' ', label=stats_label)

    # Set titles, labels, and other plot aesthetics.
    title = st.session_state.plot_title_input.strip() or f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
    ax0.set_ylabel(st.session_state.y_axis_label)
    ax0.set_title(title)
    ax0.legend(loc='best', fontsize='large')
    ax0.grid(True, linestyle=':', alpha=0.6)
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)
    if xlim: ax0.set_xlim(xlim)
    if ylim: ax0.set_ylim(ylim)

    # Create the residuals plot.
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
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
    Performs the iterative curve fit using only the included data points.
    This function handles the core fitting logic, including error propagation.
    """
    try:
        with st.spinner("Performing iterative fit..."):
            fit_func, params = st.session_state.fit_func, st.session_state.params
            
            # Filter data based on the 'Include in Fit' checkbox in the DataFrame.
            full_df = st.session_state.data_df
            include_mask = full_df['Include in Fit'].astype(bool)
            
            if include_mask.sum() <= len(params):
                st.error(f"Fit failed: You must include more data points ({include_mask.sum()}) than fit parameters ({len(params)}).")
                return False

            fit_df = full_df[include_mask]
            x_fit = fit_df['X'].to_numpy()
            y_fit = fit_df['Y'].to_numpy()
            x_err_fit = safeguard_errors(np.abs(fit_df['X_Err'].to_numpy()))
            y_err_fit = safeguard_errors(np.abs(fit_df['Y_Err'].to_numpy()))

            # Iteratively perform the fit to account for x-uncertainties.
            popt, pcov = list(initial_guesses), None
            total_err = y_err_fit.copy()
            for i in range(4):  # 4 iterations are typically sufficient.
                sigma = total_err if np.all(np.isfinite(total_err)) and np.all(total_err > 0) else None
                try:
                    popt, pcov = curve_fit(fit_func, x_fit, y_fit, sigma=sigma, p0=popt, absolute_sigma=True, maxfev=8000)
                except RuntimeError as e:
                    raise RuntimeError(f"Fit failed to converge: {e}")
                
                # Check for a valid fit result.
                if not np.all(np.isfinite(popt)) or pcov is None or np.any(np.diag(pcov) < 0):
                     raise RuntimeError("Fit resulted in non-finite parameters or an invalid covariance matrix.")

                # Update total error for the next iteration by propagating x-error.
                if i < 3:
                    slopes = numerical_derivative(fit_func, x_fit, popt)
                    total_err = safeguard_errors(np.sqrt(y_err_fit**2 + (slopes * x_err_fit)**2))

            # Calculate final results after the last iteration.
            perr = np.sqrt(np.diag(pcov))
            residuals = y_fit - fit_func(x_fit, *popt)
            dof = len(y_fit) - len(popt)
            if dof > 0:
                chi2 = np.sum((residuals / total_err)**2)
                red_chi2, red_chi2_err = chi2 / dof, np.sqrt(2.0 / dof)
            else:
                red_chi2, red_chi2_err = np.nan, np.nan

            # Store all results in the session state.
            st.session_state.fit_results = {
                "params": params, "popt": popt, "perr": perr, "dof": dof,
                "red_chi2": red_chi2, "red_chi2_err": red_chi2_err,
                "residuals_final": residuals, "total_err_final": total_err,
                "x_data_for_residuals": x_fit
            }

            # Generate and store the final plot figure and axis limits.
            fig, xlim, ylim = recreate_final_figure()
            st.session_state.final_fig = fig
            st.session_state.auto_limits = {'x': xlim, 'y': ylim}
            st.session_state.xlim_current, st.session_state.ylim_current = xlim, ylim
            return True

    except Exception as e:
        st.error(f"Error during fitting process: {e}")
        st.session_state.fit_results = None
        st.session_state.final_fig = None
        return False

def parse_data_string(data_str: str) -> list[float]:
    """Cleans and parses a string of numbers separated by spaces, commas, semicolons, or tabs."""
    if not isinstance(data_str, str) or not data_str.strip(): return []
    items = re.split(r'[\s,;\t]+', data_str.strip())
    try:
        return [float(item) for item in items if item]
    except ValueError as e:
        raise ValueError(f"Could not convert an item to a number. Please check your input. Details: {e}")

def reset_fit_state():
    """Clears all session state variables related to a specific fit, preparing for a new one."""
    keys_to_delete = [
        'fit_results', 'final_fig', 'processed_eq_string', 'params', 'fit_func',
        'legend_label_str', 'plot_title_input', 'show_guess_stage',
        'auto_limits', 'xlim_current', 'ylim_current', 'include_origin_checkbox'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    for key in [k for k in st.session_state if k.startswith("init_guess_")]:
        del st.session_state[key]

# --- UI & APP LOGIC ---

# Display the banner image.
st.components.v1.html("""
<div style="width:100%; height:150px; overflow:visible;">
    <img src="https://raw.githubusercontent.com/physicsdrted/physics_plot_py/refs/heads/main/logo.png" alt="Banner Image" style="width:auto; height:100%; object-fit:contain;">
</div>
""")

# Initialize the application's state on first run.
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data_df = None
    st.session_state.x_axis_label = "X"
    st.session_state.y_axis_label = "Y"
    st.session_state.processed_file_key = None
    st.session_state.active_data_tab = "Upload CSV File"
    st.session_state.manual_x_label = "time (s)"
    st.session_state.manual_y_label = "height (m)"
    st.session_state.manual_x_data_str = "0.0\n0.05\n0.1\n0.15\n0.2\n0.25\n0.3"
    st.session_state.manual_x_err_str = "0.001"
    st.session_state.manual_y_data_str = "0.2598\n0.3521\n0.4176\n0.4593\n0.4768\n0.4696\n0.4380"
    st.session_state.manual_y_err_str = "0.001"
    st.session_state.uploader_key_counter = 0

# --- DATA INPUT SECTION ---
# Use a radio button styled as tabs to allow programmatic switching.
selected_tab = st.radio(
    "Data Input Method", ["Upload CSV File", "Enter Data Manually"],
    key='active_data_tab', horizontal=True, label_visibility="collapsed"
)

if selected_tab == "Upload CSV File":
    def handle_file_upload():
        """Callback function to process a newly uploaded CSV file."""
        uploader_key = f"file_uploader_{st.session_state.uploader_key_counter}"
        uploaded_file = st.session_state.get(uploader_key)
        if not uploaded_file: return

        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if current_file_key == st.session_state.get('processed_file_key'): return

        reset_fit_state()
        try:
            raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
            if raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
                st.error("Invalid file structure. Requires a header row and at least one data row with 4 columns."); return

            x_label, y_label = str(raw_df.iloc[0, 0]), str(raw_df.iloc[0, 2])
            df = raw_df.iloc[1:].copy()
            df.columns = ['X', 'X_Err', 'Y', 'Y_Err']
            df = df.apply(pd.to_numeric, errors='coerce')

            if df.empty or df.isnull().values.any():
                st.error("Data contains non-numeric values or is empty."); return

            # Populate the main DataFrame and session state variables.
            df['Include in Fit'] = True
            st.session_state.data_df = df
            st.session_state.x_axis_label, st.session_state.y_axis_label = x_label, y_label
            st.session_state.data_loaded = True
            st.session_state.processed_file_key = current_file_key
            
            # Pre-fill the manual entry text boxes with the loaded data.
            st.session_state.manual_x_label, st.session_state.manual_y_label = x_label, y_label
            st.session_state.manual_x_data_str = "\n".join(df['X'].astype(str))
            st.session_state.manual_x_err_str = "\n".join(df['X_Err'].astype(str))
            st.session_state.manual_y_data_str = "\n".join(df['Y'].astype(str))
            st.session_state.manual_y_err_str = "\n".join(df['Y_Err'].astype(str))
            
            # Programmatically switch to the manual entry tab.
            st.session_state.active_data_tab = "Enter Data Manually"
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.data_loaded = False

    st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")
    
    # Provide an example CSV for users to download.
    @st.cache_data
    def get_example_csv():
        df = pd.DataFrame(np.array([[0.0, 0.001, 0.2598, 0.001], [0.05, 0.001, 0.3521, 0.001], [0.1, 0.001, 0.4176, 0.001]]),
                       columns=['time (s)', ' ', 'height (m)' ,' '])
        return df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download Example CSV", data=get_example_csv(), file_name="example_data.csv", mime="text/csv")
    
    uploader_key = f"file_uploader_{st.session_state.uploader_key_counter}"
    st.file_uploader("Choose a CSV file", type="csv", key=uploader_key, on_change=handle_file_upload)

elif selected_tab == "Enter Data Manually":
    st.write("Enter data and axis labels below. Separate numbers with spaces, commas, or new lines.")
    with st.form("manual_data_form"):
        c1, c2 = st.columns(2)
        x_label_manual = c1.text_input("X-Axis Label", value=st.session_state.manual_x_label)
        y_label_manual = c2.text_input("Y-Axis Label", value=st.session_state.manual_y_label)
        c1, c2, c3, c4 = st.columns(4)
        x_data_manual = c1.text_area("X-Values", value=st.session_state.manual_x_data_str)
        x_err_manual = c2.text_area("X-Uncertainties", value=st.session_state.manual_x_err_str)
        y_data_manual = c3.text_area("Y-Values", value=st.session_state.manual_y_data_str)
        y_err_manual = c4.text_area("Y-Uncertainties", value=st.session_state.manual_y_err_str)
        submitted = st.form_submit_button("Load/Update Data")

    if submitted:
        try:
            x_dat, y_dat = parse_data_string(x_data_manual), parse_data_string(y_data_manual)
            x_err, y_err = parse_data_string(x_err_manual), parse_data_string(y_err_manual)
            
            # Validate data integrity.
            if not x_dat or not y_dat: st.error("X and Y values cannot be empty."); st.stop()
            if len(x_err) == 1: x_err = [x_err[0]] * len(x_dat)
            if len(y_err) == 1: y_err = [y_err[0]] * len(y_dat)
            if not (len(x_dat) == len(y_dat) == len(x_err) == len(y_err)):
                st.error(f"Data length mismatch! X:{len(x_dat)}, Y:{len(y_dat)}, X_Err:{len(x_err)}, Y_Err:{len(y_err)}."); st.stop()

            reset_fit_state()
            st.session_state.uploader_key_counter += 1 # Invalidate file uploader.

            # Create the main DataFrame.
            df_manual = pd.DataFrame({'X': x_dat, 'X_Err': x_err, 'Y': y_dat, 'Y_Err': y_err})
            df_manual['Include in Fit'] = True
            st.session_state.data_df = df_manual
            
            # Update session state.
            st.session_state.x_axis_label = x_label_manual.strip() or "X"
            st.session_state.y_axis_label = y_label_manual.strip() or "Y"
            st.session_state.processed_file_key = f"manual_{hash(x_data_manual)}_{hash(y_data_manual)}"
            st.session_state.data_loaded = True
            
            # Persist the current text in the input boxes.
            st.session_state.manual_x_label, st.session_state.manual_y_label = x_label_manual, y_label_manual
            st.session_state.manual_x_data_str, st.session_state.manual_y_data_str = x_data_manual, y_data_manual
            st.session_state.manual_x_err_str, st.session_state.manual_y_err_str = x_err_manual, y_err_manual
            
            st.success("Data loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading manual data: {e}")

    # Allow downloading of the currently displayed data.
    if st.session_state.get('data_loaded'):
        st.markdown("---")
        @st.cache_data
        def convert_df_to_csv(df, x_label, y_label):
            output = io.StringIO()
            output.write(f'"{x_label}"," ","{y_label}"," "\n')
            df_to_save = df[['X', 'X_Err', 'Y', 'Y_Err']]
            df_to_save.to_csv(output, index=False, header=False)
            return output.getvalue().encode('utf-8')
        
        csv_download = convert_df_to_csv(st.session_state.data_df, st.session_state.x_axis_label, st.session_state.y_axis_label)
        st.download_button(label="Download Current Data as CSV", data=csv_download, file_name="current_data.csv", mime="text/csv")

# --- MAIN APPLICATION WORKFLOW ---
if st.session_state.get('data_loaded'):
    st.markdown("---")
    st.subheader("Select Data Points for Fitting")
    st.info("Uncheck any data points you wish to exclude from the fit.")
    
    edited_df = st.data_editor(
        st.session_state.data_df,
        column_config={ "X": st.column_config.NumberColumn(format="%.4g"), "X_Err": st.column_config.NumberColumn(format="%.3g"),
                        "Y": st.column_config.NumberColumn(format="%.4g"), "Y_Err": st.column_config.NumberColumn(format="%.3g") },
        disabled=["X", "X_Err", "Y", "Y_Err"], use_container_width=True, key="data_editor"
    )
    st.session_state.data_df = edited_df
    
    # This button appears after a fit is complete, allowing users to re-run the fit after changing their data selection.
    if st.session_state.get('fit_results'):
        if st.button("Update Fit with New Selection", type="primary", use_container_width=True):
            # Use the previous best-fit parameters as an intelligent starting guess for the new fit.
            initial_guesses = st.session_state.fit_results['popt']
            if perform_the_autofit(initial_guesses):
                st.success("Fit updated successfully!")
                st.rerun()
    st.markdown("---")

    # --- STATE 3: FIT RESULTS ---
    # If a fit has been performed, display the final results.
    if st.session_state.get('fit_results'):
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
                st.session_state.xlim_current, st.session_state.ylim_current = st.session_state.auto_limits['x'], st.session_state.auto_limits['y']
            new_fig, _, _ = recreate_final_figure(xlim=st.session_state.xlim_current, ylim=st.session_state.ylim_current)
            st.session_state.final_fig = new_fig
        st.checkbox("Include Origin (0,0)", key='include_origin_checkbox', on_change=handle_origin_toggle)
        
        c1, c2 = st.columns(2)
        xmin = c1.number_input("X-Min", value=st.session_state.xlim_current[0], step=None, format="%.3g", key="num_xmin")
        ymin = c1.number_input("Y-Min", value=st.session_state.ylim_current[0], step=None, format="%.3g", key="num_ymin")
        xmax = c2.number_input("X-Max", value=st.session_state.xlim_current[1], step=None, format="%.3g", key="num_xmax")
        ymax = c2.number_input("Y-Max", value=st.session_state.ylim_current[1], step=None, format="%.3g", key="num_ymax")

        b1, b2 = st.columns(2)
        if b1.button("Update Plot with Manual Range", use_container_width=True):
            st.session_state.xlim_current, st.session_state.ylim_current = (xmin, xmax), (ymin, ymax)
            new_fig, _, _ = recreate_final_figure(xlim=(xmin, xmax), ylim=(ymin, ymax))
            st.session_state.final_fig = new_fig
            st.rerun()
        if b2.button("Reset to Auto Range", use_container_width=True):
            st.session_state.xlim_current, st.session_state.ylim_current = st.session_state.auto_limits['x'], st.session_state.auto_limits['y']
            st.session_state.include_origin_checkbox = False 
            new_fig, _, _ = recreate_final_figure()
            st.session_state.final_fig = new_fig
            st.rerun()

        st.markdown("---")
        f1, f2 = st.columns(2)
        with f1:
            if st.session_state.final_fig:
                img_buffer = io.BytesIO()
                st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
                img_buffer.seek(0)
                fn_title = st.session_state.plot_title_input.strip() or f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}"
                file_name = re.sub(r'[^\w\.\-]+', '_', fn_title).strip('_').lower() or "fit_plot"
                st.download_button("Download Plot as SVG", img_buffer, f"{file_name}.svg", "image/svg+xml", use_container_width=True)
        if f2.button("Define New Fit", use_container_width=True):
            reset_fit_state()
            st.rerun()

    # --- STATE 2: MANUAL FIT PREVIEW ---
    # If the user has entered an equation but not finalized the fit.
    elif st.session_state.get('show_guess_stage'):
        st.subheader("Step 2: Manual Fit & Preview")
        st.info(f"Using Equation: y = {st.session_state.processed_eq_string}")
        params, fit_func = st.session_state.params, st.session_state.fit_func

        # Get user input for initial parameter guesses.
        initial_guesses = {}
        cols = st.columns(len(params))
        for i, param in enumerate(params):
            guess_key = f"init_guess_{param}"
            if guess_key not in st.session_state: st.session_state[guess_key] = 1.0
            initial_guesses[param] = cols[i].number_input(f"Parameter {param}", value=st.session_state[guess_key], key=guess_key, format="%.4g")

        st.markdown("---")
        st.write("**Preview with Current Parameter Values:**")

        try:
            # Generate and display a preview plot based on the manual guesses.
            full_df = st.session_state.data_df
            include_mask = full_df['Include in Fit'].astype(bool)
            fit_df = full_df[include_mask]
            
            guess_vals = [initial_guesses[p] for p in params]
            residuals = fit_df['Y'] - fit_func(fit_df['X'], *guess_vals)
            slopes = numerical_derivative(fit_func, fit_df['X'], guess_vals)
            total_err = safeguard_errors(np.sqrt(fit_df['Y_Err']**2 + (slopes * fit_df['X_Err'])**2))
            dof = len(fit_df) - len(params)
            
            if dof > 0: red_chi2 = np.sum((residuals / total_err)**2) / dof
            else: red_chi2 = np.nan
            st.metric("Manual Fit Reduced χ²/DoF", f"{red_chi2:.4f}", help=f"DoF = {dof}")

            fig_preview = plt.figure()
            gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax0 = fig_preview.add_subplot(gs[0])
            
            if include_mask.all():
                ax0.errorbar(full_df['X'], full_df['Y'], yerr=full_df['Y_Err'], xerr=full_df['X_Err'],
                             fmt='o', label='Data', zorder=5)
            else:
                ax0.errorbar(fit_df['X'], fit_df['Y'], yerr=fit_df['Y_Err'], xerr=fit_df['X_Err'],
                             fmt='o', label='Included Data', zorder=5)
                ax0.errorbar(full_df.loc[~include_mask, 'X'], full_df.loc[~include_mask, 'Y'],
                             yerr=full_df.loc[~include_mask, 'Y_Err'], xerr=full_df.loc[~include_mask, 'X_Err'],
                             fmt='o', markerfacecolor='none', markeredgecolor='gray', ecolor='gray',
                             label='Excluded Data', zorder=4)

            x_curve = np.linspace(full_df['X'].min(), full_df['X'].max(), 200)
            ax0.plot(x_curve, fit_func(x_curve, *guess_vals), 'r--', label="Manual Guess", zorder=10)
            
            title = st.session_state.get('plot_title_input', '').strip() or f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label}"
            ax0.set_title(title); ax0.set_ylabel(st.session_state.y_axis_label)
            ax0.legend(loc='best'); ax0.grid(True, linestyle=':', alpha=0.7)
            ax0.tick_params(axis='x', labelbottom=False)

            ax1 = fig_preview.add_subplot(gs[1], sharex=ax0)
            ax1.errorbar(fit_df['X'], residuals, yerr=total_err, fmt='o'); ax1.axhline(0, color='grey', ls='--')
            ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals"); ax1.grid(True, ls=':')
            st.pyplot(fig_preview); plt.close(fig_preview)
        except Exception as e:
            st.warning(f"Could not generate preview plot: {e}")

        st.markdown("---")
        b1, b2 = st.columns(2)
        if b1.button("Define New Fit"): reset_fit_state(); st.rerun()
        if b2.button("Perform Autofit"):
            if perform_the_autofit([initial_guesses[p] for p in params]):
                st.session_state.show_guess_stage = False
                st.rerun()

    # --- STATE 1: EQUATION INPUT ---
    # If data is loaded but no fit has been started, prompt for an equation.
    else:
        st.subheader("Initial Data Plot")
        try:
            # Display the initial plot of the loaded data.
            full_df = st.session_state.data_df
            include_mask = full_df['Include in Fit'].astype(bool)
            fig, ax = plt.subplots()

            if include_mask.all():
                ax.errorbar(full_df['X'], full_df['Y'], yerr=full_df['Y_Err'], xerr=full_df['X_Err'],
                            fmt='o', capsize=5, label='Data')
            else:
                ax.errorbar(full_df.loc[include_mask, 'X'], full_df.loc[include_mask, 'Y'],
                            yerr=full_df.loc[include_mask, 'Y_Err'], xerr=full_df.loc[include_mask, 'X_Err'],
                            fmt='o', capsize=5, label='Included Data')
                ax.errorbar(full_df.loc[~include_mask, 'X'], full_df.loc[~include_mask, 'Y'],
                            yerr=full_df.loc[~include_mask, 'Y_Err'], xerr=full_df.loc[~include_mask, 'X_Err'],
                            fmt='o', markerfacecolor='none', markeredgecolor='gray',
                            ecolor='gray', capsize=5, label='Excluded Data')

            ax.set_xlabel(st.session_state.x_axis_label); ax.set_ylabel(st.session_state.y_axis_label)
            ax.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)")
            ax.grid(True, linestyle=':', alpha=0.7); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.error(f"Error generating initial plot: {e}")

        st.markdown("---")
        st.subheader("Step 1: Enter Fit Details")
        st.markdown("""**Instructions:** Use `x` for the independent variable and uppercase letters (A-Z) for fit parameters.
                    **Example:** `A * sin(B * x) + C`""")
        
        eq_input = st.text_input("Equation:", value=st.session_state.get('last_eq_input', ""), key="equation_input")
        st.session_state.plot_title_input = st.text_input("Optional Plot Title:", value=st.session_state.get('plot_title_input', ""), key="plot_title_input_widget")
        
        b1, b2 = st.columns(2)
        manual_button = b1.button("Set Equation & Try a Manual Fit")
        auto_button = b2.button("Set Equation & Perform Autofit")

        if (manual_button or auto_button) and eq_input:
            st.session_state.last_eq_input = eq_input
            try:
                processed_eq, params_list = validate_and_parse_equation(eq_input)
                st.session_state.processed_eq_string = processed_eq
                st.session_state.params = params_list
                st.session_state.fit_func = create_fit_function(processed_eq, params_list)
                st.session_state.legend_label_str = format_equation_mathtext(processed_eq)
                
                if manual_button:
                    st.session_state.show_guess_stage = True
                    st.rerun()
                elif auto_button:
                    if perform_the_autofit([1.0] * len(params_list)):
                        st.session_state.show_guess_stage = False
                        st.rerun()
                    else: # Autofit failed, so drop user into manual mode.
                        st.warning("Automatic fit failed. Please provide a manual starting fit.")
                        st.session_state.show_guess_stage = True
                        st.rerun()
            except Exception as e:
                st.error(f"Equation Error: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Updated 12/4/2025 | [Old Version of Physics Plot](https://physicsplot.shinyapps.io/PhysicsPlot20231011/)")
