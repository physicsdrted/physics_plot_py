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
def validate_and_parse_equation(eq_string):
    """Validates equation, finds 'x' and parameters (A-Z)."""
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

# <<< NEW Function to Format Equation for Mathtext >>>
def format_equation_mathtext(eq_string_orig):
    """Attempts to format the equation string for Matplotlib's mathtext."""
    formatted = eq_string_orig
    st.write(f"DEBUG: Initial format string: `{formatted}`") # Debug

    # Handle functions (use raw strings for replacement!)
    # Use \1, \2 etc for capture groups in replacement
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
    formatted = re.sub(r'np\.ln\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE) # Keep extra \ for ln? Test this.
    formatted = re.sub(r'\bln\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.log\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\blog\((.*?)\)', r'\\ln(\\1)', formatted, flags=re.IGNORECASE)
    # Add more functions as needed (e.g., abs)
    formatted = re.sub(r'np\.abs\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\babs\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'np\.absolute\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r'\babsolute\((.*?)\)', r'|{\1}|', formatted, flags=re.IGNORECASE)

    st.write(f"DEBUG: After func replace: `{formatted}`") # Debug

    # Handle pi AFTER functions
    formatted = formatted.replace('np.pi', r'\pi')
    formatted = formatted.replace('pi', r'\pi')

    # Handle operators and powers
    formatted = formatted.replace('**', '^')
    formatted = formatted.replace('*', r'\cdot ')
    formatted = formatted.replace('/', r'/') # Keep simple slash

    st.write(f"DEBUG: After ops replace: `{formatted}`") # Debug

    # Prepare for math mode: put variables/params in braces {}
    # Need to be careful not to replace inside already processed parts e.g. ^{...}
    # Using placeholders might be safer but more complex. Let's try direct replacement first.
    formatted_final = formatted
    try:
        # Replace single uppercase letters ONLY if not preceded/followed by alphanumeric or _ or { or }
        # This avoids replacing 'A' in 'Abs' or '{A}'
        formatted_final = re.sub(r'(?<![a-zA-Z0-9_{])([A-Z])(?![a-zA-Z0-9_}])', r'{\1}', formatted_final)
        # Replace 'x' ONLY if not preceded/followed by alphanumeric or _ or { or }
        formatted_final = re.sub(r'(?<![a-zA-Z0-9_{])(x)(?![a-zA-Z0-9_}])', r'{x}', formatted_final)
    except Exception as e_re:
        st.warning(f"Regex warning during brace insertion: {e_re}. Using previous format.")
        # Fallback to formatted before brace insertion if regex fails

    st.write(f"DEBUG: After brace insert: `{formatted_final}`") # Debug

    # Add $ signs and y =
    formatted_final = f'$y = {formatted_final}$'
    # Simple cleanup
    formatted_final = formatted_final.replace('$$', '$') # Remove double dollars
    formatted_final = re.sub(r'\s{2,}', ' ', formatted_final) # Condense multiple spaces

    st.write(f"DEBUG: Final formatted string: `{formatted_final}`") # Debug
    return formatted_final


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
#st.title("Physics Data Plotter and Fitter")
st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")

@st.cache_data
def get_data():
    df = pd.DataFrame(np.array([[0.0, 0.001, 0.2598, 0.001], [0.05, 0.001, 0.3521, 0.001], [0.1, 0.001, 0.4176, 0.001], [0.15, 0.001, 0.4593, 0.001], [0.2, 0.001, 0.4768, 0.001], [0.25, 0.001, 0.4696, 0.001], [0.3, 0.001, 0.4380, 0.001]]),
                   columns=['height (m)', ' ', 'time (s)' ,' '])
    return df

@st.cache_data
def convert_for_download(df):
    return df.to_csv(index=False).encode("utf-8")

df = get_data()
csv = convert_for_download(df)

st.download_button(
    label="Download Example CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv",
    icon=":material/download:",
)

# --- Session State Initialization (Add new variables) ---
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
# --- New State Variables ---
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
if 'plot_title_input' not in st.session_state: # Store title input too
    st.session_state.plot_title_input = ""

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
if uploaded_file is not None:
    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if current_file_key != st.session_state.get('processed_file_key', None):
        st.info(f"Processing new uploaded file: {uploaded_file.name}")

        # --- !!! ADD RESET LOGIC HERE !!! ---
        # Reset all fitting-related states before processing the new file's data
        st.session_state.fit_results = None
        st.session_state.final_fig = None
        st.session_state.show_guess_stage = False
        st.session_state.processed_eq_string = None
        st.session_state.params = []
        st.session_state.fit_func = None
        st.session_state.legend_label_str = ""
        st.session_state.plot_title_input = ""
        st.session_state.last_eq_input = "" # Clear last equation attempt too

        # Clear any previous initial guess values stored in session state
        # This prevents carrying over guesses if parameter names are reused
        keys_to_remove = [k for k in st.session_state if k.startswith("init_guess_")]
        for key in keys_to_remove:
            del st.session_state[key]
        # --- End Reset Logic ---

        # --- Now, proceed with processing the new file ---
        try: # File processing logic
            raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
            if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
                st.error("Invalid file structure: Ensure header row and at least one data row with 4 columns.")
                st.session_state.data_loaded = False # Ensure data isn't marked loaded
                st.session_state.processed_file_key = None # Clear key as processing failed
                st.stop() # Stop if structure is wrong

            try:
                x_label = str(raw_df.iloc[0, 0])
                y_label = str(raw_df.iloc[0, 2])
            except Exception:
                x_label = "X (Col 1)"
                y_label = "Y (Col 3)"
                st.warning("Could not read labels from header row.")

            df = raw_df.iloc[1:].copy()
            if df.empty or df.shape[1] != 4:
                st.error("No data rows found or incorrect number of columns (expected 4).")
                st.session_state.data_loaded = False
                st.session_state.processed_file_key = None
                st.stop()

            df.columns = ['x', 'x_err', 'y', 'y_err']
            converted_cols = {}
            conversion_failed = False
            for col in df.columns:
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.isnull().any():
                        first_bad_index = numeric_col.index[numeric_col.isnull()][0] + 2 # +2 for 0-based index and header row
                        st.error(f"Column '{col}' contains non-numeric data near row {first_bad_index}. Please check your CSV.")
                        conversion_failed = True
                        break
                    else:
                        converted_cols[col] = numeric_col # Use the already converted Series
                except Exception as e:
                    st.error(f"Error converting column '{col}': {e}")
                    conversion_failed = True
                    break

            if conversion_failed:
                st.session_state.data_loaded = False
                st.session_state.processed_file_key = None
                st.stop() # Stop if conversion fails

            # If all conversions successful
            df = pd.DataFrame(converted_cols)

            # --- Store Processed Data in Session State ---
            st.session_state.x_data = df['x'].to_numpy()
            st.session_state.y_data = df['y'].to_numpy()
            # Ensure errors are non-negative before safeguarding
            st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy()))
            st.session_state.y_err_safe = safeguard_errors(np.abs(df['y_err'].to_numpy())) # Y errors usually shouldn't be abs()'d unless specifically needed
            st.session_state.x_axis_label = x_label
            st.session_state.y_axis_label = y_label
            st.session_state.df_head = df.head(10)

            # --- Mark as Loaded and Update Key ---
            st.session_state.data_loaded = True
            st.session_state.processed_file_key = current_file_key
            st.success("New data loaded successfully!")
            # No st.rerun() needed here, file upload triggers it

        except pd.errors.ParserError as pe:
            st.error(f"CSV Parsing Error: {pe}. Check file format and delimiters.")
            st.session_state.data_loaded = False
            st.session_state.processed_file_key = None
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while processing the file: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show full traceback for debugging
            st.session_state.data_loaded = False
            st.session_state.processed_file_key = None
            st.stop()
# --- End of File Uploader Block ---

# --- Display Data Preview and Initial Plot if data loaded ---
if st.session_state.data_loaded:
    if st.session_state.df_head is not None:
        st.markdown("---")
        st.subheader("Loaded Data Preview")
        st.dataframe(st.session_state.df_head, use_container_width=True)
        st.markdown("---")

    st.subheader("Initial Data Plot")
    try:
        fig_initial, ax_initial = plt.subplots(figsize=(10, 6))
        ax_initial.errorbar(st.session_state.x_data, st.session_state.y_data, yerr=st.session_state.y_err_safe, xerr=st.session_state.x_err_safe, fmt='o', linestyle='None', capsize=5, label='Data', zorder=5)
        ax_initial.set_xlabel(st.session_state.x_axis_label)
        ax_initial.set_ylabel(st.session_state.y_axis_label)
        ax_initial.set_title(f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} (Raw Data)")
        ax_initial.grid(True, linestyle=':', alpha=0.7)
        ax_initial.legend()
        plt.tight_layout()
        st.pyplot(fig_initial)
        plt.close(fig_initial) # Close the figure to free memory
    except Exception as plot_err:
        st.error(f"Error generating initial plot: {plot_err}")

    # --- Stages: Equation -> Guesses/Preview -> Results ---
    st.markdown("---")

    # --- Stage 1: Equation Input ---
    # Only show if we haven't successfully processed an equation OR if results are cleared
    if not st.session_state.show_guess_stage and not st.session_state.fit_results:
        st.subheader("Step 1: Enter Fit Details")
        st.markdown("""
        **Instructions:**
        *   Use `x` for the independent variable.
        *   Use single uppercase letters (A-Z) for fit parameters.
        *   Use standard Python math operators: `+`, `-`, `*`, `/`, `**` (power).
        *   Allowed functions:
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

        eq_string_input = st.text_input(
            "Equation:",
            value=st.session_state.get('last_eq_input', ""), # Remember last attempt
            help="Use x, params A-Z, funcs. Ex: A * exp(-B * x) + C",
            key="equation_input"
        )
        st.session_state.plot_title_input = st.text_input( # Store title in session state
            "Optional Plot Title:",
            value=st.session_state.get('plot_title_input', ""),
            help="Leave blank for default title.",
            key="plot_title_input_widget"
        )

        # --- Button to Validate Equation and Move to Guess Stage ---
        set_eq_button = st.button("Set Equation & Enter Guesses", key="set_equation_button")

        if set_eq_button and eq_string_input:
            st.session_state.last_eq_input = eq_string_input # Store for convenience
            st.session_state.fit_results = None # Clear previous results
            st.session_state.final_fig = None
            with st.spinner("Validating equation..."):
                try:
                    processed_eq, params_list = validate_and_parse_equation(eq_string_input)
                    legend_label = format_equation_mathtext(processed_eq) # Format for display later
                    fit_function = create_fit_function(processed_eq, params_list)

                    # Store validated info in session state
                    st.session_state.processed_eq_string = processed_eq
                    st.session_state.params = params_list
                    st.session_state.fit_func = fit_function
                    st.session_state.legend_label_str = legend_label
                    st.session_state.show_guess_stage = True # Flag to move to next stage
                    st.rerun() # Rerun to show the guess section

                except (ValueError, SyntaxError, RuntimeError) as e_setup:
                    st.error(f"Input Error: {e_setup}")
                except Exception as e_setup:
                    st.error(f"Unexpected error during setup: {e_setup}")
                    import traceback
                    st.error(traceback.format_exc())

    # --- Stage 2: Guess Input & Preview ---
    elif st.session_state.show_guess_stage and not st.session_state.fit_results:
        st.subheader("Step 2: Initial Guesses & Preview")
        st.info(f"Using Equation: y = {st.session_state.processed_eq_string}")

        # Retrieve from session state
        params = st.session_state.params
        fit_func = st.session_state.fit_func
        processed_eq_string = st.session_state.processed_eq_string

        if not params or fit_func is None:
             st.error("Error: Parameters or fit function not available. Please re-enter equation.")
             st.session_state.show_guess_stage = False
             st.rerun()

        initial_guesses = {}
        cols = st.columns(len(params))
        for i, param in enumerate(params):
            with cols[i]:
                # Use a unique key and provide a default value (e.g., 1.0)
                # Use session state to store the *last entered* guess value for persistence
                guess_key = f"init_guess_{param}"
                if guess_key not in st.session_state:
                    st.session_state[guess_key] = 1.0 # Default initial guess

                initial_guesses[param] = st.number_input(
                    f"Guess for {param}",
                    value=st.session_state[guess_key], # Use stored value
                    key=guess_key, # Use the unique key
                    step=0.1, # Optional: add step for easier adjustment
                    format="%.3f" # Optional: format display
                )

        # --- Preview Plot ---
        st.markdown("---")
        st.write("**Preview Plot with Current Guesses:**")
        try:
            # Generate points for the preview curve
            x_min_data, x_max_data = np.min(st.session_state.x_data), np.max(st.session_state.x_data)
            x_range = x_max_data - x_min_data
            x_preview = np.linspace(
                x_min_data - 0.05 * x_range, # Extend slightly for better visualization
                x_max_data + 0.05 * x_range,
                200
            )
            current_guess_values = [initial_guesses[p] for p in params]
            y_preview = fit_func(x_preview, *current_guess_values)

            fig_preview, ax_preview = plt.subplots(figsize=(10, 6))
            # Plot data
            ax_preview.errorbar(
                st.session_state.x_data,
                st.session_state.y_data,
                yerr=st.session_state.y_err_safe,
                xerr=st.session_state.x_err_safe,
                fmt='o', markersize=4,
                linestyle='None', capsize=3,
                label='Data', zorder=5
            )
            # Plot guess curve
            ax_preview.plot(
                x_preview,
                y_preview,
                'r--', # Red dashed line for guess
                label='Initial Guess Curve', zorder=10
            )
            ax_preview.set_xlabel(st.session_state.x_axis_label)
            ax_preview.set_ylabel(st.session_state.y_axis_label)
            ax_preview.set_title("Data vs. Initial Guess")
            ax_preview.legend()
            ax_preview.grid(True, linestyle=':', alpha=0.7)
            ax_preview.set_ylim(bottom=min(np.min(st.session_state.y_data)*0.9, np.min(y_preview)*0.9) if len(y_preview) > 0 and not np.isnan(y_preview).all() else None,
                                top=max(np.max(st.session_state.y_data)*1.1, np.max(y_preview)*1.1) if len(y_preview) > 0 and not np.isnan(y_preview).all() else None) # Basic auto-scaling
            plt.tight_layout()
            st.pyplot(fig_preview)
            plt.close(fig_preview) # Close the figure

        except Exception as preview_err:
            st.warning(f"Could not generate preview plot: {preview_err}. Check guesses and equation.")
            # Optionally show traceback in debug mode
            # import traceback
            # st.error(traceback.format_exc())


        st.markdown("---")
        # --- Autofit Button ---
        autofit_button = st.button("Perform Autofit", key="autofit_button")

        if autofit_button:
            # --- Stage 3: Perform Fit ---
            with st.spinner("Performing iterative fit... Please wait."):
                try:
                    # Get final guesses just before fitting
                    final_initial_guesses = [st.session_state[f"init_guess_{p}"] for p in params]

                    # --- Fit with User's Initial Guesses ---
                    x_data = st.session_state.x_data
                    y_data = st.session_state.y_data
                    x_err_safe = st.session_state.x_err_safe
                    y_err_safe = st.session_state.y_err_safe

                    popt_current = list(final_initial_guesses) # Start from user guesses
                    pcov_current = None
                    total_err_current = y_err_safe.copy() # Start with y-errors only
                    fit_successful = True

                    # --- Iterative Fitting Loop ---
                    fit_progress_area = st.empty()
                    max_iterations = 4 # Or make this configurable
                    for i in range(max_iterations):
                        fit_num = i + 1
                        fit_progress_area.info(f"Running Fit Iteration {fit_num}/{max_iterations}...")
                        sigma_to_use = total_err_current.copy() # Use errors from previous iteration (or initial y-err)

                        # Check if sigma has valid values before fitting
                        if not np.all(np.isfinite(sigma_to_use)) or np.any(sigma_to_use <= 0):
                             st.warning(f"Fit Iteration {fit_num}: Invalid sigma values detected. Using uniform weights for this iteration.")
                             sigma_to_use = None # Fallback to unweighted fit for this step
                        elif np.all(sigma_to_use < 1e-15): # Avoid extremely small sigmas
                             st.warning(f"Fit Iteration {fit_num}: Very small sigma values detected. Using uniform weights for this iteration.")
                             sigma_to_use = None

                        # Check if initial guesses are valid
                        if not all(np.isfinite(p) for p in popt_current):
                             st.error(f"Fit Iteration {fit_num}: Invalid initial parameter guess detected ({popt_current}). Fit aborted.")
                             fit_successful = False
                             break

                        try:
                            popt_current, pcov_current = curve_fit(
                                fit_func,
                                x_data,
                                y_data,
                                sigma=sigma_to_use,
                                p0=popt_current, # Use current params as guess for next iter
                                absolute_sigma=True, # Treat sigma as absolute std deviations
                                maxfev=5000 + i*3000, # Increase max evaluations
                                check_finite=(True, True) # Ensure input data is finite
                            )
                            # Basic check on results
                            if not np.all(np.isfinite(popt_current)):
                                raise RuntimeError("Fit resulted in non-finite parameters.")
                            if pcov_current is None or not np.all(np.isfinite(np.diag(pcov_current))) or np.any(np.diag(pcov_current) < 0):
                                st.warning(f"Fit Iteration {fit_num}: Covariance matrix calculation issues. Uncertainties might be unreliable.")
                                # Try to estimate pcov again if possible or proceed cautiously
                                if pcov_current is None: pcov_current = np.full((len(params), len(params)), np.inf) # Assign infinite variance


                        except RuntimeError as fit_error:
                            st.error(f"Error during fit iteration {fit_num}: {fit_error}. Trying to continue or aborting...")
                            # Option 1: Abort (Safer)
                            fit_successful = False
                            break
                            # Option 2: Try to proceed cautiously (might give bad results)
                            # if i > 0: # If not the first iteration, maybe use previous result?
                            #     st.warning("Using parameters from previous successful iteration.")
                            # else: # First iteration failed
                            #     fit_successful = False
                            #     break # Abort if first fails
                        except Exception as fit_error:
                             st.error(f"Unexpected error during fit iteration {fit_num}: {fit_error} ({type(fit_error).__name__})")
                             fit_successful = False
                             break

                        # Update errors for the *next* iteration (if not the last one)
                        if i < max_iterations - 1 and fit_successful:
                            try:
                                slopes = numerical_derivative(fit_func, x_data, popt_current)
                                # Check if slopes are valid before using them
                                if np.all(np.isfinite(slopes)):
                                    total_err_sq = y_err_safe**2 + (slopes * x_err_safe)**2
                                    total_err_current = safeguard_errors(np.sqrt(total_err_sq))
                                else:
                                    st.warning(f"Fit Iteration {fit_num}: Could not calculate valid slopes for error propagation. Using only y-errors for next iteration.")
                                    total_err_current = y_err_safe.copy()
                            except Exception as deriv_err:
                                 st.warning(f"Fit Iteration {fit_num}: Error calculating derivative for error propagation: {deriv_err}. Using only y-errors for next iteration.")
                                 total_err_current = y_err_safe.copy()


                    fit_progress_area.empty() # Clear progress message

                    if not fit_successful or popt_current is None or pcov_current is None:
                        st.error("Fit failed to converge or produce valid results.")
                        # Keep showing guess stage
                        st.session_state.fit_results = None
                        st.rerun() # Rerun to stay on guess page with error message

                    # --- Fit Succeeded: Process Results ---
                    popt_final = popt_current
                    pcov_final = pcov_current
                    # Use the sigma that was *used* for the final successful fit
                    total_err_final = sigma_to_use if sigma_to_use is not None else np.ones_like(y_data) # Use ones if unweighted was forced

                    try:
                        diag_pcov = np.diag(pcov_final)
                        if np.any(diag_pcov < 0):
                             st.warning("Negative variance found in covariance matrix. Uncertainties are unreliable (set to NaN).")
                             perr_final = np.full(len(popt_final), np.nan)
                        else:
                            perr_final = np.sqrt(diag_pcov)
                    except Exception as perr_calc_err:
                         st.warning(f"Could not calculate parameter errors from covariance: {perr_calc_err}. Setting errors to NaN.")
                         perr_final = np.full(len(popt_final), np.nan)


                    residuals_final = y_data - fit_func(x_data, *popt_final)

                    # --- Chi-squared Calculation ---
                    chi_squared = np.inf
                    chi_squared_err = np.nan
                    chi_squared_red = np.inf
                    red_chi_squared_err = np.nan
                    dof = len(y_data) - len(popt_final)

                    if dof > 0:
                        # Check if final errors are valid for chi2 calc
                        valid_err_mask = np.isfinite(total_err_final) & (total_err_final > 1e-15) # Avoid division by zero/NaN/Inf
                        if np.sum(valid_err_mask) == len(y_data): # All errors are valid
                             chi_squared = np.sum((residuals_final / total_err_final)**2)
                             chi_squared_err = np.sqrt(2.0 * dof)
                             chi_squared_red = chi_squared / dof
                             red_chi_squared_err = np.sqrt(2.0 / dof)
                        elif np.sum(valid_err_mask) > len(popt_final): # Some valid errors, calculate with subset? Or report NaN?
                            st.warning(f"Chi-squared calculated using only {np.sum(valid_err_mask)} points with valid errors.")
                            chi_squared = np.sum((residuals_final[valid_err_mask] / total_err_final[valid_err_mask])**2)
                            dof_subset = np.sum(valid_err_mask) - len(popt_final)
                            if dof_subset > 0 :
                                chi_squared_err = np.sqrt(2.0 * dof_subset)
                                chi_squared_red = chi_squared / dof_subset
                                red_chi_squared_err = np.sqrt(2.0 / dof_subset)
                            else:
                                chi_squared = np.inf # Not enough points for subset calc
                                chi_squared_red = np.inf

                        else:
                             st.warning("Insufficient valid error values to calculate Chi-squared reliably.")
                             chi_squared = np.nan
                             chi_squared_red = np.nan
                    else:
                         st.warning("Degrees of freedom <= 0. Cannot calculate reduced Chi-squared or its uncertainty.")


                    user_title_str = st.session_state.plot_title_input.strip() # Retrieve from session state
                    final_plot_title = user_title_str if user_title_str else f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} with fit."

                    # Store results in session state
                    st.session_state.fit_results = {
                        "eq_string": st.session_state.processed_eq_string,
                        "params": params,
                        "popt": popt_final,
                        "perr": perr_final,
                        "chi2": chi_squared,
                        "chi2_err": chi_squared_err,
                        "dof": dof,
                        "red_chi2": chi_squared_red,
                        "red_chi2_err": red_chi_squared_err,
                        "total_err_final": total_err_final,
                        "residuals_final": residuals_final,
                        "plot_title": final_plot_title,
                        "legend_label": st.session_state.legend_label_str # Use stored label
                    }

                    # --- Generate Final Plot Figure ---
                    fig = plt.figure(figsize=(10, 9.8))
                    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
                    ax0 = fig.add_subplot(gs[0])
                    ax0.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
                    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200)
                    y_fit_curve = fit_func(x_fit_curve, *popt_final)
                    ax0.plot(x_fit_curve, y_fit_curve, '-', label=st.session_state.legend_label_str, zorder=10, linewidth=1.5)
                    ax0.set_ylabel(st.session_state.y_axis_label)
                    ax0.set_title(final_plot_title)
                    ax0.legend(loc='best', fontsize='large')
                    ax0.grid(True, linestyle=':', alpha=0.6)
                    ax0.tick_params(axis='x', labelbottom=False)
                    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)

                    ax1 = fig.add_subplot(gs[1], sharex=ax0)
                    ax1.errorbar(x_data, residuals_final, yerr=total_err_final, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
                    ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
                    ax1.set_xlabel(st.session_state.x_axis_label)
                    ax1.set_ylabel("Residuals\n(Data - Fit)")
                    ax1.grid(True, linestyle=':', alpha=0.6)
                    fig.tight_layout(pad=1.0)
                    st.session_state.final_fig = fig # Store the figure itself

                    # Clear the flag that shows the guess stage
                    st.session_state.show_guess_stage = False
                    st.rerun() # Rerun to display results

                except Exception as e:
                    st.error(f"Error during fitting process: {e}")
                    import traceback
                    st.error(traceback.format_exc()) # Show full traceback for debugging
                    # Reset state partially to allow user to try again?
                    st.session_state.fit_results = None # Clear results
                    st.session_state.final_fig = None
                    # Keep st.session_state.show_guess_stage = True so they stay on guess page

    # --- Stage 4: Show Results ---
    elif st.session_state.fit_results:
        st.subheader("Step 3: Fit Results")
        res = st.session_state.fit_results # Retrieve results

        # Display Plot
        if st.session_state.final_fig:
            st.pyplot(st.session_state.final_fig)
            plt.close(st.session_state.final_fig) # Close figure after displaying
        else:
            st.warning("Final plot figure not found in session state.")

        # Display Results Table
        table_rows = []
        # Use the original equation string stored in results for display
        table_rows.append({"Category": "Equation", "Value": f"y = {res['eq_string']}", "Uncertainty": ""})

        for i, p_name in enumerate(res['params']):
            table_rows.append({
                "Category": f"Parameter: {p_name}",
                "Value": f"{res['popt'][i]:.5g}" if np.isfinite(res['popt'][i]) else "NaN",
                "Uncertainty": f"{res['perr'][i]:.3g}" if np.isfinite(res['perr'][i]) else "NaN"
            })

        table_rows.append({
            "Category": "Chi-squared (χ²)",
            "Value": f"{res['chi2']:.4f}" if np.isfinite(res['chi2']) else "N/A",
            "Uncertainty": f"{res['chi2_err']:.3f}" if res['dof'] > 0 and np.isfinite(res['chi2_err']) else ""
        })

        table_rows.append({
            "Category": "Degrees of Freedom (DoF)",
            "Value": f"{res['dof']}",
            "Uncertainty": ""
        })

        table_rows.append({
            "Category": "Reduced χ²/DoF",
            "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 and np.isfinite(res['red_chi2']) else "N/A",
            "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 and np.isfinite(res['red_chi2_err']) else ""
        })

        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df.set_index('Category'), use_container_width=True)

        # Download Button
        if st.session_state.final_fig: # Check again just in case
            try:
                plot_title_for_filename = res.get('plot_title', f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit")
                fn = re.sub(r'[^\w\.\-]+', '_', plot_title_for_filename).strip('_').lower() or "fit_plot"
                fn += ".svg"

                img_buffer = io.BytesIO()
                # Re-use the stored figure object
                st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
                img_buffer.seek(0)

                st.download_button(
                    label="Download Plot as SVG",
                    data=img_buffer,
                    file_name=fn,
                    mime="image/svg+xml"
                )
            except Exception as dl_err:
                 st.warning(f"Could not prepare plot for download: {dl_err}")

        # Button to go back and define a new fit
        if st.button("Define New Fit"):
            # Clear relevant state variables
            st.session_state.show_guess_stage = False
            st.session_state.fit_results = None
            st.session_state.final_fig = None
            st.session_state.processed_eq_string = None
            st.session_state.params = []
            st.session_state.fit_func = None
            # Maybe clear guess keys too?
            # for key in list(st.session_state.keys()):
            #      if key.startswith("init_guess_"):
            #          del st.session_state[key]
            st.rerun()

    else: # If data is loaded AND results exist, display them
        # --- Display Results Section ---
        st.markdown("---")
        st.subheader("Fit Results")
        if st.session_state.final_fig:
            st.pyplot(st.session_state.final_fig)
        
        res = st.session_state.fit_results
        table_rows = []
        table_rows.append({"Category": "Equation", "Value": f"y = {res['eq_string']}", "Uncertainty": ""})
        
        for i, p_name in enumerate(res['params']):
            table_rows.append({
                "Category": f"Parameter: {p_name}",
                "Value": f"{res['popt'][i]:.5g}",
                "Uncertainty": f"{res['perr'][i]:.3g}"
            })
        
        table_rows.append({
            "Category": "Chi-squared (χ²)",
            "Value": f"{res['chi2']:.4f}",
            "Uncertainty": f"{res['chi2_err']:.3f}" if res['dof'] > 0 else ""
        })
        
        table_rows.append({
            "Category": "Degrees of Freedom (DoF)",
            "Value": f"{res['dof']}",
            "Uncertainty": ""
        })
        
        table_rows.append({
            "Category": "Reduced χ²/DoF",
            "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 else "N/A",
            "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ""
        })
        
        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df.set_index('Category'), use_container_width=True)
        
        if st.session_state.final_fig:
            plot_title_for_filename = res.get('plot_title', f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit")
            fn = re.sub(r'[^\w\.\-]+', '_', plot_title_for_filename).strip('_').lower() or "fit_plot"
            fn += ".svg"
            
            img_buffer = io.BytesIO()
            st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
            img_buffer.seek(0)
            
            st.download_button(
                label="Download Plot as SVG",
                data=img_buffer,
                file_name=fn,
                mime="image/svg+xml"
            )

# --- Footer ---
st.markdown("---")
st.caption("Watermark 'physicsplot.com' added to the main plot.")
