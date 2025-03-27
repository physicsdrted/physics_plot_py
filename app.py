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

# --- Paste existing helper functions here ---
# validate_and_parse_equation, create_fit_function,
# numerical_derivative, safeguard_errors
# ... (Keep these functions as defined in the previous script) ...

# --- Main App Logic ---
st.title("Physics Data Plotter and Fitter")
st.write("Upload a 4-column CSV (Labels in Row 1: X, X_Err, Y, Y_Err; Data from Row 2).")

# --- Session State Initialization ---
# Ensures variables persist across reruns after interactions
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.x_data = None
    st.session_state.y_data = None
    st.session_state.x_err_safe = None
    st.session_state.y_err_safe = None
    st.session_state.x_axis_label = "X"
    st.session_state.y_axis_label = "Y"
    st.session_state.fit_results = None # To store final params, chi2 etc.
    st.session_state.final_fig = None   # To store the final plot figure


# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Process file only if it hasn't been processed or a new file is uploaded
    # This prevents reprocessing on every interaction after upload
    # A more robust check might involve comparing file IDs or names if needed
    if not st.session_state.data_loaded:
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        try:
            # Read and validate data (similar logic to original script's Step 1)
            raw_df = pd.read_csv(uploaded_file, header=None)
            if raw_df.empty or raw_df.shape[0] < 2 or raw_df.shape[1] < 4:
                st.error("Invalid file structure: Needs labels (row 1), 4 data cols, >=1 data row.")
                st.stop() # Stop execution for this run

            # Extract Labels safely
            try:
                x_label = str(raw_df.iloc[0, 0]); y_label = str(raw_df.iloc[0, 2])
            except Exception:
                x_label = "X (Col 1)"; y_label = "Y (Col 3)"
                st.warning("Could not read labels from row 1. Using defaults.")

            # Extract and validate data rows
            df = raw_df.iloc[1:].copy()
            if df.empty or df.shape[1] != 4:
                 st.error("No data rows found or incorrect number of data columns (expected 4).")
                 st.stop()
            df.columns = ['x', 'x_err', 'y', 'y_err']
            all_numeric = True
            for col in df.columns:
                try: df[col] = pd.to_numeric(df[col])
                except ValueError: st.error(f"Column '{col}' contains non-numeric data."); all_numeric = False; break
            if not all_numeric: st.stop()
            if df.isnull().values.any(): st.error("NaN/empty cells detected in data rows."); st.stop()

            # Store processed data in session state
            st.session_state.x_data = df['x'].to_numpy()
            st.session_state.y_data = df['y'].to_numpy()
            st.session_state.x_err_safe = safeguard_errors(np.abs(df['x_err'].to_numpy()))
            st.session_state.y_err_safe = safeguard_errors(df['y_err'].to_numpy())
            st.session_state.x_axis_label = x_label
            st.session_state.y_axis_label = y_label
            st.session_state.data_loaded = True
            st.session_state.fit_results = None # Clear previous fit results if new data loaded
            st.session_state.final_fig = None
            st.success("Data loaded and validated successfully!")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.data_loaded = False # Reset flag on error
            st.stop()


# --- Show Input Section only if data is loaded ---
if st.session_state.data_loaded:
    st.markdown("---")
    st.subheader("Enter Fit Equation")
    st.markdown("Use `x`, parameters (A-Z), functions (e.g., `sin`, `exp`, `log`, `np.cos`). Example: `A * exp(-B * x) + C`")

    # Equation Input
    eq_string_input = st.text_input("Equation:", key="equation_input") # Use key for potential later access

    # Fit Button
    fit_button = st.button("Perform Fit")

    if fit_button and eq_string_input:
        st.session_state.fit_results = None # Clear previous results before new fit
        st.session_state.final_fig = None

        with st.spinner("Performing iterative fit... Please wait."):
            try:
                # --- Get Equation and Create Function ---
                processed_eq_string, params = validate_and_parse_equation(eq_string_input)
                fit_func = create_fit_function(processed_eq_string, params)

                # --- Access data from session state ---
                x_data = st.session_state.x_data
                y_data = st.session_state.y_data
                x_err_safe = st.session_state.x_err_safe
                y_err_safe = st.session_state.y_err_safe

                # --- Iterative Fitting ---
                popt_current = None; pcov_current = None
                total_err_current = y_err_safe # Start with safeguarded y_err
                fit_successful = True

                fit_progress_area = st.empty() # Placeholder for progress messages

                for i in range(4):
                    fit_num = i + 1
                    fit_progress_area.info(f"Running Fit {fit_num}/4...")
                    p0 = popt_current if i > 0 else None
                    try:
                        popt_current, pcov_current = curve_fit(fit_func, x_data, y_data, sigma=total_err_current, absolute_sigma=True, p0=p0, maxfev=5000 + i*2000, check_finite=False)
                        if pcov_current is None or not np.all(np.isfinite(pcov_current)):
                            st.warning(f"Fit {fit_num} converged but covariance matrix non-finite. Stopping.")
                            fit_successful = False; break
                    except RuntimeError as fit_error: st.error(f"RuntimeError during fit {fit_num}: {fit_error}"); fit_successful = False; break
                    except Exception as fit_error: st.error(f"Unexpected error during fit {fit_num}: {fit_error}"); fit_successful = False; break

                    if i < 3: # If not the last fit, recalculate total_err
                        slopes = numerical_derivative(fit_func, x_data, popt_current)
                        total_err_sq = y_err_safe**2 + (slopes * x_err_safe)**2
                        total_err_current = safeguard_errors(np.sqrt(total_err_sq))

                fit_progress_area.empty() # Clear progress message

                if not fit_successful:
                    st.error("Fit failed during iterations. Check equation/data.")
                else:
                    # --- Process Final Results ---
                    popt_final = popt_current; pcov_final = pcov_current
                    total_err_final = total_err_current # Error used in the last successful fit
                    perr_final = np.sqrt(np.diag(pcov_final))
                    residuals_final = y_data - fit_func(x_data, *popt_final)
                    chi_squared = np.sum((residuals_final / total_err_final)**2)
                    dof = len(y_data) - len(popt_final)
                    chi_squared_err = np.nan; chi_squared_red = np.nan; red_chi_squared_err = np.nan
                    if dof > 0: chi_squared_err = np.sqrt(2.0 * dof); chi_squared_red = chi_squared / dof; red_chi_squared_err = np.sqrt(2.0 / dof)

                    # --- Store Results for Display ---
                    st.session_state.fit_results = {
                        "eq_string": processed_eq_string,
                        "params": params,
                        "popt": popt_final,
                        "perr": perr_final,
                        "chi2": chi_squared,
                        "chi2_err": chi_squared_err,
                        "dof": dof,
                        "red_chi2": chi_squared_red,
                        "red_chi2_err": red_chi_squared_err,
                        "total_err_final": total_err_final, # Needed for residual plot bars
                        "residuals_final": residuals_final,
                    }
                    st.success("Fit completed successfully!")

                    # --- Generate Plot Figure ---
                    fig = plt.figure(figsize=(10, 9.8))
                    gs = GridSpec(3, 1, height_ratios=[6, 2, 0.1], hspace=0.08) # Row 3 is minimal height initially, table goes below

                    # Top Plot
                    ax0 = fig.add_subplot(gs[0]); ax0.errorbar(x_data, y_data, yerr=y_err_safe, xerr=x_err_safe, fmt='o', markersize=4, linestyle='None', capsize=3, label='Data', zorder=5)
                    x_fit_curve = np.linspace(np.min(x_data), np.max(x_data), 200); y_fit_curve = fit_func(x_fit_curve, *popt_final)
                    ax0.plot(x_fit_curve, y_fit_curve, '-', label='Fit Line (Final Iteration)', zorder=10, linewidth=1.5); ax0.set_ylabel(st.session_state.y_axis_label)
                    plot_title = f"{st.session_state.y_axis_label} vs {st.session_state.x_axis_label} with Final Fit"
                    ax0.set_title(plot_title); ax0.legend(loc='best'); ax0.grid(True, linestyle=':', alpha=0.6); ax0.tick_params(axis='x', labelbottom=False)
                    ax0.text(0.5, 0.5, 'physicsplot.com', transform=ax0.transAxes, fontsize=40, color='lightgrey', alpha=0.4, ha='center', va='center', rotation=30, zorder=0)

                    # Middle Plot (Residuals)
                    ax1 = fig.add_subplot(gs[1], sharex=ax0)
                    ax1.errorbar(x_data, residuals_final, yerr=total_err_final, fmt='o', markersize=4, linestyle='None', capsize=3, zorder=5)
                    ax1.axhline(0, color='grey', linestyle='--', linewidth=1); ax1.set_xlabel(st.session_state.x_axis_label); ax1.set_ylabel("Residuals\n(Data - Final Fit)"); ax1.grid(True, linestyle=':', alpha=0.6)

                    # Store figure in session state
                    st.session_state.final_fig = fig


            except ValueError as e: st.error(f"Input Error: {e}")
            except SyntaxError as e: st.error(f"Syntax Error: {e}")
            except RuntimeError as e: st.error(f"Fit Error during setup: {e}")
            except TypeError as e: st.error(f"Eval Error: {e}")
            except Exception as e: st.error(f"An unexpected error occurred: {e} ({type(e).__name__})")


# --- Display Results Section ---
if st.session_state.fit_results:
    st.markdown("---")
    st.subheader("Fit Results")

    # Display Plot
    if st.session_state.final_fig:
        st.pyplot(st.session_state.final_fig)

    # Prepare and Display Results Table Data
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
    table_rows.append({"Category": "Degrees of Freedom (DoF)", "Value": f"{res['dof']}", "Uncertainty": ""})
    table_rows.append({
        "Category": "Reduced χ²/DoF",
        "Value": f"{res['red_chi2']:.4f}" if res['dof'] > 0 else "N/A",
        "Uncertainty": f"{res['red_chi2_err']:.3f}" if res['dof'] > 0 else ""
    })

    results_df = pd.DataFrame(table_rows)
    st.dataframe(results_df.set_index('Category')) # Use st.dataframe for better web table

    # SVG Download Button
    if st.session_state.final_fig:
        fn = f"{st.session_state.y_axis_label}_vs_{st.session_state.x_axis_label}_fit.svg"
        fn = re.sub(r'[^\w\.\-]+', '_', fn).strip('_').lower() or "fit_plot.svg"

        img_buffer = io.BytesIO()
        # Save figure to buffer, explicitly using tight bbox needed for table *if* using matplotlib table
        # If using st.dataframe, tight bbox is less critical for the table part
        st.session_state.final_fig.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1)
        img_buffer.seek(0)

        st.download_button(
           label="Download Plot as SVG",
           data=img_buffer,
           file_name=fn,
           mime="image/svg+xml"
        )

elif st.session_state.data_loaded and not fit_button:
     # If data is loaded but fit hasn't been run yet, maybe show the initial plot?
     # Or just wait for user to input equation and press button.
     pass

# --- Footer or additional info ---
st.markdown("---")
# st.info("Watermark 'physicsplot.com' added to the main plot.")