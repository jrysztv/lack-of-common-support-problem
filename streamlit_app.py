# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Union, List

# Adjust the import path if your folder structure is different:
from lack_of_common_support_problem.simulation import (
    CarTruckRepairsSimulator,
    CarTruckSummary,
    CarTruckPlotter,
)


# ------------
# Helper function for bin parsing
# ------------
def parse_bins(bin_str: str) -> Union[int, List[float]]:
    """
    Parses a string specifying bins into either an integer or a list of floats.
    If the string contains commas, it's interpreted as a list of numeric cutoffs.
    If it's a single integer-like string (no commas), it's interpreted as an integer.
    """
    bin_str = bin_str.strip()
    if "," in bin_str:
        # Split by commas, convert each to float
        parts = [x.strip() for x in bin_str.split(",")]
        return [float(x) for x in parts if x]  # handle empty strings
    else:
        # Try to parse as integer
        return int(bin_str)


def main():
    # Set page config for wide layout
    st.set_page_config(layout="wide")

    # Title / Explanation
    st.title("Car vs Truck Repairs Simulation")
    st.markdown(
        """
        This app simulates car and truck data, where **repairs** can be a treatment variable. 
        A confounder (vehicle type) can create a situation where **naive OLS** vs. 
        **coarsened matching** show very different estimates for the effect of repairs. 

        In particular, the distributions of age and mileage for cars vs trucks are disjoint,
        and trucks are **more likely** to have had a lot of repairs. 
        This lack of common support can bias the naive OLS estimate if we omit the vehicle type. 
        Coarsened matching helps mitigate the bias by matching within specified bins.
        """
    )

    # ------------
    # Sidebar parameters
    # ------------
    st.sidebar.header("Binning Options")
    st.sidebar.markdown(
        """
        **Age Bins** / **Mileage Bins**:  
        - Enter a single integer (no commas) to specify the **number** of bins.  
        - Enter comma-separated values (e.g. `0, 60000, 250000`) to specify **cutoff points**.  
        Make sure to include the endpoints if you do cutoffs!  
        """
    )

    default_age_bins_str = "1"  # from the example usage
    default_mileage_bins_str = "0, 60000, 250000"

    age_bins_str = st.sidebar.text_input("Age bins", default_age_bins_str)
    mileage_bins_str = st.sidebar.text_input("Mileage bins", default_mileage_bins_str)

    # Parse bin inputs
    try:
        age_bins = parse_bins(age_bins_str)
    except ValueError:
        st.sidebar.error("Could not parse age_bins. Reverting to default 1.")
        age_bins = 1

    try:
        mileage_bins = parse_bins(mileage_bins_str)
    except ValueError:
        st.sidebar.error(
            "Could not parse mileage_bins. Reverting to default [0, 60000, 250000]."
        )
        mileage_bins = [0, 60000, 250000]

    # Collapsible containers for simulation parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameter Options")

    with st.sidebar.expander("Basic Simulation Setup", expanded=False):
        n_samples = st.number_input(
            "n_samples", min_value=100, max_value=100000, value=2000
        )
        seed = st.number_input(
            "seed (0 = None)", min_value=0, max_value=999999, value=42
        )
        prob_truck = st.slider("prob_truck", min_value=0.0, max_value=1.0, value=0.4)
        prob_repair_if_car = st.slider("prob_repair_if_car", 0.0, 1.0, 0.2)
        prob_repair_if_truck = st.slider("prob_repair_if_truck", 0.0, 1.0, 0.8)

    with st.sidebar.expander("Vehicle Age & Mileage Means/Stds", expanded=False):
        age_mean_car = st.number_input("age_mean_car", value=5.0)
        age_mean_truck = st.number_input("age_mean_truck", value=7.0)
        mileage_mean_car = st.number_input("mileage_mean_car", value=30000.0)
        mileage_mean_truck = st.number_input("mileage_mean_truck", value=160000.0)

        age_std_car = st.number_input("age_std_car", value=2.0)
        age_std_truck = st.number_input("age_std_truck", value=3.0)
        mileage_std_car = st.number_input("mileage_std_car", value=8000.0)
        mileage_std_truck = st.number_input("mileage_std_truck", value=30000.0)

    with st.sidebar.expander("Coefficients & Premiums", expanded=False):
        base_price_car = st.number_input("base_price_car", value=20000.0)
        truck_premium = st.number_input("truck_premium", value=50000.0)
        b_age = st.number_input("b_age", value=-500.0)
        b_mileage = st.number_input("b_mileage", value=-0.1)
        b_repairs = st.number_input("b_repairs", value=-3000.0)
        noise_std = st.number_input("noise_std", value=2000.0)

    # ------------
    # "Run Simulation" button
    # ------------
    run_simulation = st.sidebar.button("Run Simulation")

    # We will store the simulation results (dataframe) in session_state, so that
    # changes to the bins or plot_type can update the visualization without re-running
    # the entire simulation each time (unless you explicitly click "Run Simulation").
    if run_simulation:
        # Create a simulator instance with the user inputs (defaults from example usage).
        sim = CarTruckRepairsSimulator(
            n_samples=int(n_samples),
            seed=None if seed == 0 else int(seed),
            prob_truck=prob_truck,
            prob_repair_if_car=prob_repair_if_car,
            prob_repair_if_truck=prob_repair_if_truck,
            age_mean_car=age_mean_car,
            age_mean_truck=age_mean_truck,
            age_std_car=age_std_car,
            age_std_truck=age_std_truck,
            mileage_mean_car=mileage_mean_car,
            mileage_mean_truck=mileage_mean_truck,
            mileage_std_car=mileage_std_car,
            mileage_std_truck=mileage_std_truck,
            base_price_car=base_price_car,
            truck_premium=truck_premium,
            b_age=b_age,
            b_mileage=b_mileage,
            b_repairs=b_repairs,
            noise_std=noise_std,
        )

        # Generate data and store in session_state
        st.session_state["data_df"] = sim.generate_data()
        st.session_state["simulator"] = sim

    # ------------
    # Main content (two columns)
    # ------------
    col_left, col_right = st.columns([3, 2])  # ratio 3:2

    # We only proceed if there's data in session_state (after "Run Simulation")
    if "data_df" in st.session_state and "simulator" in st.session_state:
        data_df = st.session_state["data_df"]
        simulator = st.session_state["simulator"]

        # Build the summary object with the chosen bins
        summary = CarTruckSummary(
            simulator, age_bins=age_bins, mileage_bins=mileage_bins
        )

        # ------------
        # Left column: Plot controls & visualization
        # ------------
        with col_left:
            st.subheader("Visualization Controls")

            # The user wants a descriptive set of plot types
            plot_type_options = {
                "Simple Scatter": "scatter",
                "Binned (color by bin)": "binned",
                "Treatment (Repaired vs Not)": "treatment",
                "Binned + Treatment Highlight": "binned_treatment",
            }
            selected_plot_desc = st.selectbox(
                "Select Plot Type:", list(plot_type_options.keys())
            )
            plot_type = plot_type_options[selected_plot_desc]

            # We create a plotter to figure out the bin indices
            # (We need the integer bins for age_bin and mileage_bin if the user gave an int or custom list.)
            # Let's create a temporary CarTruckPlotter to see what bins get assigned.
            # Calculate max indices for sliders based on user-provided bins
            max_age_bin = age_bins if isinstance(age_bins, int) else len(age_bins) - 1
            max_mileage_bin = (
                mileage_bins if isinstance(mileage_bins, int) else len(mileage_bins) - 1
            )

            bin_id = None
            if plot_type == "binned_treatment":
                st.markdown("### Bin Selection")
                # Handle Age Bin Slider
                if max_age_bin > 1:
                    age_bin_slider = st.slider("Age bin index", 0, max_age_bin - 1, 0)
                else:
                    st.info("Age has only one bin. No selection needed.")
                    age_bin_slider = 0

                # Handle Mileage Bin Slider
                if max_mileage_bin > 1:
                    mileage_bin_slider = st.slider(
                        "Mileage bin index", 0, max_mileage_bin - 1, 0
                    )
                else:
                    st.info("Mileage has only one bin. No selection needed.")
                    mileage_bin_slider = 0

                # Assign bin_id if both sliders exist
                bin_id = (age_bin_slider, mileage_bin_slider)
            # Generate the plots
            plotter = CarTruckPlotter(
                data_df,
                age_bins=age_bins if isinstance(age_bins, int) else age_bins,
                mileage_bins=mileage_bins
                if isinstance(mileage_bins, int)
                else mileage_bins,
            )
            plotter.plot_age_mileage_price(plot_type=plot_type, bin_id=bin_id)
            plotter.plot_age_vs_mileage(plot_type=plot_type, bin_id=bin_id)

            st.pyplot(plotter.age_mileage_price_plot)
            st.pyplot(plotter.age_mileage_plot)

        # ------------
        # Right column: Summary tables
        # ------------
        with col_right:
            st.subheader("Results Summary")

            # Compute OLS summary
            ols_df, cm_df = summary.display_summary()

            # Styling function for bolding the 'repaired_a_lot' row
            def highlight_repaired_a_lot(x):
                # x.name is the row index
                if x.name == "repaired_a_lot":
                    return ["font-weight: bold;"] * len(x)
                return [""] * len(x)

            # Display OLS summary with larger font
            st.markdown("**OLS Coefficients**")
            st.dataframe(ols_df.style.apply(highlight_repaired_a_lot, axis=1))

            # Display Coarsened Matching summary with larger font
            st.markdown("**Coarsened Matching Estimate**")
            st.dataframe(cm_df.style.apply(highlight_repaired_a_lot, axis=1))

    else:
        # If user hasn't run simulation yet
        st.warning(
            "Please configure parameters and click 'Run Simulation' in the sidebar."
        )


if __name__ == "__main__":
    main()
