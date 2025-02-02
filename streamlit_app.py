# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Union, List

from lack_of_common_support_problem.simulation import (
    NonOverlappingSupportSimulator,
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


def parse_float_list(list_str: str) -> List[float]:
    """
    Parses a comma-separated string into a list of floats.
    Example: "0, 50000, 100000" -> [0.0, 50000.0, 100000.0]
    """
    list_str = list_str.strip()
    if not list_str:
        return []
    parts = [x.strip() for x in list_str.split(",")]
    return [float(x) for x in parts if x]


def main():
    st.set_page_config(layout="wide")

    st.title("Simulation with Non-Overlapping Mileage Support")
    st.markdown(
        """
        This app simulates a scenario where we have a **binary treatment** 
        (represented by `repaired_a_lot`), and the **mileage distributions** 
        for treatment vs. control do **not overlap** (or overlap minimally).

        The price depends on:
        1. A **polynomial** relationship with mileage,
        2. A **linear** relationship with age,
        3. A constant **treatment effect** (negative by default).
            - We name this as repaired a lot for this example.
        
        We'll demonstrate how naive OLS (assuming mileage is linear) vs. coarsened matching
        can yield different estimates of the treatment effect, especially when the 
        mileage distribution is shifted and the functional form is misspecified.
        """
    )

    # ------------
    # Sidebar: Binning Options
    # ------------
    st.sidebar.header("Binning Options")
    st.sidebar.markdown(
        """Default simulation parameters generate a plateau between 50k and 120k miles.
        Hence, the default mileage bins are set to cutoffs at these points."""
    )
    st.sidebar.markdown(
        """The bins are used for both visualization and coarsened matching."""
    )

    st.sidebar.markdown(
        """If you want to use a single bin, just input a single number. 
        For multiple bins, input the cutoffs separated by commas."""
    )

    st.sidebar.markdown("""
        **Note:** Compared to the original simulation, setting mileage bins to 10 will result in the best possible fit""")

    default_age_bins_str = "1"
    default_mileage_bins_str = "0, 50000, 120000, 250000"

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
            "Could not parse mileage_bins. Reverting to default [0, 80000, 300000]."
        )
        mileage_bins = [0, 50000, 120000, 250000]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameter Options")

    with st.sidebar.expander("Basic Simulation Setup", expanded=False):
        n_samples = st.number_input(
            "n_samples", min_value=100, max_value=100000, value=2000
        )
        seed = st.number_input(
            "seed (0 = None)", min_value=0, max_value=999999, value=42
        )
        prob_treatment = st.slider("prob_treatment", 0.0, 1.0, 0.5)
        noise_std = st.number_input("noise_std", value=2000.0)

    with st.sidebar.expander("Age Distribution", expanded=False):
        age_mean = st.number_input("age_mean", value=5.0)
        age_std = st.number_input("age_std", value=2.0)

    with st.sidebar.expander(
        "Mileage Distribution (Treatment vs Control)", expanded=False
    ):
        mileage_mean_treated = st.number_input("mileage_mean_treated", value=105000.0)
        mileage_mean_control = st.number_input("mileage_mean_control", value=65000.0)
        mileage_std = st.number_input("mileage_std", value=40000.0)

    with st.sidebar.expander("Effect Coefficients", expanded=False):
        b_age = st.number_input("b_age (linear age effect)", value=-500.0)
        b_treatment = st.number_input(
            "b_treatment (constant treatment effect)", value=-3000.0
        )

    with st.sidebar.expander("Mileage Polynomial Setup", expanded=False):
        st.markdown(
            "Define control points for the mileage->price polynomial. "
            "These are (x_points, y_points), along with an optional polynomial degree. "
            "If degree is None, we use `len(x_points)-1`."
        )
        default_x_points = "0, 50000, 85000, 120000, 200000, 250000"
        default_y_points = "40000, 30000, 30000, 30000, 20000, 13000"

        x_points_str = st.text_input("Mileage X points", default_x_points)
        y_points_str = st.text_input(
            "Mileage Y points (default plateaus between 50k and 120k)", default_y_points
        )
        mileage_poly_degree = st.number_input(
            "Polynomial degree (0 = auto)", min_value=0, value=0
        )

        # Parse into lists
        x_points_list = parse_float_list(x_points_str)
        y_points_list = parse_float_list(y_points_str)
        # If degree=0 => we treat as None
        actual_poly_degree = None if mileage_poly_degree == 0 else mileage_poly_degree

    # "Run Simulation" button
    run_simulation = st.sidebar.button("Run Simulation")

    # Only run if button clicked
    if run_simulation:
        simulator = NonOverlappingSupportSimulator(
            n_samples=int(n_samples),
            seed=None if seed == 0 else int(seed),
            prob_treatment=prob_treatment,
            age_mean=age_mean,
            age_std=age_std,
            mileage_mean_treated=mileage_mean_treated,
            mileage_mean_control=mileage_mean_control,
            mileage_std=mileage_std,
            b_age=b_age,
            b_treatment=b_treatment,
            noise_std=noise_std,
            mileage_x_points=x_points_list if x_points_list else None,
            mileage_y_points=y_points_list if y_points_list else None,
            mileage_poly_degree=actual_poly_degree,
        )
        st.session_state["data_df"] = simulator.generate_data()
        st.session_state["simulator"] = simulator

    # Main layout
    col_left, col_right = st.columns([3, 2])

    if "data_df" in st.session_state and "simulator" in st.session_state:
        data_df = st.session_state["data_df"]
        simulator = st.session_state["simulator"]

        # Prepare summary object
        summary = CarTruckSummary(
            simulator, age_bins=age_bins, mileage_bins=mileage_bins
        )

        with col_left:
            st.subheader("Visualization Controls")
            plot_type_options = {
                "Simple Scatter": "scatter",
                "Binned (color by bin)": "binned",
                "Treatment (Repaired vs Not)": "treatment",
                "Binned + Treatment Highlight": "binned_treatment",
            }
            selected_plot_desc = st.selectbox(
                "Select Plot Type:", list(plot_type_options.keys()), index=2
            )
            plot_type = plot_type_options[selected_plot_desc]

            # Figure out bin_id if we do "binned_treatment"
            bin_id = None
            # Determine max index for bins
            if isinstance(age_bins, int):
                max_age_bin = age_bins
            else:
                max_age_bin = len(age_bins) - 1

            if isinstance(mileage_bins, int):
                max_mileage_bin = mileage_bins
            else:
                max_mileage_bin = len(mileage_bins) - 1

            if plot_type == "binned_treatment":
                st.markdown("### Bin Selection")
                # Age bin slider
                if max_age_bin > 1:
                    age_bin_slider = st.slider("Age bin index", 0, max_age_bin - 1, 0)
                else:
                    st.info("Only one Age bin. No selection needed.")
                    age_bin_slider = 0

                # Mileage bin slider
                if max_mileage_bin > 1:
                    mileage_bin_slider = st.slider(
                        "Mileage bin index", 0, max_mileage_bin - 1, 0
                    )
                else:
                    st.info("Only one Mileage bin. No selection needed.")
                    mileage_bin_slider = 0

                bin_id = (age_bin_slider, mileage_bin_slider)

            # Do the plotting
            plotter = CarTruckPlotter(data_df, age_bins, mileage_bins)
            plotter.plot_age_mileage_price(plot_type=plot_type, bin_id=bin_id)
            plotter.plot_age_vs_mileage(plot_type=plot_type, bin_id=bin_id)

            st.pyplot(plotter.age_mileage_price_plot)
            st.pyplot(plotter.age_mileage_plot)

        with col_right:
            st.subheader("Results Summary")
            ols_df, cm_df = summary.display_summary()

            # Helper to highlight the 'repaired_a_lot' row in bold
            def highlight_repaired_a_lot(x):
                if x.name == "repaired_a_lot":
                    return ["font-weight: bold;"] * len(x)
                return [""] * len(x)

            st.markdown("**OLS Coefficients**")
            st.dataframe(ols_df.style.apply(highlight_repaired_a_lot, axis=1))

            st.markdown("**Coarsened Matching Estimate**")
            st.dataframe(cm_df.style.apply(highlight_repaired_a_lot, axis=1))

            true_effect = (
                simulator.b_treatment
            )  # the constant treatment effect from the simulation
            st.markdown(f"**True Treatment Effect:** {true_effect}")

    else:
        st.warning(
            "Please configure parameters and click 'Run Simulation' on the sidebar."
        )


if __name__ == "__main__":
    main()
