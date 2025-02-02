# simulation.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Literal, Union


def build_mileage_polynomial(
    x_points: List[float], y_points: List[float], degree: Optional[int] = None
) -> np.poly1d:
    """
    Fits a polynomial to the specified (x_points, y_points).
    If degree is None, uses degree = len(x_points) - 1.
    Returns the polynomial as an np.poly1d object.
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length.")

    if degree is None:
        degree = len(x_points) - 1

    coeffs = np.polyfit(x_points, y_points, deg=degree)
    return np.poly1d(coeffs)


class NonOverlappingSupportSimulator:
    """
    Simulates data for a single population with a binary treatment 'repaired_a_lot',
    where the mileage distributions for treatment vs. control have little/no overlap.

    The price is determined by:
        price = b_age * age
               + poly_mileage(mileage)
               + b_treatment * repaired_a_lot
               + noise
    where poly_mileage is a user-defined polynomial over mileage.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seed: Optional[int] = None,
        prob_treatment: float = 0.5,
        # Age distribution
        age_mean: float = 5,
        age_std: float = 2,
        # Mileage distributions
        mileage_mean_treated: float = 150_000,
        mileage_mean_control: float = 50_000,
        mileage_std: float = 15_000,
        # Coefficients
        b_age: float = -300.0,
        b_treatment: float = -3000.0,
        noise_std: float = 2000.0,
        # Polynomial for mileage->price
        mileage_x_points: Optional[List[float]] = None,
        mileage_y_points: Optional[List[float]] = None,
        mileage_poly_degree: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate.
        seed : int or None
            Random seed for reproducibility. If None, no fixed seed.
        prob_treatment : float
            Probability that an individual is in the treatment (repaired_a_lot=1) group.
        age_mean, age_std : float
            Mean and standard deviation for the age distribution (normal, truncated at 0).
        mileage_mean_treated, mileage_mean_control, mileage_std : float
            Means (treated vs control) and std for the mileage distribution (normal, truncated at 0).
            We assume these are far enough apart to produce minimal overlap.
        b_age, b_treatment : float
            The linear coefficient on age, and constant effect of treatment.
        noise_std : float
            Standard deviation for the additive noise.
        mileage_x_points, mileage_y_points : List[float]
            Control points for the polynomial relating mileage to price.
            Must be of the same length. If not provided, defaults will be used.
        mileage_poly_degree : int or None
            Degree of polynomial to fit. If None, we use (len(mileage_x_points) - 1).
        """

        self.n_samples = n_samples
        self.seed = seed
        self.prob_treatment = prob_treatment

        self.age_mean = age_mean
        self.age_std = age_std

        self.mileage_mean_treated = mileage_mean_treated
        self.mileage_mean_control = mileage_mean_control
        self.mileage_std = mileage_std

        self.b_age = b_age
        self.b_treatment = b_treatment
        self.noise_std = noise_std

        # Default polynomial control points: ensures a strongly decreasing section,
        # some plateau, and then more decline. Adjust as needed.
        if mileage_x_points is None:
            mileage_x_points = [0, 50_000, 120_000, 200_000]
        if mileage_y_points is None:
            # Suppose these are the "base price contributions" from mileage alone
            mileage_y_points = [20_000, 10_000, 9_000, 0]

        self.mileage_poly = build_mileage_polynomial(
            mileage_x_points, mileage_y_points, degree=mileage_poly_degree
        )

    def generate_data(self) -> pd.DataFrame:
        """Generates synthetic data for the simulation, with non-overlapping mileage distributions."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Treatment assignment
        repaired_a_lot = np.random.binomial(1, self.prob_treatment, size=self.n_samples)

        # Age (same distribution for both groups)
        age = np.random.normal(self.age_mean, self.age_std, size=self.n_samples)
        age = np.clip(age, 0, None)  # No negative ages

        # Mileage (group-specific means, same std)
        mileage = np.where(
            repaired_a_lot == 1,
            np.random.normal(
                self.mileage_mean_treated, self.mileage_std, self.n_samples
            ),
            np.random.normal(
                self.mileage_mean_control, self.mileage_std, self.n_samples
            ),
        )
        mileage = np.clip(mileage, 0, None)

        # True price from polynomial + linear age + treatment effect
        mileage_component = self.mileage_poly(mileage)
        true_price = (
            mileage_component + self.b_age * age + self.b_treatment * repaired_a_lot
        )

        # Add noise
        noise = np.random.normal(0, self.noise_std, self.n_samples)
        price = true_price + noise

        return pd.DataFrame(
            {
                "repaired_a_lot": repaired_a_lot,
                "age": age,
                "mileage": mileage,
                "price": price,
                "true_price": true_price,
            }
        )

    def run_ols(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, float]:
        """
        Runs an OLS regression on the data:
            price ~ age + mileage + repaired_a_lot
        (Linear in mileage, which is *misspecified* if the true relation is polynomial.)
        """
        X_vars = ["mileage", "age", "repaired_a_lot"]
        X = df[X_vars].values
        y = df["price"].values
        model = LinearRegression().fit(X, y)
        return X_vars, model.coef_, model.intercept_

    def run_coarsened_matching(
        self,
        df: pd.DataFrame,
        age_bins: Optional[Union[int, List[float]]] = 5,
        mileage_bins: Optional[Union[int, List[float]]] = 5,
    ) -> Optional[float]:
        """
        Performs coarsened matching on mileage and age with AND condition.
        Returns the estimated treatment effect of repairs on price.
        """
        df = df.copy()

        # Determine bins for age
        if isinstance(age_bins, int):
            df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=False)
        elif isinstance(age_bins, list):
            df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=False)
        else:
            raise ValueError("age_bins must be int or a list of cutoff points.")

        # Determine bins for mileage
        if isinstance(mileage_bins, int):
            df["mileage_bin"] = pd.cut(df["mileage"], bins=mileage_bins, labels=False)
        elif isinstance(mileage_bins, list):
            df["mileage_bin"] = pd.cut(df["mileage"], bins=mileage_bins, labels=False)
        else:
            raise ValueError("mileage_bins must be int or a list of cutoff points.")

        df["combined_bin"] = (
            df["age_bin"].astype(str) + "_" + df["mileage_bin"].astype(str)
        )

        differences = []
        weights = []

        for _, group in df.groupby("combined_bin"):
            group_rep = group[group["repaired_a_lot"] == 1]
            group_norep = group[group["repaired_a_lot"] == 0]

            if not group_rep.empty and not group_norep.empty:
                diff = group_rep["price"].mean() - group_norep["price"].mean()
                differences.append(diff)
                weights.append(len(group))

        if not weights:
            return None

        return np.average(differences, weights=weights)


class CarTruckPlotter:
    """
    Handles plotting Age vs Price, Mileage vs Price, and Age vs Mileage
    for the data, optionally by treatment or bins.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        age_bins: Union[int, List[float]],
        mileage_bins: Union[int, List[float]],
    ):
        self.df = df.copy()
        self.age_bins = age_bins
        self.mileage_bins = mileage_bins

        # Add bin columns to the DataFrame
        if isinstance(self.age_bins, int):
            self.df["age_bin"] = pd.cut(
                self.df["age"], bins=self.age_bins, labels=False
            )
        else:
            self.df["age_bin"] = pd.cut(
                self.df["age"], bins=self.age_bins, labels=False
            )

        if isinstance(self.mileage_bins, int):
            self.df["mileage_bin"] = pd.cut(
                self.df["mileage"], bins=self.mileage_bins, labels=False
            )
        else:
            self.df["mileage_bin"] = pd.cut(
                self.df["mileage"], bins=self.mileage_bins, labels=False
            )

        self.age_mileage_price_plot = None
        self.age_mileage_plot = None

    def plot_age_mileage_price(
        self,
        plot_type: Literal["scatter", "binned", "treatment", "binned_treatment"],
        bin_id: Optional[tuple] = None,
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        for ax, x_col, x_label in zip(axs, ["age", "mileage"], ["Age", "Mileage"]):
            if plot_type == "scatter":
                ax.scatter(self.df[x_col], self.df["price"], alpha=0.5)

            elif plot_type == "binned":
                for _, sub in self.df.groupby(["age_bin", "mileage_bin"]):
                    ax.scatter(sub[x_col], sub["price"], alpha=0.5)

            elif plot_type == "treatment":
                for treatment, color, label in [
                    (0, "blue", "Not repaired a lot (Control, 0)"),
                    (1, "red", "Repaired a lot (Treated, 1)"),
                ]:
                    ax.scatter(
                        self.df[self.df["repaired_a_lot"] == treatment][x_col],
                        self.df[self.df["repaired_a_lot"] == treatment]["price"],
                        alpha=0.5,
                        color=color,
                        label=label,
                    )
                ax.legend()

            elif plot_type == "binned_treatment" and bin_id is not None:
                age_bin, mileage_bin = bin_id
                for treatment, color, label in [
                    (0, "blue", "Not repaired a lot (Control, 0)"),
                    (1, "red", "Repaired a lot (Treated, 1)"),
                ]:
                    highlight = self.df[
                        (self.df["age_bin"] == age_bin)
                        & (self.df["mileage_bin"] == mileage_bin)
                        & (self.df["repaired_a_lot"] == treatment)
                    ]
                    other = self.df[~self.df.index.isin(highlight.index)]

                    # Plot non-highlighted points in gray
                    ax.scatter(
                        other[x_col],
                        other["price"],
                        alpha=0.3,
                        color="gray",
                    )

                    # Plot highlighted points in their respective colors
                    ax.scatter(
                        highlight[x_col],
                        highlight["price"],
                        alpha=0.7,
                        color=color,
                        label=f"Bin {age_bin}_{mileage_bin} ({label})",
                    )
                ax.legend()

            ax.set_title(f"{x_label} vs Price")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Price")

        plt.tight_layout()
        self.age_mileage_price_plot = fig

    def plot_age_vs_mileage(
        self,
        plot_type: Literal["scatter", "binned", "treatment", "binned_treatment"],
        bin_id: Optional[tuple] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))

        if plot_type == "scatter":
            ax.scatter(self.df["age"], self.df["mileage"], alpha=0.5)

        elif plot_type == "binned":
            for _, sub in self.df.groupby(["age_bin", "mileage_bin"]):
                ax.scatter(sub["age"], sub["mileage"], alpha=0.5)

        elif plot_type == "treatment":
            for treatment, color, label in [
                (0, "blue", "Not repaired a lot (Control, 0)"),
                (1, "red", "Repaired a lot (Treated, 1)"),
            ]:
                ax.scatter(
                    self.df[self.df["repaired_a_lot"] == treatment]["age"],
                    self.df[self.df["repaired_a_lot"] == treatment]["mileage"],
                    alpha=0.5,
                    color=color,
                    label=label,
                )
            ax.legend()

        elif plot_type == "binned_treatment" and bin_id is not None:
            age_bin, mileage_bin = bin_id
            for treatment, color, label in [
                (0, "blue", "Not repaired a lot (Control, 0)"),
                (1, "red", "Repaired a lot (Treated, 1)"),
            ]:
                highlight = self.df[
                    (self.df["age_bin"] == age_bin)
                    & (self.df["mileage_bin"] == mileage_bin)
                    & (self.df["repaired_a_lot"] == treatment)
                ]
                other = self.df[~self.df.index.isin(highlight.index)]

                ax.scatter(
                    other["age"],
                    other["mileage"],
                    alpha=0.3,
                    color="gray",
                )
                ax.scatter(
                    highlight["age"],
                    highlight["mileage"],
                    alpha=0.7,
                    color=color,
                    label=f"Bin {age_bin}_{mileage_bin} ({label})",
                )
            ax.legend()

        ax.set_title("Age vs Mileage")
        ax.set_xlabel("Age")
        ax.set_ylabel("Mileage")

        plt.tight_layout()
        self.age_mileage_plot = fig


class CarTruckSummary:
    """
    Provides a summary of analysis results (OLS vs Coarsened Matching).
    """

    def __init__(
        self,
        simulator: NonOverlappingSupportSimulator,
        age_bins: Union[int, List[float]],
        mileage_bins: Union[int, List[float]],
    ):
        self.simulator = simulator
        self.df = simulator.generate_data()
        self.age_bins = age_bins
        self.mileage_bins = mileage_bins

    def calculate_ols_summary(self) -> pd.DataFrame:
        """
        Calculates results from OLS.
        """
        xvars, coefs, intercept = self.simulator.run_ols(self.df)
        # Build a table
        results = pd.DataFrame(
            {
                "Variable": xvars + ["Intercept"],
                "Coefficient": list(coefs) + [intercept],
            }
        ).set_index("Variable")
        return results

    def calculate_cm_summary(
        self,
        age_bins: Optional[Union[int, List[float]]] = None,
        mileage_bins: Optional[Union[int, List[float]]] = None,
    ) -> pd.DataFrame:
        """
        Calculates results from coarsened matching.
        """
        if age_bins is None:
            age_bins = self.age_bins
        if mileage_bins is None:
            mileage_bins = self.mileage_bins

        cm_estimate = self.simulator.run_coarsened_matching(
            self.df, age_bins, mileage_bins
        )
        cm_summary = pd.DataFrame(
            {
                "": ["repaired_a_lot"],
                "Coarsened Matching": [cm_estimate],
            }
        )
        return cm_summary.set_index("")

    def display_summary(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns both OLS and CM summaries.
        """
        ols_summary = self.calculate_ols_summary()
        cm_summary = self.calculate_cm_summary()
        return ols_summary, cm_summary

    def link_visualization(self, **plot_kwargs) -> Tuple[plt.Figure, plt.Figure]:
        """
        Integrates with the visualization workflow.
        """
        plotter = CarTruckPlotter(self.df, self.age_bins, self.mileage_bins)
        plotter.plot_age_mileage_price(**plot_kwargs)
        plotter.plot_age_vs_mileage(**plot_kwargs)
        return plotter.age_mileage_price_plot, plotter.age_mileage_plot


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    simulator = NonOverlappingSupportSimulator(
        n_samples=2000,
        seed=42,
        prob_treatment=0.5,
        age_mean=5,
        age_std=2,
        mileage_mean_treated=140_000,
        mileage_mean_control=40_000,
        mileage_std=15_000,
        b_age=-500.0,
        b_treatment=-3000.0,
        noise_std=2000.0,
        # Example polynomial: 4 points => cubic
        mileage_x_points=[0, 50_000, 120_000, 200_000],
        mileage_y_points=[20_000, 10_000, 9_000, 0],
        mileage_poly_degree=3,
    )

    summary = CarTruckSummary(simulator, age_bins=1, mileage_bins=[0, 80_000, 300_000])
    ols_summary, cm_summary = summary.display_summary()

    # Visualization example
    fig_price, fig_age_mileage = summary.link_visualization(
        plot_type="binned_treatment", bin_id=(0, 0)
    )
    # Show tables
    print("OLS Summary:\n", ols_summary)
    print("\nCM Summary:\n", cm_summary)

    # If you want to display or save the plots in Python:
    # fig_price.show()
    # fig_age_mileage.show()
