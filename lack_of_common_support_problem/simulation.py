# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Literal, Union


class CarTruckRepairsSimulator:
    """
    Simulates data for cars vs. trucks with repairs as a treatment variable.
    Demonstrates how omitting 'car_type' in a naive OLS can bias the estimate of repairs.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seed: Optional[int] = None,
        prob_truck: float = 0.4,
        prob_repair_if_car: float = 0.3,
        prob_repair_if_truck: float = 0.7,
        age_mean_car: float = 5,
        age_mean_truck: float = 7,
        age_std_car: float = 2,
        age_std_truck: float = 3,
        mileage_mean_car: float = 40000,
        mileage_mean_truck: float = 120000,
        mileage_std_car: float = 8000,
        mileage_std_truck: float = 30000,
        base_price_car: float = 20000,
        truck_premium: float = 15000,
        b_age: float = -500,
        b_mileage: float = -0.05,
        b_repairs: float = -3000,
        noise_std: float = 2000,
    ) -> None:
        self.n_samples = n_samples
        self.seed = seed

        self.prob_truck = prob_truck
        self.prob_repair_if_car = prob_repair_if_car
        self.prob_repair_if_truck = prob_repair_if_truck

        self.age_mean_car = age_mean_car
        self.age_mean_truck = age_mean_truck
        self.age_std_car = age_std_car
        self.age_std_truck = age_std_truck

        self.mileage_mean_car = mileage_mean_car
        self.mileage_mean_truck = mileage_mean_truck
        self.mileage_std_car = mileage_std_car
        self.mileage_std_truck = mileage_std_truck

        self.base_price_car = base_price_car
        self.truck_premium = truck_premium
        self.b_age = b_age
        self.b_mileage = b_mileage
        self.b_repairs = b_repairs
        self.noise_std = noise_std

    def generate_data(self) -> pd.DataFrame:
        """Generates synthetic data for the simulation."""
        if self.seed is not None:
            np.random.seed(self.seed)

        car_type = np.random.binomial(1, self.prob_truck, size=self.n_samples)
        repaired_a_lot = np.array(
            [
                np.random.binomial(
                    1, self.prob_repair_if_car if c == 0 else self.prob_repair_if_truck
                )
                for c in car_type
            ]
        )

        age = np.where(
            car_type == 0,
            np.random.normal(self.age_mean_car, self.age_std_car, self.n_samples),
            np.random.normal(self.age_mean_truck, self.age_std_truck, self.n_samples),
        )
        age = np.clip(age, 0, None)

        mileage = np.where(
            car_type == 0,
            np.random.normal(
                self.mileage_mean_car, self.mileage_std_car, self.n_samples
            ),
            np.random.normal(
                self.mileage_mean_truck, self.mileage_std_truck, self.n_samples
            ),
        )
        mileage = np.clip(mileage, 0, None)

        true_price = (
            self.base_price_car
            + self.truck_premium * car_type
            + self.b_age * age
            + self.b_mileage * mileage
            + self.b_repairs * repaired_a_lot
        )
        noise = np.random.normal(0, self.noise_std, self.n_samples)
        price = true_price + noise

        return pd.DataFrame(
            {
                "car_type": np.where(car_type == 1, "Truck", "Car"),
                "repaired_a_lot": repaired_a_lot,
                "age": age,
                "mileage": mileage,
                "price": price,
                "true_price": true_price,
            }
        )

    def run_ols(
        self, df: pd.DataFrame, include_car_type: bool = False
    ) -> Tuple[List[str], np.ndarray, float]:
        """Runs an OLS regression on the data."""
        X_vars = ["mileage", "age", "repaired_a_lot"]
        if include_car_type:
            df["is_truck"] = (df["car_type"] == "Truck").astype(int)
            X_vars.append("is_truck")

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

        Parameters:
            df (pd.DataFrame): Input data.
            age_bins (int or List[float]): Number of bins or specific cutoff points for age.
            mileage_bins (int or List[float]): Number of bins or specific cutoff points for mileage.
        """
        df = df.copy()

        # Determine bins for age
        if isinstance(age_bins, int):
            df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=False)
        elif isinstance(age_bins, list):
            df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=False)
        else:
            raise ValueError("age_bins must be an int or a list of cutoff points.")

        # Determine bins for mileage
        if isinstance(mileage_bins, int):
            df["mileage_bin"] = pd.cut(df["mileage"], bins=mileage_bins, labels=False)
        elif isinstance(mileage_bins, list):
            df["mileage_bin"] = pd.cut(df["mileage"], bins=mileage_bins, labels=False)
        else:
            raise ValueError("mileage_bins must be an int or a list of cutoff points.")

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
    A class for creating various plots related to car and truck data.
    Handles plotting Age vs Price, Mileage vs Price, and Age vs Mileage.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        age_bins: int,
        mileage_bins: int,
    ):
        self.df = df.copy()
        self.age_bins = age_bins
        self.mileage_bins = mileage_bins

        # Add bin columns to the DataFrame
        self.df["age_bin"] = pd.cut(self.df["age"], bins=self.age_bins, labels=False)
        self.df["mileage_bin"] = pd.cut(
            self.df["mileage"], bins=self.mileage_bins, labels=False
        )

        # Plot containers
        self.age_mileage_price_plot = None
        self.age_mileage_plot = None

    def plot_age_mileage_price(
        self,
        plot_type: Literal["scatter", "binned", "treatment", "binned_treatment"],
        bin_id: Optional[tuple] = None,
    ) -> None:
        """
        Plots Age vs Price and Mileage vs Price with optional visualizations.

        Parameters:
            plot_type (str): The type of plot to generate.
                Options:
                    - "scatter": Simple scatterplots without bins or treatment.
                    - "binned": Binned plots with bin coloring.
                    - "treatment": Non-binned plots showing treatment.
                    - "binned_treatment": Binned plots showing treatment by bin.
            bin_id (tuple, optional): Specific bin (age_bin, mileage_bin) to highlight for "binned_treatment".
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        for ax, x_col, x_label in zip(axs, ["age", "mileage"], ["Age", "Mileage"]):
            if plot_type == "scatter":
                ax.scatter(self.df[x_col], self.df["price"], alpha=0.5)

            elif plot_type == "binned":
                for _, sub in self.df.groupby(["age_bin", "mileage_bin"]):
                    ax.scatter(sub[x_col], sub["price"], alpha=0.5, label="Bin")

            elif plot_type == "treatment":
                for treatment, color, label in [
                    (0, "blue", "Not Repaired"),
                    (1, "red", "Repaired"),
                ]:
                    ax.scatter(
                        self.df[self.df["repaired_a_lot"] == treatment][x_col],
                        self.df[self.df["repaired_a_lot"] == treatment]["price"],
                        alpha=0.5,
                        label=label,
                        color=color,
                    )
                ax.legend()

            elif plot_type == "binned_treatment" and bin_id:
                age_bin, mileage_bin = bin_id
                for treatment, color, label in [
                    (0, "blue", "Not Repaired"),
                    (1, "red", "Repaired"),
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
                        label=f"Bin {age_bin}_{mileage_bin} ({label})",
                        color=color,
                    )
                ax.legend()

            ax.set_title(f"{x_label} vs Price")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Price")

        plt.tight_layout()
        self.age_mileage_price_plot = fig

    def plot_age_vs_mileage(
        self,
        plot_type: Literal["scatter", "binned", "treatment", "binned_treatment"],
        bin_id: Optional[tuple] = None,
    ) -> None:
        """
        Plots Age vs Mileage with optional visualizations.

        Parameters:
            plot_type (str): The type of plot to generate.
                Options:
                    - "scatter": Simple scatterplot without bins or treatment.
                    - "binned": Binned plot with bin coloring.
                    - "treatment": Non-binned plot showing treatment.
                    - "binned_treatment": Binned plot showing treatment by bin.
            bin_id (tuple, optional): Specific bin (age_bin, mileage_bin) to highlight for "binned_treatment".
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        if plot_type == "scatter":
            ax.scatter(self.df["age"], self.df["mileage"], alpha=0.5)

        elif plot_type == "binned":
            for _, sub in self.df.groupby(["age_bin", "mileage_bin"]):
                ax.scatter(sub["age"], sub["mileage"], alpha=0.5, label="Bin")

        elif plot_type == "treatment":
            for treatment, color, label in [
                (0, "blue", "Not Repaired"),
                (1, "red", "Repaired"),
            ]:
                ax.scatter(
                    self.df[self.df["repaired_a_lot"] == treatment]["age"],
                    self.df[self.df["repaired_a_lot"] == treatment]["mileage"],
                    alpha=0.5,
                    label=label,
                    color=color,
                )
            ax.legend()

        elif plot_type == "binned_treatment" and bin_id:
            age_bin, mileage_bin = bin_id
            for treatment, color, label in [
                (0, "blue", "Not Repaired"),
                (1, "red", "Repaired"),
            ]:
                highlight = self.df[
                    (self.df["age_bin"] == age_bin)
                    & (self.df["mileage_bin"] == mileage_bin)
                    & (self.df["repaired_a_lot"] == treatment)
                ]
                other = self.df[~self.df.index.isin(highlight.index)]

                # Plot non-highlighted points in gray
                ax.scatter(
                    other["age"],
                    other["mileage"],
                    alpha=0.3,
                    color="gray",
                )

                # Plot highlighted points in their respective colors
                ax.scatter(
                    highlight["age"],
                    highlight["mileage"],
                    alpha=0.7,
                    label=f"Bin {age_bin}_{mileage_bin} ({label})",
                    color=color,
                )
            ax.legend()

        ax.set_title("Age vs Mileage")
        ax.set_xlabel("Age")
        ax.set_ylabel("Mileage")

        plt.tight_layout()
        self.age_mileage_plot = fig


class CarTruckSummary:
    """
    Provides a summary of analysis results, including OLS naive,
    OLS with car_type, and coarsened matching.
    """

    def __init__(
        self,
        simulator: CarTruckRepairsSimulator,
        age_bins: int,
        mileage_bins: int,
    ):
        """
        Initialize the summary class.

        Parameters:
            simulator (CarTruckRepairsSimulator): The simulator instance.
            df (pd.DataFrame): The data generated from the simulator.
            n_age_bins (int): Number of bins for age discretization.
            n_mileage_bins (int): Number of bins for mileage discretization.
        """
        self.simulator = simulator
        self.df = simulator.generate_data()
        self.age_bins = age_bins
        self.mileage_bins = mileage_bins

    def calculate_ols_summary(self) -> pd.DataFrame:
        """
        Calculates results from naive OLS and OLS with car_type.

        Returns:
            pd.DataFrame: Summary table of OLS results.
        """
        # Naive OLS (without car_type)
        xvars_naive, coefs_naive, intercept_naive = self.simulator.run_ols(
            self.df, include_car_type=False
        )
        naive_results = {
            "Variable": xvars_naive + ["Intercept"],
            "Naive OLS Coefficients": list(coefs_naive) + [intercept_naive],
        }

        # OLS with car_type
        xvars_full, coefs_full, intercept_full = self.simulator.run_ols(
            self.df, include_car_type=True
        )
        full_results = {
            "Variable": xvars_full + ["Intercept"],
            "OLS with car_type Coefficients": list(coefs_full) + [intercept_full],
        }

        # Merge results
        summary_df = pd.DataFrame(naive_results).merge(
            pd.DataFrame(full_results), on="Variable", how="outer"
        )
        return summary_df.set_index("Variable")

    def calculate_cm_summary(
        self,
        age_bins: Optional[Union[int, List[float]]] = None,
        mileage_bins: Optional[Union[int, List[float]]] = None,
    ) -> pd.DataFrame:
        """
        Calculates results from coarsened matching.

        Returns:
            pd.DataFrame: Summary table of CM results.
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

    def display_summary(self) -> None:
        """
        Displays the summary table including OLS results and CM estimate.
        """
        # Generate summaries
        ols_summary = self.calculate_ols_summary()
        cm_summary = self.calculate_cm_summary()

        # Display tables
        return ols_summary, cm_summary

    def link_visualization(self, **plot_kwargs) -> None:
        """
        Integrates with the visualization workflow.

        Parameters:
            plot_kwargs: Additional arguments for the plotting functions.
        """
        plotter = CarTruckPlotter(self.df, self.age_bins, self.mileage_bins)

        plotter.plot_age_mileage_price(**plot_kwargs)
        plotter.plot_age_vs_mileage(**plot_kwargs)

        return plotter.age_mileage_price_plot, plotter.age_mileage_plot


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sim = CarTruckRepairsSimulator(
        n_samples=2000,
        seed=42,
        prob_truck=0.4,
        prob_repair_if_car=0.2,
        prob_repair_if_truck=0.8,
        age_mean_car=5,
        age_mean_truck=7,
        mileage_mean_car=30000,
        mileage_mean_truck=160000,
        # Make the distributions fairly separate:
        mileage_std_car=8000,
        mileage_std_truck=30000,
        truck_premium=50000,
        b_repairs=-3000,  # actual effect is negative
        b_age=-500,
        b_mileage=-0.1,
    )

    # Initialize the summary class
    summary = CarTruckSummary(sim, age_bins=1, mileage_bins=[0, 60000, 250000])

    # Return summary table
    ols_summary = summary.calculate_ols_summary()
    cm_summary = summary.calculate_cm_summary()

    # %%
    # Link visualization with binning and additional arguments
    age_mileage_price_plot, age_mileage_plot = summary.link_visualization(
        plot_type="binned_treatment", bin_id=(0, 0)
    )
    # %%
    age_mileage_plot
    # %%
    age_mileage_price_plot

    # %%
    ols_summary
    # %%
    cm_summary
    # %%
    plotter = CarTruckPlotter(summary.df, 1, [0, 60000, 250000])
    plotter.plot_age_mileage_price("binned_treatment", bin_id=(0, 0))
    plotter.age_mileage_price_plot
# %%
