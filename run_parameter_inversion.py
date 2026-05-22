from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    FERMI_CSV,
    FERMI_INVERSION_FIGURE,
    SWIFT_CSV,
    SWIFT_INVERSION_FIGURE,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass(frozen=True)
class ClosureScenario:
    label: str
    spin_period_ms: float
    mean_free_path_units: float = 1.0


@dataclass(frozen=True)
class ClosureStatistics:
    label: str
    n_events: int
    spin_period_ms: float
    mean_free_path_units: float
    lag_median_ms: float
    lag_scale_ms: float
    torus_radius_median: float
    torus_radius_std: float
    torus_radius_q16: float
    torus_radius_q84: float
    n_windings_median: float
    n_windings_std: float
    n_windings_q16: float
    n_windings_q84: float


class DiffusiveClosurePhysics:
    """
    Diffusive transport closure for GRB spectral lags.

    The measured lag constrains the transport product
    |tau| = N_diff * r_torus/r_LC * P.
    The closure ties the effective winding count to optical depth:
    N_diff = (r_torus/r_LC) / (2*pi*eta),
    where eta = lambda_mfp/r_LC.
    """

    @staticmethod
    def torus_radius_units(
        lag_ms: np.ndarray,
        spin_period_ms: float,
        mean_free_path_units: float,
    ) -> np.ndarray:
        if spin_period_ms <= 0:
            raise ValueError("spin_period_ms must be positive")
        if mean_free_path_units <= 0:
            raise ValueError("mean_free_path_units must be positive")
        return np.sqrt(
            lag_ms * 2 * np.pi * mean_free_path_units / spin_period_ms
        )

    @staticmethod
    def diffusive_windings(
        torus_radius_units: np.ndarray,
        mean_free_path_units: float,
    ) -> np.ndarray:
        return torus_radius_units / (2 * np.pi * mean_free_path_units)

    @staticmethod
    def lag_model_ms(
        torus_radius_units: np.ndarray,
        spin_period_ms: float,
        mean_free_path_units: float,
    ) -> np.ndarray:
        return spin_period_ms * torus_radius_units ** 2 / (
            2 * np.pi * mean_free_path_units
        )


class DiffusiveClosureAnalyzer:
    def __init__(self, scenario: ClosureScenario):
        self.scenario = scenario

    def analyze(self, lags_ms: np.ndarray) -> tuple[ClosureStatistics, pd.DataFrame]:
        lags_ms = np.asarray(lags_ms, dtype=float)
        lags_ms = lags_ms[np.isfinite(lags_ms) & (lags_ms > 0)]
        if len(lags_ms) == 0:
            raise ValueError("No positive finite lag magnitudes available")

        radius = DiffusiveClosurePhysics.torus_radius_units(
            lags_ms,
            self.scenario.spin_period_ms,
            self.scenario.mean_free_path_units,
        )
        windings = DiffusiveClosurePhysics.diffusive_windings(
            radius,
            self.scenario.mean_free_path_units,
        )
        model_lag = DiffusiveClosurePhysics.lag_model_ms(
            radius,
            self.scenario.spin_period_ms,
            self.scenario.mean_free_path_units,
        )

        df = pd.DataFrame({
            "lag_observed_ms": lags_ms,
            "lag_model_ms": model_lag,
            "torus_radius_units": radius,
            "n_diffusive_windings": windings,
            "transport_product_ms": lags_ms,
        })

        stats = ClosureStatistics(
            label=self.scenario.label,
            n_events=len(df),
            spin_period_ms=self.scenario.spin_period_ms,
            mean_free_path_units=self.scenario.mean_free_path_units,
            lag_median_ms=float(np.median(lags_ms)),
            lag_scale_ms=float(np.mean(lags_ms)),
            torus_radius_median=float(np.median(radius)),
            torus_radius_std=float(np.std(radius, ddof=1)),
            torus_radius_q16=float(np.percentile(radius, 16)),
            torus_radius_q84=float(np.percentile(radius, 84)),
            n_windings_median=float(np.median(windings)),
            n_windings_std=float(np.std(windings, ddof=1)),
            n_windings_q16=float(np.percentile(windings, 16)),
            n_windings_q84=float(np.percentile(windings, 84)),
        )
        return stats, df


class ClosurePlotter:
    def __init__(
        self,
        output_path: str,
        instrument_label: str,
        scenario: ClosureScenario,
        results: pd.DataFrame,
    ):
        self.output_path = output_path
        self.instrument_label = instrument_label
        self.scenario = scenario
        self.results = results

    def create_plot(self) -> str | None:
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Cannot generate plot.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        self._plot_lag_distribution(axes[0, 0])
        self._plot_radius_distribution(axes[0, 1])
        self._plot_winding_distribution(axes[1, 0])
        self._plot_model_check(axes[1, 1])

        fig.suptitle(
            (
                f"{self.instrument_label} diffusive-closure inversion "
                f"(P={self.scenario.spin_period_ms:g} ms, "
                f"eta={self.scenario.mean_free_path_units:g})"
            ),
            fontsize=13,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(self.output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Closure plot saved to {self.output_path}")
        return self.output_path

    def _plot_lag_distribution(self, ax):
        values = self.results["lag_observed_ms"].to_numpy()
        bins = np.logspace(0, np.log10(values.max()), 40)
        ax.hist(values, bins=bins, color="#4c72b0", alpha=0.75, edgecolor="black")
        ax.axvline(np.median(values), color="#c44e52", linestyle="--",
                   label=f"Median: {np.median(values) / 1000:.1f} s")
        ax.set_xscale("log")
        ax.set_xlabel("|Lag| (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Observed Lag Magnitudes")
        ax.legend()
        ax.grid(alpha=0.3, which="both")

    def _plot_radius_distribution(self, ax):
        values = self.results["torus_radius_units"].to_numpy()
        ax.hist(values, bins=40, color="#55a868", alpha=0.75, edgecolor="black")
        ax.axvline(np.median(values), color="#c44e52", linestyle="--",
                   label=f"Median: {np.median(values):.2f} r_LC")
        ax.set_xlabel("Torus Radius (r_LC)")
        ax.set_ylabel("Count")
        ax.set_title("Diffusive Torus Scale")
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_winding_distribution(self, ax):
        values = self.results["n_diffusive_windings"].to_numpy()
        ax.hist(values, bins=40, color="#dd8452", alpha=0.75, edgecolor="black")
        ax.axvline(np.median(values), color="#c44e52", linestyle="--",
                   label=f"Median: {np.median(values):.2f}")
        ax.set_xlabel("N_diff (windings)")
        ax.set_ylabel("Count")
        ax.set_title("Diffusive Winding Count")
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_model_check(self, ax):
        observed = self.results["lag_observed_ms"].to_numpy()
        modeled = self.results["lag_model_ms"].to_numpy()
        ax.scatter(observed, modeled, alpha=0.45, s=16, color="#8172b3")
        low = min(observed.min(), modeled.min())
        high = max(observed.max(), modeled.max())
        ax.plot([low, high], [low, high], color="#c44e52",
                linestyle="--", label="Closure identity")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Observed |Lag| (ms)")
        ax.set_ylabel("Closure Model |Lag| (ms)")
        ax.set_title("Forward Check")
        ax.legend()
        ax.grid(alpha=0.3, which="both")


class ClosureStudy:
    def __init__(
        self,
        filepath: str,
        output_prefix: str,
        output_figure: str,
        primary_scenario: ClosureScenario,
        comparison_scenarios: Iterable[ClosureScenario],
    ):
        self.filepath = filepath
        self.output_prefix = output_prefix
        self.output_figure = output_figure
        self.primary_scenario = primary_scenario
        self.comparison_scenarios = list(comparison_scenarios)
        self.df = self._load_data()
        self.lags_ms = np.abs(self.df["lag_ms"].to_numpy(dtype=float))

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        if "is_significant" in df.columns:
            df = df[df["is_significant"].astype(bool)].copy()
        return df

    def run(self) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print(f"{self.output_prefix.upper()}: DIFFUSIVE CLOSURE INVERSION")
        print("=" * 70)
        print(f"Input file: {self.filepath}")
        print(f"Significant lags: {len(self.lags_ms)}")

        all_stats = []
        primary_results = None

        for scenario in [self.primary_scenario, *self.comparison_scenarios]:
            stats, results = DiffusiveClosureAnalyzer(scenario).analyze(self.lags_ms)
            all_stats.append(stats)
            self._print_statistics(stats)
            enriched_results = self._with_event_metadata(results)

            if scenario == self.primary_scenario:
                primary_results = enriched_results
                csv_path = f"{self.output_prefix}_diffusive_closure.csv"
                enriched_results.to_csv(csv_path, index=False)
                print(f"Saved primary closure table to {csv_path}")

        if primary_results is not None:
            ClosurePlotter(
                output_path=self.output_figure,
                instrument_label=self.output_prefix.upper(),
                scenario=self.primary_scenario,
                results=primary_results,
            ).create_plot()

        stats_df = pd.DataFrame([stats.__dict__ for stats in all_stats])
        stats_path = f"{self.output_prefix}_diffusive_closure_summary.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved closure summary to {stats_path}")
        return stats_df

    def _with_event_metadata(self, results: pd.DataFrame) -> pd.DataFrame:
        metadata_columns = [
            "instrument",
            "filename",
            "grb_name",
            "grb_time_utc",
            "ra",
            "dec",
            "lag_ms",
            "lag_error",
            "lag_significance",
            "lag_type",
            "lag_class",
            "t90_s",
            "duration_class",
        ]
        available_columns = [
            column for column in metadata_columns if column in self.df.columns
        ]
        metadata = self.df[available_columns].reset_index(drop=True)
        enriched = pd.concat(
            [metadata, results.reset_index(drop=True)],
            axis=1,
        )
        return enriched

    @staticmethod
    def _print_statistics(stats: ClosureStatistics) -> None:
        print(f"\nScenario: {stats.label}")
        print(
            f"  P = {stats.spin_period_ms:g} ms, "
            f"eta = {stats.mean_free_path_units:g}"
        )
        print(
            f"  Lag median/mean scale: "
            f"{stats.lag_median_ms / 1000:.2f} / "
            f"{stats.lag_scale_ms / 1000:.2f} s"
        )
        print(
            f"  r_torus: median {stats.torus_radius_median:.2f} r_LC "
            f"(16-84%: {stats.torus_radius_q16:.2f}-"
            f"{stats.torus_radius_q84:.2f})"
        )
        print(
            f"  N_diff:  median {stats.n_windings_median:.2f} "
            f"(16-84%: {stats.n_windings_q16:.2f}-"
            f"{stats.n_windings_q84:.2f})"
        )


if __name__ == "__main__":
    primary = ClosureScenario(
        label="effective transport scale",
        spin_period_ms=1500.0,
        mean_free_path_units=1.0,
    )
    comparisons = [
        ClosureScenario(
            label="fast engine scale",
            spin_period_ms=1.5,
            mean_free_path_units=1.0,
        ),
        ClosureScenario(
            label="more transparent transport",
            spin_period_ms=1500.0,
            mean_free_path_units=3.0,
        ),
        ClosureScenario(
            label="denser transport",
            spin_period_ms=1500.0,
            mean_free_path_units=0.3,
        ),
    ]

    fermi_summary = ClosureStudy(
        filepath=FERMI_CSV,
        output_prefix="fermi",
        output_figure=FERMI_INVERSION_FIGURE,
        primary_scenario=primary,
        comparison_scenarios=comparisons,
    ).run()

    swift_summary = ClosureStudy(
        filepath=SWIFT_CSV,
        output_prefix="swift",
        output_figure=SWIFT_INVERSION_FIGURE,
        primary_scenario=primary,
        comparison_scenarios=comparisons,
    ).run()
