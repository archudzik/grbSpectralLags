from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from config import FERMI_CSV, INVERSION_RANDOM_SEED, SWIFT_CSV

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class PhysicalConstants:
    SPEED_OF_LIGHT_CM_S: float = 3e10


@dataclass
class InversionResult:
    spin_period_ms: float
    torus_radius_units: float
    n_windings: float
    error: float
    lag_model_ms: float
    lag_observed_ms: float


@dataclass
class PopulationStatistics:
    n_solutions: int
    n_failed: int
    success_rate: float
    spin_period_median: float
    spin_period_std: float
    torus_radius_median: float
    torus_radius_std: float
    n_windings_median: float
    n_windings_std: float
    mean_error: float


class MagnetosphericPhysics:
    """
    Magnetospheric wind model for GRB spectral lag.
    
    Photons scatter through wound magnetic field structures surrounding
    nascent neutron stars. Delay arises from N windings through toroidal
    field at radius r_torus. See Eq. 12 in paper.
    """
    
    def __init__(self):
        self.constants = PhysicalConstants()

    def compute_light_cylinder_radius(self, spin_period_ms: float) -> float:
        """r_LC = c * P / (2pi) - radius where corotation velocity equals c"""
        spin_period_s = spin_period_ms * 1e-3
        return self.constants.SPEED_OF_LIGHT_CM_S * spin_period_s / (2 * np.pi)

    def compute_winding_delay(self, torus_radius_cm: float, n_windings: float) -> float:
        """dt = N * 2*pi*r_torus / c - time for N windings through toroidal field"""
        circumference = 2 * np.pi * torus_radius_cm
        return n_windings * circumference / self.constants.SPEED_OF_LIGHT_CM_S

    def compute_total_lag_ms(
        self,
        spin_period_ms: float,
        torus_radius_units: float,
        n_windings: float
    ) -> float:
        """
        Total spectral lag from magnetospheric scattering.
        
        Parameters:
            spin_period_ms: Neutron star rotation period [ms]
            torus_radius_units: Torus radius in units of r_LC
            n_windings: Number of field-line windings before escape (N_esc)
        
        Returns:
            Spectral lag [ms]
        """
        r_LC = self.compute_light_cylinder_radius(spin_period_ms)
        r_torus_cm = torus_radius_units * r_LC
        delay_s = self.compute_winding_delay(r_torus_cm, n_windings)
        return delay_s * 1e3


class ParameterBounds:
    def __init__(
        self,
        spin_period_range: Tuple[float, float] = (0.5, 10000),
        torus_radius_range: Tuple[float, float] = (1, 50),
        n_windings_range: Tuple[float, float] = (0, 10),
        random_seed: int = INVERSION_RANDOM_SEED
    ):
        self.spin_period_range = spin_period_range
        self.torus_radius_range = torus_radius_range
        self.n_windings_range = n_windings_range
        self.rng = np.random.default_rng(random_seed)

    def as_list(self) -> List[Tuple[float, float]]:
        return [
            self.spin_period_range,
            self.torus_radius_range,
            self.n_windings_range
        ]

    def generate_random_initial_guess(self) -> np.ndarray:
        return np.array([
            self.rng.uniform(*self.spin_period_range),
            self.rng.uniform(*self.torus_radius_range),
            self.rng.uniform(*self.n_windings_range)
        ])


class ObjectiveFunction:
    def __init__(self, physics: MagnetosphericPhysics, observed_lag_ms: float):
        self.physics = physics
        self.observed_lag_ms = observed_lag_ms

    def __call__(self, params: np.ndarray) -> float:
        spin_period_ms, torus_radius_units, n_windings = params
        model_lag = self.physics.compute_total_lag_ms(
            spin_period_ms, torus_radius_units, n_windings
        )
        return (model_lag - self.observed_lag_ms) ** 2


class LocalOptimizer:
    def __init__(
        self,
        physics: MagnetosphericPhysics,
        bounds: ParameterBounds,
        n_trials: int = 10
    ):
        self.physics = physics
        self.bounds = bounds
        self.n_trials = n_trials

    def optimize(self, observed_lag_ms: float) -> Optional[InversionResult]:
        objective = ObjectiveFunction(self.physics, observed_lag_ms)
        best_result = None
        best_error = np.inf

        for _ in range(self.n_trials):
            initial_guess = self.bounds.generate_random_initial_guess()
            result = minimize(
                objective,
                x0=initial_guess,
                bounds=self.bounds.as_list(),
                method='L-BFGS-B'
            )

            if not result.success:
                continue

            if not self._is_physically_reasonable(result.x):
                continue

            if result.fun < best_error:
                best_error = result.fun
                best_result = result

        if best_result is None:
            return None

        return self._create_result(best_result.x, best_result.fun, observed_lag_ms)

    def _is_physically_reasonable(self, params: np.ndarray) -> bool:
        n_windings = params[2]
        return 0 <= n_windings <= 10

    def _create_result(
        self,
        params: np.ndarray,
        error: float,
        observed_lag_ms: float
    ) -> InversionResult:
        spin_period_ms, torus_radius_units, n_windings = params
        return InversionResult(
            spin_period_ms=spin_period_ms,
            torus_radius_units=torus_radius_units,
            n_windings=n_windings,
            error=error,
            lag_model_ms=self.physics.compute_total_lag_ms(
                spin_period_ms, torus_radius_units, n_windings
            ),
            lag_observed_ms=observed_lag_ms
        )


class GlobalOptimizer:
    def __init__(
        self,
        physics: MagnetosphericPhysics,
        bounds: ParameterBounds,
        max_iterations: int = 1000,
        population_size: int = 15,
        tolerance: float = 1e-7,
        seed: int = INVERSION_RANDOM_SEED
    ):
        self.physics = physics
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.tolerance = tolerance
        self.seed = seed

    def optimize(self, observed_lag_ms: float) -> Optional[InversionResult]:
        objective = ObjectiveFunction(self.physics, observed_lag_ms)

        result = differential_evolution(
            objective,
            bounds=self.bounds.as_list(),
            strategy='best1bin',
            maxiter=self.max_iterations,
            popsize=self.population_size,
            tol=self.tolerance,
            seed=self.seed
        )

        if not result.success:
            return None

        if not self._is_physically_reasonable(result.x):
            return None

        return self._create_result(result.x, result.fun, observed_lag_ms)

    def _is_physically_reasonable(self, params: np.ndarray) -> bool:
        n_windings = params[2]
        return 0 <= n_windings <= 10

    def _create_result(
        self,
        params: np.ndarray,
        error: float,
        observed_lag_ms: float
    ) -> InversionResult:
        spin_period_ms, torus_radius_units, n_windings = params
        return InversionResult(
            spin_period_ms=spin_period_ms,
            torus_radius_units=torus_radius_units,
            n_windings=n_windings,
            error=error,
            lag_model_ms=self.physics.compute_total_lag_ms(
                spin_period_ms, torus_radius_units, n_windings
            ),
            lag_observed_ms=observed_lag_ms
        )


class PopulationAnalyzer:
    def __init__(
        self,
        optimization_method: str = 'global',
        sample_size: int = 2000
    ):
        self.physics = MagnetosphericPhysics()
        self.bounds = ParameterBounds()
        self.optimization_method = optimization_method
        self.sample_size = sample_size
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        if self.optimization_method == 'global':
            return GlobalOptimizer(self.physics, self.bounds)
        return LocalOptimizer(self.physics, self.bounds)

    def analyze(self, lags_ms: np.ndarray) -> Tuple[PopulationStatistics, pd.DataFrame]:
        self._print_input_summary(lags_ms)
        solutions = self._run_inversion(lags_ms)
        return self._compute_statistics(solutions, lags_ms)

    def _print_input_summary(self, lags_ms: np.ndarray):
        print(
            f"\nAnalyzing {len(lags_ms)} observed lags with {self.optimization_method} optimization...")
        print(f"Range: {lags_ms.min():.1f} - {lags_ms.max():.1f} ms")
        print(f"Median: {np.median(lags_ms):.1f} ms")

    def _run_inversion(self, lags_ms: np.ndarray) -> List[InversionResult]:
        effective_sample_size = min(self.sample_size, len(lags_ms))
        lags_sample = lags_ms[:effective_sample_size]
        solutions = []

        for i, lag in enumerate(lags_sample):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{effective_sample_size}")

            result = self.optimizer.optimize(lag)
            if result is not None:
                solutions.append(result)

        return solutions

    def _compute_statistics(
        self,
        solutions: List[InversionResult],
        lags_ms: np.ndarray
    ) -> Tuple[PopulationStatistics, pd.DataFrame]:
        effective_sample_size = min(self.sample_size, len(lags_ms))
        n_failed = effective_sample_size - len(solutions)

        print(f"\nSuccessful fits: {len(solutions)}/{effective_sample_size}")
        print(f"Failed fits: {n_failed}/{effective_sample_size}")

        if len(solutions) == 0:
            print("ERROR: No solutions found!")
            return None, None

        df_solutions = self._solutions_to_dataframe(solutions)

        statistics = PopulationStatistics(
            n_solutions=len(solutions),
            n_failed=n_failed,
            success_rate=len(solutions) / effective_sample_size,
            spin_period_median=df_solutions['spin_period_ms'].median(),
            spin_period_std=df_solutions['spin_period_ms'].std(),
            torus_radius_median=df_solutions['torus_radius_units'].median(),
            torus_radius_std=df_solutions['torus_radius_units'].std(),
            n_windings_median=df_solutions['n_windings'].median(),
            n_windings_std=df_solutions['n_windings'].std(),
            mean_error=df_solutions['error'].mean()
        )

        return statistics, df_solutions

    def _solutions_to_dataframe(self, solutions: List[InversionResult]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'spin_period_ms': s.spin_period_ms,
                'torus_radius_units': s.torus_radius_units,
                'n_windings': s.n_windings,
                'error': s.error,
                'lag_model_ms': s.lag_model_ms,
                'lag_observed_ms': s.lag_observed_ms
            }
            for s in solutions
        ])


class PopulationPlotter:
    def __init__(self, df_solutions: pd.DataFrame, output_prefix: str):
        self.df = df_solutions
        self.output_prefix = output_prefix

    def create_population_plot(self):
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Cannot generate plot.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        self._plot_spin_period_distribution(axes[0, 0])
        self._plot_torus_radius_distribution(axes[0, 1])
        self._plot_n_windings_distribution(axes[1, 0])
        self._plot_fit_quality(axes[1, 1])

        plt.tight_layout()
        output_path = f'figures/{self.output_prefix}_population_optimized.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Population plot saved to {output_path}")
        return output_path

    def _plot_spin_period_distribution(self, ax):
        ax.hist(
            self.df['spin_period_ms'],
            bins=30,
            alpha=0.7,
            color='blue',
            edgecolor='black'
        )
        median_value = self.df['spin_period_ms'].median()
        ax.axvline(
            median_value,
            color='red',
            linestyle='--',
            label=f'Median: {median_value:.2f} ms'
        )
        ax.set_xlabel('Spin Period (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('NS Spin Period Distribution', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_torus_radius_distribution(self, ax):
        ax.hist(
            self.df['torus_radius_units'],
            bins=30,
            alpha=0.7,
            color='green',
            edgecolor='black'
        )
        median_value = self.df['torus_radius_units'].median()
        ax.axvline(
            median_value,
            color='red',
            linestyle='--',
            label=f'Median: {median_value:.1f} r_LC'
        )
        ax.set_xlabel('Torus Radius (r_LC)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Magnetospheric Scale', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_n_windings_distribution(self, ax):
        ax.hist(
            self.df['n_windings'],
            bins=30,
            alpha=0.7,
            color='orange',
            edgecolor='black'
        )
        median_value = self.df['n_windings'].median()
        ax.axvline(
            median_value,
            color='red',
            linestyle='--',
            label=f'Median: {median_value:.2f}'
        )
        ax.set_xlabel('N_esc (windings)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Escape Windings', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_fit_quality(self, ax):
        ax.scatter(
            self.df['lag_observed_ms'],
            self.df['lag_model_ms'],
            alpha=0.5,
            s=20,
            color='purple'
        )
        lag_min = self.df['lag_observed_ms'].min()
        lag_max = self.df['lag_observed_ms'].max()
        ax.plot([lag_min, lag_max], [lag_min, lag_max],
                'r--', label='Perfect fit')
        ax.set_xlabel('Observed Lag (ms)', fontsize=11)
        ax.set_ylabel('Model Lag (ms)', fontsize=11)
        ax.set_title('Model Fit Quality', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3, which='both')


class ParameterInversionStudy:
    def __init__(
        self,
        filepath: str,
        output_prefix: str,
        optimization_method: str = 'global',
        sample_size: int = 2000
    ):
        self.filepath = filepath
        self.output_prefix = output_prefix
        self.optimization_method = optimization_method
        self.sample_size = sample_size
        self.df = self._load_data()
        self.lags_ms = np.abs(self.df['lag_ms'].values)
        self.statistics = None
        self.solutions_df = None

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        if 'is_significant' in df.columns:
            df = df[df['is_significant']].copy()
        return df

    def run_analysis(self) -> PopulationStatistics:
        self._print_header()

        analyzer = PopulationAnalyzer(
            optimization_method=self.optimization_method,
            sample_size=self.sample_size
        )

        self.statistics, self.solutions_df = analyzer.analyze(self.lags_ms)

        if self.statistics is not None:
            self._create_plots()
            self._print_results()

        return self.statistics

    def _print_header(self):
        print("\n" + "=" * 70)
        print(f"{self.output_prefix.upper()}: {len(self.lags_ms)} lags")
        print("=" * 70)

    def _create_plots(self):
        if self.solutions_df is not None:
            plotter = PopulationPlotter(self.solutions_df, self.output_prefix)
            plotter.create_population_plot()

    def _print_results(self):
        print("\nRESULTS:")
        print(f"  Success rate: {self.statistics.success_rate * 100:.1f}%")
        print(
            f"  Spin period: {self.statistics.spin_period_median:.2f} +/- {self.statistics.spin_period_std:.2f} ms")
        print(
            f"  Torus scale: {self.statistics.torus_radius_median:.1f} +/- {self.statistics.torus_radius_std:.1f} r_LC")
        print(
            f"  Escape fraction: {self.statistics.n_windings_median:.2f} +/- {self.statistics.n_windings_std:.2f}")
        print(f"  Mean fit error: {self.statistics.mean_error:.2e}")


if __name__ == "__main__":
    try:
        fermi_study = ParameterInversionStudy(
            filepath=FERMI_CSV,
            output_prefix='fermi',
            optimization_method='global'
        )
        fermi_stats = fermi_study.run_analysis()
    except Exception as e:
        print(f"\nFermi data error: {e}")

    try:
        swift_study = ParameterInversionStudy(
            filepath=SWIFT_CSV,
            output_prefix='swift',
            optimization_method='global'
        )
        swift_stats = swift_study.run_analysis()
    except Exception as e:
        print(f"\nSwift data error: {e}")
