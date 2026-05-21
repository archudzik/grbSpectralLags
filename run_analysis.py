import re
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
import glob
import os

from config import (
    COMPARISON_FIGURE,
    DIRECTIONAL_TEST_FAMILY_SIZE as CONFIG_DIRECTIONAL_TEST_FAMILY_SIZE,
    FERMI_CSV,
    FERMI_LAG_FIGURE,
    OPTIMAL_AXIS_RANDOM_SEED,
    SWIFT_CSV,
    SWIFT_LAG_FIGURE,
)
from reporting import p_value_interpretation, print_rule

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CoordinateTransform:
    NGP_RA = np.radians(192.85948)
    NGP_DEC = np.radians(27.12825)
    L_NCP = np.radians(122.93192)

    @classmethod
    def equatorial_to_galactic(cls, ra: float, dec: float) -> Tuple[float, float]:
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        sin_b = (np.sin(dec_rad) * np.sin(cls.NGP_DEC) +
                 np.cos(dec_rad) * np.cos(cls.NGP_DEC) * np.cos(ra_rad - cls.NGP_RA))
        b = np.arcsin(np.clip(sin_b, -1, 1))

        numerator = np.cos(dec_rad) * np.sin(ra_rad - cls.NGP_RA)
        denominator_part = (np.cos(dec_rad) * np.sin(cls.NGP_DEC) * np.cos(ra_rad - cls.NGP_RA) -
                            np.sin(dec_rad) * np.cos(cls.NGP_DEC))

        l = np.degrees(np.arctan2(numerator, denominator_part)
                       ) + np.degrees(cls.L_NCP)

        return l % 360, np.degrees(b)

    @staticmethod
    def to_cartesian(l: float, b: float) -> np.ndarray:
        l_rad = np.radians(l)
        b_rad = np.radians(b)
        return np.array([
            np.cos(b_rad) * np.cos(l_rad),
            np.cos(b_rad) * np.sin(l_rad),
            np.sin(b_rad)
        ])

    @staticmethod
    def angular_separation(l1: float, b1: float, l2: float, b2: float) -> float:
        b1_rad = np.radians(b1)
        b2_rad = np.radians(b2)
        l1_rad = np.radians(l1)
        l2_rad = np.radians(l2)

        cos_theta = (np.sin(b1_rad) * np.sin(b2_rad) +
                     np.cos(b1_rad) * np.cos(b2_rad) * np.cos(l1_rad - l2_rad))

        return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))


class DataProcessor:
    @staticmethod
    def load_and_prepare(filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Please run preprocessing first to generate the data file."
            )

        df = pd.read_csv(filepath)

        if 'l' not in df.columns or 'b' not in df.columns:
            df = DataProcessor._add_galactic_coordinates(df)

        df = DataProcessor._add_lag_metadata(df)

        return df

    @staticmethod
    def _add_lag_metadata(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'lag_error' in df.columns:
            error_column = 'lag_error'
        elif 'lag_error_ms' in df.columns:
            error_column = 'lag_error_ms'
        else:
            error_column = None

        if error_column is not None:
            df['lag_significance'] = (
                df['lag_ms'].abs() / df[error_column]
            ).replace([np.inf, -np.inf], np.nan)
        elif 'lag_significance' not in df.columns:
            df['lag_significance'] = np.nan

        if 'is_significant' not in df.columns:
            df['is_significant'] = df['lag_significance'] >= 2.0
        else:
            df['is_significant'] = df['is_significant'].astype(bool)

        if 'lag_type' not in df.columns:
            df['lag_type'] = np.where(
                df['lag_ms'] >= 0, 'positive', 'negative')

        if 'lag_class' not in df.columns:
            df['lag_class'] = np.where(
                ~df['is_significant'],
                'consistent_with_zero',
                np.where(df['lag_ms'] > 0,
                         'significant_positive',
                         'significant_negative')
            )

        return df

    @staticmethod
    def significant_subset(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['is_significant']].copy()

    @staticmethod
    def _add_galactic_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        coords = [CoordinateTransform.equatorial_to_galactic(ra, dec)
                  for ra, dec in zip(df['ra'], df['dec'])]
        df['l'] = [c[0] for c in coords]
        df['b'] = [c[1] for c in coords]
        return df


class HemisphereTest:
    DIRECTIONAL_TEST_FAMILY_SIZE = CONFIG_DIRECTIONAL_TEST_FAMILY_SIZE

    @staticmethod
    def bonferroni_adjust(p_value: float, n_tests: int = None) -> float:
        if n_tests is None:
            n_tests = HemisphereTest.DIRECTIONAL_TEST_FAMILY_SIZE
        return min(float(p_value) * n_tests, 1.0)

    @staticmethod
    def galactic_hemisphere(df: pd.DataFrame) -> Dict:
        north_mask = df['b'] > 0
        south_mask = ~north_mask

        n_pos_north = ((df[north_mask]['lag_type'] == 'positive').sum())
        n_neg_north = ((df[north_mask]['lag_type'] == 'negative').sum())
        n_pos_south = ((df[south_mask]['lag_type'] == 'positive').sum())
        n_neg_south = ((df[south_mask]['lag_type'] == 'negative').sum())

        total_north = n_pos_north + n_neg_north
        total_south = n_pos_south + n_neg_south

        contingency_table = np.array([[n_pos_north, n_neg_north],
                                      [n_pos_south, n_neg_south]])

        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        return {
            'north_total': total_north,
            'north_positive': n_pos_north,
            'north_negative': n_neg_north,
            'north_pos_fraction': n_pos_north / total_north,
            'south_total': total_south,
            'south_positive': n_pos_south,
            'south_negative': n_neg_south,
            'south_pos_fraction': n_pos_south / total_south,
            'chi2': chi2,
            'dof': dof,
            'p_value': p,
            'p_bonferroni': HemisphereTest.bonferroni_adjust(p),
            'bonferroni_n_tests': HemisphereTest.DIRECTIONAL_TEST_FAMILY_SIZE
        }

    @staticmethod
    def cmb_dipole(df: pd.DataFrame) -> Dict:
        cmb_ra = 168.0
        cmb_dec = -7.0
        cmb_l, cmb_b = CoordinateTransform.equatorial_to_galactic(
            cmb_ra, cmb_dec)

        separations = np.array([
            CoordinateTransform.angular_separation(l, b, cmb_l, cmb_b)
            for l, b in zip(df['l'], df['b'])
        ])

        close_mask = separations < 90
        far_mask = ~close_mask

        n_pos_close = ((df[close_mask]['lag_type'] == 'positive').sum())
        n_neg_close = ((df[close_mask]['lag_type'] == 'negative').sum())
        n_pos_far = ((df[far_mask]['lag_type'] == 'positive').sum())
        n_neg_far = ((df[far_mask]['lag_type'] == 'negative').sum())

        total_close = n_pos_close + n_neg_close
        total_far = n_pos_far + n_neg_far

        contingency_table = np.array([[n_pos_close, n_neg_close],
                                      [n_pos_far, n_neg_far]])

        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        return {
            'close_total': total_close,
            'close_positive': n_pos_close,
            'close_negative': n_neg_close,
            'close_pos_fraction': n_pos_close / total_close,
            'far_total': total_far,
            'far_positive': n_pos_far,
            'far_negative': n_neg_far,
            'far_pos_fraction': n_pos_far / total_far,
            'chi2': chi2,
            'dof': dof,
            'p_value': p,
            'p_bonferroni': HemisphereTest.bonferroni_adjust(p),
            'bonferroni_n_tests': HemisphereTest.DIRECTIONAL_TEST_FAMILY_SIZE,
            'cmb_direction': (cmb_l, cmb_b)
        }


class OptimalAxisSearch:
    def __init__(self, df: pd.DataFrame, random_seed: int = OPTIMAL_AXIS_RANDOM_SEED):
        self.vectors = np.array([CoordinateTransform.to_cartesian(l, b)
                                 for l, b in zip(df['l'], df['b'])])
        self.signs = np.where(df['lag_type'] == 'positive', 1, -1)
        self.rng = np.random.default_rng(random_seed)

    def objective_function(self, params: np.ndarray) -> float:
        theta, phi = params
        axis = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        projections = self.vectors @ axis
        hemisphere_signs = np.sign(projections)

        correct_classifications = np.sum(hemisphere_signs == self.signs)
        accuracy = correct_classifications / len(self.signs)

        return -accuracy

    def find_optimal(self, n_trials: int = 20) -> Dict:
        best_accuracy = 0.0
        best_params = None
        best_result = None

        convergence_results = []

        for trial in range(n_trials):
            initial_theta = self.rng.uniform(0, np.pi)
            initial_phi = self.rng.uniform(0, 2 * np.pi)

            result = optimize.minimize(
                self.objective_function,
                x0=[initial_theta, initial_phi],
                method='Powell',
                options={'ftol': 1e-6, 'xtol': 1e-6}
            )

            accuracy = -result.fun
            convergence_results.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = result.x
                best_result = result

        theta_opt, phi_opt = best_params

        axis_cartesian = np.array([
            np.sin(theta_opt) * np.cos(phi_opt),
            np.sin(theta_opt) * np.sin(phi_opt),
            np.cos(theta_opt)
        ])

        b_opt = np.degrees(np.arcsin(axis_cartesian[2]))
        l_opt = np.degrees(np.arctan2(
            axis_cartesian[1], axis_cartesian[0])) % 360

        return {
            'accuracy': best_accuracy,
            'l': l_opt,
            'b': b_opt,
            'n_trials': n_trials,
            'convergence_std': np.std(convergence_results),
            'success': best_result.success
        }


class DipoleAnalysis:
    @staticmethod
    def compute_dipole(df: pd.DataFrame, subset_mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        if subset_mask is not None:
            df_subset = df[subset_mask]
        else:
            df_subset = df

        vectors = np.array([CoordinateTransform.to_cartesian(l, b)
                           for l, b in zip(df_subset['l'], df_subset['b'])])

        mean_vector = np.mean(vectors, axis=0)
        magnitude = np.linalg.norm(mean_vector)

        if magnitude == 0:
            return 0.0, 0.0, 0.0

        unit_vector = mean_vector / magnitude

        b_dipole = np.degrees(np.arcsin(unit_vector[2]))
        l_dipole = np.degrees(np.arctan2(unit_vector[1], unit_vector[0])) % 360

        return l_dipole, b_dipole, magnitude

    @staticmethod
    def analyze_all(df: pd.DataFrame) -> Dict:
        pos_mask = df['lag_type'] == 'positive'
        neg_mask = df['lag_type'] == 'negative'

        l_all, b_all, mag_all = DipoleAnalysis.compute_dipole(df)
        l_pos, b_pos, mag_pos = DipoleAnalysis.compute_dipole(df, pos_mask)
        l_neg, b_neg, mag_neg = DipoleAnalysis.compute_dipole(df, neg_mask)

        separation = CoordinateTransform.angular_separation(
            l_pos, b_pos, l_neg, b_neg)

        return {
            'all_sample': {
                'l': l_all,
                'b': b_all,
                'magnitude': mag_all,
                'n_bursts': len(df)
            },
            'positive_lags': {
                'l': l_pos,
                'b': b_pos,
                'magnitude': mag_pos,
                'n_bursts': pos_mask.sum()
            },
            'negative_lags': {
                'l': l_neg,
                'b': b_neg,
                'magnitude': mag_neg,
                'n_bursts': neg_mask.sum()
            },
            'pos_neg_separation': separation
        }


class SampleStatistics:
    @staticmethod
    def compute_basic_stats(df: pd.DataFrame) -> Dict:
        n_total = len(df)
        n_positive = (df['lag_type'] == 'positive').sum()
        n_negative = (df['lag_type'] == 'negative').sum()

        abs_lags = np.abs(df['lag_ms'])

        return {
            'n_total': n_total,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'positive_fraction': n_positive / n_total,
            'negative_fraction': n_negative / n_total,
            'median_abs_lag': np.median(abs_lags),
            'q1_abs_lag': np.percentile(abs_lags, 25),
            'q3_abs_lag': np.percentile(abs_lags, 75),
            'max_positive_lag': df[df['lag_type'] == 'positive']['lag_ms'].max(),
            'max_negative_lag': df[df['lag_type'] == 'negative']['lag_ms'].min()
        }

    @staticmethod
    def analyze_lag_distributions(df: pd.DataFrame) -> Dict:
        positive_lags = df[df['lag_type'] == 'positive']['lag_ms'].values
        negative_lags = df[df['lag_type'] == 'negative']['lag_ms'].values

        pos_stats = {
            'n': len(positive_lags),
            'min': np.min(positive_lags),
            'max': np.max(positive_lags),
            'mean': np.mean(positive_lags),
            'median': np.median(positive_lags),
            'std': np.std(positive_lags),
            'q1': np.percentile(positive_lags, 25),
            'q3': np.percentile(positive_lags, 75),
            'iqr': np.percentile(positive_lags, 75) - np.percentile(positive_lags, 25)
        }

        neg_stats = {
            'n': len(negative_lags),
            'min': np.min(negative_lags),
            'max': np.max(negative_lags),
            'mean': np.mean(negative_lags),
            'median': np.median(negative_lags),
            'std': np.std(negative_lags),
            'q1': np.percentile(negative_lags, 25),
            'q3': np.percentile(negative_lags, 75),
            'iqr': np.percentile(negative_lags, 75) - np.percentile(negative_lags, 25)
        }

        return {
            'positive': pos_stats,
            'negative': neg_stats
        }

    @staticmethod
    def test_linearity(values: np.ndarray, distribution_name: str = "distribution") -> Dict:
        sorted_values = np.sort(values)
        n = len(sorted_values)

        expected_uniform = np.linspace(sorted_values[0], sorted_values[-1], n)

        ss_total = np.sum((sorted_values - np.mean(sorted_values))**2)
        ss_residual = np.sum((sorted_values - expected_uniform)**2)
        r_squared = 1 - (ss_residual / ss_total)

        ks_statistic, ks_pvalue = stats.kstest(
            (sorted_values - sorted_values[0]) /
            (sorted_values[-1] - sorted_values[0]),
            'uniform'
        )

        bins = np.linspace(sorted_values[0], sorted_values[-1], 21)
        observed_counts, _ = np.histogram(sorted_values, bins=bins)
        expected_counts = np.full_like(
            observed_counts, n / len(observed_counts), dtype=float)

        chi2_stat = np.sum((observed_counts - expected_counts)
                           ** 2 / expected_counts)
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=len(observed_counts)-1)

        return {
            'distribution_name': distribution_name,
            'n': n,
            'range': (sorted_values[0], sorted_values[-1]),
            'linear_r_squared': r_squared,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'chi2_statistic': chi2_stat,
            'chi2_pvalue': chi2_pvalue
        }

    @staticmethod
    def fit_exponential(values: np.ndarray, distribution_name: str = "distribution") -> Dict:
        positive_values = values[values > 0]

        if len(positive_values) == 0:
            return {'distribution_name': distribution_name, 'fit_failed': True}

        try:
            shape, loc, scale = stats.expon.fit(positive_values, floc=0)
        except Exception:
            return {'distribution_name': distribution_name, 'fit_failed': True}

        ks_stat, ks_pval = stats.kstest(
            positive_values, 'expon', args=(loc, scale))

        sorted_vals = np.sort(positive_values)
        expected_exp = stats.expon.ppf(np.linspace(
            0.01, 0.99, len(sorted_vals)), loc=loc, scale=scale)

        ss_total = np.sum((sorted_vals - np.mean(sorted_vals))**2)
        ss_residual = np.sum((sorted_vals - expected_exp)**2)
        r_squared = 1 - (ss_residual / ss_total)

        decay_constant = 1.0 / scale
        half_life = np.log(2) * scale

        return {
            'distribution_name': distribution_name,
            'n': len(positive_values),
            'scale': scale,
            'decay_constant': decay_constant,
            'half_life': half_life,
            'mean': scale,
            'median': scale * np.log(2),
            'r_squared': r_squared,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'fit_failed': False
        }

    @staticmethod
    def compare_distributions(values: np.ndarray, distribution_name: str = "distribution") -> Dict:
        positive_values = values[values > 0]

        if len(positive_values) == 0:
            return {'distribution_name': distribution_name, 'comparison_failed': True}

        uniform_test = SampleStatistics.test_linearity(
            positive_values, distribution_name)
        exponential_test = SampleStatistics.fit_exponential(
            positive_values, distribution_name)

        if exponential_test.get('fit_failed'):
            return {
                'distribution_name': distribution_name,
                'uniform_fit': uniform_test,
                'exponential_fit': None,
                'best_fit': 'uniform',
                'comparison_failed': True
            }

        uniform_r2 = uniform_test['linear_r_squared']
        exp_r2 = exponential_test['r_squared']

        if exp_r2 > uniform_r2:
            best_fit = 'exponential'
        else:
            best_fit = 'uniform'

        aic_uniform = 2 * 2 - 2 * \
            np.sum(np.log(1.0 / (positive_values.max() - positive_values.min())))
        aic_exponential = 2 * 1 - 2 * \
            np.sum(stats.expon.logpdf(
                positive_values, 0, exponential_test['scale']))

        return {
            'distribution_name': distribution_name,
            'uniform_fit': uniform_test,
            'exponential_fit': exponential_test,
            'best_fit': best_fit,
            'r_squared_comparison': {
                'uniform': uniform_r2,
                'exponential': exp_r2,
                'difference': exp_r2 - uniform_r2
            },
            'aic_comparison': {
                'uniform': aic_uniform,
                'exponential': aic_exponential,
                'delta_aic': aic_exponential - aic_uniform
            },
            'comparison_failed': False
        }

    @staticmethod
    def test_distribution_shape(df: pd.DataFrame) -> Dict:
        positive_lags = df[df['lag_type'] == 'positive']['lag_ms'].values
        negative_lags = df[df['lag_type'] == 'negative']['lag_ms'].values
        all_abs_lags = np.abs(df['lag_ms'].values)

        pos_comparison = SampleStatistics.compare_distributions(
            positive_lags,
            "Positive lags"
        )

        neg_comparison = SampleStatistics.compare_distributions(
            np.abs(negative_lags),
            "Negative lags (absolute)"
        )

        abs_comparison = SampleStatistics.compare_distributions(
            all_abs_lags,
            "All absolute lags"
        )

        pos_skew = stats.skew(positive_lags)
        neg_skew = stats.skew(negative_lags)

        pos_kurtosis = stats.kurtosis(positive_lags)
        neg_kurtosis = stats.kurtosis(negative_lags)

        ks_pos_neg = stats.ks_2samp(positive_lags, np.abs(negative_lags))

        return {
            'positive_comparison': pos_comparison,
            'negative_comparison': neg_comparison,
            'absolute_comparison': abs_comparison,
            'positive_skewness': pos_skew,
            'negative_skewness': neg_skew,
            'positive_kurtosis': pos_kurtosis,
            'negative_kurtosis': neg_kurtosis,
            'ks_pos_vs_neg': {
                'statistic': ks_pos_neg.statistic,
                'pvalue': ks_pos_neg.pvalue
            }
        }


class GRBStudy:
    def __init__(self, filepath: str = 'data.csv'):
        self.full_df = DataProcessor.load_and_prepare(filepath)
        self.df = DataProcessor.significant_subset(self.full_df)
        self.filepath = filepath
        self.stats = SampleStatistics.compute_basic_stats(self.df)
        self.distributions = SampleStatistics.analyze_lag_distributions(
            self.df)
        self.shape_tests = SampleStatistics.test_distribution_shape(self.df)

    def run_complete_analysis(self) -> Dict:
        print(f"{'*'*70}")
        print(f"GRB SPECTRAL LAG SPATIAL ANALYSIS")
        print(f"{'*'*70}\n")

        print(f"Using file: ", self.filepath)

        self._print_sample_statistics()
        self._print_distribution_analysis()
        self._print_linearity_tests()

        galactic_results = HemisphereTest.galactic_hemisphere(self.df)
        self._print_galactic_hemisphere(galactic_results)

        cmb_results = HemisphereTest.cmb_dipole(self.df)
        self._print_cmb_dipole(cmb_results)

        optimal_results = OptimalAxisSearch(self.df).find_optimal(n_trials=20)
        self._print_optimal_axis(optimal_results)

        dipole_results = DipoleAnalysis.analyze_all(self.df)
        self._print_dipole_analysis(dipole_results)

        exp_results = self.analyze_exponential_distributions()

        return {
            'sample_statistics': self.stats,
            'distributions': self.distributions,
            'shape_tests': self.shape_tests,
            'galactic_hemisphere': galactic_results,
            'cmb_dipole': cmb_results,
            'optimal_axis': optimal_results,
            'dipole_analysis': dipole_results,
            'exponential_results': exp_results
        }

    def _print_sample_statistics(self):
        print(f"SAMPLE COMPOSITION")
        print(f"{'-'*70}")
        n_measured = len(self.full_df)
        n_significant = len(self.df)
        class_counts = self.full_df['lag_class'].value_counts()
        n_pos_sig = class_counts.get('significant_positive', 0)
        n_neg_sig = class_counts.get('significant_negative', 0)
        n_zero = class_counts.get('consistent_with_zero', 0)
        print(f"Total measured GRBs:        {n_measured}")
        print(f"Significant GRBs analyzed:  {n_significant}")
        print(f"Consistent with zero:       {n_zero}")
        print(f"  significant_positive:     {n_pos_sig}")
        print(f"  significant_negative:     {n_neg_sig}")
        print(f"  consistent_with_zero:     {n_zero}")
        print()
        print(
            f"Positive lags:              {self.stats['n_positive']} ({self.stats['positive_fraction']:.3f})")
        print(
            f"Negative lags:              {self.stats['n_negative']} ({self.stats['negative_fraction']:.3f})")
        print(
            f"Median |lag|:               {self.stats['median_abs_lag']:.2f} ms")
        print(
            f"Q1, Q3 |lag|:               {self.stats['q1_abs_lag']:.2f}, {self.stats['q3_abs_lag']:.2f} ms")
        print(
            f"Max positive lag:           {self.stats['max_positive_lag']:.1f} ms")
        print(
            f"Max negative lag:           {self.stats['max_negative_lag']:.1f} ms")
        print()

    def _print_distribution_analysis(self):
        print(f"LAG DISTRIBUTION STATISTICS")
        print(f"{'-'*70}")

        pos = self.distributions['positive']
        print(f"Positive lags (n={pos['n']}):")
        print(
            f"  Range:                    [{pos['min']:.1f}, {pos['max']:.1f}] ms")
        print(f"  Mean:                     {pos['mean']:.1f} ms")
        print(f"  Median:                   {pos['median']:.1f} ms")
        print(f"  Std dev:                  {pos['std']:.1f} ms")
        print(f"  IQR:                      {pos['iqr']:.1f} ms")
        print()

        neg = self.distributions['negative']
        print(f"Negative lags (n={neg['n']}):")
        print(
            f"  Range:                    [{neg['min']:.1f}, {neg['max']:.1f}] ms")
        print(f"  Mean:                     {neg['mean']:.1f} ms")
        print(f"  Median:                   {neg['median']:.1f} ms")
        print(f"  Std dev:                  {neg['std']:.1f} ms")
        print(f"  IQR:                      {neg['iqr']:.1f} ms")
        print()

    def _print_linearity_tests(self):
        print(f"DISTRIBUTION SHAPE ANALYSIS")
        print(f"{'-'*70}")

        pos_comp = self.shape_tests['positive_comparison']
        if not pos_comp.get('comparison_failed'):
            pos_uni = pos_comp['uniform_fit']
            pos_exp = pos_comp['exponential_fit']

            print(f"Positive lags:")
            print(f"  Uniform (linear) fit:")
            print(f"    R^2 = {pos_uni['linear_r_squared']:.6f}")
            print(f"    KS p-value = {pos_uni['ks_pvalue']:.4f}")
            print()
            print(f"  Exponential fit:")
            print(f"    R^2 = {pos_exp['r_squared']:.6f}")
            print(f"    Scale = {pos_exp['scale']:.1f} ms")
            print(f"    Half-life = {pos_exp['half_life']:.1f} ms")
            print(f"    Decay constant lambda = {pos_exp['decay_constant']:.6f}")
            print(f"    KS p-value = {pos_exp['ks_pvalue']:.4f}")
            print()
            print(f"  Best fit: {pos_comp['best_fit'].upper()}")
            print(
                f"    delta R^2 = {pos_comp['r_squared_comparison']['difference']:.6f}")
            if pos_comp['best_fit'] == 'exponential':
                print(f"    OK Exponential distribution fits better")
            else:
                print(f"    OK Uniform distribution fits better")
            print()

        neg_comp = self.shape_tests['negative_comparison']
        if not neg_comp.get('comparison_failed'):
            neg_uni = neg_comp['uniform_fit']
            neg_exp = neg_comp['exponential_fit']

            print(f"Negative lags (absolute values):")
            print(f"  Uniform (linear) fit:")
            print(f"    R^2 = {neg_uni['linear_r_squared']:.6f}")
            print(f"    KS p-value = {neg_uni['ks_pvalue']:.4f}")
            print()
            print(f"  Exponential fit:")
            print(f"    R^2 = {neg_exp['r_squared']:.6f}")
            print(f"    Scale = {neg_exp['scale']:.1f} ms")
            print(f"    Half-life = {neg_exp['half_life']:.1f} ms")
            print(f"    Decay constant lambda = {neg_exp['decay_constant']:.6f}")
            print(f"    KS p-value = {neg_exp['ks_pvalue']:.4f}")
            print()
            print(f"  Best fit: {neg_comp['best_fit'].upper()}")
            print(
                f"    delta R^2 = {neg_comp['r_squared_comparison']['difference']:.6f}")
            if neg_comp['best_fit'] == 'exponential':
                print(f"    OK Exponential distribution fits better")
            else:
                print(f"    OK Uniform distribution fits better")
            print()

        print(f"Distribution shape metrics:")
        print(
            f"  Positive skewness:        {self.shape_tests['positive_skewness']:.4f}")
        print(
            f"  Negative skewness:        {self.shape_tests['negative_skewness']:.4f}")
        print(
            f"  Positive kurtosis:        {self.shape_tests['positive_kurtosis']:.4f}")
        print(
            f"  Negative kurtosis:        {self.shape_tests['negative_kurtosis']:.4f}")
        print()

        ks_test = self.shape_tests['ks_pos_vs_neg']
        print(f"Positive vs Negative comparison (KS test):")
        print(f"  KS statistic:             {ks_test['statistic']:.4f}")
        print(f"  p-value:                  {ks_test['pvalue']:.4f}")
        if ks_test['pvalue'] > 0.05:
            print(f"  OK Distributions are statistically similar")
        else:
            print(f"  FAIL Distributions differ significantly")
        print()

    def _print_galactic_hemisphere(self, results: Dict):
        print(f"GALACTIC HEMISPHERE TEST")
        print(f"{'-'*70}")
        print(f"Northern hemisphere (b > 0):")
        print(f"  Total bursts:             {results['north_total']}")
        print(f"  Positive lags:            {results['north_positive']}")
        print(f"  Negative lags:            {results['north_negative']}")
        print(
            f"  Positive fraction:        {results['north_pos_fraction']:.3f}")
        print()
        print(f"Southern hemisphere (b <= 0):")
        print(f"  Total bursts:             {results['south_total']}")
        print(f"  Positive lags:            {results['south_positive']}")
        print(f"  Negative lags:            {results['south_negative']}")
        print(
            f"  Positive fraction:        {results['south_pos_fraction']:.3f}")
        print()
        print(
            f"chi2 = {results['chi2']:.2f}, dof = {results['dof']}, p = {results['p_value']:.4f}")
        print(
            f"Bonferroni-adjusted p ({results['bonferroni_n_tests']} directional tests): "
            f"{results['p_bonferroni']:.4f}"
        )
        print(
            "  Interpretation: "
            f"{p_value_interpretation(results['p_value'], adjusted_p=results['p_bonferroni'])}"
        )
        print()

    def _print_cmb_dipole(self, results: Dict):
        print(f"CMB DIPOLE HEMISPHERE TEST")
        print(f"{'-'*70}")
        print(f"CMB dipole direction: (alpha={168.0} deg, delta={-7.0} deg)")
        print(
            f"                      (l={results['cmb_direction'][0]:.1f} deg, b={results['cmb_direction'][1]:.1f} deg)")
        print()
        print(f"Within 90 deg of CMB dipole:")
        print(f"  Total bursts:             {results['close_total']}")
        print(f"  Positive lags:            {results['close_positive']}")
        print(f"  Negative lags:            {results['close_negative']}")
        print(
            f"  Positive fraction:        {results['close_pos_fraction']:.3f}")
        print()
        print(f"Beyond 90 deg from CMB dipole:")
        print(f"  Total bursts:             {results['far_total']}")
        print(f"  Positive lags:            {results['far_positive']}")
        print(f"  Negative lags:            {results['far_negative']}")
        print(f"  Positive fraction:        {results['far_pos_fraction']:.3f}")
        print()
        print(
            f"chi2 = {results['chi2']:.2f}, dof = {results['dof']}, p = {results['p_value']:.4f}")
        print(
            f"Bonferroni-adjusted p ({results['bonferroni_n_tests']} directional tests): "
            f"{results['p_bonferroni']:.4f}"
        )
        print(
            "  Interpretation: "
            f"{p_value_interpretation(results['p_value'], adjusted_p=results['p_bonferroni'])}"
        )
        print()

    def _print_optimal_axis(self, results: Dict):
        print(f"OPTIMAL SEPARATION AXIS SEARCH")
        print(f"{'-'*70}")
        print(f"Optimization trials:        {results['n_trials']}")
        print(f"Maximum accuracy:           {results['accuracy']:.4f}")
        print(
            f"Optimal axis location:      (l={results['l']:.1f} deg, b={results['b']:.1f} deg)")
        print(f"Convergence std dev:        {results['convergence_std']:.6f}")
        print(f"Optimization success:       {results['success']}")
        print()

    def _print_dipole_analysis(self, results: Dict):
        print(f"DIPOLE MOMENT ANALYSIS")
        print(f"{'-'*70}")

        all_data = results['all_sample']
        print(f"All {all_data['n_bursts']} GRBs:")
        print(
            f"  Direction:                (l={all_data['l']:.1f} deg, b={all_data['b']:.1f} deg)")
        print(f"  Magnitude:                {all_data['magnitude']:.4f}")
        print()

        pos_data = results['positive_lags']
        print(f"Positive lag subset ({pos_data['n_bursts']} GRBs):")
        print(
            f"  Direction:                (l={pos_data['l']:.1f} deg, b={pos_data['b']:.1f} deg)")
        print(f"  Magnitude:                {pos_data['magnitude']:.4f}")
        print()

        neg_data = results['negative_lags']
        print(f"Negative lag subset ({neg_data['n_bursts']} GRBs):")
        print(
            f"  Direction:                (l={neg_data['l']:.1f} deg, b={neg_data['b']:.1f} deg)")
        print(f"  Magnitude:                {neg_data['magnitude']:.4f}")
        print()

        print(
            f"Positive-Negative dipole separation: {results['pos_neg_separation']:.1f} deg")
        print()

    def analyze_exponential_distributions(self):
        df = self.df

        positive_lags = df[df['lag_type'] == 'positive']['lag_ms'].values
        negative_lags = df[df['lag_type'] == 'negative']['lag_ms'].values

        print("="*70)
        print("EXPONENTIAL DISTRIBUTION ANALYSIS")
        print("="*70)

        print("\n1. POSITIVE LAGS")
        print("-"*70)
        loc, scale = stats.expon.fit(positive_lags, floc=0)
        print(f"Fitted exponential parameters:")
        print(f"  Scale (mean) = {scale:.1f} ms = {scale/1000:.2f} s")
        print(f"  Decay constant lambda = {1/scale:.6f} per ms")
        print(
            f"  Half-life = {scale * np.log(2):.1f} ms = {scale * np.log(2)/1000:.2f} s")
        print(f"  Median (theoretical) = {scale * np.log(2):.1f} ms")
        print(f"  Median (observed) = {np.median(positive_lags):.1f} ms")

        ks_stat, ks_pval = stats.kstest(
            positive_lags, 'expon', args=(loc, scale))
        print(f"\nKolmogorov-Smirnov test:")
        print(f"  Statistic = {ks_stat:.4f}")
        print(f"  p-value = {ks_pval:.4f}")
        if ks_pval > 0.05:
            print(f"  OK Consistent with exponential distribution")
        else:
            print(f"  FAIL Deviates from exponential (p<0.05)")

        pos_scale = scale
        pos_ks_pval = ks_pval

        print("\n2. NEGATIVE LAGS (absolute values)")
        print("-"*70)
        abs_neg_lags = np.abs(negative_lags)
        loc, scale = stats.expon.fit(abs_neg_lags, floc=0)
        print(f"Fitted exponential parameters:")
        print(f"  Scale (mean) = {scale:.1f} ms = {scale/1000:.2f} s")
        print(f"  Decay constant lambda = {1/scale:.6f} per ms")
        print(
            f"  Half-life = {scale * np.log(2):.1f} ms = {scale * np.log(2)/1000:.2f} s")
        print(f"  Median (theoretical) = {scale * np.log(2):.1f} ms")
        print(f"  Median (observed) = {np.median(abs_neg_lags):.1f} ms")

        ks_stat, ks_pval = stats.kstest(
            abs_neg_lags, 'expon', args=(loc, scale))
        print(f"\nKolmogorov-Smirnov test:")
        print(f"  Statistic = {ks_stat:.4f}")
        print(f"  p-value = {ks_pval:.4f}")
        if ks_pval > 0.05:
            print(f"  OK Consistent with exponential distribution")
        else:
            print(f"  FAIL Deviates from exponential (p<0.05)")

        neg_scale = scale
        neg_ks_pval = ks_pval

        print("\n3. COMPARISON: POSITIVE vs NEGATIVE")
        print("-"*70)
        print(f"Positive scale: {pos_scale:.1f} ms ({pos_scale/1000:.2f} s)")
        print(f"Negative scale: {neg_scale:.1f} ms ({neg_scale/1000:.2f} s)")
        print(f"Ratio (neg/pos): {neg_scale/pos_scale:.3f}")
        print(
            f"Difference: {abs(neg_scale - pos_scale):.1f} ms ({abs(neg_scale - pos_scale)/1000:.2f} s)")
        print(
            f"Relative difference: {abs(neg_scale - pos_scale) / pos_scale * 100:.1f}%")

        if abs(neg_scale - pos_scale) / pos_scale < 0.1:
            print(f"\nOK Similar fitted scales within 10%")
            print(f"  This supports comparing the positive and negative lag samples")
            print(f"  with the same summary statistic.")
        else:
            print(
                f"\nWARNING Different timescales by {abs(neg_scale - pos_scale) / pos_scale * 100:.1f}%")
            print(f"  Treat the fitted scales as exploratory and inspect")
            print(f"  possible sample or selection effects.")

        print("\n4. SHAPE STATISTICS")
        print("-"*70)
        pos_skew = stats.skew(positive_lags)
        neg_skew = stats.skew(negative_lags)
        pos_kurt = stats.kurtosis(positive_lags)
        neg_kurt = stats.kurtosis(negative_lags)

        print(f"Positive skewness: {pos_skew:.2f}")
        if pos_skew > 0:
            print(f"  -> Right-tailed distribution")
        else:
            print(f"  -> No right-tail excess by skewness")
        print(f"Negative skewness: {neg_skew:.2f}")
        if neg_skew < 0:
            print(f"  -> Left-tailed distribution")
        else:
            print(f"  -> No left-tail excess by skewness")
        print()
        print(f"Positive kurtosis: {pos_kurt:.2f}")
        if pos_kurt > 0:
            print(f"  -> Heavier tails than a normal distribution")
        else:
            print(f"  -> No excess kurtosis relative to a normal distribution")
        print(f"Negative kurtosis: {neg_kurt:.2f}")
        if neg_kurt > 0:
            print(f"  -> Heavier tails than a normal distribution")
        else:
            print(f"  -> No excess kurtosis relative to a normal distribution")

        print("\n5. QUANTILE-QUANTILE COMPARISON")
        print("-"*70)
        theoretical_median = pos_scale * np.log(2)
        pos_observed_median = np.median(positive_lags)
        print(f"Positive lags:")
        print(f"  Theoretical median: {theoretical_median:.1f} ms")
        print(f"  Observed median: {pos_observed_median:.1f} ms")
        print(f"  Ratio: {pos_observed_median/theoretical_median:.3f}")

        theoretical_median = neg_scale * np.log(2)
        neg_observed_median = np.median(abs_neg_lags)
        print(f"\nNegative lags:")
        print(f"  Theoretical median: {theoretical_median:.1f} ms")
        print(f"  Observed median: {neg_observed_median:.1f} ms")
        print(f"  Ratio: {neg_observed_median/theoretical_median:.3f}")

        print("\n" + "="*70)
        print("DISTRIBUTION SUMMARY")
        print("="*70)

        avg_scale = (pos_scale + neg_scale) / 2
        avg_half_life = avg_scale * np.log(2)

        print(f"\nCharacteristic timescale: {avg_scale/1000:.2f} seconds")
        print(f"Half-life: {avg_half_life/1000:.2f} seconds")
        print()
        print("Distribution diagnostics:")
        print()
        near_zero_concentration = (
            pos_observed_median < pos_scale and
            neg_observed_median < neg_scale
        )
        if near_zero_concentration:
            print("OK Concentration near zero:")
            print("  Both observed medians are below their fitted mean scales")
        else:
            print("WARNING Concentration near zero not supported by median/scale check:")
            print("  Inspect the lag distribution before drawing shape conclusions")
        print()

        similar_scales = abs(neg_scale - pos_scale) / pos_scale < 0.1
        if similar_scales:
            print("OK Similar characteristic scales:")
            print(f"  tau ~ {avg_scale/1000:.1f} seconds")
            print("  Treat this as a descriptive population scale")
        else:
            print("WARNING Positive and negative fitted scales differ by >10%:")
            print("  Report the two fitted scales separately")
        print()

        ks_pos_neg = stats.ks_2samp(positive_lags, abs_neg_lags)
        if similar_scales and ks_pos_neg.pvalue > 0.05:
            print("OK Positive and negative samples are similar by scale and KS test:")
            print(
                f"  Scale difference: {abs(neg_scale - pos_scale) / pos_scale * 100:.1f}%")
            print(f"  KS p-value: {ks_pos_neg.pvalue:.4f}")
        else:
            print("WARNING Positive and negative sample similarity is not fully supported:")
            print(
                f"  Scale difference: {abs(neg_scale - pos_scale) / pos_scale * 100:.1f}%")
            print(f"  KS p-value: {ks_pos_neg.pvalue:.4f}")

        if pos_ks_pval <= 0.05 or neg_ks_pval <= 0.05:
            print()
            print("WARNING Exponential fits are descriptive only for this dataset:")
            print(f"  Positive KS p-value: {pos_ks_pval:.4f}")
            print(f"  Negative KS p-value: {neg_ks_pval:.4f}")

        return {
            'pos_scale': pos_scale,
            'neg_scale': neg_scale,
            'avg_scale': avg_scale,
            'avg_half_life': avg_half_life,
            'tau_obs': avg_scale/1000
        }

    def winding_calculator(self, tau_obs, T_burst=30.0, c=3e10):
        """Simple grid search for magnetosphere parameters."""

        print("\n" + "="*70)
        print("MAGNETOSPHERIC LAG CALCULATOR (SIMPLE GRID SEARCH)")
        print("="*70)
        print(f"Observed lag: tau_obs = {tau_obs:.2f} s\n")

        best_error = 1e10
        best_params = None

        # Grid search over reasonable parameter space
        for P_ms in np.linspace(0.8, 2.0, 20):           # rotation period (ms)
            # r_torus in units of 10^8 cm
            for r_8 in np.linspace(3, 10, 20):
                # escape fraction N_esc/N_rot
                for N_frac in np.linspace(0.02, 0.10, 20):

                    r_torus = r_8 * 1e8  # cm
                    N_rot = T_burst / (P_ms * 1e-3)
                    N_esc = N_rot * N_frac

                    # Simple penetration model: f_soft ~ 0.01, f_hard ~ 0.20
                    f_soft, f_hard = 0.01, 0.20

                    # Predicted lag
                    tau_pred = (f_hard - f_soft) * N_esc * 2*np.pi*r_torus / c

                    # Track best fit
                    error = abs(tau_pred - tau_obs)
                    if error < best_error:
                        best_error = error
                        best_params = {
                            'P_ms': P_ms,
                            'r_torus': r_torus,
                            'N_esc': N_esc,
                            'N_rot': N_rot,
                            'tau_pred': tau_pred
                        }

        # Print results
        p = best_params
        r_LC = c * (p['P_ms']*1e-3) / (2*np.pi)

        print(f"BEST-FIT SOLUTION:")
        print(f"  Rotation period:     P = {p['P_ms']:.2f} ms")
        print(f"  Rotation frequency:  f = {1000/p['P_ms']:.0f} Hz")
        print(
            f"  Torus radius:        r = {p['r_torus']:.2e} cm = {p['r_torus']/r_LC:.0f} r_LC")
        print(f"  Total rotations:     N_rot = {p['N_rot']:.0f}")
        print(
            f"  Escape windings:     N_esc = {p['N_esc']:.0f} ({p['N_esc']/p['N_rot']*100:.1f}%)")
        print(f"  Predicted lag:       tau_pred = {p['tau_pred']:.2f} s")
        print(f"  Observed lag:        tau_obs = {tau_obs:.2f} s")
        print(
            f"  Error:               {abs(p['tau_pred']-tau_obs):.2f} s ({abs(p['tau_pred']-tau_obs)/tau_obs*100:.1f}%)")
        print(f"\nInterpretation:")
        print(
            f"  A {p['P_ms']:.2f} ms pulsar winds its magnetosphere {p['N_rot']:.0f} times.")
        print(
            f"  Photons escape after ~{p['N_esc']:.0f} windings through the torus at {p['r_torus']/r_LC:.0f} r_LC.")
        print(f"  Hard photons (20% sampling) escape faster than soft (1% sampling).")
        print(f"  Result: {p['tau_pred']:.1f} s lag.")
        print("="*70)

        return best_params

    def lag_magnitude_spatial_test(self) -> Dict:
        """
        Test if lag magnitude correlates with position relative to optimal axis.
        Returns correlation statistics and spatial gradient information.
        """
        print("\n" + "="*70)
        print("LAG MAGNITUDE vs SPATIAL POSITION")
        print("="*70)

        # Get optimal axis from previous analysis
        axis_search = OptimalAxisSearch(self.df)
        optimal_result = axis_search.find_optimal(n_trials=20)

        l_opt = optimal_result['l']
        b_opt = optimal_result['b']

        # Calculate angular separation from optimal axis for each GRB
        separations = np.array([
            CoordinateTransform.angular_separation(l, b, l_opt, b_opt)
            for l, b in zip(self.df['l'], self.df['b'])
        ])

        # Get absolute lag magnitudes
        lag_magnitudes = np.abs(self.df['lag_ms'].values)

        # Correlation test
        spearman_corr, spearman_p = stats.spearmanr(
            separations, lag_magnitudes)
        pearson_corr, pearson_p = stats.pearsonr(separations, lag_magnitudes)

        print(f"\nOptimal separation axis: (l={l_opt:.1f} deg, b={b_opt:.1f} deg)")
        print(f"\nCorrelation between angular separation and |lag|:")
        print(f"  Spearman rho = {spearman_corr:.4f}, p = {spearman_p:.4f}")
        print(f"  Pearson r = {pearson_corr:.4f}, p = {pearson_p:.4f}")

        if abs(spearman_corr) < 0.1:
            print(f"  OK No significant spatial gradient in lag magnitude")
        else:
            print(f"  WARNING Moderate spatial gradient detected")

        # Binned analysis: divide sky into 4 angular zones
        bins = [0, 45, 90, 135, 180]
        bin_centers = []
        bin_medians = []
        bin_means = []

        print(f"\nBinned analysis (angular distance from optimal axis):")
        print(f"{'Zone':<15} {'N':<8} {'Median |lag|':<15} {'Mean |lag|'}")
        print("-"*60)

        for i in range(len(bins)-1):
            mask = (separations >= bins[i]) & (separations < bins[i+1])
            if np.sum(mask) > 0:
                zone_lags = lag_magnitudes[mask]
                median_lag = np.median(zone_lags)
                mean_lag = np.mean(zone_lags)
                bin_centers.append((bins[i] + bins[i+1])/2)
                bin_medians.append(median_lag)
                bin_means.append(mean_lag)
                print(
                    f"{bins[i]:.0f} deg-{bins[i+1]:.0f} deg{'':<6} {np.sum(mask):<8} {median_lag:<15.1f} {mean_lag:.1f}")

        return {
            'optimal_axis': (l_opt, b_opt),
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'bin_centers': bin_centers,
            'bin_medians': bin_medians,
            'separations': separations,
            'lag_magnitudes': lag_magnitudes
        }

    def temporal_analysis(self) -> Dict:
        """
        Analyze temporal patterns in spectral lag measurements.
        Extracts temporal ordering from filename patterns when possible.
        """
        print("\n" + "="*70)
        print("TEMPORAL ANALYSIS")
        print("="*70)

        temporal_data = []
        data_type = None

        for idx, row in self.df.iterrows():
            filename = str(row.get('filename', ''))
            grb_time_utc = row.get('grb_time_utc')

            if pd.notna(grb_time_utc):
                timestamp = pd.to_datetime(grb_time_utc, errors='coerce')
                if pd.notna(timestamp):
                    temporal_data.append({
                        'date_key': (
                            timestamp.year * 10000
                            + timestamp.month * 100
                            + timestamp.day
                        ),
                        'sort_key': timestamp.timestamp(),
                        'lag_ms': row['lag_ms'],
                        'lag_type': row['lag_type']
                    })
                    data_type = 'Swift' if filename.startswith('sw') else 'Fermi'
                    continue

            # Fermi format: glg_tte_n9_bn<YYMMDDFFF>_v00.fit
            if 'glg_tte' in filename and '_bn' in filename:
                try:
                    bn_part = filename.split('_bn')[1].split('_')[0]
                    if len(bn_part) >= 7:
                        yy = int(bn_part[0:2])
                        mm = int(bn_part[2:4])
                        dd = int(bn_part[4:6])
                        year = 2000 + yy if yy < 50 else 1900 + yy
                        date_key = year * 10000 + mm * 100 + dd

                        temporal_data.append({
                            'date_key': date_key,
                            'sort_key': date_key,
                            'lag_ms': row['lag_ms'],
                            'lag_type': row['lag_type']
                        })
                        data_type = 'Fermi'
                except (ValueError, IndexError):
                    continue

            # Swift filenames do not contain calendar dates. Use local event
            # metadata added by swift_add_metadata.py when catalog UTC is absent.
            elif filename.startswith('sw') and 'bevshsp' in filename:
                trigger_time_met = row.get('trigger_time_met')
                if pd.notna(trigger_time_met):
                    swift_epoch = pd.Timestamp('2001-01-01T00:00:00').timestamp()
                    temporal_data.append({
                        'date_key': row.get('swift_data_year_month', ''),
                        'sort_key': swift_epoch + float(trigger_time_met),
                        'lag_ms': row['lag_ms'],
                        'lag_type': row['lag_type']
                    })
                    data_type = 'Swift'

        if len(temporal_data) < 10:
            print("Insufficient temporal data extracted")
            print(f"  Extracted: {len(temporal_data)} events")
            print(f"  Data type: {data_type if data_type else 'Unknown'}")
            return {'status': 'insufficient_data', 'n_extracted': len(temporal_data)}

        temporal_df = pd.DataFrame(temporal_data).sort_values('sort_key')

        print(f"\nData type: {data_type}")
        print(f"Extracted temporal sequence: {len(temporal_df)} observations")
        if data_type == 'Fermi':
            print(
                f"Date range: {temporal_df['date_key'].min()} - {temporal_df['date_key'].max()} (YYYYMMDD)")
        elif data_type == 'Swift':
            print(
                f"Temporal key range: {temporal_df['sort_key'].min():.3f} - "
                f"{temporal_df['sort_key'].max():.3f}")

        lag_magnitudes = np.abs(temporal_df['lag_ms'].values)
        obs_sequence = np.arange(len(temporal_df))

        spearman_corr, spearman_p = stats.spearmanr(
            obs_sequence, lag_magnitudes)

        print(f"\nTemporal trend in |lag| magnitude:")
        print(f"  Spearman rho = {spearman_corr:.4f}, p = {spearman_p:.4f}")

        if spearman_p < 0.05:
            trend = "increasing" if spearman_corr > 0 else "decreasing"
            print(f"  WARNING Significant {trend} trend detected")
        else:
            print("  OK No significant temporal trend")

        if len(lag_magnitudes) > 50:
            autocorr = np.corrcoef(
                lag_magnitudes[:-1], lag_magnitudes[1:])[0, 1]
            print(f"\nLag-1 autocorrelation: {autocorr:.4f}")

            if abs(autocorr) > 0.2:
                print("  WARNING Consecutive observations show correlation")
                print("    (possible instrumental effects or real clustering)")
            else:
                print("  OK No significant autocorrelation")
        else:
            autocorr = None
            print("\nLag-1 autocorrelation: N/A (sample size < 50)")

        lag_signs = (temporal_df['lag_type'] == 'positive').astype(int)
        n_positive = np.sum(lag_signs)
        n_negative = len(lag_signs) - n_positive

        runs = 1
        for i in range(1, len(lag_signs)):
            if lag_signs[i] != lag_signs[i - 1]:
                runs += 1

        expected_runs = 1 + (2 * n_positive * n_negative) / len(lag_signs)
        runs_std = np.sqrt(
            (2 * n_positive * n_negative * (2 * n_positive * n_negative - len(lag_signs))) /
            (len(lag_signs)**2 * (len(lag_signs) - 1))
        )

        z_runs = (runs - expected_runs) / runs_std if runs_std > 0 else 0
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

        print(f"\nRuns test (temporal clustering of lag signs):")
        print(f"  Observed runs: {runs}")
        print(f"  Expected runs: {expected_runs:.1f}")
        print(f"  Z-score: {z_runs:.2f}, p = {p_runs:.4f}")

        if p_runs < 0.05:
            if runs < expected_runs:
                print("  WARNING Significant clustering (fewer runs than expected)")
            else:
                print("  WARNING Significant alternation (more runs than expected)")
        else:
            print("  OK Random temporal distribution of lag signs")

        return {
            'data_type': data_type,
            'n_observations': len(temporal_df),
            'temporal_trend_corr': spearman_corr,
            'temporal_trend_p': spearman_p,
            'autocorr': autocorr,
            'runs': runs,
            'expected_runs': expected_runs,
            'runs_p': p_runs,
            'temporal_df': temporal_df
        }

    def plot_distributions(self, output_file: str = 'figures/fig_lag_distributions.png'):
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Cannot generate plot.")
            return None

        import matplotlib.pyplot as plt

        positive_lags = self.df[self.df['lag_type']
                                == 'positive']['lag_ms'].values
        negative_lags = self.df[self.df['lag_type']
                                == 'negative']['lag_ms'].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(positive_lags, bins=50, alpha=0.7,
                 color='blue', edgecolor='black')
        ax1.axvline(np.median(positive_lags), color='red', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(positive_lags):.0f} ms')
        ax1.set_xlabel('Lag (ms)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Positive Lags (n={len(positive_lags)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(np.abs(negative_lags), bins=50,
                 alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(np.median(np.abs(negative_lags)), color='blue', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(np.abs(negative_lags)):.0f} ms')
        ax2.set_xlabel('|Lag| (ms)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Negative Lags (n={len(negative_lags)})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Distribution plot saved to {output_file}")
        return output_file

    def plot_comparison_skymap(self, df_fermi=None, df_swift=None, output_file='figures/fig_comparison_skymap.png'):
        """
        Create comparison sky map for Fermi and Swift data showing lag magnitudes and signs.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Cannot generate plot.")
            return None

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, TwoSlopeNorm

        fig = plt.figure(figsize=(16, 10))

        # Create 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        datasets = []
        titles = []

        if df_fermi is not None:
            datasets.append(DataProcessor.significant_subset(df_fermi))
            titles.append('Fermi-GBM')

        if df_swift is not None:
            datasets.append(DataProcessor.significant_subset(df_swift))
            titles.append('Swift-BAT')

        if len(datasets) == 0:
            datasets = [self.df]
            titles = ['Current Dataset']

        for idx, (df, title) in enumerate(zip(datasets, titles)):
            # Sky map with lag magnitude
            ax1 = fig.add_subplot(gs[idx, 0], projection='hammer')

            # Convert galactic coordinates to radians for Hammer projection
            l_rad = np.radians(df['l'].values - 180)  # Center at 0
            b_rad = np.radians(df['b'].values)

            lag_magnitudes = np.abs(df['lag_ms'].values)
            lag_signs = np.where(df['lag_type'] == 'positive', 1, -1)

            # Create signed log magnitude for color
            signed_lag = lag_signs * np.log10(lag_magnitudes + 1)

            scatter = ax1.scatter(l_rad, b_rad, c=signed_lag,
                                  cmap='RdBu_r', s=20, alpha=0.6,
                                  vmin=-np.max(np.abs(signed_lag)),
                                  vmax=np.max(np.abs(signed_lag)))

            ax1.set_xlabel('Galactic Longitude', fontsize=10)
            ax1.set_ylabel('Galactic Latitude', fontsize=10)
            ax1.set_title(
                f'{title} Sky Map\n(color = sign x log10|lag|)', fontsize=11)
            ax1.grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
            cbar.set_label('Signed log10(|lag| [ms])', fontsize=9)

            # Histogram of lag distribution
            ax2 = fig.add_subplot(gs[idx, 1])

            positive_lags = df[df['lag_type'] == 'positive']['lag_ms'].values
            negative_lags = np.abs(
                df[df['lag_type'] == 'negative']['lag_ms'].values)

            bins = np.logspace(0, np.log10(max(lag_magnitudes)), 40)

            ax2.hist(positive_lags, bins=bins, alpha=0.6,
                     color='blue', label=f'Positive (n={len(positive_lags)})', edgecolor='black', linewidth=0.5)
            ax2.hist(negative_lags, bins=bins, alpha=0.6,
                     color='red', label=f'Negative (n={len(negative_lags)})', edgecolor='black', linewidth=0.5)

            ax2.set_xscale('log')
            ax2.set_xlabel('|Lag| (ms)', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.set_title(
                f'{title} Lag Distribution\n(Median: {np.median(lag_magnitudes):.0f} ms)', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, which='both')

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Comparison sky map saved to {output_file}")
        return output_file


if __name__ == "__main__":
    swift_file = SWIFT_CSV
    fermi_file = FERMI_CSV

    # Run complete analysis on Fermi data
    print_rule("RUNNING COMPLETE ANALYSIS ON FERMI DATA", "~")
    try:
        fermi_study = GRBStudy(fermi_file)
        fermi_results = fermi_study.run_complete_analysis()
        fermi_study.plot_distributions(
            output_file=FERMI_LAG_FIGURE)
        fermi_spatial = fermi_study.lag_magnitude_spatial_test()
        fermi_temporal = fermi_study.temporal_analysis()
        fermi_available = True
    except FileNotFoundError as e:
        print(f"\nNote: Fermi data not found - {e}")
        fermi_available = False

    # Run complete analysis on Swift data
    print("\n" + "="*70)
    print("RUNNING COMPLETE ANALYSIS ON SWIFT DATA")
    print("="*70)
    swift_study = GRBStudy(swift_file)
    swift_results = swift_study.run_complete_analysis()
    swift_study.plot_distributions(
        output_file=SWIFT_LAG_FIGURE)
    swift_spatial = swift_study.lag_magnitude_spatial_test()
    swift_temporal = swift_study.temporal_analysis()

    # Create comparison sky map (if both datasets available)
    print("\n" + "*"*70)
    print("CREATING COMPARISON SKY MAP")
    print("*"*70)
    if fermi_available:
        try:
            df_fermi = DataProcessor.load_and_prepare(fermi_file)
            df_swift = DataProcessor.load_and_prepare(swift_file)
            swift_study.plot_comparison_skymap(
                df_fermi=df_fermi,
                df_swift=df_swift,
                output_file=COMPARISON_FIGURE
            )
        except Exception as e:
            print(f"\nNote: Could not create comparison plot - {e}")
    else:
        print("\nSkipping comparison plot (Fermi data not available)")

    # Summary
    print("\n" + "*"*70)
    print("ANALYSIS COMPLETE")
    print("*"*70)
    print("\nGenerated files:")
    print(f"  - {FERMI_LAG_FIGURE}")
    print(f"  - {SWIFT_LAG_FIGURE}")
    print(f"  - {COMPARISON_FIGURE}")

    print("\nKey findings:")

    if fermi_available:
        print("\nFERMI:")
        print(f"  - Spatial correlation: rho = {fermi_spatial['spearman_corr']:.4f}, "
              f"p = {fermi_spatial['spearman_p']:.4f}")
        if fermi_temporal.get('status') != 'insufficient_data':
            print(f"  - Temporal trend: rho = {fermi_temporal['temporal_trend_corr']:.4f}, "
                  f"p = {fermi_temporal['temporal_trend_p']:.4f}")
            print(f"  - Runs test: p = {fermi_temporal['runs_p']:.4f}")
            if fermi_temporal.get('autocorr') is not None:
                print(f"  - Autocorrelation: {fermi_temporal['autocorr']:.4f}")
        else:
            print(
                f"  - Temporal analysis: {fermi_temporal.get('n_extracted', 0)} events extracted (need >=10)")

    print("\nSWIFT:")
    print(f"  - Spatial correlation: rho = {swift_spatial['spearman_corr']:.4f}, "
          f"p = {swift_spatial['spearman_p']:.4f}")
    if swift_temporal.get('status') != 'insufficient_data':
        print(f"  - Temporal trend: rho = {swift_temporal['temporal_trend_corr']:.4f}, "
              f"p = {swift_temporal['temporal_trend_p']:.4f}")
        print(f"  - Runs test: p = {swift_temporal['runs_p']:.4f}")
        if swift_temporal.get('autocorr') is not None:
            print(f"  - Autocorrelation: {swift_temporal['autocorr']:.4f}")
    else:
        print(
            f"  - Temporal analysis: {swift_temporal.get('n_extracted', 0)} events extracted (need >=10)")
