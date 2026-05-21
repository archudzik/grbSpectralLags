from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal


PUBLIC_COLUMNS = [
    "instrument",
    "filename",
    "grb_name",
    "grb_time_utc",
    "ra",
    "dec",
    "lag_ms",
    "lag_error",
    "lag_significance",
    "is_significant",
    "lag_type",
    "lag_class",
    "t90_s",
    "t90_error_s",
    "t50_s",
    "t50_error_s",
    "duration_class",
]


def classify_duration(t90_s: float) -> str:
    if pd.isna(t90_s):
        return "unknown"
    return "short" if t90_s < 2.0 else "long"


def to_public_schema(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    df = df.copy()
    df["instrument"] = instrument
    for column in PUBLIC_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[PUBLIC_COLUMNS]


class CrossCorrelationAnalyzer:
    @staticmethod
    def compute_ccf(lc_soft: np.ndarray, lc_hard: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lags_bins = signal.correlation_lags(
            len(lc_soft), len(lc_hard), mode="full"
        )

        soft_centered = lc_soft - np.mean(lc_soft)
        hard_centered = lc_hard - np.mean(lc_hard)
        denominator = np.sqrt(
            np.sum(soft_centered ** 2) * np.sum(hard_centered ** 2)
        )

        if denominator <= 0:
            return np.zeros_like(lags_bins, dtype=float), lags_bins

        ccf_normalized = signal.correlate(
            soft_centered, hard_centered, mode="full"
        ) / denominator

        return ccf_normalized, lags_bins

    @staticmethod
    def refine_peak_lag(
        ccf: np.ndarray, lags: np.ndarray, bin_width: float
    ) -> Tuple[float, int]:
        peak_index = np.argmax(ccf)

        if peak_index == 0 or peak_index == len(ccf) - 1:
            return lags[peak_index] * bin_width, peak_index

        y_vals = ccf[peak_index - 1:peak_index + 2]
        x_vals = lags[peak_index - 1:peak_index + 2] * bin_width

        if len(y_vals) == 3:
            coeffs = np.polyfit(x_vals, y_vals, 2)

            if coeffs[0] != 0:
                refined_lag = -coeffs[1] / (2 * coeffs[0])
                return refined_lag, peak_index

        return lags[peak_index] * bin_width, peak_index

    @staticmethod
    def estimate_uncertainty(
        ccf: np.ndarray,
        lags: np.ndarray,
        peak_index: int,
        bin_width: float,
    ) -> float:
        if peak_index <= 0 or peak_index >= len(ccf) - 1:
            return bin_width

        exclusion_bins = max(5, int(0.5 / bin_width))
        indices = np.arange(len(ccf))
        noise_mask = np.abs(indices - peak_index) > exclusion_bins
        noise_samples = ccf[noise_mask] if np.any(noise_mask) else ccf
        noise_sigma = np.std(noise_samples)

        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            return bin_width

        y_vals = ccf[peak_index - 1:peak_index + 2]
        x_vals = lags[peak_index - 1:peak_index + 2] * bin_width
        coeffs = np.polyfit(x_vals, y_vals, 2)
        curvature = abs(2 * coeffs[0])

        if not np.isfinite(curvature) or curvature <= 0:
            return bin_width

        uncertainty = np.sqrt(noise_sigma / curvature)
        max_uncertainty = max(abs(lags[0]), abs(lags[-1])) * bin_width
        return float(np.clip(uncertainty, bin_width * 1e-3, max_uncertainty))

    @staticmethod
    def compute_spectral_lag(
        lc_soft: np.ndarray, lc_hard: np.ndarray, bin_width: float
    ) -> Tuple[float, float]:
        if len(lc_soft) == 0 or len(lc_hard) == 0:
            return np.nan, np.nan

        if np.std(lc_soft) < 1e-10 or np.std(lc_hard) < 1e-10:
            return np.nan, np.nan

        ccf, lags_bins = CrossCorrelationAnalyzer.compute_ccf(lc_soft, lc_hard)

        refined_lag_seconds, peak_idx = CrossCorrelationAnalyzer.refine_peak_lag(
            ccf, lags_bins, bin_width
        )

        uncertainty_seconds = CrossCorrelationAnalyzer.estimate_uncertainty(
            ccf, lags_bins, peak_idx, bin_width
        )

        lag_milliseconds = refined_lag_seconds * 1000.0
        uncertainty_milliseconds = uncertainty_seconds * 1000.0

        return lag_milliseconds, uncertainty_milliseconds


class QualityFilter:
    @staticmethod
    def is_placeholder(ra: float, dec: float, placeholders: list) -> bool:
        for ph_ra, ph_dec in placeholders:
            if abs(ra - ph_ra) < 0.01 and abs(dec - ph_dec) < 0.01:
                return True
        return False

    @staticmethod
    def is_significant(lag_ms: float, uncertainty_ms: float, min_sigma: float) -> bool:
        if np.isnan(lag_ms) or np.isnan(uncertainty_ms):
            return False

        if uncertainty_ms <= 0:
            return False

        significance = abs(lag_ms) / uncertainty_ms
        return significance >= min_sigma

    @staticmethod
    def lag_significance(lag_ms: float, uncertainty_ms: float) -> float:
        if np.isnan(lag_ms) or np.isnan(uncertainty_ms) or uncertainty_ms <= 0:
            return np.nan
        return abs(lag_ms) / uncertainty_ms

    @staticmethod
    def classify_lag(lag_ms: float, uncertainty_ms: float, min_sigma: float) -> str:
        if not QualityFilter.is_significant(lag_ms, uncertainty_ms, min_sigma):
            return "consistent_with_zero"
        return "significant_positive" if lag_ms > 0 else "significant_negative"
