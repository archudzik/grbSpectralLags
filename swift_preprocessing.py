from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from astropy.io import fits
import glob
import os

from grb_lag_common import CrossCorrelationAnalyzer, QualityFilter
from config import (
    MIN_SIGNIFICANCE_SIGMA,
    PLACEHOLDER_COORDS,
    SWIFT_BIN_WIDTH_SECONDS,
    SWIFT_CSV,
    SWIFT_HARD_BAND_KEV,
    SWIFT_RAW_DIR,
    SWIFT_SOFT_BAND_KEV,
    TIME_WINDOW_SECONDS,
)


@dataclass
class PreprocessingConfig:
    data_dir: str
    output_file: str = SWIFT_CSV
    energy_band_soft: Tuple[int, int] = SWIFT_SOFT_BAND_KEV
    energy_band_hard: Tuple[int, int] = SWIFT_HARD_BAND_KEV
    time_window: Tuple[float, float] = TIME_WINDOW_SECONDS
    bin_width_seconds: float = SWIFT_BIN_WIDTH_SECONDS
    min_significance_sigma: float = MIN_SIGNIFICANCE_SIGMA
    placeholder_coords: list = None

    def __post_init__(self):
        if self.placeholder_coords is None:
            self.placeholder_coords = PLACEHOLDER_COORDS


class LightCurveProcessor:
    @staticmethod
    def load_event_file(evt_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        with fits.open(evt_path) as hdul:
            data = hdul[1].data
            header = hdul[1].header

            time = data['TIME']
            energy = data['ENERGY']

            if 'TRIGTIME' in header:
                trigger_time = header['TRIGTIME']
            elif 'TSTART' in header:
                trigger_time = header['TSTART']
            else:
                trigger_time = time.min()

            return time, energy, trigger_time

    @staticmethod
    def extract_lightcurve(time: np.ndarray, energy: np.ndarray,
                           trigger_time: float, energy_range: Tuple[int, int],
                           time_window: Tuple[float, float],
                           bin_width: float) -> Tuple[np.ndarray, np.ndarray]:

        rel_time = time - trigger_time

        emin, emax = energy_range
        mask_energy = (energy >= emin) & (energy < emax)

        mask_time = ((rel_time >= time_window[0]) &
                     (rel_time <= time_window[1]))

        mask = mask_energy & mask_time
        times_selected = rel_time[mask]

        if len(times_selected) == 0:
            return np.array([]), np.array([])

        bins = np.arange(time_window[0],
                         time_window[1] + bin_width,
                         bin_width)

        counts, _ = np.histogram(times_selected, bins=bins)
        centroids = bins[:-1] + bin_width / 2
        rates = counts / bin_width

        return rates, centroids

    @staticmethod
    def normalize_lightcurve(rates: np.ndarray) -> np.ndarray:
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)

        if std_rate < 1e-10:
            return np.zeros_like(rates)

        normalized = (rates - mean_rate) / std_rate
        return normalized


class SkyLocalizationExtractor:
    @staticmethod
    def extract_from_header(evt_path: str) -> Optional[Tuple[float, float]]:
        try:
            with fits.open(evt_path) as hdul:
                header = hdul[0].header
                ra = header.get('RA_OBJ', None)
                dec = header.get('DEC_OBJ', None)

                if ra is None or dec is None:
                    return None

                return float(ra), float(dec)
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def get_sky_coordinates(evt_path: str) -> Optional[Tuple[float, float]]:
        return SkyLocalizationExtractor.extract_from_header(evt_path)


class SwiftEventProcessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process_single_file(self, filepath: str) -> Optional[dict]:
        try:
            time, energy, trigger_time = LightCurveProcessor.load_event_file(
                filepath)

            soft_rates, _ = LightCurveProcessor.extract_lightcurve(
                time, energy, trigger_time,
                self.config.energy_band_soft,
                self.config.time_window,
                self.config.bin_width_seconds
            )

            hard_rates, _ = LightCurveProcessor.extract_lightcurve(
                time, energy, trigger_time,
                self.config.energy_band_hard,
                self.config.time_window,
                self.config.bin_width_seconds
            )

            lag_ms, uncertainty_ms = CrossCorrelationAnalyzer.compute_spectral_lag(
                soft_rates, hard_rates, self.config.bin_width_seconds
            )

            coords = SkyLocalizationExtractor.get_sky_coordinates(filepath)

            if coords is None:
                return None

            ra, dec = coords

            if QualityFilter.is_placeholder(ra, dec, self.config.placeholder_coords):
                return None

            is_significant = QualityFilter.is_significant(
                lag_ms, uncertainty_ms, self.config.min_significance_sigma)
            lag_significance = QualityFilter.lag_significance(
                lag_ms, uncertainty_ms)
            lag_class = QualityFilter.classify_lag(
                lag_ms, uncertainty_ms, self.config.min_significance_sigma)

            return {
                'filename': os.path.basename(filepath),
                'ra': ra,
                'dec': dec,
                'lag_ms': lag_ms,
                'lag_error': uncertainty_ms,
                'lag_significance': lag_significance,
                'is_significant': is_significant,
                'lag_type': 'positive' if lag_ms > 0 else 'negative',
                'lag_class': lag_class
            }

        except Exception as e:
            return None

    def process_dataset(self) -> pd.DataFrame:
        evt_files = []
        for root, dirs, files in os.walk(self.config.data_dir):
            for f in files:
                if 'bevshsp_uf.evt' in f:
                    evt_files.append(os.path.join(root, f))

        if not evt_files:
            print(f"No event files found in {self.config.data_dir}")
            return pd.DataFrame(columns=['filename', 'ra', 'dec', 'lag_ms',
                                         'lag_error', 'lag_significance',
                                         'is_significant', 'lag_type',
                                         'lag_class'])

        print(f"Found {len(evt_files)} event files")

        results = []
        for i, filepath in enumerate(evt_files):
            if i % 100 == 0:
                print(f"Processing: {i}/{len(evt_files)} files...")

            burst_data = self.process_single_file(filepath)
            if burst_data is not None:
                results.append(burst_data)

        if not results:
            print("No valid bursts found after quality filtering")
            return pd.DataFrame(columns=['filename', 'ra', 'dec', 'lag_ms',
                                         'lag_error', 'lag_significance',
                                         'is_significant', 'lag_type',
                                         'lag_class'])

        df = pd.DataFrame(results)

        print(f"Successfully processed {len(df)} bursts with valid lag measurements")
        print(f"Significant lags: {df['is_significant'].sum()}")
        print(f"Consistent with zero: {(~df['is_significant']).sum()}")
        df.to_csv(self.config.output_file, index=False)
        print(f"Data saved to {self.config.output_file}")

        return df


def main():
    config = PreprocessingConfig(
        data_dir=SWIFT_RAW_DIR,
        output_file=SWIFT_CSV
    )

    processor = SwiftEventProcessor(config)
    dataset = processor.process_dataset()

    return dataset


if __name__ == "__main__":
    main()
