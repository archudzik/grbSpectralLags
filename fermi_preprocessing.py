from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import glob
import os

from grb_lag_common import CrossCorrelationAnalyzer, QualityFilter
from config import (
    FERMI_BIN_WIDTH_SECONDS,
    FERMI_CSV,
    FERMI_DETECTOR,
    FERMI_HARD_BAND_KEV,
    FERMI_RAW_DIR,
    FERMI_SOFT_BAND_KEV,
    MIN_SIGNIFICANCE_SIGMA,
    PLACEHOLDER_COORDS,
    TIME_WINDOW_SECONDS,
)

try:
    from gbm.binning.unbinned import bin_by_time
    from gbm.data import TTE, GbmHealPix
    GBM_AVAILABLE = True
except ImportError:
    GBM_AVAILABLE = False


@dataclass
class PreprocessingConfig:
    data_dir: str
    output_file: str = FERMI_CSV
    detector: str = FERMI_DETECTOR
    energy_band_soft: Tuple[int, int] = FERMI_SOFT_BAND_KEV
    energy_band_hard: Tuple[int, int] = FERMI_HARD_BAND_KEV
    time_window: Tuple[float, float] = TIME_WINDOW_SECONDS
    bin_width_seconds: float = FERMI_BIN_WIDTH_SECONDS
    min_significance_sigma: float = MIN_SIGNIFICANCE_SIGMA
    placeholder_coords: list = None
    
    def __post_init__(self):
        if self.placeholder_coords is None:
            self.placeholder_coords = PLACEHOLDER_COORDS


class LightCurveProcessor:
    @staticmethod
    def extract_lightcurve(tte: 'TTE', energy_range: Tuple[int, int], 
                          time_window: Tuple[float, float], 
                          bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
        
        lightcurve_data = tte.slice_energy(energy_range).to_phaii(
            bin_by_time, bin_width, time_ref=0
        ).to_lightcurve()
        
        time_mask = ((lightcurve_data.centroids >= time_window[0]) & 
                     (lightcurve_data.centroids <= time_window[1]))
        
        rates = lightcurve_data.rates[time_mask]
        times = lightcurve_data.centroids[time_mask]
        
        return rates, times
    
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
    def extract_from_healpix(tte_filepath: str) -> Optional[Tuple[float, float]]:
        burst_id = os.path.basename(tte_filepath).split('_')[3]
        search_dir = os.path.dirname(tte_filepath)
        
        healpix_pattern = os.path.join(search_dir, f"glg_healpix_all_{burst_id}*.fit")
        healpix_files = glob.glob(healpix_pattern)
        
        if not healpix_files:
            return None
        
        try:
            localization = GbmHealPix.open(healpix_files[0])
            ra, dec = localization.centroid
            return ra, dec
        except Exception:
            return None
    
    @staticmethod
    def extract_from_header(tte: 'TTE') -> Optional[Tuple[float, float]]:
        try:
            ra = tte.headers['PRIMARY']['RA_OBJ']
            dec = tte.headers['PRIMARY']['DEC_OBJ']
            
            if ra is None or dec is None:
                return None
            
            return float(ra), float(dec)
        except (KeyError, ValueError, TypeError):
            return None
    
    @staticmethod
    def get_sky_coordinates(tte: 'TTE', filepath: str) -> Optional[Tuple[float, float]]:
        coords = SkyLocalizationExtractor.extract_from_healpix(filepath)
        
        if coords is None:
            coords = SkyLocalizationExtractor.extract_from_header(tte)
        
        return coords


class FermiTTEProcessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        if not GBM_AVAILABLE:
            raise ImportError("GBM tools not available. Install gbm-data-tools package.")
    
    def process_single_file(self, filepath: str) -> Optional[dict]:
        try:
            tte = TTE.open(filepath)
            
            soft_rates, _ = LightCurveProcessor.extract_lightcurve(
                tte, 
                self.config.energy_band_soft,
                self.config.time_window,
                self.config.bin_width_seconds
            )
            
            hard_rates, _ = LightCurveProcessor.extract_lightcurve(
                tte,
                self.config.energy_band_hard,
                self.config.time_window,
                self.config.bin_width_seconds
            )
            
            lag_ms, uncertainty_ms = CrossCorrelationAnalyzer.compute_spectral_lag(
                soft_rates, hard_rates, self.config.bin_width_seconds
            )
            
            coords = SkyLocalizationExtractor.get_sky_coordinates(tte, filepath)
            
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
        search_pattern = os.path.join(
            self.config.data_dir, 
            "**", 
            f"glg_tte_{self.config.detector}_*.fit"
        )
        
        tte_files = glob.glob(search_pattern, recursive=True)
        
        if not tte_files:
            print(f"No TTE files found in {self.config.data_dir}")
            return pd.DataFrame(columns=['filename', 'ra', 'dec', 'lag_ms', 
                                        'lag_error', 'lag_significance',
                                        'is_significant', 'lag_type',
                                        'lag_class'])
        
        print(f"Found {len(tte_files)} TTE files for detector {self.config.detector}")
        
        results = []
        for i, filepath in enumerate(tte_files):
            if i % 100 == 0:
                print(f"Processing: {i}/{len(tte_files)} files...")
            
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
        data_dir=FERMI_RAW_DIR,
        output_file=FERMI_CSV
    )
    
    processor = FermiTTEProcessor(config)
    dataset = processor.process_dataset()
    
    return dataset


if __name__ == "__main__":
    if GBM_AVAILABLE:
        main()
    else:
        print("GBM tools not available. Cannot process TTE files.")
        print("Install with: pip install gbm-data-tools")
