from __future__ import annotations

from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPO_DIR / "figures"

FERMI_CSV = "fermi_full_data.csv"
SWIFT_CSV = "swift_full_data.csv"

FERMI_RAW_DIR = "fermi_tte_data"
SWIFT_RAW_DIR = "swift_bat_data"

FERMI_DETECTOR = "n9"
FERMI_SOFT_BAND_KEV = (30, 100)
FERMI_HARD_BAND_KEV = (300, 1000)
FERMI_BIN_WIDTH_SECONDS = 0.010

SWIFT_SOFT_BAND_KEV = (15, 50)
SWIFT_HARD_BAND_KEV = (50, 150)
SWIFT_BIN_WIDTH_SECONDS = 0.064

TIME_WINDOW_SECONDS = (0, 300)
MIN_SIGNIFICANCE_SIGMA = 2.0
PLACEHOLDER_COORDS = [(0.0, 0.0), (30.0, -15.0)]

FERMI_LAG_FIGURE = "figures/fig_fermi_lag_distributions.png"
SWIFT_LAG_FIGURE = "figures/fig_swift_lag_distributions.png"
COMPARISON_FIGURE = "figures/fig_fermi_swift_comparison.png"
FERMI_CLOSURE_FIGURE = "figures/fig_fermi_population_optimized.png"
SWIFT_CLOSURE_FIGURE = "figures/fig_swift_population_optimized.png"

DIRECTIONAL_TEST_FAMILY_SIZE = 4
OPTIMAL_AXIS_RANDOM_SEED = 12345

FERMI_HEASARC_BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/"
FERMI_XAMIN_URL = "https://heasarc.gsfc.nasa.gov/xamin/query"
SWIFT_HEASARC_BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/"
SWIFT_SUMMARY_URL = (
    "https://swift.gsfc.nasa.gov/results/batgrbcat/"
    "summary_cflux/summary_general_info/summary_general.txt"
)
