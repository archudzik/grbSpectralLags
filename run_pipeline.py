from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import FERMI_CSV, SWIFT_CSV
from grb_lag_common import PUBLIC_COLUMNS
from reporting import print_rule, print_section


REPO_DIR = Path(__file__).resolve().parent


def run_step(label: str, command: list[str]) -> None:
    print_rule(label)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_DIR, check=True)


def validate_catalog(path: str) -> None:
    csv_path = REPO_DIR / path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocessing/metadata first or restore the processed CSV."
        )

    df = pd.read_csv(csv_path, nrows=5)
    if list(df.columns) != PUBLIC_COLUMNS:
        raise ValueError(
            f"{path} has unexpected columns.\n"
            f"Expected: {PUBLIC_COLUMNS}\n"
            f"Found:    {list(df.columns)}"
        )


def validate_inputs() -> None:
    print_section("Validating processed catalogs")
    for path in (FERMI_CSV, SWIFT_CSV):
        validate_catalog(path)
        print(f"OK: {path}", flush=True)
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GRB spectral-lag workflow from processed CSVs or raw files."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw NASA files before preprocessing. This is slow and network-dependent.",
    )
    parser.add_argument(
        "--with-preprocessing",
        action="store_true",
        help="Rebuild processed CSVs from local raw files before analysis.",
    )
    parser.add_argument(
        "--with-inversion",
        "--with-conversion",
        dest="with_inversion",
        action="store_true",
        help="Run the parameter-inversion step after the statistical analysis.",
    )
    args = parser.parse_args()

    python = sys.executable

    if args.download:
        run_step("Downloading Fermi raw files", [python, "fermi_download.py"])
        run_step("Downloading Swift raw files", [python, "swift_download.py"])
        args.with_preprocessing = True

    if args.with_preprocessing:
        run_step("Preprocessing Fermi TTE files", [python, "fermi_preprocessing.py"])
        run_step("Adding Fermi metadata", [python, "fermi_add_metadata.py"])
        run_step("Preprocessing Swift BAT files", [python, "swift_preprocessing.py"])
        run_step("Adding Swift metadata", [python, "swift_add_metadata.py"])

    validate_inputs()
    run_step("Running statistical analysis", [python, "run_analysis.py"])

    if args.with_inversion:
        run_step("Running parameter inversion", [python, "run_parameter_inversion.py"])

    print_rule("Pipeline complete")


if __name__ == "__main__":
    main()
