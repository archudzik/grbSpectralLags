import argparse
import re
import urllib.parse
import urllib.request
from io import StringIO

import pandas as pd

from grb_lag_common import classify_duration, to_public_schema
from config import FERMI_CSV, FERMI_XAMIN_URL


FERMI_FIELDS = [
    "trigger_name",
    "name",
    "trigger_time",
    "t90",
    "t90_error",
    "t50",
    "t50_error",
    "duration_energy_low",
    "duration_energy_high",
]

def burst_id_from_filename(filename: str) -> str:
    match = re.search(r"(bn\d{9})", str(filename))
    return match.group(1) if match else ""


def fetch_fermi_metadata() -> pd.DataFrame:
    params = {
        "table": "fermigbrst",
        "fields": ",".join(FERMI_FIELDS),
        "format": "text",
        "resultmax": "0",
    }
    url = f"{FERMI_XAMIN_URL}?{urllib.parse.urlencode(params)}"

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8", errors="replace")

    table_lines = [
        line for line in text.splitlines()
        if "|" in line and not line.startswith("----")
    ]
    if not table_lines:
        raise RuntimeError("No Fermi metadata rows returned by HEASARC Xamin")

    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|", dtype=str)
    df.columns = [column.strip() for column in df.columns]
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()

    numeric_columns = [
        "t90",
        "t90_error",
        "t50",
        "t50_error",
        "duration_energy_low",
        "duration_energy_high",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.rename(columns={
        "trigger_name": "fermi_trigger_name",
        "name": "grb_name",
        "trigger_time": "grb_time_utc",
        "t90": "t90_s",
        "t90_error": "t90_error_s",
        "t50": "t50_s",
        "t50_error": "t50_error_s",
    })
    df["burst_id"] = df["fermi_trigger_name"]
    df["duration_class"] = df["t90_s"].apply(classify_duration)
    return df


def add_metadata(input_file: str, output_file: str, cache_file: str | None) -> pd.DataFrame:
    lag_df = pd.read_csv(input_file)
    metadata_columns = [
        "instrument",
        "burst_id",
        "fermi_trigger_name",
        "grb_name",
        "grb_time_utc",
        "t90_s",
        "t90_error_s",
        "t50_s",
        "t50_error_s",
        "duration_energy_low",
        "duration_energy_high",
        "duration_class",
    ]
    lag_df = lag_df.drop(
        columns=[column for column in metadata_columns if column in lag_df.columns]
    )
    lag_df["burst_id"] = lag_df["filename"].apply(burst_id_from_filename)

    if cache_file:
        try:
            metadata_df = pd.read_csv(cache_file)
        except FileNotFoundError:
            metadata_df = fetch_fermi_metadata()
            metadata_df.to_csv(cache_file, index=False)
    else:
        metadata_df = fetch_fermi_metadata()

    keep_columns = [
        "burst_id",
        "fermi_trigger_name",
        "grb_name",
        "grb_time_utc",
        "t90_s",
        "t90_error_s",
        "t50_s",
        "t50_error_s",
        "duration_energy_low",
        "duration_energy_high",
        "duration_class",
    ]
    enriched = lag_df.merge(
        metadata_df[keep_columns],
        on="burst_id",
        how="left",
        validate="many_to_one",
    )
    public_df = to_public_schema(enriched, "Fermi/GBM")
    public_df.to_csv(output_file, index=False)

    matched = enriched["t90_s"].notna().sum()
    print(f"Updated {output_file}")
    print(f"Rows: {len(enriched)}")
    print(f"Rows with T90 metadata: {matched}")
    print(f"Rows without T90 metadata: {len(enriched) - matched}")
    print(enriched["duration_class"].fillna("unknown").value_counts(dropna=False))

    return public_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Fermi GBM trigger time and duration metadata to lag CSV."
    )
    parser.add_argument("--input", default=FERMI_CSV)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    output_file = args.output or args.input
    add_metadata(args.input, output_file, args.cache)


if __name__ == "__main__":
    main()
