import argparse
import gzip
import re
import urllib.request
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
from astropy.io import fits

from grb_lag_common import classify_duration, to_public_schema
from config import SWIFT_CSV, SWIFT_RAW_DIR, SWIFT_SUMMARY_URL


SWIFT_COLUMNS = [
    "grb_name",
    "swift_trigger_id",
    "trigger_time_met",
    "grb_time_utc",
    "bat_ra",
    "bat_dec",
    "image_position_err",
    "image_snr",
    "t90_s",
    "t90_error_s",
    "t50_s",
    "t50_error_s",
    "event_start_since_trigger_s",
    "event_stop_since_trigger_s",
    "partial_coding_fraction",
    "trigger_method",
    "xrt_detection",
    "comment",
]

def obs_id_from_filename(filename: str) -> str:
    match = re.search(r"sw(\d{11})", str(filename))
    return match.group(1) if match else ""


def obs_id_from_trigger_id(trigger_id: str) -> str:
    trigger_id = str(trigger_id).strip()
    if trigger_id.endswith(".0"):
        trigger_id = trigger_id[:-2]
    if not trigger_id or trigger_id.upper() == "N/A":
        return ""
    if trigger_id.isdigit() and len(trigger_id) == 11:
        return trigger_id
    if trigger_id.isdigit():
        return f"{int(trigger_id):08d}000"
    return ""


def is_valid_obs_id(obs_id: str) -> bool:
    return bool(re.fullmatch(r"\d{11}", str(obs_id)))


def fetch_swift_metadata() -> pd.DataFrame:
    request = urllib.request.Request(
        SWIFT_SUMMARY_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8", errors="replace")

    table_lines = [
        line for line in text.splitlines()
        if "|" in line and not line.lstrip().startswith("#")
    ]
    if not table_lines:
        raise RuntimeError("No Swift BAT metadata rows returned")

    df = pd.read_csv(
        StringIO("\n".join(table_lines)),
        sep="|",
        names=SWIFT_COLUMNS,
        dtype=str,
        engine="python",
    )
    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()
        df.loc[df[column].str.upper().eq("N/A"), column] = pd.NA

    numeric_columns = [
        "swift_trigger_id",
        "trigger_time_met",
        "bat_ra",
        "bat_dec",
        "image_position_err",
        "image_snr",
        "t90_s",
        "t90_error_s",
        "t50_s",
        "t50_error_s",
        "event_start_since_trigger_s",
        "event_stop_since_trigger_s",
        "partial_coding_fraction",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["swift_trigger_id"] = df["swift_trigger_id"].astype("Int64")
    df["obs_id"] = df["swift_trigger_id"].apply(
        lambda value: obs_id_from_trigger_id(value)
        if not pd.isna(value) else ""
    )
    df["duration_class"] = df["t90_s"].apply(classify_duration)
    return df


def build_local_event_metadata(data_dir: str) -> pd.DataFrame:
    rows = []
    for path in Path(data_dir).rglob("*bevshsp_uf.evt*"):
        obs_id = obs_id_from_filename(path.name)
        if not obs_id:
            continue

        year_month = ""
        for part in path.parts:
            if re.fullmatch(r"\d{4}_\d{2}", part):
                year_month = part
                break

        trigger_time_met = pd.NA
        gzip_mtime_utc = pd.NA
        if path.suffix == ".gz":
            try:
                with gzip.open(path, "rb") as gzip_file:
                    gzip_file.read(1)
                    if gzip_file.mtime is not None:
                        gzip_mtime_utc = datetime.fromtimestamp(
                            gzip_file.mtime, tz=timezone.utc
                        ).isoformat().replace("+00:00", "Z")
            except (OSError, EOFError, gzip.BadGzipFile):
                pass

        try:
            with fits.open(path, memmap=False) as hdul:
                for hdu in hdul:
                    if "TRIGTIME" in hdu.header:
                        trigger_time_met = hdu.header["TRIGTIME"]
                        break
                if pd.isna(trigger_time_met):
                    for hdu in hdul:
                        if "TSTART" in hdu.header:
                            trigger_time_met = hdu.header["TSTART"]
                            break
        except (OSError, EOFError, gzip.BadGzipFile):
            pass

        rows.append({
            "obs_id": obs_id,
            "swift_data_year_month": year_month,
            "local_trigger_time_met": trigger_time_met,
            "local_gzip_mtime_utc": gzip_mtime_utc,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "obs_id",
            "swift_data_year_month",
            "local_trigger_time_met",
            "local_gzip_mtime_utc",
        ])

    df = pd.DataFrame(rows)
    df["local_trigger_time_met"] = pd.to_numeric(
        df["local_trigger_time_met"], errors="coerce"
    )
    return df.drop_duplicates(subset=["obs_id"], keep="first")


def fill_from_time_matches(
    enriched: pd.DataFrame,
    metadata_df: pd.DataFrame,
    columns_to_fill: list[str],
    tolerance_seconds: float = 1.0,
) -> pd.DataFrame:
    metadata_with_time = metadata_df[
        metadata_df["trigger_time_met"].notna() & metadata_df["t90_s"].notna()
    ].copy()
    if metadata_with_time.empty or "trigger_time_met" not in enriched.columns:
        return enriched

    metadata_with_time["trigger_time_met"] = pd.to_numeric(
        metadata_with_time["trigger_time_met"], errors="coerce"
    )
    metadata_with_time = metadata_with_time.dropna(subset=["trigger_time_met"])
    catalog_times = metadata_with_time["trigger_time_met"].to_numpy()

    match_methods = enriched.get(
        "metadata_match_method",
        pd.Series("obs_id", index=enriched.index, dtype=object),
    ).copy()
    missing_mask = enriched["t90_s"].isna() & enriched["trigger_time_met"].notna()

    for idx, row in enriched[missing_mask].iterrows():
        local_time = pd.to_numeric(row["trigger_time_met"], errors="coerce")
        if pd.isna(local_time):
            continue

        deltas = abs(catalog_times - float(local_time))
        nearest_position = deltas.argmin()
        if deltas[nearest_position] > tolerance_seconds:
            continue

        catalog_row = metadata_with_time.iloc[nearest_position]
        for column in columns_to_fill:
            if column in enriched.columns and column in catalog_row.index:
                if pd.isna(enriched.at[idx, column]):
                    enriched.at[idx, column] = catalog_row[column]
        match_methods.at[idx] = "trigger_time_met"

    enriched["metadata_match_method"] = match_methods
    return enriched


def swift_met_to_utc(met_seconds) -> pd.Timestamp:
    met_seconds = pd.to_numeric(met_seconds, errors="coerce")
    if pd.isna(met_seconds):
        return pd.NaT
    return pd.Timestamp("2001-01-01T00:00:00") + pd.to_timedelta(
        float(met_seconds), unit="s"
    )


def fill_missing_utc_from_local_time(df: pd.DataFrame) -> pd.DataFrame:
    if "grb_time_utc" not in df.columns or "trigger_time_met" not in df.columns:
        return df

    missing_utc = df["grb_time_utc"].isna() & df["trigger_time_met"].notna()
    if missing_utc.any():
        fallback_times = df.loc[missing_utc, "trigger_time_met"].apply(
            swift_met_to_utc
        )
        df.loc[missing_utc, "grb_time_utc"] = fallback_times.dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        ).str.rstrip("0").str.rstrip(".")

    if "local_gzip_mtime_utc" in df.columns:
        missing_utc = df["grb_time_utc"].isna() & df["local_gzip_mtime_utc"].notna()
        df.loc[missing_utc, "grb_time_utc"] = df.loc[
            missing_utc, "local_gzip_mtime_utc"
        ]
    return df


def add_metadata(
    input_file: str,
    output_file: str,
    cache_file: str | None,
    data_dir: str | None,
) -> pd.DataFrame:
    lag_df = pd.read_csv(input_file, dtype={"obs_id": str})
    metadata_output_columns = [
        "instrument",
        "grb_name",
        "swift_trigger_id",
        "grb_time_utc",
        "trigger_time_met",
        "t90_s",
        "t90_error_s",
        "t50_s",
        "t50_error_s",
        "event_start_since_trigger_s",
        "event_stop_since_trigger_s",
        "partial_coding_fraction",
        "trigger_method",
        "xrt_detection",
        "duration_class",
        "swift_data_year_month",
        "local_trigger_time_met",
        "local_gzip_mtime_utc",
        "metadata_match_method",
    ]
    lag_df = lag_df.drop(
        columns=[col for col in metadata_output_columns if col in lag_df.columns]
    )

    if "obs_id" not in lag_df.columns:
        lag_df.insert(0, "obs_id", lag_df["filename"].apply(obs_id_from_filename))
    else:
        lag_df["obs_id"] = lag_df["obs_id"].astype(str).str.zfill(11)
        filename_obs_id = lag_df["filename"].apply(obs_id_from_filename)
        invalid_obs_id = ~lag_df["obs_id"].apply(is_valid_obs_id)
        lag_df.loc[invalid_obs_id, "obs_id"] = filename_obs_id[invalid_obs_id]

    if cache_file:
        try:
            metadata_df = pd.read_csv(cache_file, dtype={"obs_id": str})
            if (
                "obs_id" not in metadata_df.columns
                or metadata_df["obs_id"].fillna("").apply(is_valid_obs_id).sum() == 0
            ):
                metadata_df = fetch_swift_metadata()
                metadata_df.to_csv(cache_file, index=False)
        except FileNotFoundError:
            metadata_df = fetch_swift_metadata()
            metadata_df.to_csv(cache_file, index=False)
    else:
        metadata_df = fetch_swift_metadata()

    metadata_df["obs_id"] = metadata_df["obs_id"].fillna("").astype(str)
    needs_obs_id = ~metadata_df["obs_id"].apply(is_valid_obs_id)
    if needs_obs_id.any() and "swift_trigger_id" in metadata_df.columns:
        metadata_df.loc[needs_obs_id, "obs_id"] = metadata_df.loc[
            needs_obs_id, "swift_trigger_id"
        ].apply(obs_id_from_trigger_id)

    metadata_df = metadata_df[metadata_df["obs_id"].apply(is_valid_obs_id)].copy()
    metadata_df = metadata_df.drop_duplicates(subset=["obs_id"], keep="first")
    keep_columns = [
        "obs_id",
        "grb_name",
        "swift_trigger_id",
        "grb_time_utc",
        "trigger_time_met",
        "t90_s",
        "t90_error_s",
        "t50_s",
        "t50_error_s",
        "event_start_since_trigger_s",
        "event_stop_since_trigger_s",
        "partial_coding_fraction",
        "trigger_method",
        "xrt_detection",
        "duration_class",
    ]
    enriched = lag_df.merge(
        metadata_df[keep_columns],
        on="obs_id",
        how="left",
        validate="many_to_one",
    )

    if data_dir:
        local_metadata = build_local_event_metadata(data_dir)
        if not local_metadata.empty:
            enriched = enriched.merge(
                local_metadata,
                on="obs_id",
                how="left",
                validate="many_to_one",
            )
            enriched["trigger_time_met"] = enriched[
                "trigger_time_met"
            ].fillna(enriched["local_trigger_time_met"])

    enriched = fill_from_time_matches(enriched, metadata_df, keep_columns[1:])
    enriched = fill_missing_utc_from_local_time(enriched)
    enriched["duration_class"] = enriched["duration_class"].fillna("unknown")
    public_df = to_public_schema(enriched, "Swift/BAT")
    public_df.to_csv(output_file, index=False)

    matched = enriched["t90_s"].notna().sum()
    print(f"Updated {output_file}")
    print(f"Rows: {len(enriched)}")
    print(f"Rows with T90 metadata: {matched}")
    print(f"Rows without T90 metadata: {len(enriched) - matched}")
    if "swift_data_year_month" in enriched.columns:
        print(
            "Rows with local year/month path metadata: "
            f"{enriched['swift_data_year_month'].notna().sum()}"
        )
    if "trigger_time_met" in enriched.columns:
        print(
            "Rows with Swift MET trigger/local timing: "
            f"{enriched['trigger_time_met'].notna().sum()}"
        )
    print(enriched["duration_class"].fillna("unknown").value_counts(dropna=False))

    return public_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Swift BAT trigger time and duration metadata to lag CSV."
    )
    parser.add_argument("--input", default=SWIFT_CSV)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--data-dir", default=SWIFT_RAW_DIR)
    args = parser.parse_args()

    output_file = args.output or args.input
    add_metadata(args.input, output_file, args.cache, args.data_dir)


if __name__ == "__main__":
    main()
