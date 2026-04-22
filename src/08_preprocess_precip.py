from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import os

import numpy as np
import pandas as pd

from io_knmi import (
    get_time_from_dataset_or_filename,
    infer_station_dim,
    list_knmi_files,
    open_knmi_file,
    parse_knmi_10min_filename,
)


# This script turns raw KNMI 10-minute rainfall files into one clean time series per station.
# Main tasks:
# 1. keep only the requested time range
# 2. rebuild the full 10-minute timeline
# 3. flag missing or duplicate intervals
# 4. compute 1 h and 3 h rolling rainfall only on complete windows
# 5. save one processed file per station

EXPECTED_FREQ = "10min"
WINDOW_1H = 6
WINDOW_3H = 18


def parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def filter_files_by_timerange(
    filepaths: Iterable[str | Path],
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> list[Path]:
    
    # KNMI filenames contain the UTC time at the end of each 10-minute interval.
    # We use that filename time to keep only files inside the requested period.
    
    start_ts = parse_utc_timestamp(start_utc)
    end_ts = parse_utc_timestamp(end_utc)

    selected: list[Path] = []

    for fp in filepaths:
        fp = Path(fp)
        ts = parse_knmi_10min_filename(fp.name)

        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue

        selected.append(fp)

    if not selected:
        raise ValueError(
            f"No files remain after time filtering. start_utc={start_utc}, end_utc={end_utc}"
        )

    return selected


def _decode_text_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, encoding="utf-8", errors="ignore")
    return arr.astype(str)


def load_station_metadata(first_file: str | Path) -> pd.DataFrame:
    
    # Read station identifiers and basic metadata from the first file.
    # This assumes station metadata are stable across the selected file set.

    with open_knmi_file(first_file) as ds:
        station_dim = infer_station_dim(ds)

        meta = pd.DataFrame(
            {
                "station": _decode_text_array(ds[station_dim].values),
            }
        )

        for field in ("stationname", "wsi", "lat", "lon", "height"):
            if field in ds.variables and station_dim in ds[field].dims:
                values = ds[field].values
                if np.asarray(values).dtype.kind in {"S", "U", "O"}:
                    meta[field] = _decode_text_array(values)
                else:
                    meta[field] = values

    meta["station"] = meta["station"].astype(str)
    return meta


def read_raw_long_table(
    filepaths: Iterable[str | Path],
    precip_var: str = "rg",
) -> pd.DataFrame:
    
    # Read all selected files into one long table:
    # one row = one station at one 10-minute timestamp.
    #
    # At this stage we keep raw values as they are.
    # Cleaning, gap handling, and rolling sums are done later.
    
    rows: list[pd.DataFrame] = []

    for path in filepaths:
        with open_knmi_file(path) as ds:
            station_dim = infer_station_dim(ds)

            if precip_var not in ds.variables:
                raise KeyError(f"{precip_var!r} not found in {path}")

            da = ds[precip_var]

            if "time" in da.dims:
                if ds.sizes["time"] != 1:
                    raise ValueError(
                        f"Expected one time per file, got {ds.sizes['time']} in {path}"
                    )
                da = da.isel(time=0)

            if station_dim not in da.dims or da.ndim != 1:
                raise ValueError(
                    f"Expected {precip_var!r} to be 1D over station after time selection in {path}"
                )

            timestamp_end = get_time_from_dataset_or_filename(ds, path)
            station_ids = _decode_text_array(ds[station_dim].values).astype(str)
            raw_values = pd.to_numeric(pd.Series(da.values), errors="coerce").to_numpy()

            file_df = pd.DataFrame(
                {
                    "timestamp_end_utc": pd.Timestamp(timestamp_end),
                    "station": station_ids,
                    "raw_value": raw_values,
                    "source_file": Path(path).name,
                }
            )
            rows.append(file_df)

    if not rows:
        raise ValueError("No raw rows were read from the supplied files.")

    out = pd.concat(rows, ignore_index=True)
    out["station"] = out["station"].astype(str)
    out = out.sort_values(["station", "timestamp_end_utc"]).reset_index(drop=True)
    return out


def collapse_duplicate_timestamps(raw_station_df: pd.DataFrame) -> pd.DataFrame:

    # A station should have at most one rainfall value per 10-minute timestamp.
    # If duplicates exist:
    # - identical finite values are collapsed to one value
    # - conflicting values are treated as invalid
    #
    # This is a conservative choice.

    collapsed_rows: list[dict[str, object]] = []

    for timestamp, grp in raw_station_df.groupby("timestamp_end_utc", sort=True):
        values = pd.to_numeric(grp["raw_value"], errors="coerce")
        finite_unique = np.unique(values[np.isfinite(values)])

        record_count_at_timestamp = int(len(grp))
        duplicate_count = max(0, record_count_at_timestamp - 1)
        duplicate_timestamp = record_count_at_timestamp > 1
        duplicate_conflict = duplicate_timestamp and len(finite_unique) > 1

        if len(finite_unique) == 0:
            raw_value = np.nan
        elif len(finite_unique) == 1:
            raw_value = float(finite_unique[0])
        else:
            raw_value = np.nan

        collapsed_rows.append(
            {
                "timestamp_end_utc": pd.Timestamp(timestamp),
                "raw_value": raw_value,
                "has_raw_record": True,
                "record_count_at_timestamp": record_count_at_timestamp,
                "duplicate_count": duplicate_count,
                "duplicate_timestamp": duplicate_timestamp,
                "duplicate_conflict": duplicate_conflict,
                "station_value_missing": not np.isfinite(raw_value),
            }
        )

    out = pd.DataFrame(collapsed_rows).sort_values("timestamp_end_utc").reset_index(drop=True)
    return out


def convert_to_10min_amount_mm(raw_value: pd.Series, precip_var: str) -> pd.Series:
    if precip_var in {"rg", "pg"}:
        return raw_value / 6.0                  # valid because the variable is a 10-minute mean intensity and the units are mm/h
    raise NotImplementedError(
        f"Conversion for precip_var={precip_var!r} is not implemented. "
        "It should use 'rg'."
    )


def add_time_axes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["interval_end_utc"] = out["timestamp_end_utc"]
    out["interval_start_utc"] = out["timestamp_end_utc"] - pd.Timedelta(minutes=10)
    return out


def build_complete_station_index(collapsed_df: pd.DataFrame) -> pd.DataFrame:
    
    # Rebuild the full 10-minute sequence for this station.
    # Missing timestamps become explicit rows so later rolling windows can detect gaps.
    
    start = collapsed_df["timestamp_end_utc"].min()
    end = collapsed_df["timestamp_end_utc"].max()

    full_index = pd.date_range(start=start, end=end, freq=EXPECTED_FREQ, tz="UTC")
    base = pd.DataFrame({"timestamp_end_utc": full_index})

    out = base.merge(collapsed_df, on="timestamp_end_utc", how="left")
    out["has_raw_record"] = out["has_raw_record"].fillna(False)
    out["record_count_at_timestamp"] = out["record_count_at_timestamp"].fillna(0).astype("int64")
    out["duplicate_count"] = out["duplicate_count"].fillna(0).astype("int64")
    out["duplicate_timestamp"] = out["duplicate_timestamp"].fillna(False)
    out["duplicate_conflict"] = out["duplicate_conflict"].fillna(False)
    out["station_value_missing"] = out["station_value_missing"].fillna(False)

    out["missing_file_interval"] = ~out["has_raw_record"]
    out["missing_interval"] = out["missing_file_interval"] | out["station_value_missing"]

    return out


def add_accumulations(df: pd.DataFrame, precip_var: str) -> pd.DataFrame:
    
    # Build rainfall amounts and rolling accumulations.
    # Rolling sums are only accepted when the whole window is complete.
    # Any missing or invalid 10-minute step makes the full 1 h or 3 h window invalid.
    
    out = df.copy()

    # Negative rainfall is physically invalid here, so treat it as missing.
    out["negative_raw_value"] = out["raw_value"] < 0
    out.loc[out["negative_raw_value"], "raw_value"] = np.nan
    out.loc[out["negative_raw_value"], "station_value_missing"] = True
    out["missing_interval"] = out["missing_file_interval"] | out["station_value_missing"]

    out["precip_10min_amount_mm"] = convert_to_10min_amount_mm(out["raw_value"], precip_var)

    out["valid_10min_for_accum"] = (
        out["has_raw_record"]
        & out["precip_10min_amount_mm"].notna()
        & ~out["duplicate_timestamp"]
        & ~out["duplicate_conflict"]
    )
    
    
    # A 10-minute value is usable only if:
    # - a raw record exists
    # - the converted rainfall amount is finite
    # - the timestamp is not duplicated
    # - there is no duplicate conflict
    amount = out["precip_10min_amount_mm"]

    out["n_valid_10min_in_1h_window"] = (
        out["valid_10min_for_accum"].astype("int64").rolling(WINDOW_1H, min_periods=1).sum()
    ).astype("int64")
    out["n_valid_10min_in_3h_window"] = (
        out["valid_10min_for_accum"].astype("int64").rolling(WINDOW_3H, min_periods=1).sum()
    ).astype("int64")


    out["incomplete_1h_window"] = out["n_valid_10min_in_1h_window"] < WINDOW_1H
    out["incomplete_3h_window"] = out["n_valid_10min_in_3h_window"] < WINDOW_3H
    # Count how many valid 10-minute steps are present in each rolling window.
    # A complete 1 h window needs 6 valid steps.
    # A complete 3 h window needs 18 valid steps.
    rolling_1h = amount.rolling(WINDOW_1H, min_periods=WINDOW_1H).sum()
    rolling_3h = amount.rolling(WINDOW_3H, min_periods=WINDOW_3H).sum()
    # Compute rolling sums, then mask out incomplete windows.
    # This avoids partial sums pretending to be full 1 h or 3 h rainfall totals.
    out["rolling_1h_valid"] = ~out["incomplete_1h_window"]
    out["rolling_3h_valid"] = ~out["incomplete_3h_window"]

    out["rolling_1h_mm"] = rolling_1h.where(out["rolling_1h_valid"], np.nan)
    out["rolling_3h_mm"] = rolling_3h.where(out["rolling_3h_valid"], np.nan)

    return out


def add_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Daily totals are secondary diagnostics only.
    # They are grouped by the START of each 10-minute interval so the day assignment matches the KNMI interval convention.
    out["date_utc"] = out["interval_start_utc"].dt.floor("D")
    daily = (
            out.groupby("date_utc", as_index=False)
            .agg(
                daily_n_intervals=("timestamp_end_utc", "size"),
                daily_n_valid_10min=("valid_10min_for_accum", "sum"),
                daily_total_utc_mm=("precip_10min_amount_mm", lambda s: s.sum(min_count=144)),
            )
        )


    daily["daily_complete_utc_day"] = (
        (daily["daily_n_intervals"] == 144) & (daily["daily_n_valid_10min"] == 144)
    )
    daily.loc[~daily["daily_complete_utc_day"], "daily_total_utc_mm"] = np.nan

    out = out.merge(daily, on="date_utc", how="left")
    return out

def attach_station_metadata(
    station_df: pd.DataFrame,
    station_meta_row: pd.Series,
    precip_var: str,
) -> pd.DataFrame:
    out = station_df.copy()
    out["station"] = str(station_meta_row["station"])
    out["precip_variable"] = precip_var

    for field in ("stationname", "wsi", "lat", "lon", "height"):
        if field in station_meta_row.index:
            out[field] = station_meta_row[field]

    return out


def process_station(
    raw_station_df: pd.DataFrame,
    station_meta_row: pd.Series,
    precip_var: str = "rg",
    include_daily_totals: bool = True,
) -> pd.DataFrame:
    # Full preprocessing chain for one station:
    # raw rows -> collapse duplicates -> fill missing timestamps ->
    # add interval bounds -> compute rolling sums -> attach metadata
    station_id = str(station_meta_row["station"])

    if raw_station_df.empty:
        raise ValueError(f"No raw records found for station {station_id}")

    raw_station_df = raw_station_df.sort_values("timestamp_end_utc").reset_index(drop=True)

    collapsed = collapse_duplicate_timestamps(raw_station_df)
    complete = build_complete_station_index(collapsed)
    complete = add_time_axes(complete)
    complete = add_accumulations(complete, precip_var=precip_var)

    if include_daily_totals:
        complete = add_daily_totals(complete)

    complete = attach_station_metadata(complete, station_meta_row, precip_var=precip_var)

    ordered_cols = [
        "station",
        "stationname",
        "wsi",
        "lat",
        "lon",
        "height",
        "precip_variable",
        "timestamp_end_utc",
        "interval_start_utc",
        "interval_end_utc",
        "raw_value",
        "precip_10min_amount_mm",
        "has_raw_record",
        "missing_file_interval",
        "station_value_missing",
        "missing_interval",
        "record_count_at_timestamp",
        "duplicate_count",
        "duplicate_timestamp",
        "duplicate_conflict",
        "negative_raw_value",
        "valid_10min_for_accum",
        "n_valid_10min_in_1h_window",
        "incomplete_1h_window",
        "rolling_1h_valid",
        "rolling_1h_mm",
        "n_valid_10min_in_3h_window",
        "incomplete_3h_window",
        "rolling_3h_valid",
        "rolling_3h_mm",
        "date_end_utc",
        "daily_n_intervals",
        "daily_n_valid_10min",
        "daily_complete_utc_day",
        "daily_total_utc_mm",
    ]

    ordered_cols = [c for c in ordered_cols if c in complete.columns]
    return complete[ordered_cols].sort_values("timestamp_end_utc").reset_index(drop=True)


def write_station_parquet(
    station_df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_id = str(station_df["station"].iloc[0])
    path = output_dir / f"station_{station_id}.parquet"
    station_df.to_parquet(path, index=False)
    return path


def build_summary_record(
    processed: pd.DataFrame,
    station_meta_row: pd.Series,
    outpath: Path,
) -> dict[str, object]:
    return {
        "station": str(station_meta_row["station"]),
        "stationname": station_meta_row.get("stationname"),
        "n_rows": int(len(processed)),
        "n_missing_file_intervals": int(processed["missing_file_interval"].sum()),
        "n_station_value_missing": int(processed["station_value_missing"].sum()),
        "n_missing_intervals": int(processed["missing_interval"].sum()),
        "n_duplicate_timestamps": int(processed["duplicate_timestamp"].sum()),
        "n_valid_10min": int(processed["valid_10min_for_accum"].sum()),
        "n_valid_1h": int(processed["rolling_1h_valid"].sum()),
        "n_valid_3h": int(processed["rolling_3h_valid"].sum()),
        "frac_valid_10min": float(processed["valid_10min_for_accum"].mean()),
        "all_rg_missing": bool(processed["valid_10min_for_accum"].sum() == 0),
        "output_path": str(outpath),
    }


def _process_one_station_task(
    raw_station_df: pd.DataFrame,
    station_meta_row_dict: dict[str, object],
    precip_var: str,
    include_daily_totals: bool,
    output_dir: str | Path,
) -> dict[str, object]:
    station_meta_row = pd.Series(station_meta_row_dict)
    
    print(f"PID {os.getpid()} starting station {station_meta_row['station']}")
    
    processed = process_station(
        raw_station_df=raw_station_df,
        station_meta_row=station_meta_row,
        precip_var=precip_var,
        include_daily_totals=include_daily_totals,
    )
    outpath = write_station_parquet(processed, output_dir)
    print(f"PID {os.getpid()} finished station {station_meta_row['station']}")
    return build_summary_record(processed, station_meta_row, outpath)


def process_all_stations(
    input_dir: str | Path,
    output_dir: str | Path,
    precip_var: str = "rg",
    include_daily_totals: bool = True,
    station_ids: list[str] | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_knmi_files(input_dir)
    files = filter_files_by_timerange(files, start_utc=start_utc, end_utc=end_utc)

    print(f"Using {len(files)} raw files after time filtering.")
    print(f"First file: {files[0].name}")
    print(f"Last file:  {files[-1].name}")

    station_meta = load_station_metadata(files[0])

    if station_ids is not None:
        selected_ids = {str(s) for s in station_ids}
        station_meta = station_meta.loc[station_meta["station"].isin(selected_ids)].copy()

    if station_meta.empty:
        raise ValueError("No stations selected after applying station filter.")

    raw_long_df = read_raw_long_table(files, precip_var=precip_var)
    raw_long_df["station"] = raw_long_df["station"].astype(str)

    selected_station_ids = set(station_meta["station"])
    raw_long_df = raw_long_df.loc[raw_long_df["station"].isin(selected_station_ids)].copy()

    if raw_long_df.empty:
        raise ValueError("No raw data remain after station filtering.")

    station_groups = {
        station_id: grp.copy()
        for station_id, grp in raw_long_df.groupby("station", sort=False)
    }

    task_payloads: list[tuple[pd.DataFrame, dict[str, object], str, bool, str | Path]] = []

    for _, row in station_meta.iterrows():
        station_id = str(row["station"])
        if station_id not in station_groups:
            print(f"Skipping station {station_id}: no raw rows found.")
            continue

        task_payloads.append(
            (
                station_groups[station_id],
                row.to_dict(),
                precip_var,
                include_daily_totals,
                output_dir,
            )
        )

    if not task_payloads:
        raise ValueError("No station tasks were created.")

    summaries: list[dict[str, object]] = []

    if workers <= 1:
        for payload in task_payloads:
            summary = _process_one_station_task(*payload)
            summaries.append(summary)
            print(f"Processed station {summary['station']} -> {summary['output_path']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one_station_task, *payload) for payload in task_payloads]

            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                print(f"Processed station {summary['station']} -> {summary['output_path']}")

    summary_df = pd.DataFrame(summaries).sort_values("station").reset_index(drop=True)
    summary_path = output_dir / "preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess KNMI 10-minute rainfall into clean station series with valid rolling accumulations."
    )
    parser.add_argument(
        "--input-dir",
        default="data_raw/knmi_10min",
        help="Directory containing KNMI 10-minute NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_processed/stations",
        help="Directory for per-station parquet output.",
    )
    parser.add_argument(
        "--precip-var",
        default="rg",
        help="Precipitation variable to use. It should use 'rg'.",
    )
    parser.add_argument(
        "--no-daily-totals",
        action="store_true",
        help="Disable secondary daily-total diagnostics.",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Optional list of station ids to process.",
    )
    parser.add_argument(
        "--start-utc",
        default=None,
        help="Optional inclusive UTC start time for selecting raw files, e.g. 2012-05-01T00:00:00Z",
    )
    parser.add_argument(
        "--end-utc",
        default=None,
        help="Optional inclusive UTC end time for selecting raw files, e.g. 2012-05-07T07:40:00Z",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-station preprocessing. Use 1 for serial.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    process_all_stations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        precip_var=args.precip_var,
        include_daily_totals=not args.no_daily_totals,
        station_ids=args.stations,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()