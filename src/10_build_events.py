from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Build event candidates from processed station rainfall series.
# Main rule used here:
# 1. use rolling 1 h rainfall at station level
# 2. find local maxima at each station
# 3. keep peaks at least 6 h apart at the same station
# 4. keep the top 5 station peaks in the selected 3-month block
# 5. merge station peaks into national events within 6 h and 75 km
#
# This gives a catalogue of the strongest short-duration rainfall events
# in the selected block. It is not a climatological threshold method.


# ----------------------------
# Fixed analysis choices
# ----------------------------
ANALYSIS_MONTHS = {5, 6, 7}  # May-June-July
ANALYSIS_START_UTC = None    # Example: "2012-05-01T00:00:00Z"
ANALYSIS_END_UTC = None      # Example: "2012-07-31T23:50:00Z"

TOP_N_STATION_SEEDS = 5
MIN_STATION_PEAK_SEPARATION_H = 6
EVENT_TIME_TOL_H = 6
EVENT_SPACE_TOL_KM = 75.0

INPUT_DIR = Path("data_processed/stations")
OUTPUT_DIR = Path("data_processed/events")

# Leave this False unless you want to debug only a few stations.
USE_PREFERRED_STATIONS = False
PREFERRED_STATIONS = ["06260", "06240", "06235", "06210"]

EVENT_DEFINITION_NAME = "top5_station_localmax_1h_may_jul_6h75km"


# ----------------------------
# Column detection
# ----------------------------
CANDIDATE_COLUMNS = {
    "station": ["station", "station_id", "station_code"],
    "stationname": ["stationname", "station_name", "name"],
    "time": ["timestamp_end_utc", "interval_end_utc", "time_utc", "time", "datetime", "timestamp"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "rain_1h": ["rolling_1h_mm", "rain_1h_mm", "r1h_mm", "precip_1h_mm", "R1H"],
    "rain_3h": ["rolling_3h_mm", "rain_3h_mm", "r3h_mm", "precip_3h_mm", "R3H"],
    "valid_1h": ["rolling_1h_valid", "valid_1h", "is_valid_1h", "window_valid_1h"],
}


def find_first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    detected: Dict[str, str] = {}
    missing: list[tuple[str, list[str]]] = []

    for key, candidates in CANDIDATE_COLUMNS.items():
        col = find_first_existing(df, candidates)
        if col is None and key not in {"stationname", "rain_3h"}:
            missing.append((key, candidates))
        elif col is not None:
            detected[key] = col

    if missing:
        msg = "\n".join([f"  - {key}: one of {cands}" for key, cands in missing])
        raise KeyError(
            "Missing required columns in processed station files.\n"
            f"Detected columns: {list(df.columns)}\n"
            f"Required unresolved mappings:\n{msg}"
        )

    return detected


# ----------------------------
# Small helpers
# ----------------------------
def parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    # Great-circle distance in km between two lon/lat points.
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def mark_local_maxima(values: pd.Series) -> pd.Series:
    # Mark local peaks in a time series.
    # A point is kept if it is at least as large as the neighbours
    # and strictly larger than at least one side.
    prev_val = values.shift(1)
    next_val = values.shift(-1)

    left_ok = prev_val.isna() | (values >= prev_val)
    right_ok = next_val.isna() | (values >= next_val)
    strictly_higher_somewhere = (
        prev_val.isna()
        | next_val.isna()
        | (values > prev_val)
        | (values > next_val)
    )

    return left_ok & right_ok & strictly_higher_somewhere


# ----------------------------
# I/O
# ----------------------------
def load_processed_station_data(
    input_dir: Path,
    preferred_stations: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    files = sorted(input_dir.glob("station_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    frames = [pd.read_parquet(fp) for fp in files]
    df = pd.concat(frames, ignore_index=True)

    colmap = detect_columns(df)

    # Standardize key types before doing any event logic.
    df[colmap["station"]] = df[colmap["station"]].astype(str).str.zfill(5)
    df[colmap["time"]] = pd.to_datetime(df[colmap["time"]], utc=True, errors="coerce")

    if df[colmap["time"]].isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    for key in ["lat", "lon", "rain_1h"]:
        df[colmap[key]] = pd.to_numeric(df[colmap[key]], errors="coerce")

    if "rain_3h" in colmap:
        df[colmap["rain_3h"]] = pd.to_numeric(df[colmap["rain_3h"]], errors="coerce")

    df[colmap["valid_1h"]] = df[colmap["valid_1h"]].fillna(False).astype(bool)

    if preferred_stations is not None:
        preferred = {str(s).zfill(5) for s in preferred_stations}
        df = df.loc[df[colmap["station"]].isin(preferred)].copy()
        if df.empty:
            raise ValueError(
                f"No rows remain after filtering to preferred stations: {sorted(preferred)}"
            )

    return df, colmap


def standardize_working_columns(
    df: pd.DataFrame,
    colmap: Dict[str, str],
) -> pd.DataFrame:
    # Rename the columns used by this script to one fixed set of names.
    rename_map = {
        colmap["station"]: "station",
        colmap["time"]: "time",
        colmap["lat"]: "lat",
        colmap["lon"]: "lon",
        colmap["rain_1h"]: "rain_1h",
        colmap["valid_1h"]: "valid_1h",
    }

    if "stationname" in colmap:
        rename_map[colmap["stationname"]] = "stationname"
    if "rain_3h" in colmap:
        rename_map[colmap["rain_3h"]] = "rain_3h"

    out = df.rename(columns=rename_map).copy()
    keep_cols = list(dict.fromkeys(rename_map.values()))
    out = out[keep_cols].copy()

    if "stationname" not in out.columns:
        out["stationname"] = pd.NA
    if "rain_3h" not in out.columns:
        out["rain_3h"] = np.nan

    return out


def filter_to_analysis_block(
    df: pd.DataFrame,
    analysis_months: set[int] | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> pd.DataFrame:
    # Keep only the selected analysis block.
    # If the processed files already contain only the target block,
    # this step just confirms it.
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    if analysis_months is not None:
        mask &= out["time"].dt.month.isin(analysis_months)

    start_ts = parse_utc_timestamp(start_utc)
    end_ts = parse_utc_timestamp(end_utc)

    if start_ts is not None:
        mask &= out["time"] >= start_ts
    if end_ts is not None:
        mask &= out["time"] <= end_ts

    out = out.loc[mask].copy()

    if out.empty:
        raise ValueError("No rows remain after filtering to the selected analysis block.")

    return out


# ----------------------------
# Station seed detection
# ----------------------------
def build_station_seed_candidates(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows where 1 h rainfall is valid.
    # Local maxima are searched only within this valid 1 h series.
    work = df.loc[df["valid_1h"] & df["rain_1h"].notna()].copy()

    if work.empty:
        raise ValueError("No valid 1 h rainfall rows are available in the selected block.")

    # Station location is required later for event merging.
    work = work.loc[work["lat"].notna() & work["lon"].notna()].copy()

    if work.empty:
        raise ValueError("No valid rows with station latitude/longitude remain.")

    work = work.sort_values(["station", "time"]).reset_index(drop=True)

    # One local peak = one candidate station seed.
    work["is_local_maximum"] = (
        work.groupby("station", group_keys=False)["rain_1h"]
        .apply(mark_local_maxima)
        .astype(bool)
    )

    candidates = work.loc[work["is_local_maximum"]].copy()

    if candidates.empty:
        raise ValueError("No station-local 1 h maxima were found.")

    candidates = candidates.rename(
        columns={
            "time": "peak_time",
            "rain_1h": "peak_1h_mm",
            "rain_3h": "peak_3h_mm",
        }
    )

    candidates = candidates[
        [
            "station",
            "stationname",
            "lat",
            "lon",
            "peak_time",
            "peak_1h_mm",
            "peak_3h_mm",
        ]
    ].sort_values(["station", "peak_time"]).reset_index(drop=True)

    return candidates


def select_top_station_seeds(
    candidates: pd.DataFrame,
    top_n: int = TOP_N_STATION_SEEDS,
    min_separation_h: int = MIN_STATION_PEAK_SEPARATION_H,
) -> pd.DataFrame:
    # For each station:
    # - rank local peaks by 1 h rainfall
    # - keep peaks at least 6 h apart
    # - stop after the top N selected peaks
    selected_groups: list[pd.DataFrame] = []

    for station_id, grp in candidates.groupby("station", sort=True):
        grp = grp.sort_values(["peak_1h_mm", "peak_time"], ascending=[False, True]).reset_index(drop=True)

        chosen_rows: list[pd.Series] = []

        for _, row in grp.iterrows():
            peak_time = row["peak_time"]

            too_close = False
            for chosen in chosen_rows:
                dt_h = abs((peak_time - chosen["peak_time"]).total_seconds()) / 3600.0
                if dt_h < min_separation_h:
                    too_close = True
                    break

            if too_close:
                continue

            chosen_rows.append(row)

            if len(chosen_rows) >= top_n:
                break

        if not chosen_rows:
            continue

        chosen = pd.DataFrame(chosen_rows).copy()
        chosen["seed_rank_within_station"] = np.arange(1, len(chosen) + 1)

        # Time of the rolling 1 h peak marks the end of the 1 h support window.
        chosen["seed_window_start"] = chosen["peak_time"] - pd.Timedelta(hours=1)
        chosen["seed_window_end"] = chosen["peak_time"]

        chosen["station_seed_id"] = [
            f"{station_id}_S{rank:02d}"
            for rank in chosen["seed_rank_within_station"]
        ]

        selected_groups.append(chosen)

    if not selected_groups:
        raise ValueError("No station seeds were selected.")

    seeds = pd.concat(selected_groups, ignore_index=True)

    # Keep output in time order after selection.
    seeds = seeds.sort_values(["peak_time", "station"]).reset_index(drop=True)

    return seeds


# ----------------------------
# National event merge
# ----------------------------
def cluster_station_seeds(
    seeds: pd.DataFrame,
    time_tol_h: int = EVENT_TIME_TOL_H,
    space_tol_km: float = EVENT_SPACE_TOL_KM,
) -> pd.DataFrame:
    # Merge station seeds into national event candidates.
    # Two seeds are linked if they are close enough in time and space.
    if seeds.empty:
        raise ValueError("No station seeds are available for clustering.")

    if seeds["lat"].isna().any() or seeds["lon"].isna().any():
        raise ValueError("Some station seeds are missing latitude/longitude.")

    members = seeds.reset_index(drop=True).copy()
    n = len(members)

    parent = np.arange(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            dt_h = abs(
                (members.loc[i, "peak_time"] - members.loc[j, "peak_time"]).total_seconds()
            ) / 3600.0

            if dt_h > time_tol_h:
                continue

            dist_km = haversine_km(
                members.loc[i, "lon"], members.loc[i, "lat"],
                members.loc[j, "lon"], members.loc[j, "lat"],
            )

            if dist_km <= space_tol_km:
                union(i, j)

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)

    members["event_candidate_id"] = [f"EVT{lab:05d}" for lab in labels]

    return members


# ----------------------------
# Event summary
# ----------------------------
def summarise_event_candidates(members: pd.DataFrame) -> pd.DataFrame:
    # Build one row per national event candidate.
    # The event 3 h peak is computed separately from the event 1 h peak.
    if members.empty:
        raise ValueError("No clustered event members are available for summary.")

    peak1_idx = members.groupby("event_candidate_id")["peak_1h_mm"].idxmax()
    peak1_meta = members.loc[
        peak1_idx,
        [
            "event_candidate_id",
            "station",
            "stationname",
            "peak_time",
            "peak_1h_mm",
            "lat",
            "lon",
        ],
    ].rename(
        columns={
            "station": "peak_station",
            "stationname": "peak_stationname",
            "lat": "peak_lat",
            "lon": "peak_lon",
        }
    )

    event_peak_3h = (
        members.groupby("event_candidate_id", as_index=False)["peak_3h_mm"]
        .max()
        .rename(columns={"peak_3h_mm": "event_peak_3h_mm"})
    )

    summary = members.groupby("event_candidate_id", as_index=False).agg(
        start_time=("seed_window_start", "min"),
        end_time=("seed_window_end", "max"),
        n_station_seeds=("station_seed_id", "size"),
        n_stations=("station", "nunique"),
    )

    summary["duration_h"] = (
        summary["end_time"] - summary["start_time"]
    ).dt.total_seconds() / 3600.0

    summary = summary.merge(peak1_meta, on="event_candidate_id", how="left", validate="one_to_one")
    summary = summary.merge(event_peak_3h, on="event_candidate_id", how="left", validate="one_to_one")

    summary["event_definition"] = EVENT_DEFINITION_NAME
    summary = summary.sort_values(["peak_1h_mm", "start_time"], ascending=[False, True]).reset_index(drop=True)

    return summary


# ----------------------------
# Run
# ----------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preferred = PREFERRED_STATIONS if USE_PREFERRED_STATIONS else None

    raw_df, colmap = load_processed_station_data(INPUT_DIR, preferred_stations=preferred)
    df = standardize_working_columns(raw_df, colmap)
    df = filter_to_analysis_block(
        df,
        analysis_months=ANALYSIS_MONTHS,
        start_utc=ANALYSIS_START_UTC,
        end_utc=ANALYSIS_END_UTC,
    )

    print("\nLoaded processed station data.")
    print(f"Rows in selected block: {len(df)}")
    print(f"Stations: {df['station'].nunique()}")
    print(f"Time coverage: {df['time'].min()} to {df['time'].max()}")
    print(f"Event definition: {EVENT_DEFINITION_NAME}")
    print(f"Top station seeds kept per station: {TOP_N_STATION_SEEDS}")
    print(f"Minimum station peak separation: {MIN_STATION_PEAK_SEPARATION_H} h")
    print(f"National merge rule: {EVENT_TIME_TOL_H} h and {EVENT_SPACE_TOL_KM:.1f} km")

    station_peak_candidates = build_station_seed_candidates(df)
    station_seeds = select_top_station_seeds(
        station_peak_candidates,
        top_n=TOP_N_STATION_SEEDS,
        min_separation_h=MIN_STATION_PEAK_SEPARATION_H,
    )
    event_members = cluster_station_seeds(
        station_seeds,
        time_tol_h=EVENT_TIME_TOL_H,
        space_tol_km=EVENT_SPACE_TOL_KM,
    )
    event_candidates = summarise_event_candidates(event_members)

    station_peak_candidates.to_parquet(
        OUTPUT_DIR / "station_peak_candidates_localmax_1h_may_jul.parquet",
        index=False,
    )
    station_seeds.to_parquet(
        OUTPUT_DIR / "station_seeds_top5_localmax_1h_may_jul.parquet",
        index=False,
    )
    event_members.to_parquet(
        OUTPUT_DIR / "event_candidate_members_top5_localmax_1h_may_jul_6h75km.parquet",
        index=False,
    )
    event_candidates.to_parquet(
        OUTPUT_DIR / "event_candidates_top5_localmax_1h_may_jul_6h75km.parquet",
        index=False,
    )

    print("\nEvent building complete.")
    print(f"Station local-max candidates: {len(station_peak_candidates)}")
    print(f"Selected station seeds:       {len(station_seeds)}")
    print(f"National event candidates:    {event_candidates['event_candidate_id'].nunique()}")

    print("\nTop 10 event candidates:")
    cols = [
        "event_candidate_id",
        "start_time",
        "end_time",
        "n_stations",
        "peak_station",
        "peak_1h_mm",
        "event_peak_3h_mm",
    ]
    print(event_candidates[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()