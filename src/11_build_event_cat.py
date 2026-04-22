from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Build a clean event catalogue from the event candidates created in 10_build_events.py.
# This step adds:
# - duration
# - local peak time information
# - simple footprint proxies
# - a nearby-station missing-data check
# - a top-events table
# - a few simple plots


# ----------------------------
# Fixed choices for this step
# ----------------------------
EVENT_DIR = Path("data_processed/events")
STATIONS_DIR = Path("data_processed/stations")
FIG_DIR = Path("report/figures")

EVENT_CANDIDATES_FILE = EVENT_DIR / "event_candidates_top5_localmax_1h_may_jul_6h75km.parquet"
EVENT_MEMBERS_FILE = EVENT_DIR / "event_candidate_members_top5_localmax_1h_may_jul_6h75km.parquet"

OUTPUT_CATALOGUE = EVENT_DIR / "events_catalogue_v1.parquet"
OUTPUT_TOP20 = EVENT_DIR / "top20_events_catalogue_v1.csv"

LOCAL_TZ = "Europe/Amsterdam"
NEARBY_RADIUS_KM = 75.0
TOP_N_TABLE = 20
TOP_N_MAPS = 4


# ----------------------------
# Utilities
# ----------------------------
def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Great-circle distance in km.
    Inputs may be scalars or arrays.
    """
    r = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def meteorological_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


# ----------------------------
# I/O
# ----------------------------
def load_event_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EVENT_CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {EVENT_CANDIDATES_FILE}")
    if not EVENT_MEMBERS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {EVENT_MEMBERS_FILE}")

    event_candidates = pd.read_parquet(EVENT_CANDIDATES_FILE)
    event_members = pd.read_parquet(EVENT_MEMBERS_FILE)

    require_columns(
        event_candidates,
        [
            "event_candidate_id",
            "start_time",
            "end_time",
            "n_station_seeds",
            "n_stations",
            "peak_station",
            "peak_time",
            "peak_1h_mm",
            "event_peak_3h_mm",
            "peak_lat",
            "peak_lon",
            "event_definition",
        ],
        "event_candidates",
    )

    require_columns(
        event_members,
        [
            "event_candidate_id",
            "station_seed_id",
            "station",
            "seed_window_start",
            "seed_window_end",
            "peak_time",
            "peak_1h_mm",
            "peak_3h_mm",
            "lat",
            "lon",
        ],
        "event_members",
    )

    for col in ["start_time", "end_time", "peak_time"]:
        event_candidates[col] = pd.to_datetime(event_candidates[col], utc=True, errors="raise")

    for col in ["seed_window_start", "seed_window_end", "peak_time"]:
        event_members[col] = pd.to_datetime(event_members[col], utc=True, errors="raise")

    event_candidates["peak_station"] = event_candidates["peak_station"].astype(str).str.zfill(5)
    event_members["station"] = event_members["station"].astype(str).str.zfill(5)

    return event_candidates, event_members


def load_station_inventory_and_validity(
    stations_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load:
    1. one-row-per-station inventory with lat/lon/name
    2. time-varying 1 h validity flags for missing-data checks
    """
    files = sorted(stations_dir.glob("station_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No station parquet files found in {stations_dir}")

    inventory_rows: list[pd.DataFrame] = []
    validity_rows: list[pd.DataFrame] = []

    for fp in files:
        df = pd.read_parquet(fp)

        require_columns(
            df,
            ["station", "timestamp_end_utc", "rolling_1h_valid", "lat", "lon"],
            f"station file {fp.name}",
        )

        df["station"] = df["station"].astype(str).str.zfill(5)
        df["timestamp_end_utc"] = pd.to_datetime(df["timestamp_end_utc"], utc=True, errors="raise")

        inv_cols = ["station", "lat", "lon"]
        if "stationname" in df.columns:
            inv_cols.append("stationname")

        inv = df[inv_cols].drop_duplicates(subset=["station"]).copy()
        if len(inv) != 1:
            raise ValueError(f"{fp.name} does not contain a unique station inventory row.")
        inventory_rows.append(inv)

        valid = df[["station", "timestamp_end_utc", "rolling_1h_valid"]].copy()
        valid["rolling_1h_valid"] = valid["rolling_1h_valid"].fillna(False).astype(bool)
        validity_rows.append(valid)

    station_inventory = pd.concat(inventory_rows, ignore_index=True)
    station_inventory = station_inventory.sort_values("station").reset_index(drop=True)

    station_validity = pd.concat(validity_rows, ignore_index=True)
    station_validity = station_validity.sort_values(["station", "timestamp_end_utc"]).reset_index(drop=True)

    return station_inventory, station_validity


# ----------------------------
# Catalogue metrics
# ----------------------------
def max_pairwise_distance_km(lons: np.ndarray, lats: np.ndarray) -> float:
    # Simple footprint proxy:
    # largest distance between any two participating stations.
    n = len(lons)
    if n <= 1:
        return 0.0

    max_dist = 0.0
    for i in range(n):
        dist = haversine_km(lons[i], lats[i], lons[i + 1 :], lats[i + 1 :])
        if dist.size:
            local_max = float(np.nanmax(dist))
            if local_max > max_dist:
                max_dist = local_max

    return max_dist


def farthest_from_peak_km(
    peak_lon: float,
    peak_lat: float,
    lons: np.ndarray,
    lats: np.ndarray,
) -> float:
    # Secondary footprint proxy:
    # farthest participating station from the event peak location.
    if len(lons) == 0:
        return np.nan

    dist = haversine_km(peak_lon, peak_lat, lons, lats)
    return float(np.nanmax(dist))


def compute_nearby_missing_fraction(
    event_row: pd.Series,
    station_inventory: pd.DataFrame,
    station_validity: pd.DataFrame,
    nearby_radius_km: float = NEARBY_RADIUS_KM,
) -> tuple[int, int, float]:
    """
    Nearby stations are defined relative to the event peak location.

    A nearby station is counted as missing if it has no valid 1 h window
    anywhere during the event time window.
    """
    peak_lon = float(event_row["peak_lon"])
    peak_lat = float(event_row["peak_lat"])
    start_time = pd.Timestamp(event_row["start_time"])
    end_time = pd.Timestamp(event_row["end_time"])

    dists = haversine_km(
        peak_lon,
        peak_lat,
        station_inventory["lon"].to_numpy(),
        station_inventory["lat"].to_numpy(),
    )

    nearby_mask = dists <= nearby_radius_km
    nearby = station_inventory.loc[nearby_mask, ["station"]].copy()

    n_nearby = int(len(nearby))
    if n_nearby == 0:
        return 0, 0, np.nan

    valid_window = station_validity.loc[
        (station_validity["timestamp_end_utc"] >= start_time)
        & (station_validity["timestamp_end_utc"] <= end_time)
        & (station_validity["station"].isin(nearby["station"]))
    ].copy()

    if valid_window.empty:
        return n_nearby, n_nearby, 1.0

    any_valid = (
        valid_window.groupby("station", as_index=False)["rolling_1h_valid"]
        .any()
        .rename(columns={"rolling_1h_valid": "any_valid_1h_in_event_window"})
    )

    nearby = nearby.merge(any_valid, on="station", how="left")
    nearby["any_valid_1h_in_event_window"] = nearby["any_valid_1h_in_event_window"].fillna(False)

    n_missing = int((~nearby["any_valid_1h_in_event_window"]).sum())
    frac_missing = float(n_missing / n_nearby)

    return n_nearby, n_missing, frac_missing


def build_event_catalogue(
    event_candidates: pd.DataFrame,
    event_members: pd.DataFrame,
    station_inventory: pd.DataFrame,
    station_validity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, event_row in event_candidates.iterrows():
        event_id = event_row["event_candidate_id"]

        members = event_members.loc[event_members["event_candidate_id"] == event_id].copy()
        if members.empty:
            raise ValueError(f"No members found for event {event_id}")

        # One point per station for footprint calculations.
        station_points = (
            members[["station", "lat", "lon"]]
            .drop_duplicates(subset=["station"])
            .reset_index(drop=True)
        )

        member_lons = station_points["lon"].to_numpy(dtype=float)
        member_lats = station_points["lat"].to_numpy(dtype=float)

        footprint_max_pairwise_km = max_pairwise_distance_km(member_lons, member_lats)
        footprint_peak_to_farthest_km = farthest_from_peak_km(
            float(event_row["peak_lon"]),
            float(event_row["peak_lat"]),
            member_lons,
            member_lats,
        )

        peak_time_utc = pd.Timestamp(event_row["peak_time"])
        peak_time_local = peak_time_utc.tz_convert(LOCAL_TZ)

        peak_local_hour = peak_time_local.hour + peak_time_local.minute / 60.0
        peak_local_dayofyear = int(peak_time_local.dayofyear)
        peak_local_month = int(peak_time_local.month)
        peak_local_season = meteorological_season(peak_local_month)

        n_nearby, n_missing_nearby, frac_missing_nearby = compute_nearby_missing_fraction(
            event_row=event_row,
            station_inventory=station_inventory,
            station_validity=station_validity,
            nearby_radius_km=NEARBY_RADIUS_KM,
        )

        row = {
            "event_candidate_id": event_id,
            "start_time": pd.Timestamp(event_row["start_time"]),
            "end_time": pd.Timestamp(event_row["end_time"]),
            "duration_h": float(
                (pd.Timestamp(event_row["end_time"]) - pd.Timestamp(event_row["start_time"])).total_seconds()
                / 3600.0
            ),
            "n_station_seeds": int(event_row["n_station_seeds"]),
            "n_stations": int(event_row["n_stations"]),
            "peak_station": str(event_row["peak_station"]).zfill(5),
            "peak_time_utc": peak_time_utc,
            "peak_time_local": peak_time_local,
            "peak_local_hour": float(peak_local_hour),
            "peak_local_dayofyear": peak_local_dayofyear,
            "peak_local_season": peak_local_season,
            "peak_1h_mm": float(event_row["peak_1h_mm"]),
            "event_peak_3h_mm": (
                float(event_row["event_peak_3h_mm"])
                if pd.notna(event_row["event_peak_3h_mm"])
                else np.nan
            ),
            "peak_lat": float(event_row["peak_lat"]),
            "peak_lon": float(event_row["peak_lon"]),
            "footprint_max_pairwise_km": float(footprint_max_pairwise_km),
            "footprint_peak_to_farthest_km": float(footprint_peak_to_farthest_km),
            "n_nearby_stations_75km": int(n_nearby),
            "n_nearby_missing_stations_75km": int(n_missing_nearby),
            "frac_nearby_missing_stations_75km": (
                float(frac_missing_nearby) if pd.notna(frac_missing_nearby) else np.nan
            ),
            "event_definition": event_row["event_definition"],
        }

        if "peak_stationname" in event_row.index:
            row["peak_stationname"] = event_row["peak_stationname"]
        else:
            row["peak_stationname"] = pd.NA

        rows.append(row)

    catalogue = pd.DataFrame(rows)

    catalogue = catalogue.sort_values(
        ["peak_1h_mm", "n_stations", "start_time"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return catalogue


# ----------------------------
# Outputs
# ----------------------------
def save_top20_table(
    catalogue: pd.DataFrame,
    output_csv: Path,
    top_n: int = TOP_N_TABLE,
) -> pd.DataFrame:
    cols = [
        "event_candidate_id",
        "start_time",
        "end_time",
        "duration_h",
        "n_stations",
        "peak_station",
        "peak_1h_mm",
        "event_peak_3h_mm",
        "footprint_max_pairwise_km",
        "frac_nearby_missing_stations_75km",
    ]

    top20 = catalogue[cols].head(top_n).copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    top20.to_csv(output_csv, index=False)

    return top20


# ----------------------------
# Plotting
# ----------------------------
def plot_duration_histogram(catalogue: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(catalogue["duration_h"].dropna(), bins=15)
    plt.xlabel("Event duration (hours)")
    plt.ylabel("Count")
    plt.title("Event duration histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_footprint_histogram(catalogue: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(catalogue["footprint_max_pairwise_km"].dropna(), bins=15)
    plt.xlabel("Footprint proxy: max pairwise station distance (km)")
    plt.ylabel("Count")
    plt.title("Event footprint histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_event_maps(
    catalogue: pd.DataFrame,
    event_members: pd.DataFrame,
    station_inventory: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_N_MAPS,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_events = catalogue.head(top_n)["event_candidate_id"].tolist()
    n = len(top_events)
    if n == 0:
        return

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 5 * nrows),
        squeeze=False,
    )

    all_lon = station_inventory["lon"].to_numpy()
    all_lat = station_inventory["lat"].to_numpy()

    lon_pad = 0.4
    lat_pad = 0.3

    for ax, event_id in zip(axes.ravel(), top_events):
        event_row = catalogue.loc[catalogue["event_candidate_id"] == event_id].iloc[0]
        members = event_members.loc[event_members["event_candidate_id"] == event_id].copy()

        member_points = (
            members[["station", "lat", "lon", "peak_1h_mm"]]
            .sort_values("peak_1h_mm", ascending=False)
            .drop_duplicates(subset=["station"])
            .reset_index(drop=True)
        )

        ax.scatter(all_lon, all_lat, alpha=0.35, s=25, label="All stations")

        sc = ax.scatter(
            member_points["lon"],
            member_points["lat"],
            s=70,
            c=member_points["peak_1h_mm"],
            label="Event stations",
        )

        ax.scatter(
            [event_row["peak_lon"]],
            [event_row["peak_lat"]],
            s=120,
            marker="x",
            label="Peak station",
        )

        ax.set_xlim(all_lon.min() - lon_pad, all_lon.max() + lon_pad)
        ax.set_ylim(all_lat.min() - lat_pad, all_lat.max() + lat_pad)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"{event_id}\n"
            f"peak_1h={event_row['peak_1h_mm']:.2f} mm, "
            f"stations={int(event_row['n_stations'])}, "
            f"duration={event_row['duration_h']:.1f} h"
        )

        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Peak 1 h rainfall (mm)")
        ax.legend(loc="best")

    # Hide unused panels if top_n is odd.
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ----------------------------
# Run
# ----------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)

    event_candidates, event_members = load_event_outputs()
    station_inventory, station_validity = load_station_inventory_and_validity(STATIONS_DIR)

    catalogue = build_event_catalogue(
        event_candidates=event_candidates,
        event_members=event_members,
        station_inventory=station_inventory,
        station_validity=station_validity,
    )

    catalogue.to_parquet(OUTPUT_CATALOGUE, index=False)

    top20 = save_top20_table(catalogue, OUTPUT_TOP20, top_n=TOP_N_TABLE)

    plot_duration_histogram(catalogue, FIG_DIR / "event_duration_histogram.png")
    plot_footprint_histogram(catalogue, FIG_DIR / "event_footprint_histogram.png")
    plot_event_maps(
        catalogue,
        event_members,
        station_inventory,
        FIG_DIR / "top_event_maps.png",
        top_n=TOP_N_MAPS,
    )

    print("\nCatalogue building complete.")
    print(f"Catalogue rows: {len(catalogue)}")
    print(f"Saved catalogue -> {OUTPUT_CATALOGUE}")
    print(f"Saved top-{TOP_N_TABLE} table -> {OUTPUT_TOP20}")
    print(f"Saved figure -> {FIG_DIR / 'event_duration_histogram.png'}")
    print(f"Saved figure -> {FIG_DIR / 'event_footprint_histogram.png'}")
    print(f"Saved figure -> {FIG_DIR / 'top_event_maps.png'}")

    print("\nTop 10 catalogue rows:")
    print(top20.head(10).to_string(index=False))


if __name__ == "__main__":
    main()