from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from diagnostics import (
    compute_event_diagnostics,
    plot_tcwv_vs_cape,
    plot_transport_vs_shear,
    save_top_table,
)
from io_era5 import (
    load_event_catalogue,
    merge_context_datasets,
    open_era5_dataset,
    subset_time_and_box,
)


# Build event-level ERA5 diagnostics from:
# 1. the rainfall event catalogue
# 2. the saved event-centred ERA5 context
# 3. the full ERA5 background over the selected analysis block


# ----------------------------
# Paths
# ----------------------------
EVENT_CATALOGUE_PATH = Path("data_processed/events/events_catalogue_v1.parquet")
EVENT_CONTEXT_PATH = Path("data_processed/diagnostics/era5_context.nc")

ERA5_SINGLE_LEVEL_DIR = Path("data_raw/era5/single_levels")
ERA5_PRESSURE_LEVEL_DIR = Path("data_raw/era5/pressure_levels")

OUTPUT_DIR = Path("data_processed/diagnostics")
FIG_DIR = Path("report/figures")

DIAGNOSTICS_PATH = OUTPUT_DIR / "event_diagnostics_v1.parquet"
TOP_TABLE_PATH = OUTPUT_DIR / "event_diagnostics_top20_v1.csv"

PLOT1_PATH = FIG_DIR / "event_tcwv_vs_cape.png"
PLOT2_PATH = FIG_DIR / "event_transport_vs_shear.png"


# ----------------------------
# Analysis block and domain
# ----------------------------
# Keep these aligned with the event-building stage.
# As the ERA5 folders already contain only the target 3-month block, these can stay as None.
ANALYSIS_START_UTC = None   # Example: "2012-05-01T00:00:00Z"
ANALYSIS_END_UTC = None     # Example: "2012-07-31T23:00:00Z"

# Netherlands-centred box: keep identical to the ERA5 context step
LAT_MIN = 50.5
LAT_MAX = 55.8
LON_MIN = 2.5
LON_MAX = 7.5


def collect_era5_files(directory: Path) -> list[Path]:
    """
    Collect ERA5 files from one folder.

    This accepts common GRIB and NetCDF suffixes.
    """
    patterns = ("*.grib", "*.grb", "*.grib2", "*.nc", "*.nc4", "*.cdf")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {directory}")

    return files


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load the rainfall event catalogue.
    event_catalogue = load_event_catalogue(EVENT_CATALOGUE_PATH)
    if event_catalogue.empty:
        raise ValueError("Event catalogue is empty.")

    # Load the saved event-centred ERA5 dataset.
    # .load() keeps the rest of the workflow simpler and avoids repeated file access.
    with xr.open_dataset(EVENT_CONTEXT_PATH) as ds_evt:
        event_context = ds_evt.load()

    # Collect ERA5 files used to build the background reference.
    single_level_files = collect_era5_files(ERA5_SINGLE_LEVEL_DIR)
    pressure_level_files = collect_era5_files(ERA5_PRESSURE_LEVEL_DIR)

    print(f"Loaded event catalogue with {len(event_catalogue)} events.")
    print(f"Loaded event context with {event_context.sizes.get('event_candidate_id', 0)} events.")
    print(f"Single-level files:  {len(single_level_files)}")
    print(f"Pressure-level files: {len(pressure_level_files)}")

    # Open the ERA5 source files and build one merged dataset.
    ds_surface = open_era5_dataset(single_level_files)
    ds_pressure = open_era5_dataset(pressure_level_files)

    background_ds = merge_context_datasets(
        ds_surface=ds_surface,
        ds_pressure=ds_pressure,
        include_w700=True,
    )

    # Restrict the background to the same Netherlands-centred box.
    # If ANALYSIS_START_UTC / ANALYSIS_END_UTC are set, also restrict the time period
    # to the intended analysis block.
    background_start = (
        ANALYSIS_START_UTC
        if ANALYSIS_START_UTC is not None
        else pd.Timestamp(background_ds["time"].values.min())
    )
    background_end = (
        ANALYSIS_END_UTC
        if ANALYSIS_END_UTC is not None
        else pd.Timestamp(background_ds["time"].values.max())
    )

    background_ds = subset_time_and_box(
        background_ds,
        start_time_utc=background_start,
        end_time_utc=background_end,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
    )

    diagnostics = compute_event_diagnostics(
        event_catalogue=event_catalogue,
        event_context=event_context,
        background_ds=background_ds,
    )

    diagnostics.to_parquet(DIAGNOSTICS_PATH, index=False)
    top20 = save_top_table(diagnostics, TOP_TABLE_PATH, top_n=20)

    plot_tcwv_vs_cape(diagnostics, PLOT1_PATH, annotate_top_n=5)
    plot_transport_vs_shear(diagnostics, PLOT2_PATH, annotate_top_n=5)

    print("\nEvent diagnostics complete.")
    print(f"Rows: {len(diagnostics)}")
    print(f"Saved -> {DIAGNOSTICS_PATH}")
    print(f"Saved -> {TOP_TABLE_PATH}")
    print(f"Saved -> {PLOT1_PATH}")
    print(f"Saved -> {PLOT2_PATH}")

    print("\nTop 10 diagnostics rows:")
    print(top20.head(10).to_string(index=False))


if __name__ == "__main__":
    main()