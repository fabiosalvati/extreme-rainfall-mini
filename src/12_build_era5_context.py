from __future__ import annotations

from pathlib import Path

from io_era5 import (
    build_event_context,
    load_event_catalogue,
    merge_context_datasets,
    open_era5_dataset,
    save_event_context,
)


# Build event-centred ERA5 context for the rainfall catalogue.
# This script:
# 1. loads the event catalogue
# 2. opens ERA5 single-level and pressure-level files
# 3. keeps only the variables needed for this project
# 4. builds one ERA5 time window around each event
# 5. saves the final dataset


# ----------------------------
# Fixed choices for this step
# ----------------------------
EVENT_CATALOGUE_PATH = Path("data_processed/events/events_catalogue_v1.parquet")

# Adjust these folders to match your local ERA5 layout.
ERA5_SINGLE_LEVEL_DIR = Path("data_raw/era5/single_levels")
ERA5_PRESSURE_LEVEL_DIR = Path("data_raw/era5/pressure_levels")

OUTPUT_PATH = Path("data_processed/diagnostics/era5_context.nc")

# Netherlands-centred box
LAT_MIN = 50.5
LAT_MAX = 55.8
LON_MIN = 2.5
LON_MAX = 7.5

# Fixed event-centred ERA5 window
HOURS_BEFORE_PEAK = 12
HOURS_AFTER_PEAK = 12

# Optional 700 hPa vertical velocity
INCLUDE_W700 = True


def collect_era5_files(directory: Path) -> list[Path]:
    """
    Collect ERA5 files from one folder.

    This allows common GRIB and NetCDF suffixes instead of only *.grib.
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
    # Load the event catalogue built from the rainfall analysis.
    event_catalogue = load_event_catalogue(EVENT_CATALOGUE_PATH)

    if event_catalogue.empty:
        raise ValueError("Event catalogue is empty.")

    # Collect ERA5 files for single-level and pressure-level variables.
    single_level_files = collect_era5_files(ERA5_SINGLE_LEVEL_DIR)
    pressure_level_files = collect_era5_files(ERA5_PRESSURE_LEVEL_DIR)

    print(f"Loaded event catalogue with {len(event_catalogue)} events.")
    print(f"Single-level files:  {len(single_level_files)}")
    print(f"Pressure-level files: {len(pressure_level_files)}")

    # Open the two ERA5 datasets.
    ds_surface = open_era5_dataset(single_level_files)
    ds_pressure = open_era5_dataset(pressure_level_files)

    # Merge the selected ERA5 variables into one common dataset.
    merged_context = merge_context_datasets(
        ds_surface=ds_surface,
        ds_pressure=ds_pressure,
        include_w700=INCLUDE_W700,
    )

    # Build one event-centred ERA5 window for each rainfall event.
    event_context = build_event_context(
        merged_context=merged_context,
        event_catalogue=event_catalogue,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        hours_before_peak=HOURS_BEFORE_PEAK,
        hours_after_peak=HOURS_AFTER_PEAK,
    )

    saved = save_event_context(event_context, OUTPUT_PATH)

    print("\nERA5 context build complete.")
    print(f"Events: {event_context.sizes.get('event_candidate_id', 0)}")
    print(f"Relative hours: {event_context.sizes.get('relative_hour', 0)}")
    print(f"Latitude points: {event_context.sizes.get('latitude', 0)}")
    print(f"Longitude points: {event_context.sizes.get('longitude', 0)}")
    print(f"Variables: {list(event_context.data_vars)}")
    print(f"Saved -> {saved}")


if __name__ == "__main__":
    main()