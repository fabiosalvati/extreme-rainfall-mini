from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr


# Variables kept from single-level ERA5 files.
SURFACE_KEEP = ("tcwv", "cape", "msl")

# If both humidity measures exist on pressure levels, prefer specific humidity q.
# If q is missing, fall back to relative humidity r.
PRESSURE_HUMIDITY_PREFERENCE = ("q", "r")


def ensure_utc(ts) -> pd.Timestamp:
    """
    Return a timezone-aware UTC timestamp.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def to_utc_naive(ts) -> pd.Timestamp:
    """
    Return a UTC timestamp without timezone info.

    xarray time coordinates are usually easier to handle in this form.
    """
    return ensure_utc(ts).tz_localize(None)


def ts_to_datetime64ns(ts) -> np.datetime64:
    """
    Convert a timestamp into numpy datetime64[ns].
    """
    return np.datetime64(to_utc_naive(ts).to_datetime64())


def load_event_catalogue(path: str | Path) -> pd.DataFrame:
    """
    Load the event catalogue and standardize key time columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing event catalogue: {path}")

    df = pd.read_parquet(path)

    required = ["event_candidate_id", "start_time", "end_time"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Event catalogue is missing required column: {col}")

    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="raise")

    if "peak_time_utc" in df.columns:
        df["peak_time_utc"] = pd.to_datetime(df["peak_time_utc"], utc=True, errors="raise")
    elif "peak_time" in df.columns:
        df["peak_time_utc"] = pd.to_datetime(df["peak_time"], utc=True, errors="raise")
    else:
        raise KeyError("Event catalogue must contain 'peak_time_utc' or 'peak_time'.")

    if "peak_station" in df.columns:
        df["peak_station"] = df["peak_station"].astype(str).str.zfill(5)

    return df.sort_values(["peak_time_utc", "event_candidate_id"]).reset_index(drop=True)


def _clean_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    """
    Remove helper files such as .idx and keep only actual data files.
    """
    cleaned: list[Path] = []

    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".idx":
            continue
        cleaned.append(p)

    if not cleaned:
        raise ValueError("No usable ERA5 files were supplied after excluding .idx files.")

    return cleaned


def _infer_engine(paths: Sequence[str | Path]) -> str:
    """
    Guess whether files should be opened as NetCDF or GRIB.
    """
    paths = _clean_input_paths(paths)
    suffixes = {Path(p).suffix.lower() for p in paths}

    if suffixes.issubset({".nc", ".nc4", ".cdf"}):
        return "netcdf"

    if suffixes.issubset({".grib", ".grb", ".grib2"}):
        return "cfgrib"

    raise ValueError(f"Could not infer a single ERA5 engine from suffixes: {sorted(suffixes)}")


def _normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize the ERA5 time coordinate to naive UTC datetime64[ns].
    """
    if "time" not in ds.coords:
        raise KeyError("ERA5 dataset must contain a 'time' coordinate.")

    t = pd.to_datetime(ds["time"].values, utc=True, errors="raise")
    t = pd.DatetimeIndex(t).tz_convert("UTC").tz_localize(None)
    ds = ds.assign_coords(time=t.to_numpy(dtype="datetime64[ns]"))

    return ds.sortby("time")


def _drop_extra_coords(
    ds: xr.Dataset,
    keep_level: bool = False,
    keep_names: Sequence[str] = (),
) -> xr.Dataset:
    """
    Drop auxiliary coordinates that are not needed.

    This keeps the core spatial/time coordinates and any extra names
    explicitly listed in keep_names.
    """
    keep = {"time", "latitude", "longitude", *keep_names}
    if keep_level:
        keep.add("level")

    drop_names = [c for c in ds.coords if c not in keep and c not in ds.dims]
    if drop_names:
        ds = ds.drop_vars(drop_names)

    return ds


def _standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize common ERA5 coordinate names.

    This handles common variants such as:
    - valid_time -> time
    - pressure_level or isobaricInhPa -> level
    """
    rename_map = {}

    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename_map["valid_time"] = "time"

    if "pressure_level" in ds.coords and "level" not in ds.coords:
        rename_map["pressure_level"] = "level"

    if "isobaricInhPa" in ds.coords and "level" not in ds.coords:
        rename_map["isobaricInhPa"] = "level"

    if rename_map:
        ds = ds.rename(rename_map)

    if "time" not in ds.coords:
        raise KeyError("ERA5 dataset must contain 'time' after standardization.")

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise KeyError("ERA5 dataset must contain 'latitude' and 'longitude' coordinates.")

    ds = _normalize_time_coord(ds)

    # Convert longitude to [-180, 180] if needed.
    lon = ds["longitude"]
    if float(lon.max()) > 180.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=new_lon).sortby("longitude")

    ds = _drop_extra_coords(ds, keep_level=("level" in ds.coords))
    return ds


def open_era5_dataset(
    paths: Iterable[str | Path],
    engine: str | None = None,
) -> xr.Dataset:
    """
    Open one or more ERA5 files and return a standardized xarray dataset.
    """
    paths = _clean_input_paths(paths)

    if engine is None:
        engine = _infer_engine(paths)

    if engine == "netcdf":
        ds = xr.open_mfdataset(
            [str(p) for p in paths],
            combine="by_coords",
            parallel=False,
        )
        return _standardize_dataset(ds)

    if engine == "cfgrib":
        opened = []

        for p in paths:
            ds = xr.open_dataset(str(p), engine="cfgrib")
            ds = _standardize_dataset(ds)
            opened.append(ds)

        if len(opened) == 1:
            return opened[0]

        ds = xr.concat(
            opened,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="override",
        )

        ds = ds.sortby("time")

        # If overlapping files were included by mistake, keep the first copy of each time.
        time_index = pd.Index(ds["time"].values)
        if time_index.has_duplicates:
            keep = ~time_index.duplicated(keep="first")
            ds = ds.isel(time=np.where(keep)[0])

        ds = _standardize_dataset(ds)
        return ds

    raise ValueError(f"Unsupported ERA5 engine: {engine}")


def subset_time_and_box(
    ds: xr.Dataset,
    start_time_utc,
    end_time_utc,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    """
    Extract a time window and lat/lon box from an ERA5 dataset.
    """
    start_time_utc = to_utc_naive(start_time_utc)
    end_time_utc = to_utc_naive(end_time_utc)

    if end_time_utc < start_time_utc:
        raise ValueError("end_time_utc must be >= start_time_utc")

    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min

    ds = ds.sel(time=slice(start_time_utc, end_time_utc))

    # ERA5 latitude is often stored from north to south, so the slice direction matters.
    lat_values = ds["latitude"].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    ds = ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    return ds


def extract_surface_context(ds_surface: xr.Dataset) -> xr.Dataset:
    """
    Keep only the single-level variables used in this project.
    """
    keep = [v for v in SURFACE_KEEP if v in ds_surface.data_vars]
    if not keep:
        raise KeyError(f"No expected single-level variables found. Tried: {list(SURFACE_KEEP)}")

    out = ds_surface[keep]
    out = _drop_extra_coords(out, keep_level=False)
    return out


def _select_level_no_coord(da: xr.DataArray, level_value: int) -> xr.DataArray:
    """
    Extract one pressure level from a variable and drop leftover level metadata.
    """
    out = da.sel(level=level_value)

    if "level" in out.coords:
        out = out.drop_vars("level")

    drop_names = [
        c for c in out.coords
        if c not in {"time", "latitude", "longitude"} and c not in out.dims
    ]
    if drop_names:
        out = out.drop_vars(drop_names)

    return out


def extract_pressure_context(
    ds_pressure: xr.Dataset,
    include_w700: bool = True,
) -> xr.Dataset:
    """
    Extract the pressure-level variables used for event diagnostics.
    """
    if "level" not in ds_pressure.coords:
        raise KeyError("Pressure-level ERA5 dataset must contain a 'level' coordinate.")

    available_levels = set(int(x) for x in np.asarray(ds_pressure["level"].values).tolist())
    out = {}

    if "u" in ds_pressure.data_vars:
        if 850 in available_levels:
            out["u850"] = _select_level_no_coord(ds_pressure["u"], 850)
        if 500 in available_levels:
            out["u500"] = _select_level_no_coord(ds_pressure["u"], 500)

    if "v" in ds_pressure.data_vars:
        if 850 in available_levels:
            out["v850"] = _select_level_no_coord(ds_pressure["v"], 850)
        if 500 in available_levels:
            out["v500"] = _select_level_no_coord(ds_pressure["v"], 500)

    humidity_var = None
    for candidate in PRESSURE_HUMIDITY_PREFERENCE:
        if candidate in ds_pressure.data_vars:
            humidity_var = candidate
            break

    if humidity_var is not None and 850 in available_levels:
        out[f"{humidity_var}850"] = _select_level_no_coord(ds_pressure[humidity_var], 850)

    if include_w700 and "w" in ds_pressure.data_vars and 700 in available_levels:
        out["w700"] = _select_level_no_coord(ds_pressure["w"], 700)

    if not out:
        raise KeyError("No expected pressure-level variables found. Need at least one of u/v/q/r/w.")

    ds = xr.Dataset(out)
    ds = _drop_extra_coords(ds, keep_level=False)
    return ds


def merge_context_datasets(
    ds_surface: xr.Dataset,
    ds_pressure: xr.Dataset,
    include_w700: bool = True,
) -> xr.Dataset:
    """
    Merge the selected single-level and pressure-level variables.
    """
    surface = extract_surface_context(ds_surface)
    pressure = extract_pressure_context(ds_pressure, include_w700=include_w700)

    merged = xr.merge([surface, pressure], compat="override")
    merged = _drop_extra_coords(merged, keep_level=False)
    merged = merged.sortby("time")

    return merged


def _nearest_hour_anchor(ts) -> pd.Timestamp:
    """
    Map a timestamp to the nearest full UTC hour.

    This is used because ERA5 is hourly while event peaks come from 10-minute data.
    """
    ts = ensure_utc(ts)
    return (ts + pd.Timedelta(minutes=30)).floor("h")


def build_event_context(
    merged_context: xr.Dataset,
    event_catalogue: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    hours_before_peak: int = 12,
    hours_after_peak: int = 12,
) -> xr.Dataset:
    """
    Build an event-centered ERA5 dataset.

    For each event:
    - anchor the event on the nearest ERA5 hour to the event peak
    - extract a time window around that anchor
    - subset to the chosen lat/lon box
    - reindex to a complete hourly relative-time axis
    """
    if hours_before_peak < 0 or hours_after_peak < 0:
        raise ValueError("hours_before_peak and hours_after_peak must be >= 0")

    expected_relative_hour = np.arange(-hours_before_peak, hours_after_peak + 1, dtype=int)
    out: list[xr.Dataset] = []

    for row in event_catalogue.itertuples(index=False):
        event_id = row.event_candidate_id
        peak_time = ensure_utc(row.peak_time_utc)

        # Hourly ERA5 cannot match a 10-minute event peak exactly.
        # Use the nearest ERA5 hour as the event anchor.
        anchor_time = _nearest_hour_anchor(peak_time)

        start_time = anchor_time - pd.Timedelta(hours=hours_before_peak)
        end_time = anchor_time + pd.Timedelta(hours=hours_after_peak)

        expected_times = pd.date_range(
            start=start_time,
            end=end_time,
            freq="1h",
            tz="UTC",
        )
        expected_times_naive = expected_times.tz_localize(None).to_numpy(dtype="datetime64[ns]")

        event_ds = subset_time_and_box(
            merged_context,
            start_time_utc=start_time,
            end_time_utc=end_time,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        # Reindex so every event has the same relative-hour axis,
        # even if some ERA5 times are missing.
        event_ds = event_ds.reindex(time=expected_times_naive)

        event_ds = event_ds.assign_coords(relative_hour=("time", expected_relative_hour))
        event_ds = event_ds.swap_dims({"time": "relative_hour"})

        # After switching to relative_hour as the main axis,
        # drop the original absolute time coordinate.
        # Event-level absolute timing is stored separately below.
        if "time" in event_ds.coords:
            event_ds = event_ds.drop_vars("time")

        event_ds = event_ds.expand_dims(event_candidate_id=[event_id])

        event_coord_names = [
            "event_start_time_utc",
            "event_end_time_utc",
            "event_peak_time_utc",
            "event_anchor_time_utc",
        ]

        event_coords = {
            "event_start_time_utc": (
                "event_candidate_id",
                np.array([ts_to_datetime64ns(row.start_time)], dtype="datetime64[ns]"),
            ),
            "event_end_time_utc": (
                "event_candidate_id",
                np.array([ts_to_datetime64ns(row.end_time)], dtype="datetime64[ns]"),
            ),
            "event_peak_time_utc": (
                "event_candidate_id",
                np.array([ts_to_datetime64ns(peak_time)], dtype="datetime64[ns]"),
            ),
            "event_anchor_time_utc": (
                "event_candidate_id",
                np.array([ts_to_datetime64ns(anchor_time)], dtype="datetime64[ns]"),
            ),
        }

        if hasattr(row, "peak_station"):
            event_coords["event_peak_station"] = (
                "event_candidate_id",
                np.array([str(row.peak_station).zfill(5)], dtype="<U5"),
            )
            event_coord_names.append("event_peak_station")

        if hasattr(row, "peak_1h_mm"):
            event_coords["event_peak_1h_mm"] = (
                "event_candidate_id",
                np.array([float(row.peak_1h_mm)], dtype="float64"),
            )
            event_coord_names.append("event_peak_1h_mm")

        if hasattr(row, "n_stations"):
            event_coords["event_n_stations"] = (
                "event_candidate_id",
                np.array([int(row.n_stations)], dtype="int64"),
            )
            event_coord_names.append("event_n_stations")

        event_ds = event_ds.assign_coords(event_coords)

        # Keep the event metadata coordinates.
        event_ds = _drop_extra_coords(
            event_ds,
            keep_level=False,
            keep_names=event_coord_names,
        )

        out.append(event_ds)

    if not out:
        raise ValueError("No event contexts were built.")

    combined = xr.concat(
        out,
        dim="event_candidate_id",
        combine_attrs="override",
        coords="minimal",
        compat="override",
    )

    combined = _drop_extra_coords(
        combined,
        keep_level=False,
        keep_names=[
            "event_start_time_utc",
            "event_end_time_utc",
            "event_peak_time_utc",
            "event_anchor_time_utc",
            "event_peak_station",
            "event_peak_1h_mm",
            "event_n_stations",
        ],
    )

    return combined


def save_event_context(ds: xr.Dataset, output_path: str | Path) -> Path:
    """
    Save the event-centered ERA5 dataset as Zarr or NetCDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".zarr":
        ds.to_zarr(output_path, mode="w")
        return output_path

    if output_path.suffix in {".nc", ".nc4"}:
        ds.to_netcdf(output_path)
        return output_path

    raise ValueError("Output path must end with .zarr, .nc, or .nc4")