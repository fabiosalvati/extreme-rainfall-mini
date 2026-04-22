from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
import xarray as xr


# KNMI Open Data endpoints
ANONYMOUS_OPEN_DATA_DOC_URL = "https://developer.dataplatform.knmi.nl/open-data-api"
OPEN_DATA_BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1"

# Default dataset used in this project
DEFAULT_DATASET = "10-minute-in-situ-meteorological-observations"
DEFAULT_VERSION = "1"

# KNMI filenames end with a UTC timestamp like YYYYMMDDHHMM.nc
FILENAME_PATTERN = re.compile(r"(?P<yyyymmddhhmm>\d{12})\.nc$")

# Possible variable names that may contain precipitation information
PRECIP_CANDIDATES = ("rg", "R1H", "R6H", "R12H", "R24H", "pg", "dr", "pr")

# Possible names for the station dimension
STATION_DIM_CANDIDATES = ("station", "stations", "location", "locations", "point", "points")

# Possible names for station identifiers and metadata fields
STATION_ID_CANDIDATES = (
    "station",
    "station_id",
    "stationId",
    "station_name",
    "name",
    "wigos_station_identifier",
    "wigos_id",
    "wsi",
)
LAT_CANDIDATES = ("lat", "latitude", "station_latitude")
LON_CANDIDATES = ("lon", "longitude", "station_longitude")
NAME_CANDIDATES = ("stationname", "station_name", "name", "station")
HEIGHT_CANDIDATES = ("height", "station_height")
WIGOS_CANDIDATES = ("wigos_station_identifier", "wigos_id", "wsi")
TIME_CANDIDATES = ("time", "valid_time", "datetime", "observation_time")
MISSING_ATTR_CANDIDATES = ("_FillValue", "missing_value", "fill_value")


@dataclass(frozen=True)
class KNMIInterval:
    """
    One 10-minute observation window.

    start and end are both UTC timestamps.
    """
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class PrecipitationSpec:
    """
    Basic description of the precipitation variable found in a file.

    conversion_to_10min_mm is written as plain text on purpose.
    It records the assumption used later in the pipeline.
    """
    variable: str
    interpretation: str
    units: str | None
    conversion_to_10min_mm: str


def parse_knmi_10min_filename(filename: str | Path) -> pd.Timestamp:
    """
    Read the UTC timestamp from a KNMI 10-minute filename.

    Important:
    the filename timestamp marks the END of the 10-minute interval.

    Example
    -------
    ...202301010900.nc means the interval 08:50-09:00 UTC,
    not a point measurement exactly at 09:00.
    """
    name = Path(filename).name
    match = FILENAME_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Could not parse KNMI timestamp from filename: {name}")
    return pd.to_datetime(match.group("yyyymmddhhmm"), format="%Y%m%d%H%M", utc=True)


def interval_from_filename(filename: str | Path) -> KNMIInterval:
    """
    Build the full 10-minute interval from the filename timestamp.
    """
    end = parse_knmi_10min_filename(filename)
    return KNMIInterval(start=end - pd.Timedelta(minutes=10), end=end)


def list_knmi_files(directory: str | Path, pattern: str = "*.nc") -> list[Path]:
    """
    Return all matching KNMI NetCDF files in time-sorted order.
    """
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {directory!s}")
    return files


def open_knmi_file(path: str | Path, decode_times: bool = True) -> xr.Dataset:
    """
    Open one KNMI NetCDF file with xarray.
    """
    return xr.open_dataset(path, engine="netcdf4", decode_times=decode_times)


def file_inventory_dataframe(filepaths: Sequence[str | Path]) -> pd.DataFrame:
    """
    Build a simple file inventory table.

    This is useful for checking whether files are continuous in time
    or whether some 10-minute files are missing.
    """
    rows: list[dict[str, object]] = []

    for p in filepaths:
        interval = interval_from_filename(p)
        rows.append(
            {
                "path": str(p),
                "filename": Path(p).name,
                "interval_start_utc": interval.start,
                "interval_end_utc": interval.end,
            }
        )

    df = pd.DataFrame(rows).sort_values("interval_end_utc").reset_index(drop=True)

    # If files are complete, the difference between consecutive end times
    # should be 10 minutes.
    df["gap_from_previous_minutes"] = (
        df["interval_end_utc"].diff() / pd.Timedelta(minutes=1)
    ).astype("float64")

    return df


def dataset_variable_summary(ds: xr.Dataset) -> pd.DataFrame:
    """
    Summarize all variables and coordinates in a dataset.

    This is mainly a structure-inspection tool:
    what variables exist, their dimensions, shape, type, units,
    and whether missing-value attributes are defined.
    """
    rows: list[dict[str, object]] = []

    for name, da in ds.variables.items():
        attrs = dict(da.attrs)
        row = {
            "name": name,
            "kind": "coordinate" if name in ds.coords else "data_var",
            "dims": tuple(da.dims),
            "shape": tuple(int(s) for s in da.shape),
            "dtype": str(da.dtype),
            "units": attrs.get("units"),
            "long_name": attrs.get("long_name") or attrs.get("standard_name"),
        }

        for attr_name in MISSING_ATTR_CANDIDATES:
            row[attr_name] = attrs.get(attr_name)

        rows.append(row)

    return pd.DataFrame(rows)


def dataset_global_attrs(ds: xr.Dataset) -> pd.DataFrame:
    """
    Return global dataset attributes as a two-column table.
    """
    return pd.DataFrame(
        {"attribute": list(ds.attrs.keys()), "value": list(ds.attrs.values())}
    )


def infer_station_dim(ds: xr.Dataset) -> str:
    """
    Guess which dimension represents stations.

    Search order:
    1. explicit dimension names like 'station'
    2. 1D variables that look like station IDs
    3. largest dimension that does not look like time

    This is a convenience function.
    If KNMI file structure changes, this is one of the first places to check.
    """
    # First pass: obvious station-like dimension names.
    for dim in ds.dims:
        if dim in STATION_DIM_CANDIDATES:
            return dim

    # Second pass: look for a 1D variable that seems to identify stations.
    for candidate in STATION_ID_CANDIDATES:
        if candidate in ds.variables and ds[candidate].ndim == 1:
            return ds[candidate].dims[0]

    # Third pass: take the largest non-time dimension as a fallback.
    time_dims = {d for c in TIME_CANDIDATES if c in ds.variables for d in ds[c].dims}
    non_time_dims = [d for d in ds.dims if d not in time_dims]

    if len(non_time_dims) == 1:
        return non_time_dims[0]
    if non_time_dims:
        return max(non_time_dims, key=lambda d: ds.dims[d])

    raise ValueError("Could not infer station dimension from dataset")


def _pick_station_field(ds: xr.Dataset, station_dim: str, candidates: Sequence[str]) -> str | None:
    """
    Find the first variable or coordinate that:
    - matches one of the candidate names
    - depends on the station dimension
    """
    for name in candidates:
        if name in ds.variables and station_dim in ds[name].dims:
            return name
        if name in ds.coords and station_dim in ds[name].dims:
            return name
    return None


def station_metadata_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Extract station metadata into a pandas table.

    The function tries to collect useful station fields such as:
    ID, latitude, longitude, height, station name, and WIGOS ID.
    """
    station_dim = infer_station_dim(ds)
    n = int(ds.sizes[station_dim])
    index = np.arange(n)

    # _row is the dataset row index for each station.
    # It is useful later when selecting a station.
    meta = pd.DataFrame({"_row": index})

    station_field = _pick_station_field(ds, station_dim, STATION_ID_CANDIDATES)
    if station_field is not None:
        meta[station_field] = _as_object_array(ds[station_field].values)
    else:
        meta[station_dim] = index

    # Add common metadata fields if they exist.
    for name in LAT_CANDIDATES + LON_CANDIDATES + HEIGHT_CANDIDATES + NAME_CANDIDATES + WIGOS_CANDIDATES:
        if name in meta.columns:
            continue
        field = _pick_station_field(ds, station_dim, (name,))
        if field is not None:
            meta[field] = _as_object_array(ds[field].values)

    # Some files may include active date ranges for each station.
    for extra in ("start_date", "end_date", "station_start_date", "station_end_date"):
        field = _pick_station_field(ds, station_dim, (extra,))
        if field is not None:
            meta[field] = _as_object_array(ds[field].values)

    return meta


def infer_precipitation_spec(ds: xr.Dataset) -> PrecipitationSpec:
    """
    Guess which dataset variable should be used for precipitation.

    For this project, 'rg' and 'pg' are the most useful cases because
    they can be converted into 10-minute rainfall amount if the unit is mm/h.

    Important:
    the conversion rule is an assumption based on the variable meaning.
    It should be checked against the KNMI metadata once and then frozen.
    """
    for name in PRECIP_CANDIDATES:
        if name not in ds.variables:
            continue

        units = ds[name].attrs.get("units")
        long_name = str(ds[name].attrs.get("long_name") or "")

        if name == "rg":
            return PrecipitationSpec(
                variable="rg",
                interpretation="10-minute mean rain-gauge precipitation intensity",
                units=units,
                conversion_to_10min_mm="amount_mm = rg_mm_per_h / 6",
            )

        if name == "pg":
            return PrecipitationSpec(
                variable="pg",
                interpretation="10-minute mean present-weather-sensor precipitation intensity",
                units=units,
                conversion_to_10min_mm="amount_mm = pg_mm_per_h / 6",
            )

        if name == "R1H":
            return PrecipitationSpec(
                variable="R1H",
                interpretation="past-hour rainfall accumulation",
                units=units,
                conversion_to_10min_mm="not a native 10-minute amount; use only as a cross-check",
            )

        return PrecipitationSpec(
            variable=name,
            interpretation=long_name or "candidate precipitation-related variable",
            units=units,
            conversion_to_10min_mm="inspect file metadata before using",
        )

    raise KeyError("No known precipitation-related variable found in dataset")


def get_time_from_dataset_or_filename(ds: xr.Dataset, filename: str | Path) -> pd.Timestamp:
    """
    Get the main timestamp for a file.

    Prefer the timestamp stored inside the dataset.
    If no usable time field is present, fall back to the filename timestamp.
    """
    for name in TIME_CANDIDATES:
        if name not in ds.variables and name not in ds.coords:
            continue

        da = ds[name]
        if da.size == 0:
            continue

        value = pd.to_datetime(np.ravel(da.values)[0], utc=True)
        return value

    return parse_knmi_10min_filename(filename)


def extract_station_series(
    filepaths: Sequence[str | Path],
    station_selector: str | int | None = None,
    precip_var: str | None = None,
) -> pd.DataFrame:
    """
    Extract one station time series across many KNMI files.

    Parameters
    ----------
    filepaths
        List of 10-minute NetCDF files.
    station_selector
        Either:
        - integer row index
        - string matching one of the station metadata fields
        - None, meaning use the first station row
    precip_var
        Name of the precipitation variable to read.
        If None, the function tries to infer it.

    Returns
    -------
    DataFrame with:
    - interval start/end
    - raw precipitation value
    - simple 10-minute amount estimate when possible

    Important:
    for 'rg' and 'pg', this function assumes the raw values are mean
    intensity over the 10-minute interval in mm/h, so 10-minute amount = value / 6.
    """
    if not filepaths:
        raise ValueError("No filepaths supplied")

    records: list[dict[str, object]] = []
    first_meta: pd.DataFrame | None = None
    station_dim: str | None = None
    station_lookup_columns: list[str] = []

    for path in filepaths:
        with open_knmi_file(path) as ds:
            if station_dim is None:
                station_dim = infer_station_dim(ds)

            if precip_var is None:
                precip_var = infer_precipitation_spec(ds).variable

            if first_meta is None:
                first_meta = station_metadata_dataframe(ds)
                station_lookup_columns = [c for c in first_meta.columns if c != "_row"]

            # Skip files that do not contain the requested precipitation variable.
            if precip_var not in ds:
                continue

            da = ds[precip_var]
            if station_dim not in da.dims:
                raise ValueError(
                    f"Variable {precip_var!r} does not use inferred station dimension {station_dim!r}"
                )

            meta = station_metadata_dataframe(ds)
            station_row = _resolve_station_row(meta, station_selector, station_lookup_columns)

            # If no selector is given, default to the first station row.
            if station_row is None:
                station_row = 0

            selector = {station_dim: station_row}
            value = da.isel(**selector).values

            # Convert array-like output into a scalar if needed.
            if np.ndim(value) > 0:
                value = np.ravel(value)[0]

            timestamp = get_time_from_dataset_or_filename(ds, path)
            interval = interval_from_filename(path)

            record = {
                "timestamp_end_utc": timestamp,
                "interval_start_utc": interval.start,
                "interval_end_utc": interval.end,
                "precip_variable": precip_var,
                "raw_value": _decode_scalar(value),
            }

            # Keep station identification fields in the output for traceability.
            for col in station_lookup_columns:
                if col in meta.columns:
                    record[col] = _decode_scalar(meta.iloc[station_row][col])

            records.append(record)

    if not records:
        raise ValueError("No station records extracted")

    out = pd.DataFrame(records).sort_values("timestamp_end_utc").reset_index(drop=True)
    out["raw_value"] = pd.to_numeric(out["raw_value"], errors="coerce")

    # Convert intensity to 10-minute amount only when the variable meaning is clear.
    if precip_var in {"rg", "pg"}:
        out["precip_10min_amount_mm"] = out["raw_value"] / 6.0
    elif precip_var == "R1H":
        out["precip_10min_amount_mm"] = np.nan
    else:
        out["precip_10min_amount_mm"] = np.nan

    return out


def missing_value_summary(ds: xr.Dataset) -> pd.DataFrame:
    """
    Summarize missing-value metadata and NaN counts for each data variable.

    This checks only NaN values directly visible after loading.
    It does not prove that all KNMI missing-value conventions were decoded correctly.
    """
    rows: list[dict[str, object]] = []

    for name, da in ds.data_vars.items():
        attrs = da.attrs
        row = {"variable": name}

        for attr in MISSING_ATTR_CANDIDATES:
            row[attr] = attrs.get(attr)

        if np.issubdtype(da.dtype, np.number):
            row["n_nan"] = int(np.isnan(np.asarray(da.values, dtype="float64")).sum())
        else:
            row["n_nan"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def list_remote_files(
    api_key: str,
    dataset_name: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
    max_keys: int = 10,
    begin: str | None = None,
) -> dict:
    """
    List available files from the KNMI Open Data API.
    """
    url = f"{OPEN_DATA_BASE_URL}/datasets/{dataset_name}/versions/{version}/files"
    params: dict[str, object] = {"maxKeys": max_keys}
    if begin is not None:
        params["begin"] = begin

    response = requests.get(
        url,
        headers={"Authorization": api_key},
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_download_url(
    api_key: str,
    filename: str,
    dataset_name: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
) -> str:
    """
    Request a temporary download URL for one KNMI file.
    """
    url = (
        f"{OPEN_DATA_BASE_URL}/datasets/{dataset_name}/versions/{version}"
        f"/files/{filename}/url"
    )

    response = requests.get(url, headers={"Authorization": api_key}, timeout=60)
    response.raise_for_status()

    payload = response.json()
    if "temporaryDownloadUrl" not in payload:
        raise KeyError(f"temporaryDownloadUrl missing from response: {payload}")

    return str(payload["temporaryDownloadUrl"])


def download_file(
    api_key: str,
    filename: str,
    destination: str | Path,
    dataset_name: str = DEFAULT_DATASET,
    version: str = DEFAULT_VERSION,
) -> Path:
    """
    Download one KNMI file and save it locally.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = get_download_url(api_key, filename, dataset_name=dataset_name, version=version)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    destination.write_bytes(response.content)
    return destination


def _resolve_station_row(
    metadata: pd.DataFrame,
    selector: str | int | None,
    lookup_columns: Sequence[str],
) -> int | None:
    """
    Convert a station selector into a row index.

    If selector is:
    - None: return None
    - int: use it directly as row index
    - str: search matching text in known station metadata columns
    """
    if selector is None:
        return None

    if isinstance(selector, int):
        return int(selector)

    text = str(selector)
    for col in lookup_columns:
        if col not in metadata.columns:
            continue

        values = metadata[col].astype(str)
        matches = np.where(values == text)[0]
        if len(matches) > 0:
            return int(matches[0])

    return None


def _decode_scalar(value: object) -> object:
    """
    Convert common NetCDF/xarray scalar types into plain Python values.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_object_array(values: np.ndarray) -> np.ndarray:
    """
    Convert arrays with bytes, unicode, or mixed objects into a cleaner object array.
    """
    flat = np.asarray(values)

    if flat.dtype.kind == "S":
        return np.vectorize(lambda x: x.decode("utf-8", errors="ignore"))(flat)

    if flat.dtype.kind == "U":
        return flat.astype(object)

    if flat.dtype == object:
        return np.vectorize(_decode_scalar, otypes=[object])(flat)

    return flat