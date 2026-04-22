from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Compute event-level ERA5 diagnostics from the event-centred ERA5 dataset.
# These diagnostics are simple environmental proxies, not full process diagnostics.
#
# Main outputs:
# - TCWV anomaly
# - CAPE percentile within the analysis block
# - 850 hPa wind speed
# - low-level moisture-transport proxy
# - 850-500 hPa shear proxy
# - mean sea-level pressure gradient proxy


def _require_vars(ds: xr.Dataset, names: list[str], ds_name: str) -> None:
    """
    Check that a dataset contains the expected variables.
    """
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"{ds_name} is missing required variables: {missing}")


def _ensure_utc_timestamp(ts) -> pd.Timestamp:
    """
    Return a timezone-aware UTC timestamp.
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def area_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Cos(latitude)-weighted mean over latitude/longitude.

    This gives one box-mean value for each remaining dimension.
    """
    if "latitude" not in da.dims or "longitude" not in da.dims:
        raise ValueError("DataArray must have latitude and longitude dimensions.")

    weights = np.cos(np.deg2rad(da["latitude"]))
    weights = xr.DataArray(weights, coords={"latitude": da["latitude"]}, dims=("latitude",))

    return da.weighted(weights).mean(dim=("latitude", "longitude"), skipna=True)


def percentile_rank(value: float, reference: np.ndarray) -> float:
    """
    Percentile rank of one value relative to a reference distribution.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]

    if ref.size == 0 or not np.isfinite(value):
        return np.nan

    return float(100.0 * np.mean(ref <= value))


def mean_mslp_gradient_hpa_per_100km(msl_field: xr.DataArray) -> float:
    """
    Mean horizontal MSLP gradient magnitude over the box.

    Input:
    - msl_field in Pa on a latitude/longitude grid

    Output:
    - mean gradient in hPa per 100 km

    This is a simple regular-grid approximation.
    """
    if set(msl_field.dims) != {"latitude", "longitude"}:
        raise ValueError("msl_field must have exactly latitude and longitude dimensions.")

    lat = np.asarray(msl_field["latitude"].values, dtype=float)
    lon = np.asarray(msl_field["longitude"].values, dtype=float)
    field_hpa = np.asarray(msl_field.values, dtype=float) / 100.0

    if field_hpa.ndim != 2 or len(lat) < 2 or len(lon) < 2:
        return np.nan

    dlat_deg = float(np.nanmean(np.abs(np.diff(lat))))
    dlon_deg = float(np.nanmean(np.abs(np.diff(lon))))

    if dlat_deg == 0.0 or dlon_deg == 0.0:
        return np.nan

    dy_km = 111.32 * dlat_deg
    dx_km = 111.32 * dlon_deg * np.cos(np.deg2rad(float(np.nanmean(lat))))

    if dx_km <= 0.0 or dy_km <= 0.0:
        return np.nan

    dfdlat, dfdlon = np.gradient(field_hpa, dy_km, dx_km)
    grad_hpa_per_km = np.sqrt(dfdlat**2 + dfdlon**2)

    return float(np.nanmean(grad_hpa_per_km) * 100.0)


def compute_background_references(background_ds: xr.Dataset) -> dict[str, object]:
    """
    Build simple background references from the full ERA5 analysis block.

    Background here means:
    the full selected ERA5 time period over the Netherlands box.
    """
    _require_vars(background_ds, ["tcwv", "cape"], "background_ds")

    tcwv_boxmean_ts = area_mean(background_ds["tcwv"]).to_series().dropna()
    cape_boxmean_ts = area_mean(background_ds["cape"]).to_series().dropna()

    if tcwv_boxmean_ts.empty:
        raise ValueError("No valid tcwv background values found.")
    if cape_boxmean_ts.empty:
        raise ValueError("No valid cape background values found.")

    background = {
        "tcwv_background_mean_kgm2": float(tcwv_boxmean_ts.mean()),
        "cape_background_distribution_jkg": cape_boxmean_ts.to_numpy(dtype=float),
        "background_start_time_utc": _ensure_utc_timestamp(background_ds["time"].values.min()),
        "background_end_time_utc": _ensure_utc_timestamp(background_ds["time"].values.max()),
    }

    return background


def _require_q850_for_moisture_transport(event_context: xr.Dataset) -> None:
    """
    Require q850 explicitly for the moisture-transport proxy.

    If only r850 is available, the metric would no longer mean the same thing.
    """
    if "q850" in event_context.data_vars:
        return

    if "r850" in event_context.data_vars:
        raise KeyError(
            "event_context contains r850 but not q850. "
            "This diagnostics step defines low-level moisture transport as q850 × wind850, "
            "so the ERA5 pressure-level download must include specific humidity q at 850 hPa."
        )

    raise KeyError(
        "event_context is missing q850. "
        "Low-level moisture-transport proxy requires specific humidity q at 850 hPa."
    )


def compute_event_diagnostics(
    event_catalogue: pd.DataFrame,
    event_context: xr.Dataset,
    background_ds: xr.Dataset,
) -> pd.DataFrame:
    """
    Compute one row of ERA5 diagnostics per event.

    Diagnostics are evaluated at relative_hour = 0,
    which is the ERA5 hour nearest to the event peak.
    """
    required_context = ["tcwv", "cape", "msl", "u850", "u500", "v850", "v500"]
    _require_vars(event_context, required_context, "event_context")
    _require_q850_for_moisture_transport(event_context)

    background = compute_background_references(background_ds)
    tcwv_bg = background["tcwv_background_mean_kgm2"]
    cape_ref = background["cape_background_distribution_jkg"]

    rows: list[dict[str, object]] = []

    for row in event_catalogue.itertuples(index=False):
        event_id = row.event_candidate_id

        # Select one event and then the anchor hour relative to the event peak.
        ds_evt = event_context.sel(event_candidate_id=event_id, drop=True)
        ds0 = ds_evt.sel(relative_hour=0, drop=True)

        tcwv_mean = float(area_mean(ds0["tcwv"]).item())
        cape_mean = float(area_mean(ds0["cape"]).item())

        wind850_field = np.sqrt(ds0["u850"] ** 2 + ds0["v850"] ** 2)
        wind850_mean = float(area_mean(wind850_field).item())

        q850_mean = float(area_mean(ds0["q850"]).item())

        # Simple low-level moisture-transport proxy.
        # This is not a full moisture flux integral.
        moisture_transport_field = ds0["q850"] * wind850_field
        moisture_transport_mean = float(area_mean(moisture_transport_field).item())

        # Simple deep-layer shear proxy between 850 and 500 hPa.
        shear_field = np.sqrt((ds0["u500"] - ds0["u850"]) ** 2 + (ds0["v500"] - ds0["v850"]) ** 2)
        shear_mean = float(area_mean(shear_field).item())

        mslp_grad = float(mean_mslp_gradient_hpa_per_100km(ds0["msl"]))

        tcwv_anom = tcwv_mean - tcwv_bg
        cape_pct = percentile_rank(cape_mean, cape_ref)

        rows.append(
            {
                "event_candidate_id": event_id,
                "start_time": _ensure_utc_timestamp(row.start_time),
                "end_time": _ensure_utc_timestamp(row.end_time),
                "peak_time_utc": _ensure_utc_timestamp(row.peak_time_utc),
                "peak_station": str(row.peak_station).zfill(5) if hasattr(row, "peak_station") else None,
                "peak_1h_mm": float(row.peak_1h_mm) if hasattr(row, "peak_1h_mm") else np.nan,
                "n_stations": int(row.n_stations) if hasattr(row, "n_stations") else np.nan,
                "duration_h": float(row.duration_h) if hasattr(row, "duration_h") else np.nan,
                "tcwv_boxmean_kgm2": tcwv_mean,
                "tcwv_background_mean_kgm2": tcwv_bg,
                "tcwv_anomaly_kgm2": tcwv_anom,
                "cape_boxmean_jkg": cape_mean,
                "cape_percentile_vs_background": cape_pct,
                "wind850_boxmean_ms": wind850_mean,
                "q850_boxmean_kgkg": q850_mean,
                "moisture_transport_proxy_qv": moisture_transport_mean,
                "shear_850_500_boxmean_ms": shear_mean,
                "mslp_gradient_proxy_hpa_per_100km": mslp_grad,
                "background_start_time_utc": background["background_start_time_utc"],
                "background_end_time_utc": background["background_end_time_utc"],
            }
        )

    diagnostics = pd.DataFrame(rows)

    diagnostics = diagnostics.sort_values(
        ["peak_1h_mm", "tcwv_anomaly_kgm2"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return diagnostics


def save_top_table(
    diagnostics: pd.DataFrame,
    output_csv: str | Path,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Save a compact top-events diagnostics table.
    """
    cols = [
        "event_candidate_id",
        "peak_time_utc",
        "peak_station",
        "peak_1h_mm",
        "n_stations",
        "tcwv_anomaly_kgm2",
        "cape_percentile_vs_background",
        "wind850_boxmean_ms",
        "moisture_transport_proxy_qv",
        "shear_850_500_boxmean_ms",
        "mslp_gradient_proxy_hpa_per_100km",
    ]

    out = diagnostics[cols].head(top_n).copy()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return out


def plot_tcwv_vs_cape(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    annotate_top_n: int = 5,
) -> None:
    """
    Scatter plot of moisture anomaly versus CAPE percentile.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        diagnostics["tcwv_anomaly_kgm2"],
        diagnostics["cape_percentile_vs_background"],
        s=np.clip(diagnostics["peak_1h_mm"].fillna(0).to_numpy() * 8.0, 20.0, 250.0),
    )

    ax.set_xlabel("TCWV anomaly vs analysis-block mean (kg m$^{-2}$)")
    ax.set_ylabel("CAPE percentile vs analysis-block background")
    ax.set_title("Event moisture anomaly vs instability proxy")

    top = diagnostics.nlargest(annotate_top_n, "peak_1h_mm")
    for _, r in top.iterrows():
        ax.annotate(
            r["event_candidate_id"],
            (r["tcwv_anomaly_kgm2"], r["cape_percentile_vs_background"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transport_vs_shear(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    annotate_top_n: int = 5,
) -> None:
    """
    Scatter plot of low-level moisture transport proxy versus shear proxy.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        diagnostics["moisture_transport_proxy_qv"],
        diagnostics["shear_850_500_boxmean_ms"],
        s=np.clip(diagnostics["peak_1h_mm"].fillna(0).to_numpy() * 8.0, 20.0, 250.0),
    )

    ax.set_xlabel("Moisture-transport proxy (q850 × wind850)")
    ax.set_ylabel("Deep-layer shear proxy (850–500 hPa, m s$^{-1}$)")
    ax.set_title("Event moisture transport vs shear proxy")

    top = diagnostics.nlargest(annotate_top_n, "peak_1h_mm")
    for _, r in top.iterrows():
        ax.annotate(
            r["event_candidate_id"],
            (r["moisture_transport_proxy_qv"], r["shear_850_500_boxmean_ms"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)