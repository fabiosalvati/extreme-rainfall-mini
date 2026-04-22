from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from io_knmi import (
    list_knmi_files,
    open_knmi_file,
    station_metadata_dataframe,
    extract_station_series,
)

DATA_DIR = "data_raw/knmi_10min"
OUT_PARQUET = Path("data_processed/stations/knmi_10min_sample_station_series.parquet")
OUT_PNG = Path("report/figures/knmi_10min_sample_station_series.png")

PREFERRED_STATIONS = ["06260", "06240", "06235", "06210"]


def choose_station(files):
    with open_knmi_file(files[0]) as ds:
        meta = station_metadata_dataframe(ds)

    station_ids = meta["station"].astype(str).tolist()

    for sid in PREFERRED_STATIONS + station_ids:
        if sid not in station_ids:
            continue
        ts = extract_station_series(files, station_selector=sid, precip_var="rg")
        if ts["precip_10min_amount_mm"].notna().sum() > 0:
            row = meta.loc[meta["station"].astype(str) == sid].iloc[0]
            return sid, row.to_dict(), ts

    raise RuntimeError("Could not find a station with non-missing rg values.")


def main():
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    files = list_knmi_files(DATA_DIR)
    station_id, station_meta, ts = choose_station(files)

    ts.to_parquet(OUT_PARQUET, index=False)

    print("station_id:", station_id)
    if "stationname" in station_meta:
        print("stationname:", station_meta["stationname"])
    if "wsi" in station_meta:
        print("wsi:", station_meta["wsi"])
    if "lat" in station_meta and "lon" in station_meta:
        print("lat/lon:", station_meta["lat"], station_meta["lon"])

    print("n_rows:", len(ts))
    print("n_valid_precip:", int(ts["precip_10min_amount_mm"].notna().sum()))
    print("n_missing_precip:", int(ts["precip_10min_amount_mm"].isna().sum()))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts["timestamp_end_utc"], ts["precip_10min_amount_mm"])
    ax.set_title(f"Sample 10-minute rainfall series (station {station_id})")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("mm per 10 min")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print("Saved parquet:", OUT_PARQUET)
    print("Saved figure:", OUT_PNG)


if __name__ == "__main__":
    main()