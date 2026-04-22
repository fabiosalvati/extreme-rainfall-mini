from pathlib import Path
import pandas as pd

from io_knmi import list_knmi_files, file_inventory_dataframe

DATA_DIR = "data_raw/knmi_10min"
OUT_CSV = Path("data_processed/diagnostics/file_inventory_knmi_10min.csv")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    files = list_knmi_files(DATA_DIR)
    inv = file_inventory_dataframe(files)

    print("n_files:", len(inv))
    print("start_interval_start_utc:", inv["interval_start_utc"].min())
    print("end_interval_end_utc:", inv["interval_end_utc"].max())

    gaps = inv[
        inv["gap_from_previous_minutes"].notna()
        & (inv["gap_from_previous_minutes"] != 10)
    ]

    print("n_non_10min_gaps:", len(gaps))
    if len(gaps) > 0:
        print("\nNON-10-MIN GAPS\n")
        print(gaps.head(20).to_string(index=False))

    inv.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()