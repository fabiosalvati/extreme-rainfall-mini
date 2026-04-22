import numpy as np
import pandas as pd

from io_knmi import list_knmi_files, open_knmi_file, missing_value_summary

DATA_DIR = "data_raw/knmi_10min"


def inspect_key_precip_vars(ds, variables=("rg", "pg", "R1H", "R6H", "R12H", "R24H")):
    rows = []

    for var in variables:
        if var not in ds:
            continue

        values = np.asarray(ds[var].values)

        row = {
            "variable": var,
            "dtype": str(values.dtype),
            "shape": values.shape,
            "n_total": int(values.size),
        }

        if np.issubdtype(values.dtype, np.number):
            row["n_nan"] = int(np.isnan(values).sum())
            finite = values[np.isfinite(values)]
            row["min_finite"] = float(finite.min()) if finite.size else np.nan
            row["max_finite"] = float(finite.max()) if finite.size else np.nan

            unique_sample = np.unique(values[~np.isnan(values)])[:10]
            row["unique_sample"] = unique_sample.tolist()
        else:
            row["n_nan"] = None
            row["min_finite"] = None
            row["max_finite"] = None
            row["unique_sample"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    files = list_knmi_files(DATA_DIR)
    f = files[0]

    print(f"Checking missing-value behaviour in: {f}\n")

    with open_knmi_file(f) as ds:
        print("ATTR-BASED MISSING VALUE SUMMARY\n")
        print(missing_value_summary(ds).to_string(index=False))

        print("\nKEY PRECIPITATION VARIABLES: ACTUAL VALUE CHECK\n")
        print(inspect_key_precip_vars(ds).to_string(index=False))


if __name__ == "__main__":
    main()