from io_knmi import (
    list_knmi_files,
    open_knmi_file,
    infer_station_dim,
    station_metadata_dataframe,
    infer_precipitation_spec
)

DATA_DIR = "data_raw/knmi_10min"


def main():
    files = list_knmi_files(DATA_DIR)
    f = files[0]

    print(f"Inspecting station metadata from: {f}\n")

    with open_knmi_file(f) as ds:
        station_dim = infer_station_dim(ds)
        meta = station_metadata_dataframe(ds)

        print("station_dim:", station_dim)
        print("n_stations:", len(meta))
        print("\nSTATION METADATA (first 20 rows)\n")
        print(meta.head(20).to_string(index=False))


    print(f"Checking precipitation variable in: {f}\n")

    with open_knmi_file(f) as ds:
        spec = infer_precipitation_spec(ds)

        print("variable:", spec.variable)
        print("interpretation:", spec.interpretation)
        print("units:", spec.units)
        print("conversion_to_10min_mm:", spec.conversion_to_10min_mm)

        if spec.variable == "rg":
            print("\nDecision: use 'rg' as the primary 10-minute rainfall variable.")
        elif spec.variable == "pg":
            print("\nDecision: only use 'pg' if 'rg' is absent or unusable.")
        else:
            print("\nDecision: inspect manually before using this variable for Day 3.")



if __name__ == "__main__":
    main()
