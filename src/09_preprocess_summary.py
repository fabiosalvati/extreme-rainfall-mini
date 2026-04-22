import pandas as pd

# Output written by 08_preprocess_precip.py
SUMMARY_PATH = "data_processed/stations/preprocessing_summary.csv"

# It reports missing intervals, duplicate timestamps, and basic counts of valid 1 h and 3 h windows.

def main():
    # Read the station summary table.
    # Station IDs are kept as strings so leading zeros are not lost.
    df = pd.read_csv(SUMMARY_PATH, dtype={"station": str})
    df["station"] = df["station"].str.zfill(5)

    print("n_stations:", len(df))
    print()

    # Show stations where at least one 10-minute interval is missing.
    print("Stations with any missing intervals:")
    print(
        df.loc[
            df["n_missing_intervals"] > 0,
            ["station", "stationname", "n_missing_intervals"],
        ].to_string(index=False)
    )
    print()

    # Show stations where duplicate timestamps were found.
    print("Stations with any duplicate timestamps:")
    print(
        df.loc[
            df["n_duplicate_timestamps"] > 0,
            ["station", "stationname", "n_duplicate_timestamps"],
        ].to_string(index=False)
    )
    print()

    # Print overall counts for data completeness and valid rolling windows.
    print("Summary stats:")
    print(
        df[["n_rows", "n_missing_intervals", "n_valid_1h", "n_valid_3h"]]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()