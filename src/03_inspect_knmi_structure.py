from io_knmi import list_knmi_files, open_knmi_file, dataset_variable_summary

DATA_DIR = "data_raw/knmi_10min"


def main():
    files = list_knmi_files(DATA_DIR)
    f = files[0]

    print(f"Inspecting file: {f}\n")

    with open_knmi_file(f) as ds:
        print(ds)
        print("\nVARIABLE SUMMARY\n")
        print(dataset_variable_summary(ds).to_string(index=False))


if __name__ == "__main__":
    main()