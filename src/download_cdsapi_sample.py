from pathlib import Path

import cdsapi

client = cdsapi.Client()

single_dir = Path("data_raw/era5/single_levels")
pressure_dir = Path("data_raw/era5/pressure_levels")

single_dir.mkdir(parents=True, exist_ok=True)
pressure_dir.mkdir(parents=True, exist_ok=True)

selected_days = [f"{d:02d}" for d in range(1, 32)]
selected_hours = [f"{h:02d}:00" for h in range(24)]

single_target = single_dir / "era5_single_levels_20120501_20120601.grib"
pressure_target = pressure_dir / "era5_pressure_levels_20120501_20120601.grib"

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": [
            "total_column_water_vapour",
            "convective_available_potential_energy",
            "mean_sea_level_pressure",
        ],
        "year": ["2012"],
        "month": ["05"],
        "day": selected_days,
        "time": selected_hours,
        "area": [55.8, 2.5, 50.5, 7.5],  # north, west, south, east
        "data_format": "grib",
    },
    str(single_target),
)

client.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": ["reanalysis"],
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "vertical_velocity",
        ],
        "pressure_level": ["850", "700", "500"],
        "year": ["2012"],
        "month": ["05"],
        "day": selected_days,
        "time": selected_hours,
        "area": [55.8, 2.5, 50.5, 7.5],  # north, west, south, east
        "data_format": "grib",
    },
    str(pressure_target),
)