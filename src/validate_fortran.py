from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


FORTRAN_EXE = Path("build/rolling_accumulation")
INPUT_FILE = Path("fortran/python_demo_input.txt")
OUTPUT_FILE = Path("fortran/python_demo_output.txt")

MISSING_VALUE = -9999.0
WINDOW = 6


def rolling_sum_python(
    rain: np.ndarray,
    window: int,
    missing_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Python reference version of the rolling accumulation logic.

    Returns
    -------
    rolling_sum : np.ndarray
        Rolling sums. Invalid windows get the missing_value sentinel.
    valid_flag : np.ndarray
        1 for valid windows, 0 for invalid windows.
    """
    rain = np.asarray(rain, dtype=float)
    n = len(rain)

    rolling_sum = np.full(n, missing_value, dtype=float)
    valid_flag = np.zeros(n, dtype=int)

    for i in range(n):
        if i + 1 < window:
            continue

        current_window = rain[i - window + 1 : i + 1]

        if np.any(current_window == missing_value):
            rolling_sum[i] = missing_value
            valid_flag[i] = 0
        else:
            rolling_sum[i] = float(current_window.sum())
            valid_flag[i] = 1

    return rolling_sum, valid_flag


def write_fortran_input(
    path: Path,
    rain: np.ndarray,
    window: int,
    missing_value: float,
) -> None:
    """
    Write the text input expected by the Fortran program.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(rain)}\n")
        f.write(f"{window}\n")
        f.write(f"{missing_value}\n")
        for value in rain:
            f.write(f"{value}\n")


def run_fortran(
    exe_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Run the compiled Fortran executable.
    """
    if not exe_path.exists():
        raise FileNotFoundError(
            f"Missing executable: {exe_path}\n"
            "Compile it first with: make fortran-demo"
        )

    result = subprocess.run(
        [str(exe_path), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    print("Fortran STDOUT:")
    print(result.stdout.strip())

    if result.stderr.strip():
        print("\nFortran STDERR:")
        print(result.stderr.strip())


def read_fortran_output(path: Path) -> pd.DataFrame:
    """
    Read the output file written by the Fortran program.

    Expected format:
    # index rainfall rolling_sum valid_flag
       1   0.000  -9999.000 0
       ...
    """
    rows: list[dict[str, float | int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Unexpected output line: {line}")

            rows.append(
                {
                    "index": int(parts[0]),
                    "rain": float(parts[1]),
                    "fortran_sum": float(parts[2]),
                    "fortran_flag": int(parts[3]),
                }
            )

    return pd.DataFrame(rows)


def build_comparison_table(
    rain: np.ndarray,
    py_sum: np.ndarray,
    py_flag: np.ndarray,
    fortran_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Python reference results with Fortran output.
    """
    python_df = pd.DataFrame(
        {
            "index": np.arange(1, len(rain) + 1),
            "rain": rain,
            "python_sum": py_sum,
            "python_flag": py_flag,
        }
    )

    comparison = python_df.merge(
        fortran_df,
        on=["index", "rain"],
        how="inner",
        validate="one_to_one",
    )

    comparison["sum_match"] = np.isclose(
        comparison["python_sum"],
        comparison["fortran_sum"],
        atol=1e-12,
        rtol=0.0,
    )
    comparison["flag_match"] = comparison["python_flag"] == comparison["fortran_flag"]

    return comparison


def main() -> None:
    # Toy rainfall series with one missing value
    rain = np.array(
        [0.0, 0.2, 1.1, 0.0, 2.0, 0.3, -9999.0, 0.4, 0.5, 1.0, 0.0, 0.7],
        dtype=float,
    )

    print("Step 1: Python reference calculation")
    py_sum, py_flag = rolling_sum_python(rain, WINDOW, MISSING_VALUE)

    print("Step 2: Write Fortran input file")
    write_fortran_input(INPUT_FILE, rain, WINDOW, MISSING_VALUE)
    print(f"Wrote -> {INPUT_FILE}")

    print("Step 3: Run Fortran executable")
    run_fortran(FORTRAN_EXE, INPUT_FILE, OUTPUT_FILE)
    print(f"Wrote -> {OUTPUT_FILE}")

    print("Step 4: Read Fortran output")
    fortran_df = read_fortran_output(OUTPUT_FILE)

    print("Step 5: Compare Python and Fortran")
    comparison = build_comparison_table(rain, py_sum, py_flag, fortran_df)

    print("\nComparison table:")
    print(comparison.to_string(index=False))

    all_sum_match = bool(comparison["sum_match"].all())
    all_flag_match = bool(comparison["flag_match"].all())

    print("\nValidation summary:")
    print(f"All rolling sums match: {all_sum_match}")
    print(f"All valid flags match: {all_flag_match}")

    if not all_sum_match or not all_flag_match:
        raise ValueError("Validation failed: Python and Fortran outputs do not match.")

    print("\nValidation passed.")


if __name__ == "__main__":
    main()