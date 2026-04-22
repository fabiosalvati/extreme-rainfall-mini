from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Rule-based event regime classification.
#
# The goal is to separate events into simple process-oriented classes:
# - widespread_frontal
# - organized_convective
# - localized_afternoon_convective
# - mixed_or_uncertain
#
# These rules are project-specific and tuned to this dataset and domain.
# They should not be treated as universal thresholds.


EVENT_CATALOGUE_PATH = Path("data_processed/events/events_catalogue_v1.parquet")
DIAGNOSTICS_PATH = Path("data_processed/diagnostics/event_diagnostics_v1.parquet")

OUTPUT_DIR = Path("data_processed/events")
OUTPUT_PATH = OUTPUT_DIR / "event_regimes_v1.parquet"
SUMMARY_PATH = OUTPUT_DIR / "event_regimes_summary_v1.csv"

REGIME_METHOD = "rule_based_v1"


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def load_inputs() -> pd.DataFrame:
    """
    Load the event catalogue and diagnostics table, then merge them.
    """
    if not EVENT_CATALOGUE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {EVENT_CATALOGUE_PATH}")
    if not DIAGNOSTICS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DIAGNOSTICS_PATH}")

    event_catalogue = pd.read_parquet(EVENT_CATALOGUE_PATH)
    diagnostics = pd.read_parquet(DIAGNOSTICS_PATH)

    require_columns(
        event_catalogue,
        [
            "event_candidate_id",
            "peak_local_hour",
            "duration_h",
            "footprint_max_pairwise_km",
            "n_stations",
            "peak_1h_mm",
        ],
        "event_catalogue",
    )

    require_columns(
        diagnostics,
        [
            "event_candidate_id",
            "cape_percentile_vs_background",
            "shear_850_500_boxmean_ms",
            "mslp_gradient_proxy_hpa_per_100km",
        ],
        "diagnostics",
    )

    diagnostics_keep = diagnostics[
        [
            "event_candidate_id",
            "cape_percentile_vs_background",
            "shear_850_500_boxmean_ms",
            "mslp_gradient_proxy_hpa_per_100km",
        ]
    ].copy()

    df = event_catalogue.merge(
        diagnostics_keep,
        on="event_candidate_id",
        how="inner",
        validate="one_to_one",
    )

    if df.empty:
        raise ValueError("No rows remain after merging event catalogue and diagnostics.")

    return df


def hour_is_afternoon(hour_local: float) -> bool:
    """
    Afternoon to early evening local time window used for convective timing.
    """
    return 12.0 <= hour_local < 20.0


def classify_event(row: pd.Series) -> dict[str, object]:
    """
    Assign one regime label to one event.

    Rule order matters:
    1. widespread_frontal
    2. organized_convective
    3. localized_afternoon_convective
    4. mixed_or_uncertain

    This is intentionally conservative.
    If an event does not fit one class cleanly, keep it as mixed_or_uncertain.
    """
    needed = [
        "peak_local_hour",
        "duration_h",
        "footprint_max_pairwise_km",
        "n_stations",
        "cape_percentile_vs_background",
        "shear_850_500_boxmean_ms",
        "mslp_gradient_proxy_hpa_per_100km",
    ]

    if row[needed].isna().any():
        return {
            "regime": "mixed_or_uncertain",
            "regime_confidence": "low",
            "regime_reason": "missing key classification fields",
            "score_localized": np.nan,
            "score_organized": np.nan,
            "score_widespread": np.nan,
            "regime_method": REGIME_METHOD,
        }

    peak_local_hour = float(row["peak_local_hour"])
    duration_h = float(row["duration_h"])
    footprint_km = float(row["footprint_max_pairwise_km"])
    n_stations = int(row["n_stations"])
    cape_pct = float(row["cape_percentile_vs_background"])
    shear_ms = float(row["shear_850_500_boxmean_ms"])
    mslp_grad = float(row["mslp_gradient_proxy_hpa_per_100km"])

    afternoon_peak = hour_is_afternoon(peak_local_hour)
    non_afternoon_peak = not afternoon_peak

    # 1) Widespread / frontal
    # Long-lived, broad, and with at least some large-scale forcing signal.
    widespread_gate = (
        duration_h >= 6.0
        and (footprint_km >= 150.0 or n_stations >= 10)
        and mslp_grad >= 1.3
        and (
            cape_pct < 75.0
            or shear_ms < 8.0
            or non_afternoon_peak
        )
    )

    if widespread_gate:
        score_widespread = 0
        reasons = []

        if duration_h >= 9.0:
            score_widespread += 1
            reasons.append("long duration")
        elif duration_h >= 6.0:
            score_widespread += 1
            reasons.append("moderately long duration")

        if footprint_km >= 200.0:
            score_widespread += 1
            reasons.append("very broad footprint")
        elif footprint_km >= 150.0:
            score_widespread += 1
            reasons.append("broad footprint")

        if n_stations >= 10:
            score_widespread += 1
            reasons.append("many stations affected")

        if mslp_grad >= 1.6:
            score_widespread += 1
            reasons.append("stronger pressure-gradient signal")
        elif mslp_grad >= 1.3:
            score_widespread += 1
            reasons.append("pressure-gradient support")

        if cape_pct < 60.0:
            score_widespread += 1
            reasons.append("not strongly CAPE-dominated")

        if non_afternoon_peak:
            score_widespread += 1
            reasons.append("non-afternoon peak")

        confidence = "high" if score_widespread >= 5 else "medium"

        return {
            "regime": "widespread_frontal",
            "regime_confidence": confidence,
            "regime_reason": "; ".join(reasons),
            "score_localized": 0,
            "score_organized": 0,
            "score_widespread": score_widespread,
            "regime_method": REGIME_METHOD,
        }

    # 2) Organized convective
    # Needs both instability and shear, plus a sign that the event is
    # larger or longer-lived than a brief local shower.
    organized_gate = (
        cape_pct >= 60.0
        and shear_ms >= 7.0
        and duration_h >= 2.5
        and (footprint_km >= 75.0 or n_stations >= 5)
    )

    if organized_gate:
        score_organized = 0
        reasons = []

        if cape_pct >= 80.0:
            score_organized += 1
            reasons.append("very high CAPE percentile")
        elif cape_pct >= 60.0:
            score_organized += 1
            reasons.append("high CAPE percentile")

        if shear_ms >= 12.0:
            score_organized += 1
            reasons.append("strong deep-layer shear")
        elif shear_ms >= 8.0:
            score_organized += 1
            reasons.append("enhanced deep-layer shear")

        if duration_h >= 6.0:
            score_organized += 1
            reasons.append("multi-hour event")
        elif duration_h >= 3.0:
            score_organized += 1
            reasons.append("moderate duration")

        if footprint_km >= 150.0:
            score_organized += 1
            reasons.append("large footprint")
        elif footprint_km >= 75.0:
            score_organized += 1
            reasons.append("expanded footprint")

        if n_stations >= 10:
            score_organized += 1
            reasons.append("many stations affected")
        elif n_stations >= 5:
            score_organized += 1
            reasons.append("multi-station event")

        if mslp_grad < 1.8:
            score_organized += 1
            reasons.append("not strongly frontal by pressure-gradient proxy")

        confidence = "high" if score_organized >= 5 else "medium"

        return {
            "regime": "organized_convective",
            "regime_confidence": confidence,
            "regime_reason": "; ".join(reasons),
            "score_localized": 0,
            "score_organized": score_organized,
            "score_widespread": 0,
            "regime_method": REGIME_METHOD,
        }

    # 3) Localized afternoon convective
    # Compact, short-lived, afternoon-peaking, unstable enough,
    # and without a strong large-scale forcing signal.
    localized_gate = (
        afternoon_peak
        and duration_h <= 4.5
        and (footprint_km < 125.0 or n_stations <= 6)
        and cape_pct >= 60.0
        and shear_ms < 12.0
        and mslp_grad < 1.4
    )

    if localized_gate:
        score_localized = 0
        reasons = []

        if 14.0 <= peak_local_hour < 19.0:
            score_localized += 1
            reasons.append("afternoon or early-evening peak")
        elif afternoon_peak:
            score_localized += 1
            reasons.append("daytime peak")

        if duration_h <= 3.0:
            score_localized += 1
            reasons.append("short duration")
        elif duration_h <= 4.5:
            score_localized += 1
            reasons.append("moderately short duration")

        if footprint_km < 50.0:
            score_localized += 1
            reasons.append("very compact footprint")
        elif footprint_km < 100.0:
            score_localized += 1
            reasons.append("compact footprint")

        if n_stations <= 3:
            score_localized += 1
            reasons.append("few stations affected")
        elif n_stations <= 5:
            score_localized += 1
            reasons.append("limited station count")

        if cape_pct >= 80.0:
            score_localized += 1
            reasons.append("very high CAPE percentile")
        elif cape_pct >= 60.0:
            score_localized += 1
            reasons.append("high CAPE percentile")

        if shear_ms < 8.0:
            score_localized += 1
            reasons.append("weak shear")
        elif shear_ms < 12.0:
            score_localized += 1
            reasons.append("weak-to-moderate shear")

        if mslp_grad < 1.1:
            score_localized += 1
            reasons.append("weak pressure-gradient forcing")
        elif mslp_grad < 1.4:
            score_localized += 1
            reasons.append("modest pressure-gradient forcing")

        confidence = "high" if score_localized >= 6 else "medium"

        return {
            "regime": "localized_afternoon_convective",
            "regime_confidence": confidence,
            "regime_reason": "; ".join(reasons),
            "score_localized": score_localized,
            "score_organized": 0,
            "score_widespread": 0,
            "regime_method": REGIME_METHOD,
        }

    # 4) Mixed / uncertain
    reason_parts = []

    if non_afternoon_peak and duration_h <= 4.5 and footprint_km < 100.0:
        reason_parts.append("compact event but timing not typical of localized afternoon convection")

    if cape_pct >= 60.0 and shear_ms >= 8.0 and duration_h < 3.0:
        reason_parts.append("convective environment present but event too short or too small for organized class")

    if duration_h >= 6.0 and mslp_grad < 1.3:
        reason_parts.append("broad or long-lived event without strong frontal signal")

    if not reason_parts:
        reason_parts.append("no regime matched cleanly")

    return {
        "regime": "mixed_or_uncertain",
        "regime_confidence": "low",
        "regime_reason": "; ".join(reason_parts),
        "score_localized": 0,
        "score_organized": 0,
        "score_widespread": 0,
        "regime_method": REGIME_METHOD,
    }


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the regime rule set to all events.
    """
    classified = df.copy()

    results = classified.apply(classify_event, axis=1, result_type="expand")
    classified = pd.concat([classified, results], axis=1)

    confidence_order = pd.CategoricalDtype(
        categories=["high", "medium", "low"],
        ordered=True,
    )
    regime_order = pd.CategoricalDtype(
        categories=[
            "localized_afternoon_convective",
            "organized_convective",
            "widespread_frontal",
            "mixed_or_uncertain",
        ],
        ordered=True,
    )

    classified["regime_confidence"] = classified["regime_confidence"].astype(confidence_order)
    classified["regime"] = classified["regime"].astype(regime_order)

    classified = classified.sort_values(
        ["regime", "regime_confidence", "peak_1h_mm"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return classified


def save_summary(df: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    """
    Save a compact summary table by regime and confidence level.
    """
    summary = (
        df.groupby(["regime", "regime_confidence"], dropna=False)
        .agg(
            n_events=("event_candidate_id", "count"),
            median_peak_1h_mm=("peak_1h_mm", "median"),
            median_duration_h=("duration_h", "median"),
            median_footprint_km=("footprint_max_pairwise_km", "median"),
            median_cape_pct=("cape_percentile_vs_background", "median"),
            median_shear_ms=("shear_850_500_boxmean_ms", "median"),
            median_mslp_grad=("mslp_gradient_proxy_hpa_per_100km", "median"),
        )
        .reset_index()
        .sort_values(["n_events", "median_peak_1h_mm"], ascending=[False, False])
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_inputs()
    classified = classify_regimes(df)

    classified.to_parquet(OUTPUT_PATH, index=False)
    summary = save_summary(classified, SUMMARY_PATH)

    print("\nRegime classification complete.")
    print(f"Rows: {len(classified)}")
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Saved -> {SUMMARY_PATH}")

    print("\nRegime counts:")
    print(classified["regime"].value_counts(dropna=False).to_string())

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()