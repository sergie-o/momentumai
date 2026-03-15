# preprocess.py
# MomentumAI — Feature engineering pipeline
#
# IMPORTANT: MODEL_FEATURES defines the exact set of columns (in order) that
# both train_model.py and app.py feed into the model. Any change here must be
# reflected in both files.

import pandas as pd
import numpy as np

from config import CONFIG

# ── Canonical feature list ──────────────────────────────────────────────────────
# These are the columns the trained model expects at inference time.
MODEL_FEATURES = [
    # --- Raw inputs ---
    "target_volume",            # Packages to process this shift (primary driver)
    "shift_type_encoded",       # 0=morning, 1=afternoon, 2=night, 3=peak
    "day_of_week",              # 0=Monday … 6=Sunday
    "is_weekend",               # 1 if Saturday or Sunday
    "is_peak_season",           # 1 during high-volume events (Prime Day, Christmas…)
    "team_size",                # Number of warehouse associates on shift
    "historical_spoilage_rate", # Area's rolling spoilage/damage rate (decimal, e.g. 0.07)
    "inbound_ratio",            # Proportion of volume that is inbound (0–1)
    "has_oversized_items",      # 1 if shift includes oversized packages

    # --- Engineered features ---
    "base_pallets_formula",     # ceil(target_volume / 180) — the hard floor
    "volume_per_worker",        # Packages per associate — proxy for pace/stress
    "is_heavy_shift",           # 1 if target_volume > HIGH_VOLUME_THRESHOLD
    "is_stressed_shift",        # 1 if volume_per_worker > HIGH_STRESS threshold
    "shift_spoilage_index",     # historical_spoilage_rate × shift-type multiplier
]

# Shift-type spoilage multiplier lookup (mirrors config for offline use)
_SHIFT_SPOILAGE_MULT = {0: 1.00, 1: 1.05, 2: 1.20, 3: 1.40}


def engineer_features(df: pd.DataFrame, config: dict = CONFIG) -> pd.DataFrame:
    """
    Add all engineered features to a warehouse-shift DataFrame.

    Works on both:
    - Full training datasets (many rows, loaded from CSV)
    - Single-row inference DataFrames (built in app.py at prediction time)

    Parameters
    ----------
    df     : DataFrame with at minimum the raw input columns.
    config : CONFIG dict (default imported at module level).

    Returns
    -------
    DataFrame with all original columns PLUS the engineered ones.
    """
    df  = df.copy()
    ppp = config["packages_per_pallet"]

    # ── 1. Core formula baseline ───────────────────────────────────────────────
    df["base_pallets_formula"] = np.ceil(
        df["target_volume"] / ppp
    ).astype(int)

    # ── 2. Labour efficiency ───────────────────────────────────────────────────
    df["volume_per_worker"] = (
        df["target_volume"] / df["team_size"].replace(0, 1)
    ).round(1)

    df["is_heavy_shift"] = (
        df["target_volume"] > config["high_volume_threshold"]
    ).astype(int)

    df["is_stressed_shift"] = (
        df["volume_per_worker"] > config["high_stress_vols_per_worker"]
    ).astype(int)

    # ── 3. Spoilage composite index ────────────────────────────────────────────
    # Multiplies the area's historical rate by the shift-type risk factor,
    # giving the model a single numeric signal for "how risky is this shift?"
    df["shift_spoilage_index"] = (
        df["historical_spoilage_rate"]
        * df["shift_type_encoded"].map(_SHIFT_SPOILAGE_MULT).fillna(1.0)
    ).round(4)

    return df


def get_feature_matrix(df: pd.DataFrame, config: dict = CONFIG) -> pd.DataFrame:
    """
    Run `engineer_features` then return only MODEL_FEATURES in canonical order.
    This is the function called by both train_model.py and app.py.
    """
    df = engineer_features(df, config)
    # Validate that all expected columns are present
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"preprocess.get_feature_matrix: missing columns after engineering: {missing}"
        )
    return df[MODEL_FEATURES]
