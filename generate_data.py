# generate_data.py
# MomentumAI — Synthetic warehouse shift data generator
#
# Each row represents ONE shift. The target variable 'pallets_to_order'
# reflects what an experienced Area Manager would actually requisition —
# base pallets from volume PLUS a spoilage/damage buffer.

import os
import numpy as np
import pandas as pd

from config import CONFIG

# ── Shift type mapping ─────────────────────────────────────────────────────────
SHIFT_TYPES    = ["morning", "afternoon", "night", "peak"]
SHIFT_TYPE_MAP = {s: i for i, s in enumerate(SHIFT_TYPES)}   # morning=0 … peak=3


def generate_synthetic_data(config: dict = CONFIG) -> pd.DataFrame:
    """
    Generate synthetic warehouse shift records for model training.

    Core logic
    ----------
    1.  Draw a target_volume (packages expected per shift) from realistic ranges.
    2.  Compute base_pallets = ceil(target_volume / 180).
    3.  Compute an effective spoilage rate that varies by:
            • The area's historical spoilage rate (sampled per row)
            • Shift type multiplier  (night / peak = more damage)
            • Whether it is a peak season  (e.g. Prime Day, Black Friday)
    4.  spoilage_buffer_pallets = max(min_floor, ceil(base_pallets × eff_spoilage))
    5.  pallets_to_order = base_pallets + spoilage_buffer_pallets + oversized_extra
        This is the label the model learns to predict.
    6.  carts_needed is a formula-only target (no ML model needed — carts can be
        reordered the day before, making precision less critical).
    """

    np.random.seed(config["random_seed"])
    n   = config["n_samples"]
    rng = np.random

    # ── 1. Shift type ──────────────────────────────────────────────────────────
    # Approximate real distribution: mostly morning/afternoon, some night, rare peak
    shift_choices = rng.choice(
        SHIFT_TYPES,
        size=n,
        p=[0.35, 0.35, 0.20, 0.10],
    )

    # ── 2. Target volume (packages) ───────────────────────────────────────────
    volumes = np.array(
        [rng.uniform(*config["volume_ranges"][s]) for s in shift_choices]
    )
    # Add ±5% natural variance
    volume_noise = rng.uniform(
        1.0 - config["volume_noise_pct"],
        1.0 + config["volume_noise_pct"],
        n,
    )
    volumes = np.round(volumes * volume_noise).astype(int)

    # ── 3. Shift-level attributes ──────────────────────────────────────────────
    day_of_week             = rng.randint(0, 7, n)                    # 0=Mon … 6=Sun
    is_weekend              = (day_of_week >= 5).astype(int)
    is_peak_season          = rng.binomial(1, 0.15, n)                # ~15% of shifts
    team_size               = rng.randint(
                                  config["team_size_range"][0],
                                  config["team_size_range"][1] + 1,
                                  n)
    historical_spoilage_rate = rng.uniform(*config["historical_spoilage_range"], n)
    inbound_ratio           = rng.uniform(*config["inbound_ratio_range"], n)
    has_oversized_items     = rng.binomial(1, 0.25, n)               # 25% of shifts

    # ── 4. Base pallets (the simple formula) ──────────────────────────────────
    ppp          = config["packages_per_pallet"]
    base_pallets = np.ceil(volumes / ppp).astype(int)

    # ── 5. Effective spoilage rate ─────────────────────────────────────────────
    shift_mult   = np.array([config["spoilage_by_shift"][s] for s in shift_choices])
    season_mult  = np.where(
        is_peak_season == 1,
        config["peak_season_spoilage_multiplier"],
        1.0,
    )
    effective_spoilage = historical_spoilage_rate * shift_mult * season_mult

    # Spoilage buffer: never less than the configured minimum floor
    spoilage_pallets = np.maximum(
        config["min_extra_pallets"],
        np.ceil(base_pallets * effective_spoilage).astype(int),
    )

    # ── 6. Oversized item penalty ──────────────────────────────────────────────
    # Oversized packages reduce packing efficiency — need 8–15% more pallet space
    oversized_extra = np.where(
        has_oversized_items == 1,
        np.ceil(base_pallets * rng.uniform(0.08, 0.15, n)).astype(int),
        0,
    ).astype(int)

    # ── 7. Final pallets to order (the training target) ───────────────────────
    # Small integer noise mimics the real-world judgment calls managers make
    order_noise      = rng.randint(-1, 3, n)
    pallets_to_order = np.maximum(
        base_pallets,
        base_pallets + spoilage_pallets + oversized_extra + order_noise,
    )

    # ── 8. Carts needed (formula-only; label kept for reference) ──────────────
    ppc          = config["packages_per_cart"]
    cart_buffer  = 0.08                                              # 8% cart buffer
    carts_needed = np.ceil(volumes / ppc * (1.0 + cart_buffer)).astype(int)

    # ── 9. Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        # Raw inputs (model features before engineering)
        "target_volume":            volumes,
        "shift_type":               shift_choices,
        "shift_type_encoded":       [SHIFT_TYPE_MAP[s] for s in shift_choices],
        "day_of_week":              day_of_week,
        "is_weekend":               is_weekend,
        "is_peak_season":           is_peak_season,
        "team_size":                team_size,
        "historical_spoilage_rate": historical_spoilage_rate,
        "inbound_ratio":            inbound_ratio,
        "has_oversized_items":      has_oversized_items,
        # Intermediate (kept for analysis, not fed to model directly)
        "base_pallets":             base_pallets,
        "spoilage_buffer_pallets":  spoilage_pallets,
        "oversized_extra_pallets":  oversized_extra,
        # Targets
        "pallets_to_order":         pallets_to_order,
        "carts_needed":             carts_needed,
    })

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv("data/raw/warehouse_shift_data.csv", index=False)

    print(f"✅  Generated {len(df):,} shift records")
    print(f"\nSample output:")
    print(
        df[[
            "target_volume", "shift_type",
            "base_pallets", "spoilage_buffer_pallets",
            "pallets_to_order", "carts_needed",
        ]].head(10).to_string(index=False)
    )
    print(f"\nTarget stats:")
    print(df["pallets_to_order"].describe().round(1))
    print(f"\nSaved → data/raw/warehouse_shift_data.csv")
