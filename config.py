# config.py
# MomentumAI — Warehouse Pallet & Cart Planning System
# Central configuration. Edit these values to match your facility's real-world numbers.

CONFIG = {

    # ─── Core Warehouse Logic ──────────────────────────────────────────────────
    "packages_per_pallet":    180,     # Packages that fit on one standard pallet
    "packages_per_cart":       35,     # Packages that fit on one standard cart/dolly
    "pallet_order_lead_days":   7,     # Pallets MUST be ordered 7 days in advance
    "cart_order_lead_days":     1,     # Carts can be ordered just 1 day in advance

    # ─── Spoilage & Damage Buffer ──────────────────────────────────────────────
    # "Spoilt" pallets are damaged/unusable pallets that reduce effective stock.
    # We add a buffer at order time to account for this inevitable shrinkage.
    "base_spoilage_rate":       0.06,  # 6% baseline — i.e. 6 in every 100 pallets ordered are unusable
    "min_extra_pallets":           2,  # Minimum additional pallets always added as a safety floor

    # Shift-type spoilage multipliers — night & peak shifts cause more damage
    "spoilage_by_shift": {
        "morning":   1.00,            # Baseline
        "afternoon": 1.05,            # Slightly more rushed
        "night":     1.20,            # Reduced supervision → ~20% more spoilage
        "peak":      1.40,            # High pace, high pressure → ~40% more spoilage
    },

    # Extra multiplier applied across the board during peak seasons
    "peak_season_spoilage_multiplier": 1.25,

    # ─── Volume Ranges by Shift Type (packages processed per shift) ────────────
    # Used for synthetic data generation and UI slider limits
    "volume_ranges": {
        "morning":   (5_000,  22_000),
        "afternoon": (5_000,  22_000),
        "night":     (2_000,  10_000),
        "peak":      (18_000, 60_000),
    },

    # ─── Risk Alert Thresholds ─────────────────────────────────────────────────
    # These control the Green / Yellow / Red alert system across three dimensions.
    "risk": {

        # 1. STOCK ADEQUACY — How much of your current stock will this shift consume?
        #    Ratio = pallets_predicted / current_stock_on_hand
        "stock_green":    0.75,    # Consuming < 75% of stock       → GREEN  (healthy headroom)
        "stock_yellow":   0.90,    # Consuming 75–90% of stock      → YELLOW (monitor closely)
        #                          # Consuming > 90% of stock       → RED    (order immediately)

        # 2. ORDER DEADLINE — Days remaining before the weekly pallet order closes
        "deadline_green":   7,     # More than 7 days               → GREEN
        "deadline_yellow":  4,     # 4–6 days remaining             → YELLOW (plan your order)
        "deadline_red":     3,     # 3 days or fewer                → RED    (order today!)

        # 3. SPOILAGE RATE — Your area's rolling damage/spoilage percentage
        "spoilage_green":  0.07,   # < 7%  spoilage rate            → GREEN
        "spoilage_yellow": 0.10,   # 7–10% spoilage rate            → YELLOW (review handling)
        #                          # > 10% spoilage rate            → RED    (urgent review)
    },

    # ─── Synthetic Training Data ───────────────────────────────────────────────
    "n_samples":                    8_000,
    "random_seed":                     42,
    "team_size_range":           (12, 90),    # Warehouse team headcount range
    "historical_spoilage_range": (0.02, 0.14),
    "inbound_ratio_range":        (0.30, 0.80),
    "volume_noise_pct":             0.05,     # ±5% random variance added to volumes

    # ─── Feature Thresholds (used in preprocessing) ───────────────────────────
    "high_volume_threshold":      25_000,     # Shifts above this are classed as "heavy"
    "high_stress_vols_per_worker":   400,     # Packages/worker above this = "stressed" shift
}
