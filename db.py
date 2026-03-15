# db.py
# MomentumAI — Shift logging & spoilage tracking database
#
# Uses SQLite — a lightweight database stored as a single file (momentum_data.db)
# in your project folder. No server or signup needed. Built into Python.
#
# Every time your friend logs a completed shift, one row is added here.
# Over time this builds up the real spoilage data the app needs.

import sqlite3
import pandas as pd
from datetime  import datetime, date
from pathlib   import Path

DB_PATH = Path("momentum_data.db")


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE — creates the database and table if they don't exist yet.
# Safe to call every time the app starts.
# ══════════════════════════════════════════════════════════════════════════════
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS shift_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at           TEXT    DEFAULT (datetime('now','localtime')),

            -- When the shift happened
            shift_date          TEXT    NOT NULL,
            day_of_week         TEXT    NOT NULL,
            shift_type          TEXT    NOT NULL,    -- morning / afternoon / night / peak

            -- What was planned
            target_volume       INTEGER NOT NULL,    -- packages expected
            pallets_predicted   INTEGER NOT NULL,    -- what the app recommended
            carts_predicted     INTEGER NOT NULL,

            -- What actually happened
            pallets_ordered     INTEGER NOT NULL,    -- what was actually ordered
            pallets_spoilt      INTEGER NOT NULL,    -- how many were damaged/unusable
            carts_ordered       INTEGER NOT NULL,

            -- Shift context flags
            is_peak_season      INTEGER DEFAULT 0,
            has_oversized       INTEGER DEFAULT 0,

            -- Free-text notes (optional)
            notes               TEXT    DEFAULT ''
        )
        """)
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# LOG A SHIFT — call this when your friend submits the end-of-shift form.
# ══════════════════════════════════════════════════════════════════════════════
def log_shift(
    shift_date:       date,
    day_of_week:      str,
    shift_type:       str,
    target_volume:    int,
    pallets_predicted: int,
    carts_predicted:  int,
    pallets_ordered:  int,
    pallets_spoilt:   int,
    carts_ordered:    int,
    is_peak_season:   bool = False,
    has_oversized:    bool = False,
    notes:            str  = "",
) -> int:
    """Insert one shift record. Returns the new row id."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
        INSERT INTO shift_logs (
            shift_date, day_of_week, shift_type,
            target_volume, pallets_predicted, carts_predicted,
            pallets_ordered, pallets_spoilt, carts_ordered,
            is_peak_season, has_oversized, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            shift_date.isoformat(), day_of_week, shift_type,
            target_volume, pallets_predicted, carts_predicted,
            pallets_ordered, pallets_spoilt, carts_ordered,
            int(is_peak_season), int(has_oversized), notes,
        ))
        conn.commit()
        return cur.lastrowid


# ══════════════════════════════════════════════════════════════════════════════
# GET ALL LOGGED SHIFTS — returns a DataFrame of everything recorded so far.
# ══════════════════════════════════════════════════════════════════════════════
def get_history() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM shift_logs ORDER BY shift_date DESC, logged_at DESC",
            conn,
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE REAL SPOILAGE RATE
#
# Once there are enough logged shifts (at least 3), this replaces the
# manual slider with a rate calculated from actual recorded data.
#
# Formula:  spoilage_rate = total_spoilt / total_ordered  (rolling, all shifts)
# ══════════════════════════════════════════════════════════════════════════════
def get_real_spoilage_rate(min_shifts: int = 3) -> tuple[float | None, int]:
    """
    Returns (spoilage_rate, n_shifts).

    spoilage_rate is None if there are fewer than min_shifts logged shifts,
    meaning the app should fall back to the manual slider.
    """
    df = get_history()
    if df.empty or len(df) < min_shifts:
        return None, len(df) if not df.empty else 0

    total_ordered = df["pallets_ordered"].sum()
    total_spoilt  = df["pallets_spoilt"].sum()

    if total_ordered == 0:
        return None, len(df)

    rate = round(total_spoilt / total_ordered, 4)
    return rate, len(df)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS — used by the Shift Log tab to show quick KPIs.
# ══════════════════════════════════════════════════════════════════════════════
def get_summary_stats() -> dict:
    """Return a dict of headline numbers for display in the dashboard."""
    df = get_history()
    if df.empty:
        return {
            "total_shifts":        0,
            "total_volume":        0,
            "total_ordered":       0,
            "total_spoilt":        0,
            "spoilage_rate":       None,
            "avg_accuracy_pct":    None,   # how close predictions were to orders
        }

    total_ordered = int(df["pallets_ordered"].sum())
    total_spoilt  = int(df["pallets_spoilt"].sum())
    spoilage_rate = round(total_spoilt / total_ordered, 4) if total_ordered else None

    # Prediction accuracy: how close was the app's recommendation to what was ordered?
    df["accuracy"] = 1 - abs(
        df["pallets_predicted"] - df["pallets_ordered"]
    ) / df["pallets_ordered"].replace(0, 1)
    avg_accuracy = round(df["accuracy"].clip(0, 1).mean() * 100, 1)

    return {
        "total_shifts":     len(df),
        "total_volume":     int(df["target_volume"].sum()),
        "total_ordered":    total_ordered,
        "total_spoilt":     total_spoilt,
        "spoilage_rate":    spoilage_rate,
        "avg_accuracy_pct": avg_accuracy,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DELETE A ROW — lets your friend correct a mistake.
# ══════════════════════════════════════════════════════════════════════════════
def delete_shift(row_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM shift_logs WHERE id = ?", (row_id,))
        conn.commit()
