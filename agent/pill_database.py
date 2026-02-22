"""
Pill Database — loads the pill reference CSV at import time and provides query utilities.

The CSV is the single source of truth for available oral contraceptives.
Loaded once into a Pandas DataFrame and accessed via helper functions.

Source: data/pill_reference_db.csv (built by drugs/build_drug_reference.py)
Primary key: combo_id  (e.g. "EE20_LNG90", "NET_PO_350")
Key columns: combo_id, pill_type, estrogen, estrogen_dose_mcg, progestin,
             progestin_dose_mg, vte_risk_class, label_* text fields, faers_*, kw_*
"""

import os
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PILLS_CSV = _DATA_DIR / "pill_reference_db.csv"

# Load once at module import — kept in memory as a DataFrame
_pills_df: pd.DataFrame = pd.read_csv(_PILLS_CSV)


def get_all_pills() -> pd.DataFrame:
    """Return the full pill DataFrame (read-only reference)."""
    return _pills_df


def get_pill_ids() -> list[str]:
    """Return all pill combo_ids as a list."""
    return _pills_df["combo_id"].tolist()


def get_pill_by_id(pill_id: str) -> dict | None:
    """
    Return a single pill record as a dict, or None if not found.
    pill_id corresponds to the combo_id field in pill_reference_db.csv.
    """
    row = _pills_df[_pills_df["combo_id"] == pill_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_pills_by_ids(pill_ids: list[str]) -> pd.DataFrame:
    """Return a filtered DataFrame for the given pill combo_ids."""
    return _pills_df[_pills_df["combo_id"].isin(pill_ids)]


def query(
    pill_type_filter: str | None = None,
    exclude_ids: list[str] | None = None,
) -> list[str]:
    """
    Query the pill database with optional filters.
    Returns a list of combo_ids matching all criteria.

    Args:
        pill_type_filter: Filter by pill_type prefix (e.g. "combined" matches all combined types,
                          "progestin_only" matches the minipill)
        exclude_ids: List of combo_ids to exclude from results
    """
    df = _pills_df.copy()

    if pill_type_filter:
        df = df[df["pill_type"].str.startswith(pill_type_filter, na=False)]

    if exclude_ids:
        df = df[~df["combo_id"].isin(exclude_ids)]

    return df["combo_id"].tolist()
