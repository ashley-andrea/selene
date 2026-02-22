"""
Pill Database — loads the pill CSV at import time and provides query utilities.

The CSV is the single source of truth for available oral contraceptives.
Loaded once into a Pandas DataFrame and accessed via helper functions.

Schema: set_id, brand_name, generic_name, manufacturer_name, product_ndc, substance_name,
        adverse_reactions, warnings, warnings_and_cautions, boxed_warning, contraindications,
        drug_interactions, clinical_pharmacology, indications_and_usage, 
        dosage_and_administration, description
"""

import os
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PILLS_CSV = _DATA_DIR / "pills.csv"

# Load once at module import — kept in memory as a DataFrame
_pills_df: pd.DataFrame = pd.read_csv(_PILLS_CSV)


def get_all_pills() -> pd.DataFrame:
    """Return the full pill DataFrame (read-only reference)."""
    return _pills_df


def get_pill_ids() -> list[str]:
    """Return all pill IDs (set_id) as a list."""
    return _pills_df["set_id"].tolist()


def get_pill_by_id(pill_id: str) -> dict | None:
    """
    Return a single pill record as a dict, or None if not found.
    pill_id corresponds to the set_id field in pills.csv.
    """
    row = _pills_df[_pills_df["set_id"] == pill_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_pills_by_ids(pill_ids: list[str]) -> pd.DataFrame:
    """Return a filtered DataFrame for the given pill IDs (set_ids)."""
    return _pills_df[_pills_df["set_id"].isin(pill_ids)]


def query(
    substance_filter: str | None = None,
    exclude_ids: list[str] | None = None,
) -> list[str]:
    """
    Query the pill database with optional filters.
    Returns a list of pill IDs (set_ids) matching all criteria.
    
    Args:
        substance_filter: Filter by substring match in substance_name (e.g., "LEVONORGESTREL")
        exclude_ids: List of set_ids to exclude from results
    """
    df = _pills_df.copy()

    if substance_filter:
        df = df[df["substance_name"].str.contains(substance_filter, case=False, na=False)]

    if exclude_ids:
        df = df[~df["set_id"].isin(exclude_ids)]

    return df["set_id"].tolist()
