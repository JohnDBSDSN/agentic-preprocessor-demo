"""
Utils for Agentic Preprocessor Demo
-----------------------------------

This module provides helper functions used in the preprocessing pipeline:

1. load_csv(path)         → Safely load a CSV into pandas.
2. brief_schema(df)       → Summarize dataset (rows, cols, dtypes, null counts, duplicates, categorical candidates).
3. print_section(title)   → Nicely format section headers for console output.

Used by: agentic_preprocessor.py
"""

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(path)
        print(f" Loaded file: {path} (rows={len(df)}, cols={len(df.columns)})")
        return df
    except Exception as e:
        raise FileNotFoundError(f" Could not load {path}: {e}")


def brief_schema(df: pd.DataFrame) -> dict:
    """Return a brief schema summary of the dataset."""
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "dup_count": df.duplicated().sum(),
        "categorical_candidates": [c for c in df.columns if df[c].dtype == "object"]
    }
    return summary


def print_section(title: str):
    """Print a section header for console clarity."""
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
