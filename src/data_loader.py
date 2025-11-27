
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


def list_tables(db_path: str) -> List[str]:
    """Return a list of table names from the SQLite DB."""
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )["name"].tolist()
    finally:
        conn.close()
    return tables


def load_table(db_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a table from SQLite. If table_name is None, pick the first table
    that has a 'label' column (per spec) or otherwise the first table.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)

    try:
        if table_name is None:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';", conn
            )["name"].tolist()
            if not tables:
                raise ValueError("No tables found in database.")

            # Try to find table with 'label' column
            chosen = None
            for t in tables:
                cols = pd.read_sql(f"PRAGMA table_info({t});", conn)["name"].tolist()
                if "label" in cols:
                    chosen = t
                    break
            if chosen is None:
                chosen = tables[0]
            table_name = chosen

        df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    finally:
        conn.close()

    return df


if __name__ == "__main__":
    db = "data/phishing.db"  # adjust if needed
    print("Tables:", list_tables(db))
    df = load_table(db)
    print("Loaded shape:", df.shape)
    print(df.head())
