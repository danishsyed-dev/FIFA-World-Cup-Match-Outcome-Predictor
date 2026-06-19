"""
data_loader.py
--------------
Loads and merges the raw CSV datasets.

Expected files in data/:
  - results.csv   (International football results 1872-2026)
  - elo.csv       (International Elo ratings per team per date)
"""

import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_results(filepath: str = None) -> pd.DataFrame:
    """Load the international football results dataset."""
    path = filepath or DATA_DIR / "results.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    print(f"[data_loader] Loaded results: {len(df):,} rows")
    return df


def load_elo(filepath: str = None) -> pd.DataFrame:
    """Load the Elo ratings dataset."""
    path = filepath or DATA_DIR / "elo.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    # Normalise column names (different Kaggle datasets vary)
    df.columns = [c.lower().strip() for c in df.columns]
    if "rating" in df.columns and "elo" not in df.columns:
        df = df.rename(columns={"rating": "elo"})
    print(f"[data_loader] Loaded elo: {len(df):,} rows")
    return df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the results dataframe:
    - Drop rows with missing scores
    - Standardise team name casing
    - Add an 'outcome' column (Home Win=2, Draw=1, Away Win=0)
    - Add 'home_goals' and 'away_goals' aliases
    """
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Outcome from HOME TEAM perspective
    df["outcome"] = df.apply(
        lambda r: 2 if r["home_score"] > r["away_score"]
        else (1 if r["home_score"] == r["away_score"] else 0),
        axis=1,
    )
    print(f"[data_loader] Cleaned results: {len(df):,} rows")
    return df


def merge_elo(results: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Elo ratings into the results dataframe.

    For each match, we find the most recent Elo rating for each team
    that is *before or on* the match date using a merge_asof strategy.
    """
    # Ensure elo has required columns
    required = {"team", "date", "elo"}
    missing = required - set(elo.columns)
    if missing:
        print(f"[data_loader] WARNING: Elo dataset missing columns {missing}. Skipping Elo merge.")
        results["home_elo"] = 1500.0
        results["away_elo"] = 1500.0
        return results

    print("[data_loader] Merging Elo ratings (using optimized merge_asof)...")
    results = results.sort_values("date").reset_index(drop=True)
    elo = elo.sort_values("date").reset_index(drop=True)

    # Merge for home_team
    home_merged = pd.merge_asof(
        results,
        elo,
        on="date",
        left_by="home_team",
        right_by="team",
        direction="backward"
    )
    results["home_elo"] = home_merged["elo"].fillna(1500.0)

    # Merge for away_team
    away_merged = pd.merge_asof(
        results,
        elo,
        on="date",
        left_by="away_team",
        right_by="team",
        direction="backward"
    )
    results["away_elo"] = away_merged["elo"].fillna(1500.0)

    print("[data_loader] Elo merge complete.")
    return results


def load_all() -> pd.DataFrame:
    """Main entry point: load, clean, and merge all data."""
    results = load_results()
    results = clean_results(results)

    elo_path = DATA_DIR / "elo.csv"
    if elo_path.exists():
        elo = load_elo()
        results = merge_elo(results, elo)
    else:
        print("[data_loader] elo.csv not found — defaulting Elo to 1500.")
        results["home_elo"] = 1500.0
        results["away_elo"] = 1500.0

    return results


if __name__ == "__main__":
    df = load_all()
    print(df.head())
    print(df.dtypes)
