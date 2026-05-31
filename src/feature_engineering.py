"""
feature_engineering.py
-----------------------
Creates all model features from the cleaned results DataFrame.

Features produced (per match row):
  - elo_difference          : home_elo - away_elo
  - home_recent_form        : points from last 5 home-team matches (W=3, D=1, L=0)
  - away_recent_form        : points from last 5 away-team matches
  - home_avg_goals_scored   : avg goals scored by home team in last 5 matches
  - away_avg_goals_scored   : avg goals scored by away team in last 5 matches
  - home_avg_goals_conceded : avg goals conceded by home team in last 5 matches
  - away_avg_goals_conceded : avg goals conceded by away team in last 5 matches
  - h2h_home_wins           : home team wins in last 10 H2H meetings
  - h2h_draws               : draws in last 10 H2H meetings
  - home_advantage          : 1 if not neutral, 0 if neutral
  - tournament_weight       : encoded tournament importance (0-4)
  - outcome                 : target variable (0=Away Win, 1=Draw, 2=Home Win)
"""

import pandas as pd
import numpy as np
from typing import Tuple

# ── Tournament importance mapping ────────────────────────────────────────────
TOURNAMENT_WEIGHT = {
    "friendly": 0,
    "qualifier": 1,
    "continental cup": 2,
    "copa america": 2,
    "uefa euro": 2,
    "africa cup of nations": 2,
    "afc asian cup": 2,
    "concacaf gold cup": 2,
    "confederations cup": 3,
    "fifa world cup": 4,
}

WINDOW = 5   # recent-form lookback
H2H_WINDOW = 10  # head-to-head lookback


def _encode_tournament(tournament: str) -> int:
    """Map tournament name to a numeric importance weight."""
    t_lower = tournament.lower()
    for key, val in TOURNAMENT_WEIGHT.items():
        if key in t_lower:
            return val
    return 1  # default: qualifier-level


def _recent_stats(
    df: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    window: int = WINDOW,
) -> Tuple[float, float, float]:
    """
    Return (form_score, avg_goals_scored, avg_goals_conceded)
    for `team` in its last `window` matches before `before_date`.

    Works regardless of whether the team was home or away.
    """
    home_mask = (df["home_team"] == team) & (df["date"] < before_date)
    away_mask = (df["away_team"] == team) & (df["date"] < before_date)

    home_matches = df[home_mask].tail(window)
    away_matches = df[away_mask].tail(window)

    records = []

    for _, r in home_matches.iterrows():
        if r["home_score"] > r["away_score"]:
            form = 3
        elif r["home_score"] == r["away_score"]:
            form = 1
        else:
            form = 0
        records.append((form, r["home_score"], r["away_score"]))

    for _, r in away_matches.iterrows():
        if r["away_score"] > r["home_score"]:
            form = 3
        elif r["home_score"] == r["away_score"]:
            form = 1
        else:
            form = 0
        records.append((form, r["away_score"], r["home_score"]))

    # Take the most recent `window` overall
    records = sorted(records, key=lambda x: x[0])[-window:]

    if not records:
        return 0.0, 0.0, 0.0

    form_scores, goals_scored, goals_conceded = zip(*records)
    return (
        float(np.mean(form_scores)),
        float(np.mean(goals_scored)),
        float(np.mean(goals_conceded)),
    )


def _h2h_stats(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    before_date: pd.Timestamp,
    window: int = H2H_WINDOW,
) -> Tuple[int, int]:
    """
    Return (home_team_wins, draws) in the last `window` meetings
    between home_team and away_team before `before_date`.
    """
    mask = (
        (
            ((df["home_team"] == home_team) & (df["away_team"] == away_team))
            | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        )
        & (df["date"] < before_date)
    )
    meetings = df[mask].tail(window)

    if meetings.empty:
        return 0, 0

    def classify(r) -> str:
        if r["home_team"] == home_team:
            if r["home_score"] > r["away_score"]:
                return "home_win"
            elif r["home_score"] == r["away_score"]:
                return "draw"
            return "away_win"
        else:
            if r["away_score"] > r["home_score"]:
                return "home_win"
            elif r["home_score"] == r["away_score"]:
                return "draw"
            return "away_win"

    results = meetings.apply(classify, axis=1)
    home_wins = int((results == "home_win").sum())
    draws = int((results == "draw").sum())
    return home_wins, draws


def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Main feature engineering function.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned results dataframe with columns:
        date, home_team, away_team, home_score, away_score,
        tournament, neutral, home_elo, away_elo, outcome

    Returns
    -------
    pd.DataFrame with engineered feature columns appended.
    """
    df = df.sort_values("date").reset_index(drop=True)

    features = {
        "elo_difference": [],
        "home_recent_form": [],
        "away_recent_form": [],
        "home_avg_goals_scored": [],
        "away_avg_goals_scored": [],
        "home_avg_goals_conceded": [],
        "away_avg_goals_conceded": [],
        "h2h_home_wins": [],
        "h2h_draws": [],
        "home_advantage": [],
        "tournament_weight": [],
    }

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if verbose and i % 5000 == 0:
            print(f"  [feature_engineering] {i:,}/{total:,} rows processed...")

        ht, at, date = row["home_team"], row["away_team"], row["date"]

        # Elo difference
        features["elo_difference"].append(row["home_elo"] - row["away_elo"])

        # Recent form
        hf, hgs, hgc = _recent_stats(df, ht, date)
        af, ags, agc = _recent_stats(df, at, date)
        features["home_recent_form"].append(hf)
        features["away_recent_form"].append(af)
        features["home_avg_goals_scored"].append(hgs)
        features["away_avg_goals_scored"].append(ags)
        features["home_avg_goals_conceded"].append(hgc)
        features["away_avg_goals_conceded"].append(agc)

        # Head-to-head
        hw, draws = _h2h_stats(df, ht, at, date)
        features["h2h_home_wins"].append(hw)
        features["h2h_draws"].append(draws)

        # Home advantage
        features["home_advantage"].append(0 if row.get("neutral", False) else 1)

        # Tournament weight
        features["tournament_weight"].append(
            _encode_tournament(str(row.get("tournament", "Friendly")))
        )

    feature_df = pd.DataFrame(features, index=df.index)
    result = pd.concat([df, feature_df], axis=1)

    if verbose:
        print(f"[feature_engineering] Done. Shape: {result.shape}")

    return result


def get_feature_columns() -> list:
    """Return the ordered list of feature columns used by the model."""
    return [
        "elo_difference",
        "home_recent_form",
        "away_recent_form",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "home_avg_goals_conceded",
        "away_avg_goals_conceded",
        "h2h_home_wins",
        "h2h_draws",
        "home_advantage",
        "tournament_weight",
    ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.data_loader import load_all
    df = load_all()
    df_feats = build_features(df.head(500))
    print(df_feats[get_feature_columns() + ["outcome"]].head())
