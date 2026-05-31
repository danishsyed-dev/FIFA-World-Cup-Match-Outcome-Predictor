"""
elo_calculator.py
-----------------
Custom Elo rating calculator built from scratch.

Usage:
    from src.elo_calculator import EloSystem
    elo = EloSystem()
    ratings = elo.calculate(results_df)
"""

import pandas as pd
import numpy as np
from typing import Dict


class EloSystem:
    """
    Implements the World Football Elo Rating system.

    Key parameters:
        K       — base weight constant (match importance)
        home_advantage — Elo points added to home team
        initial_rating — default starting rating for new teams
    """

    TOURNAMENT_K = {
        "FIFA World Cup": 60,
        "Confederations Cup": 50,
        "Copa América": 50,
        "UEFA Euro": 50,
        "AFC Asian Cup": 50,
        "Africa Cup of Nations": 50,
        "CONCACAF Gold Cup": 40,
        "Friendly": 20,
    }
    DEFAULT_K = 30  # qualifiers and other tournaments

    def __init__(self, home_advantage: float = 100.0, initial_rating: float = 1500.0):
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _get_k(self, tournament: str) -> float:
        for key, k in self.TOURNAMENT_K.items():
            if key.lower() in tournament.lower():
                return k
        return self.DEFAULT_K

    def _expected(self, rating_a: float, rating_b: float) -> float:
        """Expected score for team A against team B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _actual_score(self, goals_a: int, goals_b: int) -> float:
        """Actual score: 1 = win, 0.5 = draw, 0 = loss."""
        if goals_a > goals_b:
            return 1.0
        if goals_a == goals_b:
            return 0.5
        return 0.0

    def _goal_index(self, goal_diff: int) -> float:
        """Goal difference multiplier (standard WElo formula)."""
        if goal_diff <= 1:
            return 1.0
        if goal_diff == 2:
            return 1.5
        return (11.0 + goal_diff) / 8.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> None:
        """Update Elo ratings for one match."""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        # Apply home advantage to home team (unless neutral venue)
        home_rating_adj = home_rating + (0 if neutral else self.home_advantage)

        exp_home = self._expected(home_rating_adj, away_rating)
        exp_away = 1.0 - exp_home

        actual_home = self._actual_score(home_goals, away_goals)
        actual_away = 1.0 - actual_home

        k = self._get_k(tournament)
        goal_diff = abs(home_goals - away_goals)
        gd_mult = self._goal_index(goal_diff)

        # Update ratings
        self.ratings[home_team] = home_rating + k * gd_mult * (actual_home - exp_home)
        self.ratings[away_team] = away_rating + k * gd_mult * (actual_away - exp_away)

    def calculate(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically and return a DataFrame
        with home_elo and away_elo columns representing the PRE-match ratings.

        Parameters
        ----------
        results : pd.DataFrame
            Must contain columns: date, home_team, away_team,
                                  home_score, away_score, tournament, neutral
        Returns
        -------
        pd.DataFrame  — results with added home_elo and away_elo columns
        """
        self.ratings = {}  # reset
        results = results.sort_values("date").reset_index(drop=True)

        home_elos, away_elos = [], []

        for _, row in results.iterrows():
            ht = row["home_team"]
            at = row["away_team"]
            home_elos.append(self.get_rating(ht))
            away_elos.append(self.get_rating(at))

            self.update(
                home_team=ht,
                away_team=at,
                home_goals=int(row["home_score"]),
                away_goals=int(row["away_score"]),
                tournament=row.get("tournament", "Friendly"),
                neutral=bool(row.get("neutral", False)),
            )

        results = results.copy()
        results["home_elo"] = home_elos
        results["away_elo"] = away_elos
        print(f"[elo_calculator] Calculated Elo for {len(results):,} matches.")
        return results

    def top_teams(self, n: int = 20) -> pd.DataFrame:
        """Return the top N teams by current Elo rating."""
        df = pd.DataFrame(
            self.ratings.items(), columns=["team", "elo"]
        ).sort_values("elo", ascending=False).head(n).reset_index(drop=True)
        df.index = df.index + 1
        return df


if __name__ == "__main__":
    # Quick smoke-test
    test_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-11-20", "2022-11-21", "2022-11-22"]),
            "home_team": ["Qatar", "England", "Argentina"],
            "away_team": ["Ecuador", "Iran", "Saudi Arabia"],
            "home_score": [0, 6, 1],
            "away_score": [2, 2, 2],
            "tournament": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup"],
            "neutral": [False, False, False],
        }
    )
    elo = EloSystem()
    result = elo.calculate(test_data)
    print(result[["home_team", "away_team", "home_elo", "away_elo"]])
    print("\nTop teams:")
    print(elo.top_teams(5))
