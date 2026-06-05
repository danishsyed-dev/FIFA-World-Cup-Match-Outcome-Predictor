"""
predict.py
----------
Prediction interface for the trained model.

Usage (CLI):
    python src/predict.py --team-a "Brazil" --team-b "Argentina" --neutral

Usage (Python API):
    from src.predict import Predictor
    p = Predictor()
    result = p.predict("Brazil", "Argentina", neutral=True)
    print(result)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_all, load_elo, DATA_DIR
from src.elo_calculator import EloSystem
from src.feature_engineering import (
    _recent_stats,
    _h2h_stats,
    _encode_tournament,
    get_feature_columns,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}
TOURNAMENT_OPTIONS = ["Friendly", "Qualifier", "Continental Cup", "FIFA World Cup"]


class Predictor:
    """
    Loads the trained model and Elo system, then makes predictions
    for any Team A vs Team B fixture.
    """

    def __init__(self, model_path: str = None):
        mp = Path(model_path) if model_path else MODELS_DIR / "best_model.pkl"
        if not mp.exists():
            raise FileNotFoundError(
                f"No trained model found at {mp}.\n"
                "Please run `python src/train.py` first."
            )
        self.model = joblib.load(mp)
        self.feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

        print("[predict] Loading historical data for feature computation...")
        raw = load_all()

        print("[predict] Rebuilding Elo ratings...")
        self.elo_system = EloSystem()
        self.df = self.elo_system.calculate(raw)

        # Override with latest ratings from elo.csv if present
        elo_path = DATA_DIR / "elo.csv"
        if elo_path.exists():
            try:
                elo_df = load_elo()
                # Find the latest rating for each team
                latest_ratings = elo_df.sort_values("date").groupby("team").last().reset_index()
                for _, row in latest_ratings.iterrows():
                    team_name = str(row["team"]).strip()
                    self.elo_system.ratings[team_name] = float(row["elo"])
                print(f"[predict] Overrode active Elo ratings with latest values from elo.csv (e.g. Portugal: {self.elo_system.get_rating('Portugal'):.0f}).")
            except Exception as e:
                print(f"[predict] Error overriding Elo ratings from elo.csv: {e}")

        print(f"[predict] Ready. {len(self.df):,} historical matches loaded.")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament: str = "Friendly",
    ) -> np.ndarray:
        """Build a single feature vector for the requested fixture."""
        today = pd.Timestamp.now()

        home_elo = self.elo_system.get_rating(home_team)
        away_elo = self.elo_system.get_rating(away_team)

        hf, hgs, hgc = _recent_stats(self.df, home_team, today)
        af, ags, agc = _recent_stats(self.df, away_team, today)
        hw, draws = _h2h_stats(self.df, home_team, away_team, today)
        t_weight = _encode_tournament(tournament)

        feature_vector = [
            home_elo - away_elo,   # elo_difference
            hf,                    # home_recent_form
            af,                    # away_recent_form
            hgs,                   # home_avg_goals_scored
            ags,                   # away_avg_goals_scored
            hgc,                   # home_avg_goals_conceded
            agc,                   # away_avg_goals_conceded
            hw,                    # h2h_home_wins
            draws,                 # h2h_draws
            0 if neutral else 1,   # home_advantage
            t_weight,              # tournament_weight
        ]
        return np.array(feature_vector).reshape(1, -1)

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        tournament: str = "Friendly",
    ) -> dict:
        """
        Predict the outcome of home_team vs away_team.

        Returns
        -------
        dict with keys:
            home_team, away_team,
            home_win_prob, draw_prob, away_win_prob,
            predicted_outcome, confidence,
            home_elo, away_elo
        """
        X = self._build_feature_vector(home_team, away_team, neutral, tournament)
        probs = self.model.predict_proba(X)[0]  # [away_win, draw, home_win]

        # Map to class indices used at training time
        classes = self.model.classes_
        prob_map = {cls: float(p) for cls, p in zip(classes, probs)}

        away_win_prob = prob_map.get(0, 0.0)
        draw_prob     = prob_map.get(1, 0.0)
        home_win_prob = prob_map.get(2, 0.0)

        best_class = int(classes[np.argmax(probs)])
        confidence = float(np.max(probs))

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": home_win_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_win_prob,
            "predicted_outcome": LABEL_MAP[best_class],
            "confidence": confidence,
            "home_elo": self.elo_system.get_rating(home_team),
            "away_elo": self.elo_system.get_rating(away_team),
        }

    def predict_batch(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction from a DataFrame with columns:
            home_team, away_team, neutral (bool), tournament (str)
        """
        rows = []
        for _, row in fixtures.iterrows():
            r = self.predict(
                row["home_team"],
                row["away_team"],
                neutral=bool(row.get("neutral", False)),
                tournament=str(row.get("tournament", "Friendly")),
            )
            rows.append(r)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_prediction(self, result: dict) -> None:
        ht = result["home_team"]
        at = result["away_team"]
        hw = result["home_win_prob"] * 100
        dr = result["draw_prob"] * 100
        aw = result["away_win_prob"] * 100

        def bar(pct, width=25):
            filled = int(round(pct / 100 * width))
            return "█" * filled + "░" * (width - filled)

        print()
        print("=" * 55)
        print(f"  ⚽  {ht}  vs  {at}")
        print("=" * 55)
        print(f"  {ht} Win : {hw:5.1f}%  {bar(hw)}")
        print(f"  Draw     : {dr:5.1f}%  {bar(dr)}")
        print(f"  {at} Win : {aw:5.1f}%  {bar(aw)}")
        print()
        print(f"  🏆 Predicted: {result['predicted_outcome']}")
        print(f"  📊 Confidence: {result['confidence']*100:.1f}%")
        print(f"  🔢 Elo  — {ht}: {result['home_elo']:.0f}  |  {at}: {result['away_elo']:.0f}")
        print("=" * 55)
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FIFA Match Outcome Predictor")
    parser.add_argument("--team-a", required=True, help="Home team name")
    parser.add_argument("--team-b", required=True, help="Away team name")
    parser.add_argument("--neutral", action="store_true", help="Neutral venue flag")
    parser.add_argument(
        "--tournament",
        default="Friendly",
        choices=TOURNAMENT_OPTIONS,
        help="Tournament type",
    )
    parser.add_argument("--input", help="CSV file with batch fixtures")
    parser.add_argument("--output", help="CSV file to save batch predictions")

    args = parser.parse_args()

    predictor = Predictor()

    if args.input:
        fixtures = pd.read_csv(args.input)
        predictions = predictor.predict_batch(fixtures)
        output_path = args.output or "predictions.csv"
        predictions.to_csv(output_path, index=False)
        print(f"Batch predictions saved → {output_path}")
    else:
        result = predictor.predict(
            home_team=args.team_a,
            away_team=args.team_b,
            neutral=args.neutral,
            tournament=args.tournament,
        )
        predictor.print_prediction(result)


if __name__ == "__main__":
    main()
