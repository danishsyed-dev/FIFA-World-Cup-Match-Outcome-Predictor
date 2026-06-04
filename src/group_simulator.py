"""
group_simulator.py
------------------
Monte Carlo Group Stage Simulator for the FIFA World Cup 2026.

Simulates all group-stage matches using Elo-derived probabilities,
resolves standings with FIFA tiebreakers, and estimates each team's
probability of advancing (Top 2 automatic + best 3rd-place slots).

Usage:
    from src.group_simulator import simulate_all_groups, WC2026_GROUPS
    results = simulate_all_groups(elo_ratings_dict, n_sims=10000)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# FIFA World Cup 2026 — Official Group Draw (December 5, 2025)
# ══════════════════════════════════════════════════════════════════════════════

WC2026_GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Hardcoded fallback Elo ratings (approximate pre-tournament estimates)
# Used when the EloSystem hasn't processed a team from historical data.
FALLBACK_ELO: Dict[str, float] = {
    "Mexico": 1752, "South Africa": 1541, "South Korea": 1735, "Czechia": 1680,
    "Canada": 1680, "Bosnia and Herzegovina": 1620, "Qatar": 1530, "Switzerland": 1760,
    "Brazil": 1840, "Morocco": 1770, "Scotland": 1640, "Haiti": 1370,
    "USA": 1780, "Paraguay": 1640, "Australia": 1650, "Turkey": 1710,
    "Germany": 1850, "Curacao": 1350, "Ivory Coast": 1650, "Ecuador": 1700,
    "Netherlands": 1810, "Japan": 1780, "Tunisia": 1620, "Sweden": 1680,
    "Belgium": 1780, "Iran": 1640, "Egypt": 1610, "New Zealand": 1450,
    "Spain": 1870, "Uruguay": 1790, "Saudi Arabia": 1560, "Cape Verde": 1430,
    "France": 1860, "Senegal": 1700, "Norway": 1690, "Iraq": 1530,
    "Argentina": 1880, "Austria": 1700, "Algeria": 1620, "Jordan": 1530,
    "Portugal": 1830, "Colombia": 1750, "Uzbekistan": 1560, "DR Congo": 1520,
    "England": 1810, "Croatia": 1760, "Panama": 1560, "Ghana": 1570,
}


# ══════════════════════════════════════════════════════════════════════════════
# Elo Probability Calculations
# ══════════════════════════════════════════════════════════════════════════════

def win_draw_loss_prob(elo_a: float, elo_b: float) -> Tuple[float, float, float]:
    """
    Compute (P_win_A, P_draw, P_win_B) from Elo difference.

    Uses the standard Elo expected score, then splits the expected
    result into Win/Draw/Loss using an empirical draw factor that
    mimics real World Cup match distributions (~25% draws).
    """
    expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a

    # Draw factor: ~25% base draw rate, scaled by how close the teams are
    closeness = 1.0 - abs(expected_a - expected_b)
    draw_prob = 0.25 * closeness + 0.05  # range: ~5% to ~30%

    # Distribute remaining probability proportional to expected scores
    remaining = 1.0 - draw_prob
    win_a = remaining * expected_a
    win_b = remaining * expected_b

    return (win_a, draw_prob, win_b)


def simulate_match_score(elo_a: float, elo_b: float, rng: np.random.Generator) -> Tuple[int, int]:
    """
    Simulate a single match result (goals_a, goals_b) using Poisson distributions.

    Expected goals are derived from Elo ratings:
    - Average World Cup match: ~2.5 total goals
    - Stronger team gets a proportionally larger share
    """
    expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    # Total expected goals: ~2.5 for WC matches
    total_goals = 2.5
    lambda_a = total_goals * (0.3 + 0.4 * expected_a)  # range: ~0.7 to ~1.1
    lambda_b = total_goals * (0.3 + 0.4 * (1 - expected_a))

    goals_a = rng.poisson(lambda_a)
    goals_b = rng.poisson(lambda_b)

    return (int(goals_a), int(goals_b))


# ══════════════════════════════════════════════════════════════════════════════
# Group Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_group_once(
    teams: List[str],
    elo_ratings: Dict[str, float],
    rng: np.random.Generator,
) -> List[dict]:
    """
    Play all 6 round-robin matches in a 4-team group and return final standings.

    Each team plays 3 matches. FIFA tiebreakers:
    1. Points (3W, 1D, 0L)
    2. Goal difference
    3. Goals scored
    4. If still tied: random (simplified from FIFA's head-to-head)
    """
    # Initialize standing records
    records = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}

    # Play all 6 matches (round-robin of 4)
    for team_a, team_b in combinations(teams, 2):
        elo_a = elo_ratings.get(team_a, FALLBACK_ELO.get(team_a, 1500))
        elo_b = elo_ratings.get(team_b, FALLBACK_ELO.get(team_b, 1500))

        goals_a, goals_b = simulate_match_score(elo_a, elo_b, rng)

        records[team_a]["gf"] += goals_a
        records[team_a]["ga"] += goals_b
        records[team_b]["gf"] += goals_b
        records[team_b]["ga"] += goals_a

        if goals_a > goals_b:
            records[team_a]["pts"] += 3
        elif goals_a == goals_b:
            records[team_a]["pts"] += 1
            records[team_b]["pts"] += 1
        else:
            records[team_b]["pts"] += 3

    # Calculate goal difference
    for t in teams:
        records[t]["gd"] = records[t]["gf"] - records[t]["ga"]

    # Sort by: points desc, goal diff desc, goals scored desc, random tiebreak
    standings = sorted(
        teams,
        key=lambda t: (
            records[t]["pts"],
            records[t]["gd"],
            records[t]["gf"],
            rng.random(),  # random tiebreak
        ),
        reverse=True,
    )

    results = []
    for pos, team in enumerate(standings):
        results.append({
            "team": team,
            "position": pos + 1,
            "pts": records[team]["pts"],
            "gd": records[team]["gd"],
            "gf": records[team]["gf"],
            "ga": records[team]["ga"],
        })
    return results


def simulate_group(
    group_name: str,
    teams: List[str],
    elo_ratings: Dict[str, float],
    n_sims: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run n_sims Monte Carlo simulations of a single group.

    Returns a DataFrame with columns:
        team, group, elo, 1st, 2nd, 3rd, 4th, top2_pct, avg_pts, avg_gd
    """
    rng = np.random.default_rng(seed)

    # Accumulators
    position_counts = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}
    total_pts = {t: 0 for t in teams}
    total_gd = {t: 0 for t in teams}
    # Track 3rd-place records for cross-group comparison
    third_place_records: List[dict] = []

    for _ in range(n_sims):
        standings = _simulate_group_once(teams, elo_ratings, rng)
        for entry in standings:
            t = entry["team"]
            pos = entry["position"]
            position_counts[t][pos] += 1
            total_pts[t] += entry["pts"]
            total_gd[t] += entry["gd"]

            if pos == 3:
                third_place_records.append({
                    "team": t,
                    "group": group_name,
                    "pts": entry["pts"],
                    "gd": entry["gd"],
                    "gf": entry["gf"],
                })

    rows = []
    for t in teams:
        elo = elo_ratings.get(t, FALLBACK_ELO.get(t, 1500))
        rows.append({
            "team": t,
            "group": group_name,
            "elo": elo,
            "1st": position_counts[t][1] / n_sims * 100,
            "2nd": position_counts[t][2] / n_sims * 100,
            "3rd": position_counts[t][3] / n_sims * 100,
            "4th": position_counts[t][4] / n_sims * 100,
            "top2_pct": (position_counts[t][1] + position_counts[t][2]) / n_sims * 100,
            "avg_pts": total_pts[t] / n_sims,
            "avg_gd": total_gd[t] / n_sims,
        })

    return pd.DataFrame(rows), third_place_records


def _estimate_third_place_advancement(
    all_third_place_records: List[dict],
    n_sims: int,
    n_advancing: int = 8,
) -> Dict[str, float]:
    """
    Estimate how often each 3rd-place team is among the best N advancing.

    In the 2026 format, 8 of 12 third-place teams advance to Round of 32.
    """
    if not all_third_place_records:
        return {}

    df = pd.DataFrame(all_third_place_records)
    # For each simulation run, find the best 8 third-place teams
    # We track by grouping sim index: every 12 consecutive 3rd-place records = 1 sim
    # Actually since we collect per group, we need a different approach.

    # Simpler approach: for each team, what fraction of the time they finished 3rd
    # with enough points/GD to be in the top 8 third-place teams.
    # We'll use aggregate statistics as an approximation.

    team_advance_count: Dict[str, int] = {}
    team_third_count: Dict[str, int] = {}

    # Group records by team
    for _, row in df.iterrows():
        t = row["team"]
        team_third_count[t] = team_third_count.get(t, 0) + 1

    # Estimate via points/GD distribution:
    # A 3rd-place team with 4+ points almost always advances (top 8 out of 12)
    # 3 points + positive GD: usually advances
    # 3 points + negative GD: sometimes advances
    # < 3 points: rarely advances
    for _, row in df.iterrows():
        t = row["team"]
        pts = row["pts"]
        gd = row["gd"]

        advances = False
        if pts >= 5:
            advances = True
        elif pts == 4:
            advances = True  # 4 pts as 3rd = very strong
        elif pts == 3:
            advances = gd >= -1  # 3pts with decent GD usually enough
        elif pts == 2:
            advances = gd >= 2  # unlikely but possible

        if advances:
            team_advance_count[t] = team_advance_count.get(t, 0) + 1

    # Convert to percentages
    result = {}
    for t in team_third_count:
        third_count = team_third_count[t]
        advance_count = team_advance_count.get(t, 0)
        # This gives P(advance | finished 3rd) * P(finished 3rd)
        # We want total advance probability including top 2
        result[t] = advance_count / n_sims * 100  # as percentage

    return result


def simulate_all_groups(
    elo_ratings: Dict[str, float],
    n_sims: int = 10000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for all 12 World Cup 2026 groups.

    Returns a combined DataFrame with per-team advancement probabilities
    including the 3rd-place advancement estimate.
    """
    all_results = []
    all_third_records = []

    for group_name, teams in WC2026_GROUPS.items():
        group_df, third_records = simulate_group(
            group_name, teams, elo_ratings, n_sims, seed
        )
        all_results.append(group_df)
        all_third_records.extend(third_records)

    combined = pd.concat(all_results, ignore_index=True)

    # Estimate 3rd-place advancement
    third_advance = _estimate_third_place_advancement(
        all_third_records, n_sims, n_advancing=8
    )

    # Add total advancement probability (top2 + 3rd-place advancement)
    combined["third_advance_pct"] = combined["team"].map(
        lambda t: third_advance.get(t, 0.0)
    )
    combined["advance_pct"] = combined["top2_pct"] + combined["third_advance_pct"]
    # Cap at 100%
    combined["advance_pct"] = combined["advance_pct"].clip(upper=100.0)

    return combined


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running quick test with fallback Elo ratings...\n")
    results = simulate_all_groups(FALLBACK_ELO, n_sims=1000, seed=42)

    for group_name in WC2026_GROUPS:
        group = results[results["group"] == group_name].sort_values("advance_pct", ascending=False)
        print(f"=== GROUP {group_name} ===")
        for _, row in group.iterrows():
            print(f"  {row['team']:25s}  Advance: {row['advance_pct']:5.1f}%  "
                  f"(1st: {row['1st']:5.1f}%  2nd: {row['2nd']:5.1f}%  "
                  f"3rd: {row['3rd']:5.1f}%  4th: {row['4th']:5.1f}%)")
        print()
