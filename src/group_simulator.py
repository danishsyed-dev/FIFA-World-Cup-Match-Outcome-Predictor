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


def load_real_played_matches(knockout: bool = False) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    Load all played FIFA World Cup 2026 matches from results.csv.
    Returns a dictionary mapping (team_a, team_b) -> (goals_a, goals_b).
    """
    from pathlib import Path
    
    # Try multiple paths to find results.csv
    possible_paths = [
        Path(__file__).resolve().parent.parent / "data" / "results.csv",
        Path("data/results.csv"),
        Path("../data/results.csv"),
    ]
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
            
    if not csv_path:
        return {}
        
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
        
    # Filter for played 2026 World Cup matches
    df_wc = df[
        (df["tournament"] == "FIFA World Cup") & 
        (df["date"].astype(str).str.startswith("2026")) &
        df["home_score"].notna() &
        df["away_score"].notna()
    ]
    if knockout:
        wc_matches = df_wc[df_wc["date"] >= "2026-06-29"]
    else:
        wc_matches = df_wc[df_wc["date"] < "2026-06-29"]
    
    def clean_name_for_simulator(team_name: str) -> str:
        t_clean = team_name.strip()
        if "Cura" in t_clean:
            return "Curacao"
        elif t_clean == "United States":
            return "USA"
        elif t_clean == "Czech Republic":
            return "Czechia"
        elif t_clean == "Congo DR" or t_clean == "DR Congo":
            return "DR Congo"
        elif t_clean == "Bosnia-Herz" or t_clean == "Bosnia and Herzegovina":
            return "Bosnia and Herzegovina"
        return t_clean
    
    played = {}
    for _, row in wc_matches.iterrows():
        try:
            ht = clean_name_for_simulator(str(row["home_team"]))
            at = clean_name_for_simulator(str(row["away_team"]))
            hs = int(row["home_score"])
            as_ = int(row["away_score"])
            
            played[(ht, at)] = (hs, as_)
            played[(at, ht)] = (as_, hs)
        except (ValueError, TypeError):
            continue
            
    return played


def load_real_shootouts() -> Dict[Tuple[str, str], str]:
    from pathlib import Path
    
    possible_paths = [
        Path(__file__).resolve().parent.parent / "data" / "shootouts.csv",
        Path("data/shootouts.csv"),
        Path("../data/shootouts.csv"),
    ]
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
            
    if not csv_path:
        return {}
        
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
        
    def clean_name_for_simulator(team_name: str) -> str:
        t_clean = team_name.strip()
        if "Cura" in t_clean:
            return "Curacao"
        elif t_clean == "United States":
            return "USA"
        elif t_clean == "Czech Republic":
            return "Czechia"
        elif t_clean == "Congo DR" or t_clean == "DR Congo":
            return "DR Congo"
        elif t_clean == "Bosnia-Herz" or t_clean == "Bosnia and Herzegovina":
            return "Bosnia and Herzegovina"
        return t_clean

    shootouts = {}
    for _, row in df.iterrows():
        try:
            ht = clean_name_for_simulator(str(row["home_team"]))
            at = clean_name_for_simulator(str(row["away_team"]))
            winner = clean_name_for_simulator(str(row["winner"]))
            shootouts[(ht, at)] = winner
            shootouts[(at, ht)] = winner
        except Exception:
            continue
    return shootouts


# ══════════════════════════════════════════════════════════════════════════════
# Group Simulation Engine
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_group_once(
    teams: List[str],
    elo_ratings: Dict[str, float],
    rng: np.random.Generator,
    played_matches: Dict[Tuple[str, str], Tuple[int, int]] = None,
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
        if played_matches and (team_a, team_b) in played_matches:
            goals_a, goals_b = played_matches[(team_a, team_b)]
        else:
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

    played_matches = load_real_played_matches()

    for _ in range(n_sims):
        standings = _simulate_group_once(teams, elo_ratings, rng, played_matches)
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


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    elo_ratings: Dict[str, float],
    rng: np.random.Generator,
    played_matches: Dict[Tuple[str, str], Tuple[int, int]] = None,
    real_shootouts: Dict[Tuple[str, str], str] = None,
) -> dict:
    """
    Simulates a single knockout match. If it's a draw after 90 mins,
    it simulates extra time and, if needed, a penalty shootout (50/50 split).
    Supports looking up real-world scores if played_matches is provided.
    """
    if played_matches and (team_a, team_b) in played_matches:
        goals_a, goals_b = played_matches[(team_a, team_b)]
        if goals_a > goals_b:
            winner = team_a
        elif goals_a < goals_b:
            winner = team_b
        else:
            winner = None
            if real_shootouts:
                winner = real_shootouts.get((team_a, team_b))
            if not winner:
                winner = team_a if rng.random() < 0.5 else team_b
        return {
            "team_a": team_a,
            "team_b": team_b,
            "score_a": goals_a,
            "score_b": goals_b,
            "winner": winner,
            "extra_time": goals_a == goals_b,
            "penalties": goals_a == goals_b,
        }

    elo_a = elo_ratings.get(team_a, FALLBACK_ELO.get(team_a, 1500))
    elo_b = elo_ratings.get(team_b, FALLBACK_ELO.get(team_b, 1500))
    
    goals_a, goals_b = simulate_match_score(elo_a, elo_b, rng)
    
    extra_time = False
    penalties = False
    
    score_a = goals_a
    score_b = goals_b
    
    if score_a == score_b:
        extra_time = True
        # Extra time (30 mins, total expected goals ~0.8)
        expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        lambda_a = 0.8 * (0.3 + 0.4 * expected_a)
        lambda_b = 0.8 * (0.3 + 0.4 * (1 - expected_a))
        et_a = rng.poisson(lambda_a)
        et_b = rng.poisson(lambda_b)
        
        score_a += et_a
        score_b += et_b
        
        if score_a == score_b:
            penalties = True
            
    if penalties:
        # Penalty shootout is simulated as a coin flip
        winner = team_a if rng.random() < 0.5 else team_b
    else:
        winner = team_a if score_a > score_b else team_b
        
    return {
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
        "extra_time": extra_time,
        "penalties": penalties,
    }


def allocate_third_places(third_places: List[dict]) -> Dict[str, dict]:
    """
    Allocate the 8 best third-placed teams to the 8 slots in the Round of 32.
    Obeys FIFA allocation rules and prevents group-mate rematches.
    """
    advancing_teams = third_places[:8]
    slots = {
        "1E": ["A", "B", "C", "D", "F"],
        "1I": ["C", "D", "F", "G", "H"],
        "1A": ["C", "E", "F", "H", "I"],
        "1L": ["E", "H", "I", "J", "K"],
        "1G": ["A", "E", "H", "I", "J"],
        "1D": ["B", "E", "F", "I", "J"],
        "1B": ["E", "F", "G", "I", "J"],
        "1K": ["D", "E", "I", "J", "L"],
    }
    assignment = {}
    used_teams = set()
    slot_keys = list(slots.keys())
    
    def backtrack(slot_idx):
        if slot_idx == len(slot_keys):
            return True
        slot = slot_keys[slot_idx]
        eligible_groups = slots[slot]
        for team_info in advancing_teams:
            team_name = team_info["team"]
            team_group = team_info["group"]
            if team_name not in used_teams and team_group in eligible_groups and team_group != slot[1:]:
                assignment[slot] = team_info
                used_teams.add(team_name)
                if backtrack(slot_idx + 1):
                    return True
                used_teams.remove(team_name)
                del assignment[slot]
        return False
        
    if backtrack(0):
        return assignment
    else:
        # Fallback allocation if backtrack fails (highly unlikely)
        return {slot_keys[i]: advancing_teams[i] for i in range(8)}


def simulate_single_tournament(
    elo_ratings: Dict[str, float],
    rng: np.random.Generator,
    group_played_matches: Dict[Tuple[str, str], Tuple[int, int]] = None,
    knockout_played_matches: Dict[Tuple[str, str], Tuple[int, int]] = None,
    real_shootouts: Dict[Tuple[str, str], str] = None,
) -> dict:
    """
    Simulates one full World Cup 2026 (Group Stage + 32-team Knockout Stage).
    Returns the furthest stage reached by each team and the full bracket results.
    """
    if group_played_matches is None:
        group_played_matches = load_real_played_matches(knockout=False)
    if knockout_played_matches is None:
        knockout_played_matches = load_real_played_matches(knockout=True)
    if real_shootouts is None:
        real_shootouts = load_real_shootouts()

    import json
    from pathlib import Path
    possible_json_paths = [
        Path(__file__).resolve().parent.parent / "data" / "knockout_bracket.json",
        Path("data/knockout_bracket.json"),
        Path("../data/knockout_bracket.json"),
    ]
    json_path = None
    for p in possible_json_paths:
        if p.exists():
            json_path = p
            break
    
    bracket_struct = None
    if json_path:
        try:
            with open(json_path, 'r') as f:
                bracket_struct = json.load(f)
        except Exception:
            pass

    group_stage_completed = (group_played_matches is not None and len(group_played_matches) >= 144) and (bracket_struct is not None)

    if group_stage_completed:
        # Use real-world R32 matchups directly
        def normalize_name(n):
            n_strip = str(n).strip()
            if n_strip == "Congo DR":
                return "DR Congo"
            elif n_strip == "Bosnia-Herz":
                return "Bosnia and Herzegovina"
            return n_strip
            
        r32_matches = []
        for m in bracket_struct["r32"]:
            t1 = normalize_name(m["team1"])
            t2 = normalize_name(m["team2"])
            res = simulate_knockout_match(t1, t2, elo_ratings, rng, knockout_played_matches, real_shootouts)
            r32_matches.append(res)
    else:
        # 1. Simulate Group Stage
        group_standings = {}
        third_placed = []
        
        for group_name, teams in WC2026_GROUPS.items():
            standings = _simulate_group_once(teams, elo_ratings, rng, group_played_matches)
            group_standings[group_name] = standings
            for entry in standings:
                if entry["position"] == 3:
                    third_placed.append({
                        "team": entry["team"],
                        "group": group_name,
                        "pts": entry["pts"],
                        "gd": entry["gd"],
                        "gf": entry["gf"],
                    })
                    
        # Rank 3rd place teams
        third_placed = sorted(
            third_placed,
            key=lambda x: (x["pts"], x["gd"], x["gf"], rng.random()),
            reverse=True,
        )
        
        third_place_assignments = allocate_third_places(third_placed)
        
        # Helper to resolve qualifiers
        def get_team(rank_type: str) -> str:
            if rank_type.startswith("3_"):
                slot = rank_type[2:]
                return third_place_assignments[slot]["team"]
            pos = int(rank_type[0])
            group = rank_type[1]
            return group_standings[group][pos - 1]["team"]

        # 2. Round of 32 Pairings (predetermined by FIFA)
        r32_pairings = [
            ("2A", "2B"),       # Match 1
            ("1C", "2F"),       # Match 2
            ("1E", "3_1E"),     # Match 3
            ("1F", "2C"),       # Match 4
            ("2E", "2I"),       # Match 5
            ("1I", "3_1I"),     # Match 6
            ("1A", "3_1A"),     # Match 7
            ("1L", "3_1L"),     # Match 8
            ("1G", "3_1G"),     # Match 9
            ("1D", "3_1D"),     # Match 10
            ("1H", "2J"),       # Match 11
            ("2K", "2L"),       # Match 12
            ("1B", "3_1B"),     # Match 13
            ("2D", "2G"),       # Match 14
            ("1J", "2H"),       # Match 15
            ("1K", "3_1K"),     # Match 16
        ]
        
        r32_matches = []
        for pair in r32_pairings:
            res = simulate_knockout_match(get_team(pair[0]), get_team(pair[1]), elo_ratings, rng, knockout_played_matches, real_shootouts)
            r32_matches.append(res)
        
    # 3. Round of 16
    r16_pairings = [
        (r32_matches[0]["winner"], r32_matches[2]["winner"]),   # Match 17 (W1 vs W3)
        (r32_matches[1]["winner"], r32_matches[4]["winner"]),   # Match 18 (W2 vs W5)
        (r32_matches[3]["winner"], r32_matches[5]["winner"]),   # Match 19 (W4 vs W6)
        (r32_matches[6]["winner"], r32_matches[7]["winner"]),   # Match 20 (W7 vs W8)
        (r32_matches[10]["winner"], r32_matches[11]["winner"]), # Match 21 (W11 vs W12)
        (r32_matches[8]["winner"], r32_matches[9]["winner"]),   # Match 22 (W9 vs W10)
        (r32_matches[13]["winner"], r32_matches[15]["winner"]), # Match 23 (W14 vs W16)
        (r32_matches[12]["winner"], r32_matches[14]["winner"]), # Match 24 (W13 vs W15)
    ]
    
    r16_matches = []
    for pair in r16_pairings:
        res = simulate_knockout_match(pair[0], pair[1], elo_ratings, rng, knockout_played_matches, real_shootouts)
        r16_matches.append(res)
        
    # 4. Quarterfinals
    qf_pairings = [
        (r16_matches[0]["winner"], r16_matches[1]["winner"]),   # QF1 (W17 vs W18)
        (r16_matches[4]["winner"], r16_matches[5]["winner"]),   # QF2 (W21 vs W22)
        (r16_matches[2]["winner"], r16_matches[3]["winner"]),   # QF3 (W19 vs W20)
        (r16_matches[6]["winner"], r16_matches[7]["winner"]),   # QF4 (W23 vs W24)
    ]
    
    qf_matches = []
    for pair in qf_pairings:
        res = simulate_knockout_match(pair[0], pair[1], elo_ratings, rng, knockout_played_matches, real_shootouts)
        qf_matches.append(res)
        
    # 5. Semifinals
    sf_pairings = [
        (qf_matches[0]["winner"], qf_matches[1]["winner"]),     # SF1 (W25 vs W26)
        (qf_matches[2]["winner"], qf_matches[3]["winner"]),     # SF2 (W27 vs W28)
    ]
    
    sf_matches = []
    for pair in sf_pairings:
        res = simulate_knockout_match(pair[0], pair[1], elo_ratings, rng, knockout_played_matches, real_shootouts)
        sf_matches.append(res)
        
    # 6. Final
    final_match = simulate_knockout_match(sf_matches[0]["winner"], sf_matches[1]["winner"], elo_ratings, rng, knockout_played_matches, real_shootouts)
    
    # 7. Collect furthest round
    furthest = {}
    all_teams = [t for group in WC2026_GROUPS.values() for t in group]
    for t in all_teams:
        furthest[t] = "Group Stage"
        
    for m in r32_matches:
        furthest[m["team_a"]] = "R32"
        furthest[m["team_b"]] = "R32"
    for m in r16_matches:
        furthest[m["team_a"]] = "R16"
        furthest[m["team_b"]] = "R16"
    for m in qf_matches:
        furthest[m["team_a"]] = "QF"
        furthest[m["team_b"]] = "QF"
    for m in sf_matches:
        furthest[m["team_a"]] = "SF"
        furthest[m["team_b"]] = "SF"
        
    furthest[final_match["team_a"]] = "Finalist"
    furthest[final_match["team_b"]] = "Finalist"
    furthest[final_match["winner"]] = "Champion"
    
    return {
        "furthest": furthest,
        "knockouts": {
            "r32": r32_matches,
            "r16": r16_matches,
            "qf": qf_matches,
            "sf": sf_matches,
            "final": final_match,
        }
    }


def simulate_full_tournament_monte_carlo(
    elo_ratings: Dict[str, float],
    n_sims: int = 10000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Runs full World Cup 2026 simulations (Groups + Knockouts) n_sims times.
    Calculates exact cumulative progression percentages for each team.
    """
    rng = np.random.default_rng(seed)
    
    all_teams = [t for group in WC2026_GROUPS.values() for t in group]
    stage_counts = {
        t: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "Final": 0, "Champion": 0}
        for t in all_teams
    }
    
    group_stage_stats = {t: {"top2": 0, "pts": 0, "gd": 0} for t in all_teams}
    position_counts = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in all_teams}

    group_played_matches = load_real_played_matches(knockout=False)
    knockout_played_matches = load_real_played_matches(knockout=True)
    real_shootouts = load_real_shootouts()

    for _ in range(n_sims):
        # 1. Group Stage Standing Tracking
        for group_name, teams in WC2026_GROUPS.items():
            standings = _simulate_group_once(teams, elo_ratings, rng, group_played_matches)
            for entry in standings:
                t = entry["team"]
                pos = entry["position"]
                position_counts[t][pos] += 1
                group_stage_stats[t]["pts"] += entry["pts"]
                group_stage_stats[t]["gd"] += entry["gd"]
                if pos <= 2:
                    group_stage_stats[t]["top2"] += 1
                    
        # 2. Knockout Progression Tracking
        sim_res = simulate_single_tournament(elo_ratings, rng, group_played_matches, knockout_played_matches, real_shootouts)
        furthest = sim_res["furthest"]
        for t, stage in furthest.items():
            if stage == "Champion":
                stage_counts[t]["Champion"] += 1
                stage_counts[t]["Final"] += 1
                stage_counts[t]["SF"] += 1
                stage_counts[t]["QF"] += 1
                stage_counts[t]["R16"] += 1
                stage_counts[t]["R32"] += 1
            elif stage == "Finalist":
                stage_counts[t]["Final"] += 1
                stage_counts[t]["SF"] += 1
                stage_counts[t]["QF"] += 1
                stage_counts[t]["R16"] += 1
                stage_counts[t]["R32"] += 1
            elif stage == "SF":
                stage_counts[t]["SF"] += 1
                stage_counts[t]["QF"] += 1
                stage_counts[t]["R16"] += 1
                stage_counts[t]["R32"] += 1
            elif stage == "QF":
                stage_counts[t]["QF"] += 1
                stage_counts[t]["R16"] += 1
                stage_counts[t]["R32"] += 1
            elif stage == "R16":
                stage_counts[t]["R16"] += 1
                stage_counts[t]["R32"] += 1
            elif stage == "R32":
                stage_counts[t]["R32"] += 1

    rows = []
    for group_name, teams in WC2026_GROUPS.items():
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
                "top2_pct": group_stage_stats[t]["top2"] / n_sims * 100,
                "avg_pts": group_stage_stats[t]["pts"] / n_sims,
                "avg_gd": group_stage_stats[t]["gd"] / n_sims,
                "r32_pct": stage_counts[t]["R32"] / n_sims * 100,
                "r16_pct": stage_counts[t]["R16"] / n_sims * 100,
                "qf_pct": stage_counts[t]["QF"] / n_sims * 100,
                "sf_pct": stage_counts[t]["SF"] / n_sims * 100,
                "final_pct": stage_counts[t]["Final"] / n_sims * 100,
                "champion_pct": stage_counts[t]["Champion"] / n_sims * 100,
                "advance_pct": stage_counts[t]["R32"] / n_sims * 100,
            })
            
    return pd.DataFrame(rows)


def simulate_all_groups(
    elo_ratings: Dict[str, float],
    n_sims: int = 10000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper. Calls the full Monte Carlo tournament engine.
    """
    return simulate_full_tournament_monte_carlo(elo_ratings, n_sims, seed)


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
