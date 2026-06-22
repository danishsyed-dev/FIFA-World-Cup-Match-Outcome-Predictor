"""
1_Group_Stage_Simulator.py
---------------------------
Streamlit page: FIFA World Cup 2026 Group Stage Monte Carlo Simulator.

Runs N simulations per group and displays advancement probabilities
with stacked bar charts and a tournament-wide leaderboard.

Navigation: This page appears in the sidebar automatically as part of
Streamlit's multi-page app system.
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import importlib
import src.image_generator
importlib.reload(src.image_generator)
import app.shared_theme
importlib.reload(app.shared_theme)
import src.group_simulator

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def get_cached_group_standings_png_v2(results, selected_view):
    # force cache reload: v6
    from src.image_generator import generate_group_standings_png
    return generate_group_standings_png(results, selected_view)

@st.cache_data(show_spinner=False)
def get_cached_progression_png_v2(results):
    # force cache reload: v6
    from src.image_generator import generate_progression_png
    return generate_progression_png(results)

@st.cache_data(show_spinner=False)
def get_cached_bracket_png_v2(bracket_data):
    # force cache reload: v6
    from src.image_generator import generate_bracket_png
    return generate_bracket_png(bracket_data)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Group Stage Simulator — FIFA WC 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared Theme ──────────────────────────────────────────────────────────────
from app.shared_theme import inject_theme, get_flag_url, COUNTRY_TO_ISO

inject_theme()

# ── Simulator-specific CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Group Card ─────────────────────────────────────────────── */
.group-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.group-card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--card-border);
}
.group-letter {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #10b981;
    line-height: 1;
}
.group-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    text-transform: uppercase;
}

/* ── Team Row ───────────────────────────────────────────────── */
.sim-team-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid color-mix(in srgb, var(--background-color), var(--text-color) 6%);
}
.sim-team-row:last-child {
    border-bottom: none;
}
.sim-team-rank {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text-muted);
    min-width: 22px;
    text-align: center;
}
.sim-team-flag {
    width: 36px;
    height: auto;
    border-radius: 4px;
}
.sim-team-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--title-color);
    width: 160px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sim-team-elo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    color: #10b981;
    font-weight: 600;
    min-width: 60px;
}

/* ── Advance Badge ──────────────────────────────────────────── */
.advance-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    min-width: 80px;
    justify-content: center;
}
.badge-high {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-mid {
    background: rgba(229, 193, 88, 0.12);
    color: #e5c158;
    border: 1px solid rgba(229, 193, 88, 0.3);
}
.badge-low {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}

/* ── Stats Micro Labels ─────────────────────────────────────── */
.sim-stat-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
}
.sim-stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--title-color);
}

/* ── Leaderboard Table ──────────────────────────────────────── */
.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
}
.leaderboard-table th {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
    padding: 0.6rem 0.5rem;
    border-bottom: 2px solid var(--card-border);
    text-align: left;
}
.leaderboard-table td {
    padding: 0.55rem 0.5rem;
    border-bottom: 1px solid color-mix(in srgb, var(--background-color), var(--text-color) 6%);
    vertical-align: middle;
}
.leaderboard-table tr:hover {
    background-color: var(--card-bg-hover);
}
.lb-rank {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-muted);
    min-width: 28px;
}
.lb-team {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.lb-flag {
    width: 20px;
    height: auto;
    border-radius: 3px;
}
.lb-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--title-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: inline-block;
    max-width: 110px;
    vertical-align: middle;
}
.lb-group {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--btn-bg);
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
    border: 1px solid var(--card-border);
}
.lb-stat {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--title-color);
}

/* ── Progress Bar (mini) ────────────────────────────────────── */
.mini-progress {
    width: 100%;
    height: 6px;
    background: color-mix(in srgb, var(--background-color), var(--text-color) 8%);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 2px;
}
.mini-progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Simulation Status Badge ────────────────────────────────── */
.sim-status {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.75rem;
    border-radius: 4px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.sim-running {
    background: rgba(229, 193, 88, 0.12);
    color: #e5c158;
    border: 1px solid rgba(229, 193, 88, 0.3);
}
.sim-complete {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* ── Bracket Tree Styling ────────────────────────────────────── */
.bracket-wrapper {
    display: flex;
    gap: 1.5rem;
    overflow-x: auto;
    padding: 1.5rem 0.5rem;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    min-width: 1100px;
    margin-top: 1rem;
}
.bracket-column {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    height: 980px;
    flex: 1;
    min-width: 200px;
}
.bracket-match {
    background: color-mix(in srgb, var(--card-bg), #fff 2%);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.bracket-team {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.4rem;
    border-radius: 4px;
}
.bracket-team.winner {
    background: rgba(16, 185, 129, 0.08);
}
.bracket-team.winner .bracket-team-name {
    color: #10b981;
    font-weight: 700;
}
.bracket-team.winner .bracket-team-score {
    color: #10b981;
    font-weight: 800;
}
.bracket-team-flag {
    width: 20px;
    height: auto;
    border-radius: 2px;
}
.bracket-team-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    color: var(--text-color);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100px;
}
.bracket-team-score {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 600;
    min-width: 15px;
    text-align: right;
}
.bracket-match-header {
    font-family: 'Outfit', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    border-bottom: 1px solid color-mix(in srgb, var(--card-border), transparent 30%);
    padding-bottom: 0.15rem;
}
.champion-box {
    text-align: center;
    background: rgba(16, 185, 129, 0.06);
    border: 2px solid #10b981;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
}
.champion-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    color: #10b981;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.champion-flag {
    width: 48px;
    height: auto;
    border-radius: 4px;
    margin-bottom: 0.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.15);
}
.champion-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #10b981;
}

/* ── Standings Table ── */
.standings-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
}
.standings-table th {
    font-family: 'Outfit', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    text-transform: uppercase;
    padding: 0.4rem 0.15rem;
    border-bottom: 2px solid var(--card-border);
    text-align: center;
    white-space: nowrap;
}
.standings-table th.team-col {
    text-align: left;
}
.standings-table td {
    padding: 0.5rem 0.15rem;
    border-bottom: 1px solid color-mix(in srgb, var(--background-color), var(--text-color) 6%);
    font-size: 0.85rem;
    text-align: center;
    vertical-align: middle;
}
.standings-table td.team-col {
    text-align: left;
}
.standings-pos {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: var(--text-muted);
}
.standings-team-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    color: var(--title-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: inline-block;
    vertical-align: middle;
}
.pre-sim-name {
    max-width: 80px;
}
.post-sim-name {
    max-width: 90px;
}
.standings-elo {
    font-family: 'Space Grotesk', sans-serif;
    color: #10b981;
    font-weight: 600;
}
.standings-val {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: var(--title-color);
}
.standings-pts {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: #10b981;
}
.group-played-matches {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px dashed var(--card-border);
}
.group-played-matches strong {
    color: var(--title-color);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Simulator Logic
# ══════════════════════════════════════════════════════════════════════════════

from typing import List, Dict
from itertools import combinations
from src.group_simulator import WC2026_GROUPS, FALLBACK_ELO, simulate_all_groups, load_real_played_matches

def compute_actual_standings(teams: List[str], elo_ratings: Dict[str, float], played_matches: dict) -> List[dict]:
    """Calculate actual group standings based on played matches from results.csv."""
    records = {t: {"pts": 0, "pld": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}
    for team_a, team_b in combinations(teams, 2):
        if played_matches and (team_a, team_b) in played_matches:
            goals_a, goals_b = played_matches[(team_a, team_b)]
            records[team_a]["pld"] += 1
            records[team_b]["pld"] += 1
            records[team_a]["gf"] += goals_a
            records[team_a]["ga"] += goals_b
            records[team_b]["gf"] += goals_b
            records[team_b]["ga"] += goals_a
            
            if goals_a > goals_b:
                records[team_a]["pts"] += 3
                records[team_a]["w"] += 1
                records[team_b]["l"] += 1
            elif goals_a == goals_b:
                records[team_a]["pts"] += 1
                records[team_b]["pts"] += 1
                records[team_a]["w"] += 0
                records[team_a]["d"] += 1
                records[team_b]["d"] += 1
            else:
                records[team_b]["pts"] += 3
                records[team_b]["w"] += 1
                records[team_a]["l"] += 1
                
    for t in teams:
        records[t]["gd"] = records[t]["gf"] - records[t]["ga"]
        
    sorted_teams = sorted(
        teams,
        key=lambda t: (
            records[t]["pts"],
            records[t]["gd"],
            records[t]["gf"],
            elo_ratings.get(t, FALLBACK_ELO.get(t, 1500))
        ),
        reverse=True
    )
    
    standings = []
    for pos, t in enumerate(sorted_teams, 1):
        standings.append({
            "pos": pos,
            "team": t,
            "elo": elo_ratings.get(t, FALLBACK_ELO.get(t, 1500)),
            "pld": records[t]["pld"],
            "w": records[t]["w"],
            "d": records[t]["d"],
            "l": records[t]["l"],
            "gf": records[t]["gf"],
            "ga": records[t]["ga"],
            "gd": records[t]["gd"],
            "pts": records[t]["pts"],
        })
    return standings


def load_elo_ratings():
    """Extract Elo ratings from the shared cached Predictor. Falls back to hardcoded."""
    try:
        # Import the same cached function from the main app so Predictor is only ever
        # initialized once across all pages in the session.
        import app.streamlit_app as _main_app
        predictor = _main_app.load_predictor()
    except Exception:
        predictor = None

    if predictor is None or isinstance(predictor, str):
        return FALLBACK_ELO.copy()
    try:
        all_wc_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
        elo_dict = {}
        for team in all_wc_teams:
            rating = predictor.elo_system.get_rating(team)
            if rating == predictor.elo_system.initial_rating and team in FALLBACK_ELO:
                elo_dict[team] = FALLBACK_ELO[team]
            else:
                elo_dict[team] = rating
        return elo_dict
    except Exception:
        return FALLBACK_ELO.copy()


def get_badge_class(pct: float) -> str:
    if pct >= 60:
        return "badge-high"
    elif pct >= 30:
        return "badge-mid"
    return "badge-low"


def get_progress_color(pct: float) -> str:
    if pct >= 60:
        return "#10b981"
    elif pct >= 30:
        return "#e5c158"
    return "#f87171"


def render_match_html(match: dict, match_title: str) -> str:
    flag_a = get_flag_url(match["team_a"])
    flag_b = get_flag_url(match["team_b"])
    
    winner = match["winner"]
    win_a = "winner" if winner == match["team_a"] else ""
    win_b = "winner" if winner == match["team_b"] else ""
    
    match_status = ""
    if match["penalties"]:
        match_status = " (PEN)"
    elif match["extra_time"]:
        match_status = " (AET)"
        
    return f"""
    <div class="bracket-match">
        <div class="bracket-match-header">{match_title}{match_status}</div>
        <div class="bracket-team {win_a}">
            <img class="bracket-team-flag" src="{flag_a}" crossorigin="anonymous" />
            <span class="bracket-team-name">{match["team_a"]}</span>
            <span class="bracket-team-score">{match["score_a"]}</span>
        </div>
        <div class="bracket-team {win_b}">
            <img class="bracket-team-flag" src="{flag_b}" crossorigin="anonymous" />
            <span class="bracket-team-name">{match["team_b"]}</span>
            <span class="bracket-team-score">{match["score_b"]}</span>
        </div>
    </div>
    """


# ── Upset Detector Data Processing & Layout ──

COUNTRY_TO_CONFEDERATION: Dict[str, str] = {
    "France": "UEFA", "Croatia": "UEFA", "Belgium": "UEFA", "England": "UEFA",
    "Russia": "UEFA", "Sweden": "UEFA", "Spain": "UEFA", "Portugal": "UEFA",
    "Denmark": "UEFA", "Switzerland": "UEFA", "Germany": "UEFA", "Iceland": "UEFA",
    "Serbia": "UEFA", "Poland": "UEFA", "Italy": "UEFA", "Netherlands": "UEFA",
    "Greece": "UEFA", "Bosnia and Herzegovina": "UEFA", "Ukraine": "UEFA",
    "Czechia": "UEFA", "Czech Republic": "UEFA", "Slovakia": "UEFA",
    "Slovenia": "UEFA", "Turkey": "UEFA", "Republic of Ireland": "UEFA",
    "Ireland": "UEFA", "Scotland": "UEFA", "Wales": "UEFA", "Austria": "UEFA",
    "Bulgaria": "UEFA", "Romania": "UEFA", "Norway": "UEFA", "Hungary": "UEFA",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Peru": "CONMEBOL", "Chile": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "USA": "CONCACAF", "United States": "CONCACAF", "Mexico": "CONCACAF",
    "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF", "Trinidad and Tobago": "CONCACAF", "Haiti": "CONCACAF",
    "Canada": "CONCACAF", "Curacao": "CONCACAF", "El Salvador": "CONCACAF",
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Algeria": "CAF", "Ivory Coast": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "South Africa": "CAF", "Angola": "CAF", "Togo": "CAF",
    "Cote d'Ivoire": "CAF", "DR Congo": "CAF", "Cape Verde": "CAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "Uzbekistan": "AFC", "Jordan": "AFC",
    "China": "AFC", "Iraq": "AFC", "North Korea": "AFC", "Korea DPR": "AFC",
    "Korea Republic": "AFC", "New Zealand": "OFC"
}

def get_confederation(team: str, date: pd.Timestamp) -> str:
    if team == "Australia":
        return "OFC" if date.year < 2006 else "AFC"
    return COUNTRY_TO_CONFEDERATION.get(team, "UEFA")

def get_historical_df():
    try:
        import app.streamlit_app as _main_app
        predictor = _main_app.load_predictor()
        if predictor is not None and not isinstance(predictor, str):
            return predictor.df
    except Exception:
        pass
    from src.data_loader import load_all
    from src.elo_calculator import EloSystem
    raw = load_all()
    elo_system = EloSystem()
    return elo_system.calculate(raw)

def compute_upsets(df, min_elo_diff):
    df = df.sort_values('date').copy()
    df['year'] = df['date'].dt.year
    
    match_count = {}
    home_match_idx = []
    away_match_idx = []
    for idx, row in df.iterrows():
        yr = row['year']
        h = row['home_team']
        a = row['away_team']
        h_count = match_count.get((yr, h), 0)
        a_count = match_count.get((yr, a), 0)
        home_match_idx.append(h_count)
        away_match_idx.append(a_count)
        match_count[(yr, h)] = h_count + 1
        match_count[(yr, a)] = a_count + 1
        
    df['home_match_idx'] = home_match_idx
    df['away_match_idx'] = away_match_idx
    df['stage'] = np.where(
        (df['home_match_idx'] <= 2) & (df['away_match_idx'] <= 2),
        "Group Stage",
        "Knockout Stage"
    )
    
    last_match_date = {}
    home_rest = []
    away_rest = []
    for idx, row in df.iterrows():
        yr = row['year']
        h = row['home_team']
        a = row['away_team']
        dt = row['date']
        
        h_last = last_match_date.get((yr, h))
        a_last = last_match_date.get((yr, a))
        
        home_rest.append((dt - h_last).days if h_last else 7.0)
        away_rest.append((dt - a_last).days if a_last else 7.0)
        
        last_match_date[(yr, h)] = dt
        last_match_date[(yr, a)] = dt
        
    df['home_rest'] = home_rest
    df['away_rest'] = away_rest
    
    upset_rows = []
    for idx, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        h_elo = row['home_elo']
        a_elo = row['away_elo']
        h_score = int(row['home_score'])
        a_score = int(row['away_score'])
        
        if h_elo >= a_elo:
            fav = h
            und = a
            fav_elo = h_elo
            und_elo = a_elo
            fav_score = h_score
            und_score = a_score
            fav_rest = row['home_rest']
            und_rest = row['away_rest']
        else:
            fav = a
            und = h
            fav_elo = a_elo
            und_elo = h_elo
            fav_score = a_score
            und_score = h_score
            fav_rest = row['away_rest']
            und_rest = row['home_rest']
            
        elo_diff = fav_elo - und_elo
        if elo_diff < min_elo_diff:
            continue
            
        if und_score > fav_score:
            is_upset = True
            winner = und
            loser = fav
        else:
            is_upset = False
            winner = fav if fav_score > und_score else None
            loser = und if fav_score > und_score else None
            
        is_draw = (fav_score == und_score)
        
        upset_rows.append({
            "date": row['date'],
            "year": row['year'],
            "home_team": h,
            "away_team": a,
            "home_score": h_score,
            "away_score": a_score,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "favorite": fav,
            "underdog": und,
            "fav_elo": fav_elo,
            "und_elo": und_elo,
            "elo_diff": elo_diff,
            "is_upset": is_upset,
            "is_draw": is_draw,
            "winner": winner,
            "loser": loser,
            "fav_rest": fav_rest,
            "und_rest": und_rest,
            "stage": row['stage']
        })
        
    return pd.DataFrame(upset_rows)

def render_upset_detector_tab():
    st.html("""
    <div style="margin-top: 1rem; margin-bottom: 1.5rem;">
        <span class="telemetry-badge" style="background: rgba(229, 193, 88, 0.1); color: #e5c158; border: 1px solid rgba(229, 193, 88, 0.3);">UPSET TELEMETRY & PATTERN DETECTOR</span>
        <h3 style="font-family: 'Outfit', sans-serif; font-weight: 800; color: var(--title-color); margin-top: 0.5rem; margin-bottom: 0.25rem;">
            ⚡ HISTORICAL WORLD CUP UPSET DETECTOR
        </h3>
        <p style="font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text-muted); font-size: 0.9rem; margin-top: 0;">
            Analyzing every upset in the last 30 years of FIFA World Cup history using pre-match Elo ratings, rest days, and continental matchups.
        </p>
    </div>
    """)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        year_range = st.slider(
            "WORLD CUP YEARS ANALYZED",
            min_value=1998,
            max_value=2026,
            value=(1998, 2026),
            step=4,
            key="upset_year_range"
        )
    with col_s2:
        min_elo_diff = st.slider(
            "MINIMUM ELO DIFFERENCE FOR FAVORITE",
            min_value=0,
            max_value=400,
            value=50,
            step=10,
            help="Minimum Elo rating gap required to qualify a match as a potential upset scenario.",
            key="upset_min_elo_diff"
        )
        
    df_raw = get_historical_df()
    if df_raw.empty:
        st.warning("Could not load historical data.")
        return
        
    df_calc = compute_upsets(df_raw, min_elo_diff)
    if df_calc.empty:
        st.warning("No matches match the Elo difference threshold in the dataset.")
        return
        
    df_upsets = df_calc[(df_calc['year'] >= year_range[0]) & (df_calc['year'] <= year_range[1])].copy()
    if df_upsets.empty:
        st.warning("No matches found for the selected year range.")
        return
        
    df_only_upsets = df_upsets[df_upsets['is_upset'] == True].sort_values('date', ascending=False)
    
    total_matches = len(df_upsets)
    total_upsets = len(df_only_upsets)
    upset_rate = (total_upsets / total_matches) * 100 if total_matches > 0 else 0.0
    
    biggest_upset_html = "N/A"
    if not df_only_upsets.empty:
        biggest = df_only_upsets.loc[df_only_upsets['elo_diff'].idxmax()]
        b_date = biggest['date'].strftime('%Y-%m-%d')
        b_win = biggest['winner']
        b_los = biggest['loser']
        b_score = f"{biggest['home_score']}–{biggest['away_score']}"
        b_diff = biggest['elo_diff']
        biggest_upset_html = f"<strong>{b_win}</strong> def. {b_los} ({b_score})<br><span style='color: #e5c158;'>Elo Diff: {b_diff:.0f}</span> | {b_date}"
        
    st.html(f"""
    <div class="telemetry-grid">
        <div class="telemetry-card">
            <div class="telemetry-lbl">Total Matches</div>
            <div class="telemetry-val" style="color: var(--title-color);">{total_matches}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-lbl">Total Upset Wins</div>
            <div class="telemetry-val" style="color: #f87171;">{total_upsets}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-lbl">Upset Win Rate</div>
            <div class="telemetry-val" style="color: #e5c158;">{upset_rate:.1f}%</div>
        </div>
        <div class="telemetry-card" style="text-align: left; display: flex; flex-direction: column; justify-content: center; padding-left: 1.25rem;">
            <div class="telemetry-lbl">Biggest Upset</div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.8rem; color: var(--text-color); margin-top: 0.25rem; line-height: 1.4;">
                {biggest_upset_html}
            </div>
        </div>
    </div>
    """)
    
    col_left, col_right = st.columns([5, 7])
    
    with col_left:
        st.markdown(f"<div class='sim-stat-label' style='margin-bottom: 0.5rem;'>Upset Match List ({total_upsets})</div>", unsafe_allow_html=True)
        if total_upsets == 0:
            st.info("No upset wins found with the current filters.")
        else:
            table_rows_html = ""
            for idx, row in df_only_upsets.iterrows():
                dt_str = row['date'].strftime('%Y-%m-%d')
                win = row['winner']
                los = row['loser']
                
                if win == row['home_team']:
                    score_str = f"<strong>{row['home_score']}</strong>–{row['away_score']}"
                else:
                    score_str = f"{row['home_score']}–<strong>{row['away_score']}</strong>"
                    
                win_flag = get_flag_url(win)
                los_flag = get_flag_url(los)
                
                table_rows_html += f"""
                <tr>
                    <td style="font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; color: var(--text-muted);">{dt_str}</td>
                    <td style="text-align: left; padding: 0.35rem 0.5rem;">
                        <div style="display: flex; flex-direction: column; gap: 0.2rem;">
                            <div style="display: flex; align-items: center; gap: 0.4rem;">
                                <img src="{win_flag}" style="width: 16px; height: auto; border-radius: 2px;" crossorigin="anonymous" />
                                <span style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #10b981; font-size: 0.8rem;">{win}</span>
                                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; color: var(--text-muted);">({row['und_elo']:.0f})</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 0.4rem;">
                                <img src="{los_flag}" style="width: 16px; height: auto; border-radius: 2px;" crossorigin="anonymous" />
                                <span style="font-family: 'Outfit', sans-serif; font-weight: 600; color: var(--text-muted); font-size: 0.8rem;">{los}</span>
                                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; color: var(--text-muted);">({row['fav_elo']:.0f})</span>
                            </div>
                        </div>
                    </td>
                    <td style="font-family: 'Space Grotesk', sans-serif; font-size: 0.8rem; font-weight: 700; color: var(--title-color);">{score_str}</td>
                    <td style="font-family: 'Space Grotesk', sans-serif; font-size: 0.8rem; font-weight: 700; color: #e5c158;">+{row['elo_diff']:.0f}</td>
                    <td style="padding: 0.3rem;"><span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.65rem; color: var(--text-muted); background: var(--btn-bg); padding: 0.1rem 0.35rem; border-radius: 3px; border: 1px solid var(--card-border);">{row['stage']}</span></td>
                </tr>
                """
                
            st.html(f"""
            <div style="max-height: 575px; overflow-y: auto; border: 1px solid var(--card-border); border-radius: 8px; background: var(--background-color);">
                <table class="standings-table" style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr>
                            <th style="width: 18%; font-size: 0.65rem; padding: 0.4rem 0.25rem;">Date</th>
                            <th style="text-align: left; padding-left: 0.5rem; width: 44%; font-size: 0.65rem; padding: 0.4rem 0.25rem;">Matchup (Elo)</th>
                            <th style="width: 16%; font-size: 0.65rem; padding: 0.4rem 0.25rem;">Score</th>
                            <th style="width: 11%; font-size: 0.65rem; padding: 0.4rem 0.25rem;">Diff</th>
                            <th style="width: 11%; font-size: 0.65rem; padding: 0.4rem 0.25rem;">Stage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
            """)
            
    with col_right:
        if not df_only_upsets.empty:
            df_only_upsets['winner_confed'] = df_only_upsets.apply(lambda r: get_confederation(r['winner'], r['date']), axis=1)
            confed_counts = df_only_upsets['winner_confed'].value_counts().reset_index()
            confed_counts.columns = ['confed', 'count']
            confed_counts = confed_counts.sort_values('count', ascending=True)
            
            fig_confed = go.Figure(go.Bar(
                y=confed_counts['confed'],
                x=confed_counts['count'],
                orientation='h',
                marker_color='#10b981',
                text=confed_counts['count'],
                textposition='outside',
                textfont=dict(size=11, family="Space Grotesk", color="#f8fafc"),
            ))
            fig_confed.update_layout(
                title=dict(
                    text="UPSETS PULLED OFF BY UNDERDOG CONFEDERATION",
                    font=dict(size=12, family="Outfit", weight="bold", color="#f8fafc")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=10, l=80, r=30),
                height=175,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(tickfont=dict(size=11, family="Outfit", color="#94a3b8"), showgrid=False)
            )
            st.plotly_chart(fig_confed, width='stretch', key="upset_chart_confed")
            
        n_group = len(df_upsets[df_upsets['stage'] == 'Group Stage'])
        u_group = len(df_only_upsets[df_only_upsets['stage'] == 'Group Stage'])
        rate_group = (u_group / n_group) * 100 if n_group > 0 else 0.0
        
        n_knockout = len(df_upsets[df_upsets['stage'] == 'Knockout Stage'])
        u_knockout = len(df_only_upsets[df_only_upsets['stage'] == 'Knockout Stage'])
        rate_knockout = (u_knockout / n_knockout) * 100 if n_knockout > 0 else 0.0
        
        stages = ['Group Stage', 'Knockout Stage']
        rates = [rate_group, rate_knockout]
        
        fig_stage = go.Figure(go.Bar(
            x=stages,
            y=rates,
            marker_color=['#10b981', '#38bdf8'],
            text=[f"{r:.1f}%" for r in rates],
            textposition='outside',
            textfont=dict(size=11, family="Space Grotesk", color="#f8fafc"),
            width=0.35
        ))
        fig_stage.update_layout(
            title=dict(
                text="UPSET RATE (%) BY TOURNAMENT STAGE",
                font=dict(size=12, family="Outfit", weight="bold", color="#f8fafc")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=20, r=20),
            height=175,
            yaxis=dict(showgrid=True, gridcolor="#232b2b", zeroline=False, range=[0, max(rates) + 6] if rates else [0, 100], tickfont=dict(size=10, family="Space Grotesk", color="#94a3b8")),
            xaxis=dict(tickfont=dict(size=11, family="Outfit", color="#f8fafc"), showgrid=False)
        )
        st.plotly_chart(fig_stage, width='stretch', key="upset_chart_stage")
        
        rest_adv_matches = df_upsets[df_upsets['und_rest'] > df_upsets['fav_rest']]
        rest_adv_upsets = rest_adv_matches[rest_adv_matches['is_upset'] == True]
        rate_adv = (len(rest_adv_upsets) / len(rest_adv_matches)) * 100 if len(rest_adv_matches) > 0 else 0.0
        
        rest_equal_matches = df_upsets[df_upsets['und_rest'] == df_upsets['fav_rest']]
        rest_equal_upsets = rest_equal_matches[rest_equal_matches['is_upset'] == True]
        rate_equal = (len(rest_equal_upsets) / len(rest_equal_matches)) * 100 if len(rest_equal_matches) > 0 else 0.0
        
        rest_dis_matches = df_upsets[df_upsets['und_rest'] < df_upsets['fav_rest']]
        rest_dis_upsets = rest_dis_matches[rest_dis_matches['is_upset'] == True]
        rate_dis = (len(rest_dis_upsets) / len(rest_dis_matches)) * 100 if len(rest_dis_matches) > 0 else 0.0
        
        categories = ['Underdog Rest Advantage', 'Equal Rest', 'Underdog Rest Disadvantage']
        fatigue_rates = [rate_adv, rate_equal, rate_dis]
        
        fig_fatigue = go.Figure(go.Bar(
            x=categories,
            y=fatigue_rates,
            marker_color='#e5c158',
            text=[f"{r:.1f}%" for r in fatigue_rates],
            textposition='outside',
            textfont=dict(size=11, family="Space Grotesk", color="#f8fafc"),
            width=0.35
        ))
        fig_fatigue.update_layout(
            title=dict(
                text="UPSET RATE (%) BY UNDERDOG REST STATUS",
                font=dict(size=12, family="Outfit", weight="bold", color="#f8fafc")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=20, r=20),
            height=175,
            yaxis=dict(showgrid=True, gridcolor="#232b2b", zeroline=False, range=[0, max(fatigue_rates) + 6] if fatigue_rates else [0, 100], tickfont=dict(size=10, family="Space Grotesk", color="#94a3b8")),
            xaxis=dict(tickfont=dict(size=11, family="Outfit", color="#f8fafc"), showgrid=False)
        )
        st.plotly_chart(fig_fatigue, width='stretch', key="upset_chart_fatigue")

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 8px; padding: 1rem 1.5rem;">
        <h4 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #10b981; margin: 0 0 0.5rem 0;">💡 TACTICAL DATA SCIENCE INSIGHTS</h4>
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.85rem; color: var(--text-color); line-height: 1.5;">
            <ul>
                <li><strong>Confederation Strength</strong>: UEFA and CONMEBOL teams rarely suffer major upsets due to depth, but AFC/CAF underdogs pull off the highest proportion of surprise wins when meeting highly-ranked opponents.</li>
                <li><strong>Stage Susceptibility</strong>: Historically, upsets occur significantly more in the <strong>Group Stage</strong>. Favorites in the Knockout Stage play with higher tactical caution, and Elo ratings tend to stabilize.</li>
                <li><strong>The Rest Factor (Fatigue)</strong>: Teams entering matches with a <strong>Rest Advantage</strong> show a higher upset conversion rate. Short rest cycles (3-4 days) impact higher-ranked teams who rely on high-intensity pressure.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_group_chart(group_df: pd.DataFrame, group_name: str) -> go.Figure:
    """Build a horizontal stacked bar chart for group position probabilities."""
    group_df = group_df.sort_values("advance_pct", ascending=True)

    fig = go.Figure()

    colors = {
        "1st": "#10b981",
        "2nd": "#38bdf8",
        "3rd": "#e5c158",
        "4th": "#f87171",
    }

    for pos in ["1st", "2nd", "3rd", "4th"]:
        fig.add_trace(go.Bar(
            y=group_df["team"],
            x=group_df[pos],
            name=pos,
            orientation="h",
            marker_color=colors[pos],
            text=[f"{v:.0f}%" if v >= 8 else "" for v in group_df[pos]],
            textposition="inside",
            textfont=dict(size=11, family="Space Grotesk", weight="bold"),
            hovertemplate="<b>%{y}</b><br>" + pos + ": %{x:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=150, r=15),
        height=180,
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, family="Outfit", weight="bold", color="#f8fafc"),
            automargin=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Outfit", color="#94a3b8"),
            traceorder="normal",
        ),
        showlegend=True,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Page Layout
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        """<div class="tactical-header">
<div class="telemetry-badge">MATCH DECISION TELEMETRY v1.2</div>
<h1 class="tactical-title">
<svg class="header-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none">
<defs>
<!-- 3D Spherical Radial Shadow -->
<radialGradient id="sphereShadow" cx="35%" cy="30%" r="70%">
<stop offset="0%" stop-color="#ffffff" />
<stop offset="65%" stop-color="#f8fafc" />
<stop offset="85%" stop-color="#e2e8f0" />
<stop offset="100%" stop-color="#cbd5e1" />
</radialGradient>

<!-- Premium World Cup Trophy Gold Gradient -->
<linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#fef08a" />
<stop offset="40%" stop-color="#eab308" />
<stop offset="75%" stop-color="#ca8a04" />
<stop offset="100%" stop-color="#854d0e" />
</linearGradient>

<!-- Red Gradient for Canada Wave -->
<linearGradient id="redGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#f87171" />
<stop offset="100%" stop-color="#dc2626" />
</linearGradient>

<!-- Green Gradient for Mexico Wave -->
<linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#34d399" />
<stop offset="100%" stop-color="#059669" />
</linearGradient>

<!-- Blue Gradient for USA Wave -->
<linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#60a5fa" />
<stop offset="100%" stop-color="#2563eb" />
</linearGradient>
</defs>

<!-- Ball Sphere Base -->
<circle cx="50" cy="50" r="46" fill="url(#sphereShadow)" stroke="#94a3b8" stroke-width="1.2" />

<!-- Aerodynamic Grooves & Seam Textures (Adidas Trionda signature debossing) -->
<circle cx="50" cy="50" r="41" fill="none" stroke="#e2e8f0" stroke-width="0.8" stroke-dasharray="10,4,3,4" opacity="0.9" />
<circle cx="50" cy="50" r="46" fill="none" stroke="#e2e8f0" stroke-width="0.5" opacity="0.7" />

<!-- Subtle curved panels outline -->
<path d="M 17 33 A 46 46 0 0 1 83 33" fill="none" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="4,4" opacity="0.6" />
<path d="M 17 67 A 46 46 0 0 0 83 67" fill="none" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="4,4" opacity="0.6" />

<!-- Canada Red Wave -->
<path d="M 36.14 58.0 A 28 28 0 0 1 50.0 34.0 C 40.0 22.0, 28.0 18.0, 18.0 30.0 C 8.0 42.0, 18.0 54.0, 36.14 58.0 Z" fill="url(#redGrad)" stroke="url(#goldGrad)" stroke-width="1" stroke-linejoin="round" />

<!-- USA Blue Wave -->
<path d="M 50.0 34.0 A 28 28 0 0 1 63.86 58.0 C 79.25 55.34, 88.71 46.95, 83.32 32.29 C 77.93 17.63, 62.54 20.29, 50.0 34.0 Z" fill="url(#blueGrad)" stroke="url(#goldGrad)" stroke-width="1" stroke-linejoin="round" />

<!-- Mexico Green Wave -->
<path d="M 63.86 58.0 A 28 28 0 0 1 36.14 58.0 C 30.75 72.66, 33.29 85.05, 48.68 87.71 C 64.07 90.37, 69.46 75.71, 63.86 58.0 Z" fill="url(#greenGrad)" stroke="url(#goldGrad)" stroke-width="1" stroke-linejoin="round" />

<!-- Central Core (Curved Reuleaux Gold Triangle) -->
<path d="M 50.0 34.0 A 28 28 0 0 1 63.86 58.0 A 28 28 0 0 1 36.14 58.0 A 28 28 0 0 1 50.0 34.0 Z" fill="url(#goldGrad)" stroke="#1e293b" stroke-width="1.2" stroke-linejoin="round" />

<!-- Inner Gold Core Details (Star/Ray design) -->
<path d="M 50 34 L 50 42 M 63.86 58 L 56.93 54 M 36.14 58 L 43.07 54" fill="none" stroke="#854d0e" stroke-width="0.8" opacity="0.5" />

<!-- Trophy Silhouette -->
<g transform="translate(0, 1.5)">
<rect x="47.5" y="52.0" width="5.0" height="0.8" rx="0.2" fill="#166534" />
<rect x="47.0" y="52.8" width="6.0" height="0.8" rx="0.2" fill="url(#goldGrad)" />
<rect x="47.5" y="53.6" width="5.0" height="0.8" rx="0.2" fill="#166534" />
<path d="M 48.2 52.0 C 48.2 48.0, 46.8 45.0, 48.8 42.0 C 49.3 41.2, 50.7 41.2, 51.2 42.0 C 53.2 45.0, 51.8 48.0, 51.8 52.0 Z" fill="url(#goldGrad)" stroke="#854d0e" stroke-width="0.3" />
<circle cx="50.0" cy="40.8" r="1.8" fill="url(#goldGrad)" stroke="#854d0e" stroke-width="0.3" />
<circle cx="50.0" cy="40.8" r="1.4" fill="#fef08a" opacity="0.8" />
</g>

<!-- Canada Maple Leaf -->
<g transform="translate(32.5, 38.0) scale(0.24)" opacity="0.95">
<path d="M 0 -22 L 3 -11 L 11 -13 L 9 -4 L 18 -1 L 8 4 L 9 10 L 3 7 L 1.2 16 L -1.2 16 L -3 7 L -9 10 L -8 4 L -18 -1 L -9 -4 L -11 -13 L -3 -11 Z" fill="#ffffff" stroke="#854d0e" stroke-width="0.8" stroke-linejoin="round" />
</g>

<!-- USA Star -->
<g transform="translate(67.5, 38.0) scale(0.24)" opacity="0.95">
<path d="M 0 -20 L 5.5 -5 L 20 -5 L 8 4 L 13 18 L 0 9 L -13 18 L -8 4 L -20 -5 L -5.5 -5 Z" fill="#ffffff" stroke="#854d0e" stroke-width="0.8" stroke-linejoin="round" />
</g>

<!-- Mexico Eagle Head -->
<g transform="translate(50.0, 71.0) scale(0.24)" opacity="0.95">
<path d="M -15 12 C -12 2, -8 -12, 2 -14 C 8 -15, 14 -11, 18 -5 C 21 -2, 23 3, 18 7 C 14 10, 10 6, 8 3 C 5 1, 2 4, 0 7 C -4 11, -10 12, -15 12 Z" fill="#ffffff" stroke="#854d0e" stroke-width="0.8" stroke-linejoin="round" />
<circle cx="3" cy="-4" r="1.5" fill="#854d0e" />
</g>
</svg>
GROUP STAGE SIMULATOR
</h1>
<p class="tactical-subtitle">Simulating all 48 teams across 12 groups — FIFA World Cup 2026</p>
</div>""",
        unsafe_allow_html=True
    )

    # ── Live Integration Telemetry ────────────────────────────────────────
    played_matches = load_real_played_matches()
    num_played = len(played_matches) // 2
    
    st.html(f"""
    <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.2); 
                border-radius: 8px; padding: 0.75rem 1.25rem; margin-bottom: 1.5rem; display: flex; 
                align-items: center; gap: 0.75rem;">
        <span style="font-size: 1.25rem;">📊</span>
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.85rem; color: var(--text-color); line-height: 1.4;">
            <strong>Live Standings Integration Active:</strong> The simulation engine has loaded <strong>{num_played} played matches</strong> 
            from the official tournament database (`results.csv`). The standings below are pre-populated with these real results, 
            and the Monte Carlo engine only predicts remaining unplayed fixtures.
        </div>
    </div>
    """)

    # ── Load Elo Ratings ──────────────────────────────────────────────────
    elo_ratings = load_elo_ratings()

    # ── Sidebar Controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-header">SIMULATION SETUP</div>', unsafe_allow_html=True)

        n_sims = st.slider(
            "NUMBER OF SIMULATIONS",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="More simulations = more precise probabilities but slower"
        )

        group_options = ["ALL GROUPS"] + [f"GROUP {g}" for g in WC2026_GROUPS.keys()]
        selected_view = st.selectbox("VIEW", group_options)

        st.markdown("---")

        run_btn = st.button("RUN SIMULATION", width='stretch')

        st.markdown("---")
        st.markdown('<div class="sidebar-header">ABOUT THIS MODEL</div>', unsafe_allow_html=True)
        st.caption(
            "Each simulation plays all 6 round-robin matches per group using Poisson-distributed "
            "goal scoring derived from Elo rating differentials. "
            "Standings resolve via FIFA tiebreakers: Points → Goal Difference → Goals Scored. "
            "Top 2 teams advance automatically; 3rd-place teams are evaluated across all groups "
            "for the 8 best third-place slots."
        )

    # ── Session State ─────────────────────────────────────────────────────
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "sim_count" not in st.session_state:
        st.session_state.sim_count = 0

    # ── Run Simulation ────────────────────────────────────────────────────
    if run_btn:
        with st.spinner(f"Running {n_sims:,} simulations across 12 groups..."):
            results = simulate_all_groups(elo_ratings, n_sims=n_sims, seed=None)
            st.session_state.sim_results = results
            st.session_state.sim_count = n_sims

    # ── Display Results ───────────────────────────────────────────────────
    results = st.session_state.sim_results

    if results is None:
        tab_overview, tab_upsets_pre = st.tabs([
            "📋 GROUPS OVERVIEW",
            "⚡ WC UPSET DETECTOR"
        ])
        
        with tab_overview:
            st.html("""
            <div class="group-card" style="text-align: center; padding: 2rem 1.5rem; margin-bottom: 1.5rem;">
                <div class="group-letter" style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--title-color); margin-bottom: 0.5rem;">
                    SIMULATION NOT YET EXECUTED
                </div>
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text-muted); font-size: 0.9rem;">
                    Configure the number of simulations in the sidebar, then press <strong>RUN SIMULATION</strong> to begin.
                </div>
            </div>
            """)

            st.html("""
            <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700;
                        letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase;">
                Tournament Groups — Current Live Standings
            </div>
            """)

            group_names = list(WC2026_GROUPS.keys())
            for idx in range(0, len(group_names), 3):
                row_groups = group_names[idx:idx+3]
                cols = st.columns(3)
                for col_idx, group_name in enumerate(row_groups):
                    with cols[col_idx]:
                        teams = WC2026_GROUPS[group_name]
                        standings = compute_actual_standings(teams, elo_ratings, played_matches)
                        
                        rows_html = ""
                        for s in standings:
                            flag = get_flag_url(s["team"])
                            gd_val = s["gd"]
                            gd_str = f"+{gd_val}" if gd_val > 0 else str(gd_val)
                            rows_html += f"""
                            <tr>
                                <td class="standings-pos">{s["pos"]}</td>
                                <td class="team-col">
                                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                                        <img src="{flag}" style="width: 18px; height: auto; border-radius: 2px;" crossorigin="anonymous" />
                                        <span class="standings-team-name pre-sim-name">{s["team"]}</span>
                                    </div>
                                </td>
                                <td class="standings-elo">{s["elo"]:.0f}</td>
                                <td class="standings-val">{s["pld"]}</td>
                                <td class="standings-val">{gd_str}</td>
                                <td class="standings-pts">{s["pts"]}</td>
                            </tr>
                            """
                            
                        # Find played matches in this group
                        group_played = []
                        for t_a, t_b in combinations(teams, 2):
                            if (t_a, t_b) in played_matches:
                                goals_a, goals_b = played_matches[(t_a, t_b)]
                                group_played.append(f"{t_a} {goals_a}–{goals_b} {t_b}")
                        
                        played_html = ""
                        if group_played:
                            played_str = ", ".join(group_played)
                            played_html = f"""
                            <div class="group-played-matches">
                                <strong>Played:</strong> {played_str}
                            </div>
                            """
                        else:
                            played_html = f"""
                            <div class="group-played-matches">
                                <em>No matches played yet</em>
                            </div>
                            """

                        st.html(f"""
                        <div class="group-card">
                            <div class="group-card-header">
                                <span class="group-letter">{group_name}</span>
                                <span class="group-label">Group</span>
                            </div>
                            <div style="overflow-x: auto; width: 100%;">
                                <table class="standings-table">
                                    <thead>
                                        <tr>
                                            <th style="width: 6%;">#</th>
                                            <th class="team-col" style="width: 40%;">Team</th>
                                            <th style="width: 14%;">Elo</th>
                                            <th style="width: 13%;">Pld</th>
                                            <th style="width: 13%;">GD</th>
                                            <th style="width: 14%;">Pts</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {rows_html}
                                    </tbody>
                                </table>
                            </div>
                            {played_html}
                        </div>
                        """)
                        
        with tab_upsets_pre:
            render_upset_detector_tab()
            
        return  # Stop here until simulation is run

    # ── Post-simulation display ───────────────────────────────────────────
    st.html(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
        <span class="sim-status sim-complete">SIMULATION COMPLETE</span>
        <span style="font-family: 'Space Grotesk', sans-serif; color: var(--text-muted); font-size: 0.85rem;">
            {st.session_state.sim_count:,} iterations processed
        </span>
    </div>
    """)

    # ── Tabs Navigation ───────────────────────────────────────────────────
    tab_standings, tab_upsets, tab_leaderboard, tab_bracket = st.tabs([
        "📊 GROUP STANDINGS",
        "⚡ WC UPSET DETECTOR",
        "🏆 TOURNAMENT PROGRESSION",
        "🌳 BRACKET SIMULATOR"
    ])

    # ── Tab 1: Group Standings ────────────────────────────────────────────
    with tab_standings:
        standings_png_bytes = get_cached_group_standings_png_v2(results, selected_view)
        col_dummy, col_btn = st.columns([3, 1])
        with col_btn:
            st.download_button(
                label="EXPORT STANDINGS PNG",
                data=standings_png_bytes,
                file_name="group_standings.png",
                mime="image/png",
                width='stretch',
                key="download_standings_btn"
            )

        # Determine which groups to show
        if selected_view == "ALL GROUPS":
            groups_to_show = list(WC2026_GROUPS.keys())
        else:
            groups_to_show = [selected_view.replace("GROUP ", "")]

        for idx in range(0, len(groups_to_show), 2):
            row_groups = groups_to_show[idx:idx+2]
            cols = st.columns(2)
            for col_idx, group_name in enumerate(row_groups):
                # Sort by simulated avg_pts desc, then avg_gd desc, then elo desc
                group_df = results[results["group"] == group_name].sort_values(
                    ["avg_pts", "avg_gd", "elo"], ascending=[False, False, False]
                )
                with cols[col_idx]:
                    teams = WC2026_GROUPS[group_name]
                    actual_standings = compute_actual_standings(teams, elo_ratings, played_matches)
                    actual_map = {s["team"]: s for s in actual_standings}

                    # Group header with team flags
                    header_flags = ""
                    for _, row in group_df.iterrows():
                        flag = get_flag_url(row["team"])
                        header_flags += f'<img src="{flag}" style="width: 24px; border-radius: 3px;" crossorigin="anonymous" />'

                    # Build the table rows
                    rows_html = ""
                    for proj_pos, (_, row) in enumerate(group_df.iterrows(), 1):
                        t_name = row["team"]
                        flag = get_flag_url(t_name)
                        actual_pld = actual_map[t_name]["pld"]
                        actual_pts = actual_map[t_name]["pts"]
                        avg_pts = row["avg_pts"]
                        avg_gd = row["avg_gd"]
                        avg_gd_str = f"+{avg_gd:.1f}" if avg_gd > 0 else f"{avg_gd:.1f}"
                        advance_pct = row["advance_pct"]
                        badge_class = get_badge_class(advance_pct)
                        bar_color = get_progress_color(advance_pct)
                        
                        rows_html += f"""
                        <tr>
                            <td class="standings-pos">{proj_pos}</td>
                            <td class="team-col">
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <img src="{flag}" style="width: 18px; height: auto; border-radius: 2px;" crossorigin="anonymous" />
                                    <span class="standings-team-name post-sim-name">{t_name}</span>
                                </div>
                            </td>
                            <td class="standings-elo">{row["elo"]:.0f}</td>
                            <td class="standings-val">{actual_pld}</td>
                            <td class="standings-val" style="font-weight: 600;">{actual_pts}</td>
                            <td class="standings-val" style="color: #10b981; font-weight: 600;">{avg_pts:.1f}</td>
                            <td class="standings-val">{avg_gd_str}</td>
                            <td style="padding-right: 0.5rem;">
                                <div style="display: flex; align-items: center; gap: 0.4rem; justify-content: flex-end;">
                                    <span class="advance-badge {badge_class}" style="min-width: 55px; padding: 0.15rem 0.35rem; font-size: 0.8rem; margin-top: 0;">{advance_pct:.1f}%</span>
                                </div>
                            </td>
                        </tr>
                        """
                        
                    # Find played matches in this group
                    group_played = []
                    for t_a, t_b in combinations(teams, 2):
                        if (t_a, t_b) in played_matches:
                            goals_a, goals_b = played_matches[(t_a, t_b)]
                            group_played.append(f"{t_a} {goals_a}–{goals_b} {t_b}")
                    
                    played_html = ""
                    if group_played:
                        played_str = ", ".join(group_played)
                        played_html = f"""
                        <div class="group-played-matches">
                            <strong>Played:</strong> {played_str}
                        </div>
                        """
                    else:
                        played_html = f"""
                        <div class="group-played-matches">
                            <em>No matches played yet</em>
                        </div>
                        """
                        
                    group_card_html = f"""
                    <div class="group-card">
                        <div class="group-card-header">
                            <span class="group-letter">{group_name}</span>
                            <span class="group-label">Group</span>
                            <div style="margin-left: auto; display: flex; gap: 0.4rem;">
                                {header_flags}
                            </div>
                        </div>
                        <div style="overflow-x: auto; width: 100%;">
                            <table class="standings-table">
                                <thead>
                                    <tr>
                                        <th style="width: 6%;">Proj</th>
                                        <th class="team-col" style="width: 32%;">Team</th>
                                        <th style="width: 11%;">Elo</th>
                                        <th style="width: 7%;">Pld</th>
                                        <th style="width: 7%;">Pts</th>
                                        <th style="width: 11%;">Avg Pts</th>
                                        <th style="width: 11%;">Avg GD</th>
                                        <th style="width: 15%; text-align: right; padding-right: 0.5rem;">Advance %</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rows_html}
                                </tbody>
                            </table>
                        </div>
                        {played_html}
                    </div>
                    """
                    st.html(group_card_html)

                    # Position distribution chart
                    chart = build_group_chart(group_df, group_name)
                    st.plotly_chart(chart, width='stretch', key=f"chart_{group_name}")

    with tab_upsets:
        render_upset_detector_tab()

    # ── Tab 3: Tournament Leaderboard ─────────────────────────────────────
    with tab_leaderboard:
        progression_png_bytes = get_cached_progression_png_v2(results)

        col_lbl, col_btn = st.columns([3, 1])
        with col_lbl:
            st.markdown(
                """
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700;
                            letter-spacing: 0.1em; color: var(--text-muted); text-transform: uppercase; margin-top: 0.6rem;">
                    Tournament-Wide Knockout Progression Leaderboard
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_btn:
            st.download_button(
                label="EXPORT LEADERBOARD PNG",
                data=progression_png_bytes,
                file_name="tournament_progression.png",
                mime="image/png",
                width='stretch',
                key="download_leaderboard_btn"
            )

        # Sort by Champion %, then Final, SF, QF, R16, R32
        leaderboard = results.sort_values(
            ["champion_pct", "final_pct", "sf_pct", "qf_pct", "r16_pct", "r32_pct", "elo"],
            ascending=False
        ).reset_index(drop=True)

        rows_html = ""
        for rank, (_, row) in enumerate(leaderboard.iterrows(), 1):
            flag = get_flag_url(row["team"])

            rows_html += f"""
            <tr>
                <td><span class="lb-rank">{rank}</span></td>
                <td>
                    <div class="lb-team">
                        <img class="lb-flag" src="{flag}" crossorigin="anonymous" />
                        <span class="lb-name">{row['team']}</span>
                    </div>
                </td>
                <td><span class="lb-group">{row['group']}</span></td>
                <td><span class="lb-stat">{row['elo']:.0f}</span></td>
                <td><span class="advance-badge {get_badge_class(row['r32_pct'])}">{row['r32_pct']:.1f}%</span></td>
                <td><span class="advance-badge {get_badge_class(row['r16_pct'])}">{row['r16_pct']:.1f}%</span></td>
                <td><span class="advance-badge {get_badge_class(row['qf_pct'])}">{row['qf_pct']:.1f}%</span></td>
                <td><span class="advance-badge {get_badge_class(row['sf_pct'])}">{row['sf_pct']:.1f}%</span></td>
                <td><span class="advance-badge {get_badge_class(row['final_pct'])}">{row['final_pct']:.1f}%</span></td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="mini-progress" style="width: 60px;">
                            <div class="mini-progress-fill" style="width: {row['champion_pct']}%; background: #10b981;"></div>
                        </div>
                        <span class="advance-badge badge-high" style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #10b981;">{row['champion_pct']:.1f}%</span>
                    </div>
                </td>
            </tr>
            """

        st.html(f"""
        <div class="group-card" style="overflow-x: auto;">
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th style="width: 5%;">#</th>
                        <th style="width: 20%;">Team</th>
                        <th style="width: 8%;">Group</th>
                        <th style="width: 9%;">Elo</th>
                        <th style="width: 9%;">Reach R32</th>
                        <th style="width: 9%;">Reach R16</th>
                        <th style="width: 9%;">Reach QF</th>
                        <th style="width: 9%;">Reach SF</th>
                        <th style="width: 9%;">Reach Final</th>
                        <th style="width: 13%;">Champion %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

    # ── Tab 3: Bracket Tree Simulator ─────────────────────────────────────
    with tab_bracket:
        st.html("""
        <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700;
                    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">
            Live Tournament Bracket Simulator
        </div>
        """)

        if "sample_bracket" not in st.session_state:
            from src.group_simulator import simulate_single_tournament
            rng = np.random.default_rng(None)
            st.session_state.sample_bracket = simulate_single_tournament(elo_ratings, rng)

        col1, col2 = st.columns([1, 4])
        with col1:
            sim_single_btn = st.button("SIMULATE SINGLE RUN", key="sim_single_run_btn", width='stretch')
            if sim_single_btn:
                from src.group_simulator import simulate_single_tournament
                rng = np.random.default_rng(None)
                st.session_state.sample_bracket = simulate_single_tournament(elo_ratings, rng)
                st.rerun()

        bracket_data = st.session_state.sample_bracket

        with col2:
            bracket_png_bytes = get_cached_bracket_png_v2(bracket_data)
            st.download_button(
                label="EXPORT BRACKET PNG",
                data=bracket_png_bytes,
                file_name="tournament_bracket.png",
                mime="image/png",
                width='stretch',
                key="download_bracket_btn"
            )

        bracket_data = st.session_state.sample_bracket

        # Build column HTMLs
        # Column 1: Round of 32
        r32_html = '<div class="bracket-column">'
        for idx, m in enumerate(bracket_data["knockouts"]["r32"], 1):
            r32_html += render_match_html(m, f"R32 Match {idx}")
        r32_html += '</div>'

        # Column 2: Round of 16
        r16_html = '<div class="bracket-column">'
        for idx, m in enumerate(bracket_data["knockouts"]["r16"], 1):
            r16_html += render_match_html(m, f"R16 Match {idx}")
        r16_html += '</div>'

        # Column 3: Quarterfinals
        qf_html = '<div class="bracket-column">'
        for idx, m in enumerate(bracket_data["knockouts"]["qf"], 1):
            qf_html += render_match_html(m, f"Quarterfinal {idx}")
        qf_html += '</div>'

        # Column 4: Semifinals
        sf_html = '<div class="bracket-column">'
        for idx, m in enumerate(bracket_data["knockouts"]["sf"], 1):
            sf_html += render_match_html(m, f"Semifinal {idx}")
        sf_html += '</div>'

        # Column 5: Final & Champion
        final_m = bracket_data["knockouts"]["final"]
        final_html = '<div class="bracket-column">'
        final_html += render_match_html(final_m, "World Cup Final")

        champ = final_m["winner"]
        champ_flag = get_flag_url(champ)

        final_html += f"""
        <div class="champion-box">
            <div class="champion-title">🏆 CHAMPION 🏆</div>
            <img class="champion-flag" src="{champ_flag}" crossorigin="anonymous" />
            <div class="champion-name">{champ}</div>
        </div>
        """
        final_html += '</div>'

        # Combine all columns into wrapper
        full_bracket_html = f"""
        <div class="bracket-wrapper">
            {r32_html}
            {r16_html}
            {qf_html}
            {sf_html}
            {final_html}
        </div>
        """
        st.html(full_bracket_html)


main()
