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

import src.image_generator
import src.group_simulator

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data(show_spinner=False)
def get_cached_group_standings_png(results, selected_view):
    from src.image_generator import generate_group_standings_png
    return generate_group_standings_png(results, selected_view)

@st.cache_data(show_spinner=False)
def get_cached_progression_png(results):
    from src.image_generator import generate_progression_png
    return generate_progression_png(results)

@st.cache_data(show_spinner=False)
def get_cached_bracket_png(bracket_data):
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
    width: 28px;
    height: auto;
    border-radius: 3px;
}
.lb-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--title-color);
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
            tickfont=dict(size=12, family="Outfit", weight="bold"),
            automargin=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Outfit"),
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
    st.html("""
    <div class="tactical-header">
        <div class="telemetry-badge">MONTE CARLO SIMULATION ENGINE v1.0</div>
        <h1 class="tactical-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 32px; height: 32px; vertical-align: middle; margin-right: 12px; margin-top: -4px;">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
            GROUP STAGE SIMULATOR
        </h1>
        <p class="tactical-subtitle">Simulating all 48 teams across 12 groups — FIFA World Cup 2026</p>
    </div>
    """)

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
        # Show initial state — group overview with Elo ratings
        st.html("""
        <div class="group-card" style="text-align: center; padding: 3rem 2rem;">
            <div class="group-letter" style="font-size: 3rem; margin-bottom: 1rem;">⚙</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--title-color); margin-bottom: 0.5rem;">
                SIMULATION NOT YET EXECUTED
            </div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text-muted); font-size: 0.9rem;">
                Configure the number of simulations in the sidebar, then press <strong>RUN SIMULATION</strong> to begin.
            </div>
        </div>
        """)

        # Show groups preview
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
    tab_standings, tab_leaderboard, tab_bracket = st.tabs([
        "📊 GROUP STANDINGS",
        "🏆 TOURNAMENT PROGRESSION",
        "🌳 BRACKET SIMULATOR"
    ])

    # ── Tab 1: Group Standings ────────────────────────────────────────────
    with tab_standings:
        standings_png_bytes = get_cached_group_standings_png(results, selected_view)
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

    # ── Tab 2: Tournament Leaderboard ─────────────────────────────────────
    with tab_leaderboard:
        progression_png_bytes = get_cached_progression_png(results)

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
                        <th>#</th>
                        <th>Team</th>
                        <th>Group</th>
                        <th>Elo</th>
                        <th>Reach R32</th>
                        <th>Reach R16</th>
                        <th>Reach QF</th>
                        <th>Reach SF</th>
                        <th>Reach Final</th>
                        <th>Champion %</th>
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
            bracket_png_bytes = get_cached_bracket_png(bracket_data)
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
