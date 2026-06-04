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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    min-width: 160px;
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
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Simulator Logic
# ══════════════════════════════════════════════════════════════════════════════

from src.group_simulator import WC2026_GROUPS, FALLBACK_ELO, simulate_all_groups


@st.cache_resource(show_spinner="Loading analytical engine and computing Elo ratings…")
def load_elo_ratings():
    """Load the EloSystem from historical data. Falls back to hardcoded ratings."""
    try:
        from src.predict import Predictor
        predictor = Predictor()
        # Extract current Elo for all WC teams
        all_wc_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
        elo_dict = {}
        for team in all_wc_teams:
            rating = predictor.elo_system.get_rating(team)
            # If team has no historical data, use fallback
            if rating == predictor.elo_system.initial_rating and team in FALLBACK_ELO:
                elo_dict[team] = FALLBACK_ELO[team]
            else:
                elo_dict[team] = rating
        return elo_dict
    except Exception:
        # Fallback to hardcoded Elo ratings
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
        margin=dict(t=10, b=10, l=0, r=15),
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
            automargin=True,
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
    st.markdown("""
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
    """, unsafe_allow_html=True)

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

        run_btn = st.button("RUN SIMULATION", use_container_width=True)

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
        st.markdown("""
        <div class="group-card" style="text-align: center; padding: 3rem 2rem;">
            <div class="group-letter" style="font-size: 3rem; margin-bottom: 1rem;">⚙</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--title-color); margin-bottom: 0.5rem;">
                SIMULATION NOT YET EXECUTED
            </div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text-muted); font-size: 0.9rem;">
                Configure the number of simulations in the sidebar, then press <strong>RUN SIMULATION</strong> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show groups preview
        st.markdown("""
        <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700;
                    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1rem; text-transform: uppercase;">
            Tournament Groups — Pre-Simulation Overview
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, (group_name, teams) in enumerate(WC2026_GROUPS.items()):
            with cols[idx % 3]:
                flags_html = ""
                for t in teams:
                    flag = get_flag_url(t)
                    elo = elo_ratings.get(t, FALLBACK_ELO.get(t, 1500))
                    flags_html += f"""
                    <div class="sim-team-row">
                        <img class="sim-team-flag" src="{flag}" alt="{t}" />
                        <span class="sim-team-name">{t}</span>
                        <span class="sim-team-elo">{elo:.0f}</span>
                    </div>
                    """

                st.markdown(f"""
                <div class="group-card">
                    <div class="group-card-header">
                        <span class="group-letter">{group_name}</span>
                        <span class="group-label">Group</span>
                    </div>
                    {flags_html}
                </div>
                """, unsafe_allow_html=True)

        return  # Stop here until simulation is run

    # ── Post-simulation display ───────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
        <span class="sim-status sim-complete">SIMULATION COMPLETE</span>
        <span style="font-family: 'Space Grotesk', sans-serif; color: var(--text-muted); font-size: 0.85rem;">
            {st.session_state.sim_count:,} iterations processed
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Determine which groups to show
    if selected_view == "ALL GROUPS":
        groups_to_show = list(WC2026_GROUPS.keys())
    else:
        groups_to_show = [selected_view.replace("GROUP ", "")]

    # ── Group Cards ───────────────────────────────────────────────────────
    cols = st.columns(2)
    for idx, group_name in enumerate(groups_to_show):
        group_df = results[results["group"] == group_name].sort_values("advance_pct", ascending=False)

        with cols[idx % 2]:
            # Group header with team flags
            header_flags = ""
            for _, row in group_df.iterrows():
                flag = get_flag_url(row["team"])
                header_flags += f'<img src="{flag}" style="width: 24px; border-radius: 3px;" />'

            st.markdown(f"""
            <div class="group-card">
                <div class="group-card-header">
                    <span class="group-letter">{group_name}</span>
                    <span class="group-label">Group</span>
                    <div style="margin-left: auto; display: flex; gap: 0.4rem;">
                        {header_flags}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Team rows with advancement badges
            for rank, (_, row) in enumerate(group_df.iterrows(), 1):
                flag = get_flag_url(row["team"])
                badge_class = get_badge_class(row["advance_pct"])
                bar_color = get_progress_color(row["advance_pct"])

                st.markdown(f"""
                <div class="sim-team-row">
                    <span class="sim-team-rank">{rank}</span>
                    <img class="sim-team-flag" src="{flag}" alt="{row['team']}" />
                    <span class="sim-team-name">{row['team']}</span>
                    <span class="sim-team-elo">{row['elo']:.0f}</span>
                    <div style="flex: 1;">
                        <div class="mini-progress">
                            <div class="mini-progress-fill" style="width: {row['advance_pct']}%; background: {bar_color};"></div>
                        </div>
                    </div>
                    <span class="advance-badge {badge_class}">{row['advance_pct']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Position distribution chart
            chart = build_group_chart(group_df, group_name)
            st.plotly_chart(chart, use_container_width=True, key=f"chart_{group_name}")

    # ── Tournament Leaderboard ────────────────────────────────────────────
    if selected_view == "ALL GROUPS":
        st.markdown("---")
        st.markdown("""
        <div style="font-family: 'Outfit', sans-serif; font-size: 0.8rem; font-weight: 700;
                    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">
            Tournament-Wide Advancement Leaderboard
        </div>
        """, unsafe_allow_html=True)

        leaderboard = results.sort_values("advance_pct", ascending=False).reset_index(drop=True)

        # Build the leaderboard as HTML table
        rows_html = ""
        for rank, (_, row) in enumerate(leaderboard.iterrows(), 1):
            flag = get_flag_url(row["team"])
            badge_class = get_badge_class(row["advance_pct"])
            bar_color = get_progress_color(row["advance_pct"])

            rows_html += f"""
            <tr>
                <td><span class="lb-rank">{rank}</span></td>
                <td>
                    <div class="lb-team">
                        <img class="lb-flag" src="{flag}" />
                        <span class="lb-name">{row['team']}</span>
                    </div>
                </td>
                <td><span class="lb-group">{row['group']}</span></td>
                <td><span class="lb-stat">{row['elo']:.0f}</span></td>
                <td>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="mini-progress" style="width: 80px;">
                            <div class="mini-progress-fill" style="width: {row['advance_pct']}%; background: {bar_color};"></div>
                        </div>
                        <span class="advance-badge {badge_class}">{row['advance_pct']:.1f}%</span>
                    </div>
                </td>
                <td><span class="lb-stat">{row['1st']:.1f}%</span></td>
                <td><span class="lb-stat">{row['avg_pts']:.1f}</span></td>
                <td><span class="lb-stat">{row['avg_gd']:+.1f}</span></td>
            </tr>
            """

        st.markdown(f"""
        <div class="group-card" style="overflow-x: auto;">
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Team</th>
                        <th>Group</th>
                        <th>Elo</th>
                        <th>Advance %</th>
                        <th>Win Group</th>
                        <th>Avg Pts</th>
                        <th>Avg GD</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)


main()
