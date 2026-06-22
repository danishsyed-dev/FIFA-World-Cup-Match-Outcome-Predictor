"""
streamlit_app.py
----------------
Streamlit dashboard for the FIFA Match Outcome Predictor.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all, load_elo, DATA_DIR
import importlib
import src.image_generator
importlib.reload(src.image_generator)
from src.image_generator import generate_matchup_png

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def get_cached_matchup_png_v2(home_team, away_team, home_elo, away_elo, hw, dr, aw, predicted, confidence):
    # force cache reload: v6
    return generate_matchup_png(home_team, away_team, home_elo, away_elo, hw, dr, aw, predicted, confidence)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared Theme ──────────────────────────────────────────────────────────────
from app.shared_theme import inject_theme, get_flag_url, COUNTRY_TO_ISO

inject_theme()

# ── Page-Specific CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Matchup Billboard */
    .matchup-billboard {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-left: 6px solid var(--home-color) !important;
        border-right: 6px solid var(--away-color) !important;
        border-radius: 12px;
        padding: 1.5rem 2.5rem;
        margin-bottom: 2rem;
        gap: 1rem;
    }
    .team-panel {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        flex: 1;
        min-width: 0;
    }
    .home-panel {
        justify-content: flex-start;
    }
    .away-panel {
        flex-direction: row-reverse;
        justify-content: flex-start;
        text-align: right;
    }
    .home-panel .team-role {
        background: color-mix(in srgb, var(--home-color), transparent 88%) !important;
        color: var(--home-color) !important;
        border: 1px solid color-mix(in srgb, var(--home-color), transparent 70%) !important;
        padding: 0.15rem 0.6rem !important;
        border-radius: 4px !important;
        display: inline-block !important;
        align-self: flex-start;
    }
    .away-panel .team-role {
        background: color-mix(in srgb, var(--away-color), transparent 88%) !important;
        color: var(--away-color) !important;
        border: 1px solid color-mix(in srgb, var(--away-color), transparent 70%) !important;
        padding: 0.15rem 0.6rem !important;
        border-radius: 4px !important;
        display: inline-block !important;
        align-self: flex-end;
    }
    .team-flag {
        width: 75px;
        height: auto;
        border-radius: 6px;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
    }
    .team-flag:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .team-info {
        display: flex;
        flex-direction: column;
        min-width: 0;
    }
    .team-role {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }
    .team-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--title-color);
        margin: 0;
        line-height: 1.2;
    }
    .team-elo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 4px;
    }
    .vs-divider {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 130px;
        padding: 0 1.5rem;
        border-left: 1px solid var(--card-border);
        border-right: 1px solid var(--card-border);
    }
    .vs-circle {
        background: var(--btn-bg);
        border: 1px solid var(--card-border);
        color: var(--draw-color);
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.95rem;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .vs-tournament {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-muted);
        text-align: center;
        white-space: nowrap;
    }
    .vs-venue {
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-top: 3px;
        text-align: center;
        opacity: 0.8;
    }

    /* Prediction Telemetry Slider */
    .prediction-panel {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
    }
    .panel-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: var(--text-muted);
    }
    .confidence-badge {
        background: rgba(229, 193, 88, 0.15);
        color: var(--draw-color);
        border: 1px solid rgba(229, 193, 88, 0.3);
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    .probability-track {
        display: flex;
        height: 52px;
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid var(--card-border);
        margin-bottom: 1.25rem;
        background-color: var(--background-color);
    }
    .prob-segment {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #0f1414;
        overflow: hidden;
        white-space: nowrap;
        padding: 0 0.5rem;
        transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .seg-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        line-height: 1;
    }
    .seg-lbl {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-top: 3px;
        opacity: 0.85;
    }
    .predicted-winner-banner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-top: 1px solid var(--card-border);
        padding-top: 1rem;
        font-family: 'Outfit', sans-serif;
    }
    .banner-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-muted);
        letter-spacing: 0.05em;
    }
    .banner-value {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .winner-home { color: var(--home-color); }
    .winner-away { color: var(--away-color); }
    .winner-draw { color: var(--draw-color); }

    /* Match Stats Comparison Table */
    .stats-sheet {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        box-sizing: border-box;
    }
    .sheet-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 1.25rem;
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 0.5rem;
    }
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
    }
    .comparison-table th {
        font-size: 0.75rem;
        color: var(--text-muted);
        letter-spacing: 0.05em;
        padding: 0.5rem 0.25rem;
        border-bottom: 1px solid var(--card-border);
        font-weight: 700;
    }
    .comparison-table td {
        padding: 0.75rem 0.25rem;
        border-bottom: 1px solid color-mix(in srgb, var(--background-color), var(--text-color) 8%);
    }
    .stat-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .home-color { color: var(--home-color); }
    .away-color { color: var(--away-color); }
    .stat-lbl {
        text-align: center;
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .stat-lbl-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        padding: 0.75rem 0 0.25rem 0;
        opacity: 0.85;
    }

    /* Plotly Custom Theme-Adaptability Overrides */
    .js-line, .gridlayer path {
        stroke: var(--card-border) !important;
    }
    .xtick text, .ytick text {
        fill: var(--text-muted) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .gtitle {
        fill: var(--title-color) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .bartext, .bartext text {
        fill: var(--title-color) !important;
        color: var(--title-color) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Small screen layout adjustments */
    @media (max-width: 768px) {
        .matchup-billboard {
            flex-direction: column;
            padding: 1.5rem;
            text-align: center;
        }
        .team-panel {
            flex-direction: column;
            align-items: center;
            gap: 0.75rem;
        }
        .away-panel {
            flex-direction: column;
            text-align: center;
        }
        .vs-divider {
            border-left: none;
            border-right: none;
            border-top: 1px solid var(--card-border);
            border-bottom: 1px solid var(--card-border);
            padding: 1rem 0;
            width: 100%;
        }
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load predictor (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Booting analytical engine and loading historical dataset…")
def load_predictor():
    """Cache the Predictor object so it is only loaded once."""
    try:
        import importlib
        import src.predict
        importlib.reload(src.predict)
        from src.predict import Predictor
        return Predictor()
    except FileNotFoundError as e:
        return str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_all_teams(predictor) -> list:
    """Return sorted list of all teams in the historical dataset."""
    teams = set(predictor.df["home_team"].unique()) | set(predictor.df["away_team"].unique())
    return sorted(teams)

def elo_comparison_chart(home_team, away_team, home_elo, away_elo):
    """Bar chart comparing Elo ratings."""
    fig = go.Figure(go.Bar(
        x=[home_team, away_team],
        y=[home_elo, away_elo],
        marker_color=["#38bdf8", "#f87171"],  # Home and Away Team Accents from theme
        text=[f"{home_elo:.0f}", f"{away_elo:.0f}"],
        textposition="outside",
        textfont=dict(size=13, family="Space Grotesk", color="#f8fafc"),
        width=0.4,
    ))
    fig.update_layout(
        title={
            "text": "ELO RATING COMPARISON",
            "font": {"size": 13, "family": "Outfit", "weight": "bold", "color": "#f8fafc"},
            "pad": {"b": 10}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            gridcolor="#232b2b", 
            showgrid=True, 
            zeroline=False,
            range=[min(home_elo, away_elo) - 150, max(home_elo, away_elo) + 150],
            tickfont=dict(family="Space Grotesk", color="#94a3b8")
        ),
        xaxis=dict(
            tickfont=dict(size=13, family="Outfit", color="#f8fafc"),
            showgrid=False
        ),
        margin=dict(t=50, b=10, l=10, r=10),
        height=260,
    )
    return fig

# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Tactical Header Badge & Title with Geometric SVG Soccer Ball
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
FIFA MATCH OUTCOME PREDICTOR
</h1>
<p class="tactical-subtitle">Data-driven machine learning models forecasting international fixtures</p>
</div>""",
        unsafe_allow_html=True
    )

    predictor = load_predictor()

    if isinstance(predictor, str):
        st.error(f"Configuration Error: Model file not found. Please run training first.\n\n```\npython src/train.py\n```\n\n{predictor}")
        st.stop()

    all_teams = get_all_teams(predictor)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-header">MATCH SETUP</div>', unsafe_allow_html=True)

        # Session State Initializations
        if "home_team" not in st.session_state:
            st.session_state.home_team = "Portugal"
        if "away_team" not in st.session_state:
            st.session_state.away_team = "Germany"
        if "tournament" not in st.session_state:
            st.session_state.tournament = "FIFA World Cup"
        if "neutral" not in st.session_state:
            st.session_state.neutral = False

        # Presets Setup
        presets = {
            "CUSTOM MATCHUP": None,
            "CLASSIC: BRAZIL VS ARGENTINA": ("Brazil", "Argentina", "FIFA World Cup", False),
            "EURO 2024: GERMANY VS SPAIN": ("Germany", "Spain", "Continental Cup", True),
            "WORLD CUP FINAL: ARGENTINA VS FRANCE": ("Argentina", "France", "FIFA World Cup", True),
            "CLASSIC: ENGLAND VS GERMANY": ("England", "Germany", "FIFA World Cup", False),
            "BORDER RIVALRY: USA VS MEXICO": ("USA", "Mexico", "Friendly", False),
        }
        
        selected_preset = st.selectbox("PRESET FIXTURE", list(presets.keys()))
        if selected_preset != "CUSTOM MATCHUP" and presets[selected_preset] is not None:
            p_home, p_away, p_tournament, p_neutral = presets[selected_preset]
            st.session_state.home_team = p_home
            st.session_state.away_team = p_away
            st.session_state.tournament = p_tournament
            st.session_state.neutral = p_neutral
            # Reset selectbox to custom matchup and rerun to apply
            st.rerun()

        # Find select index
        try:
            home_idx = all_teams.index(st.session_state.home_team)
        except ValueError:
            home_idx = 0

        try:
            away_idx = all_teams.index(st.session_state.away_team)
        except ValueError:
            away_idx = min(1, len(all_teams) - 1)

        home_team = st.selectbox("HOME TEAM", all_teams, index=home_idx)
        st.session_state.home_team = home_team

        # Swap button between teams (no emojis)
        if st.button("SWAP HOME & AWAY", width='stretch'):
            st.session_state.home_team, st.session_state.away_team = st.session_state.away_team, st.session_state.home_team
            st.rerun()

        away_team = st.selectbox("AWAY TEAM", all_teams, index=away_idx)
        st.session_state.away_team = away_team

        tournament_list = ["Friendly", "Qualifier", "Continental Cup", "FIFA World Cup"]
        try:
            tourney_idx = tournament_list.index(st.session_state.tournament)
        except ValueError:
            tourney_idx = 3

        tournament = st.selectbox("TOURNAMENT TYPE", tournament_list, index=tourney_idx)
        st.session_state.tournament = tournament

        neutral = st.toggle("NEUTRAL VENUE", value=st.session_state.neutral)
        st.session_state.neutral = neutral

        st.markdown("---")
        predict_btn = st.button("RUN PREDICTION TELEMETRY", width='stretch')

        st.markdown("---")
        st.markdown('<div class="sidebar-header">TECHNICAL ABOUT</div>', unsafe_allow_html=True)
        st.caption(
            "This model incorporates pre-match Elo rating differentials, rolling form scores, "
            "rolling average goals scored & conceded, and head-to-head history. "
            "Model parameters are optimized for a 55% - 70% confidence interval."
        )

    # ── Main content ──────────────────────────────────────────────────────────
    if home_team == away_team:
        st.warning("Invalid Matchup: Select two different teams in the match setup panel.")
        st.stop()

    # Predictions automatically update reactively
    if predict_btn or True:
        with st.spinner("Executing prediction telemetry..."):
            result = predictor.predict(
                home_team=home_team,
                away_team=away_team,
                neutral=neutral,
                tournament=tournament,
            )

        hw = result["home_win_prob"]
        dr = result["draw_prob"]
        aw = result["away_win_prob"]
        predicted = result["predicted_outcome"]
        confidence = result["confidence"]
        home_elo = result["home_elo"]
        away_elo = result["away_elo"]

        # Generate flag URLs from CDN
        home_flag_url = get_flag_url(home_team)
        away_flag_url = get_flag_url(away_team)
        venue_text = "NEUTRAL VENUE" if neutral else "HOME ADVANTAGE"

        # ── Match Billboard ──────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="matchup-billboard">
                <div class="team-panel home-panel">
                    <img class="team-flag" src="{home_flag_url}" alt="{home_team}" crossorigin="anonymous" />
                    <div class="team-info">
                        <span class="team-role">HOME</span>
                        <h2 class="team-name">{home_team}</h2>
                        <span class="team-elo">ELO {home_elo:.0f}</span>
                    </div>
                </div>
                <div class="vs-divider">
                    <div class="vs-circle">VS</div>
                    <div class="vs-tournament">{tournament.upper()}</div>
                    <div class="vs-venue">{venue_text}</div>
                </div>
                <div class="team-panel away-panel">
                    <img class="team-flag" src="{away_flag_url}" alt="{away_team}" crossorigin="anonymous" />
                    <div class="team-info">
                        <span class="team-role">AWAY</span>
                        <h2 class="team-name">{away_team}</h2>
                        <span class="team-elo">ELO {away_elo:.0f}</span>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # ── Unified Telemetry Prediction Slider ──────────────────────────────
        if predicted == "Home Win":
            winner_class = "winner-home"
            decision_text = f"{home_team} predicted Win"
        elif predicted == "Away Win":
            winner_class = "winner-away"
            decision_text = f"{away_team} predicted Win"
        else:
            winner_class = "winner-draw"
            decision_text = "Draw outcome predicted"

        st.markdown(
            f"""
            <div class="prediction-panel">
                <div class="panel-header">
                    <span class="panel-title">TELEMETRY PREDICTION OUTCOME</span>
                    <span class="confidence-badge">CONFIDENCE: {confidence*100:.1f}%</span>
                </div>
                <div class="probability-track">
                    <div class="prob-segment" style="width: {hw*100}%; background-color: #38bdf8;">
                        <span class="seg-val">{hw*100:.1f}%</span>
                        <span class="seg-lbl">HOME WIN</span>
                    </div>
                    <div class="prob-segment" style="width: {dr*100}%; background-color: #e5c158;">
                        <span class="seg-val">{dr*100:.1f}%</span>
                        <span class="seg-lbl">DRAW</span>
                    </div>
                    <div class="prob-segment" style="width: {aw*100}%; background-color: #f87171;">
                        <span class="seg-val">{aw*100:.1f}%</span>
                        <span class="seg-lbl">AWAY WIN</span>
                    </div>
                </div>
                <div class="predicted-winner-banner">
                    <span class="banner-label">MODEL DECISION:</span>
                    <span class="banner-value {winner_class}">{decision_text.upper()}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Details section ───────────────────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                elo_comparison_chart(home_team, away_team, home_elo, away_elo),
                width='stretch',
            )

        with right:
            # Compute supporting stats
            today = pd.Timestamp.now()
            from src.feature_engineering import _recent_stats, _h2h_stats

            hf, hgs, hgc = _recent_stats(predictor.df, home_team, today)
            af, ags, agc = _recent_stats(predictor.df, away_team, today)
            
            # Head-to-head calculation details
            h2h_hw, h2h_draws = _h2h_stats(predictor.df, home_team, away_team, today)
            
            # Find total actual matches in H2H history (last 10)
            mask = (
                ((predictor.df["home_team"] == home_team) & (predictor.df["away_team"] == away_team)) |
                ((predictor.df["home_team"] == away_team) & (predictor.df["away_team"] == home_team))
            ) & (predictor.df["date"] < today)
            total_meetings = len(predictor.df[mask].tail(10))
            h2h_aw = total_meetings - h2h_hw - h2h_draws

            st.markdown(
                f"""
                <div class="stats-sheet">
                    <div class="sheet-title">TACTICAL METRICS DATA SHEET</div>
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th style="width: 40%; text-align: left;">{home_team.upper()}</th>
                                <th style="width: 20%; text-align: center;">METRIC</th>
                                <th style="width: 40%; text-align: right;">{away_team.upper()}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="stat-val home-color">{home_elo:.0f}</td>
                                <td class="stat-lbl">Elo Rating</td>
                                <td class="stat-val away-color" style="text-align: right;">{away_elo:.0f}</td>
                            </tr>
                            <tr>
                                <td class="stat-val home-color">{hf:.1f} pts</td>
                                <td class="stat-lbl">Recent Form (5 matches)</td>
                                <td class="stat-val away-color" style="text-align: right;">{af:.1f} pts</td>
                            </tr>
                            <tr>
                                <td class="stat-val home-color">{hgs:.2f}</td>
                                <td class="stat-lbl">Avg Goals Scored</td>
                                <td class="stat-val away-color" style="text-align: right;">{ags:.2f}</td>
                            </tr>
                            <tr>
                                <td class="stat-val home-color">{hgc:.2f}</td>
                                <td class="stat-lbl">Avg Goals Conceded</td>
                                <td class="stat-val away-color" style="text-align: right;">{agc:.2f}</td>
                            </tr>
                            <tr>
                                <td class="stat-val home-color">{h2h_hw} Wins</td>
                                <td class="stat-lbl">Head-to-Head</td>
                                <td class="stat-val away-color" style="text-align: right;">{h2h_aw} Wins</td>
                            </tr>
                            <tr>
                                <td colspan="3" class="stat-lbl-sub" style="text-align: center;">
                                    {h2h_draws} matches ended in a draw (last 10 historical meetings)
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )

        matchup_png_bytes = get_cached_matchup_png_v2(
            home_team=home_team,
            away_team=away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            hw=hw,
            dr=dr,
            aw=aw,
            predicted=predicted,
            confidence=confidence
        )

        col_dummy, col_btn = st.columns([3, 1])
        with col_btn:
            st.download_button(
                label="EXPORT MATCHUP PNG",
                data=matchup_png_bytes,
                file_name=f"{home_team.lower()}_vs_{away_team.lower()}_prediction.png",
                mime="image/png",
                width='stretch',
                key="download_matchup_btn"
            )

        # ── Batch simulation ──────────────────────────────────────────────
        with st.expander("BATCH PREDICTION SIMULATOR"):
            st.caption("Upload a CSV file containing columns: home_team, away_team, neutral (bool), tournament (str)")
            uploaded = st.file_uploader("Upload fixtures CSV", type=["csv"], label_visibility="collapsed")
            if uploaded:
                fixtures = pd.read_csv(uploaded)
                predictions = predictor.predict_batch(fixtures)
                st.dataframe(predictions, width='stretch')
                csv_bytes = predictions.to_csv(index=False).encode()
                st.download_button("DOWNLOAD PREDICTIONS", csv_bytes, "predictions.csv", "text/csv")

        # ── Model info ────────────────────────────────────────────────────
        with st.expander("MODEL INFORMATION & FEATURES"):
            st.markdown("""
            **Features used by the model:**
            | Feature | Description |
            |---------|-------------|
            | `elo_difference` | Pre-match Elo rating difference |
            | `home/away_recent_form` | Points (W=3, D=1, L=0) from last 5 matches |
            | `home/away_avg_goals_scored` | Average goals scored in last 5 matches |
            | `home/away_avg_goals_conceded` | Average goals conceded in last 5 matches |
            | `h2h_home_wins` | Head-to-head wins (last 10 meetings) |
            | `h2h_draws` | Head-to-head draws (last 10 meetings) |
            | `home_advantage` | 1 if home venue, 0 if neutral |
            | `tournament_weight` | Encoded tournament importance (0–4) |

            **Target classes:** 0=Away Win · 1=Draw · 2=Home Win

            **Data source:** International Football Results 1872–2026 (Kaggle)
            """)

if __name__ == "__main__":
    main()
