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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

    /* Global Dynamic Theme System */
    :root {
        color-scheme: light dark;
        --background-color: light-dark(#ffffff, #0c0f0f);
        --text-color: light-dark(#1f2937, #f8fafc);
        --home-color: light-dark(#0284c7, #38bdf8);
        --away-color: light-dark(#dc2626, #f87171);
        --draw-color: light-dark(#d97706, #e5c158);
        --text-muted: light-dark(#4b5563, #8b9e9b);
        --title-color: light-dark(#0f172a, #f8fafc);
        --card-bg: color-mix(in srgb, var(--background-color), var(--text-color) 4%);
        --card-border: color-mix(in srgb, var(--background-color), var(--text-color) 12%);
        --card-bg-hover: color-mix(in srgb, var(--background-color), var(--text-color) 8%);
        --btn-bg: color-mix(in srgb, var(--background-color), var(--text-color) 6%);
        --btn-hover: color-mix(in srgb, var(--background-color), var(--text-color) 10%);
    }

    /* Global CSS Overrides */
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }

    .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }

    [data-testid="stSidebar"] {
        background-color: color-mix(in srgb, var(--background-color), var(--text-color) 3%) !important;
        border-right: 1px solid var(--card-border) !important;
    }

    /* Style selectboxes and inputs in sidebar */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        cursor: pointer !important;
    }
    /* Force pointer cursor on all parts of streamlit selectbox */
    div[data-testid="stSelectbox"] div, 
    div[data-testid="stSelectbox"] span, 
    div[data-testid="stSelectbox"] svg {
        cursor: pointer !important;
    }
    div[role="listbox"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
    }
    div[role="option"], 
    div[role="option"] * {
        color: var(--text-color) !important;
        background-color: transparent !important;
        cursor: pointer !important;
    }
    div[role="option"]:hover {
        background-color: var(--card-bg-hover) !important;
    }
    
    /* Expander containers */
    div[data-testid="stExpander"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    div[data-testid="stExpander"] details summary p {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        color: var(--title-color) !important;
        text-transform: uppercase !important;
    }

    /* Custom button overrides */
    div.stButton > button {
        background: var(--btn-bg) !important;
        color: var(--draw-color) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
        height: 2.5rem;
    }
    div.stButton > button:hover {
        background: var(--btn-hover) !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
    }

    /* Sidebar Headings & Labels */
    .sidebar-header {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        margin-top: 1rem;
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 0.25rem;
    }
    div[data-testid="stSidebar"] label p {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        margin-bottom: 0.2rem !important;
    }

    /* Tactical Header */
    .tactical-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--card-border);
    }
    .telemetry-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: 0.12em;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .tactical-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--title-color);
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .tactical-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

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
        transition: transform 0.3s ease;
    }
    .team-flag:hover {
        transform: scale(1.06);
    }
    .team-info {
        display: flex;
        flex-direction: column;
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

# ── Country ISO-2 Mapper (FlagCDN) ───────────────────────────────────────────
COUNTRY_TO_ISO = {
    "england": "gb-eng", "france": "fr", "germany": "de", "italy": "it", "spain": "es",
    "portugal": "pt", "belgium": "be", "netherlands": "nl", "croatia": "hr", "denmark": "dk",
    "switzerland": "ch", "poland": "pl", "ukraine": "ua", "scotland": "gb-sct", "wales": "gb-wls",
    "sweden": "se", "austria": "at", "turkey": "tr", "hungary": "hu", "czechia": "cz",
    "czech republic": "cz", "slovakia": "sk", "romania": "ro", "serbia": "rs", "slovenia": "si",
    "norway": "no", "finland": "fi", "greece": "gr", "republic of ireland": "ie", "ireland": "ie",
    "northern ireland": "gb-nir", "albania": "al", "georgia": "ge", "iceland": "is",
    "north macedonia": "mk", "bosnia and herzegovina": "ba", "bulgaria": "bg", "montenegro": "me",
    "belarus": "by", "lithuania": "lt", "latvia": "lv", "estonia": "ee", "luxembourg": "lu",
    "cyprus": "cy", "malta": "mt", "andorra": "ad", "san marino": "sm", "liechtenstein": "li",
    "gibraltar": "gi", "faroe islands": "fo", "kosovo": "xk", "israel": "il", "azerbaijan": "az",
    "armenia": "am", "kazakhstan": "kz",
    "argentina": "ar", "brazil": "br", "uruguay": "uy", "colombia": "co", "peru": "pe",
    "chile": "cl", "paraguay": "py", "ecuador": "ec", "venezuela": "ve", "bolivia": "bo",
    "usa": "us", "united states": "us", "mexico": "mx", "canada": "ca", "costa rica": "cr",
    "jamaica": "jm", "panama": "pa", "honduras": "hn", "el salvador": "sv", "haiti": "ht",
    "curacao": "cw", "trinidad and tobago": "tt", "martinique": "mq", "guadeloupe": "gp",
    "guatemala": "gt", "cuba": "cu", "suriname": "sr", "french guiana": "gf", "bermuda": "bm",
    "grenada": "gd", "barbados": "bb", "dominica": "dm", "saint lucia": "lc",
    "saint vincent and the grenadines": "vc", "saint kitts and nevis": "kn", "antigua and barbuda": "ag",
    "montserrat": "ms", "virgin islands": "vi", "puerto rico": "pr", "cayman islands": "ky",
    "bahamas": "bs", "belize": "bz", "nicaragua": "ni", "guyana": "gy",
    "senegal": "sn", "morocco": "ma", "nigeria": "ng", "egypt": "eg", "tunisia": "tn",
    "cameroon": "cm", "algeria": "dz", "ghana": "gh", "ivory coast": "ci", "cote d'ivoire": "ci",
    "mali": "ml", "burkina faso": "bf", "dr congo": "cd", "congo dr": "cd", "south africa": "za",
    "cape verde": "cv", "cabo verde": "cv", "guinea": "gn", "equatorial guinea": "gq", "gabon": "ga",
    "togo": "tg", "angola": "ao", "zambia": "zm", "uganda": "ug", "kenya": "ke", "zimbabwe": "zw",
    "tanzania": "tz", "libya": "ly", "madagascar": "mg", "mauritania": "mr", "namibia": "na",
    "guinea-bissau": "gw", "mozambique": "mz", "malawi": "mw", "sudan": "sd", "south sudan": "ss",
    "ethiopia": "et", "rwanda": "rw", "burundi": "bi", "central african republic": "cf",
    "congo": "cg", "republic of the congo": "cg", "chad": "td", "niger": "ne", "benin": "bj",
    "sierra leone": "sl", "liberia": "lr", "gambia": "gm", "eritrea": "er", "somalia": "so",
    "djibouti": "dj", "seychelles": "sc", "mauritius": "mu", "comoros": "km", "reunion": "re",
    "japan": "jp", "iran": "ir", "ir iran": "ir", "south korea": "kr", "korea republic": "kr",
    "australia": "au", "saudi arabia": "sa", "qatar": "qa", "iraq": "iq", "uae": "ae",
    "united arab emirates": "ae", "oman": "om", "uzbekistan": "uz", "china": "cn", "china pr": "cn",
    "jordan": "jo", "bahrain": "bh", "syria": "sy", "palestine": "ps", "vietnam": "vn",
    "kyrgyzstan": "kg", "lebannon": "lb", "lebanon": "lb", "india": "in", "tajikistan": "tj",
    "thailand": "th", "north korea": "kp", "korea dpr": "kp", "yemen": "ye", "kuwait": "kw",
    "afghanistan": "af", "turkmenistan": "tm", "maldives": "mv", "nepal": "np", "bangladesh": "bd",
    "pakistan": "pk", "sri lanka": "lk", "bhutan": "bt", "guam": "gu", "macau": "mo",
    "chinese taipei": "tw", "taiwan": "tw", "hong kong": "hk", "myanmar": "mm", "cambodia": "kh",
    "laos": "la", "brunei": "bn", "timor-leste": "tl", "singapore": "sg", "malaysia": "my",
    "indonesia": "id", "philippines": "ph",
    "new zealand": "nz", "solomon islands": "sb", "new caledonia": "nc", "tahiti": "pf",
    "fiji": "fj", "vanuatu": "vu", "papua new guinea": "pg", "samoa": "ws", "american samoa": "as",
    "tonga": "to", "cook islands": "ck"
}

def get_flag_url(team_name: str) -> str:
    name_clean = team_name.lower().strip()
    iso = COUNTRY_TO_ISO.get(name_clean)
    if not iso:
        return "https://flagcdn.com/w80/un.png"  # UN flag default
    return f"https://flagcdn.com/w80/{iso}.png"

# ── Load predictor (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Booting analytical engine and loading historical dataset…")
def load_predictor():
    """Cache the Predictor object so it is only loaded once."""
    try:
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
        marker_color=["#0284c7", "#dc2626"],  # Uses mid-tones that look great in both dark and light
        text=[f"{home_elo:.0f}", f"{away_elo:.0f}"],
        textposition="outside",
        textfont=dict(size=13, family="Space Grotesk"),
        width=0.4,
    ))
    fig.update_layout(
        title={
            "text": "ELO RATING COMPARISON",
            "font": {"size": 13, "family": "Outfit", "weight": "bold"},
            "pad": {"b": 10}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            gridcolor="#232b2b", 
            showgrid=True, 
            zeroline=False,
            range=[min(home_elo, away_elo) - 150, max(home_elo, away_elo) + 150],
            tickfont=dict(family="Space Grotesk")
        ),
        xaxis=dict(
            tickfont=dict(size=13, family="Outfit"),
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
        """
        <div class="tactical-header">
            <div class="telemetry-badge">MATCH DECISION TELEMETRY v1.2</div>
            <h1 class="tactical-title">
                <svg class="header-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 32px; height: 32px; vertical-align: middle; margin-right: 12px; margin-top: -4px;">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="m12 2-2 3h4Z"/>
                    <path d="M12 22v-4"/>
                    <path d="m2 12 5-1v2Z"/>
                    <path d="m22 12-5 1v-2Z"/>
                    <path d="m10 5 1 5-3 2v4l4 2 4-2v-4l-3-2 1-5"/>
                </svg>
                FIFA MATCH OUTCOME PREDICTOR
            </h1>
            <p class="tactical-subtitle">Data-driven machine learning models forecasting international fixtures</p>
        </div>
        """, 
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
        if st.button("SWAP HOME & AWAY", use_container_width=True):
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
        predict_btn = st.button("RUN PREDICTION TELEMETRY", use_container_width=True)

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
                    <img class="team-flag" src="{home_flag_url}" alt="{home_team}" />
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
                    <img class="team-flag" src="{away_flag_url}" alt="{away_team}" />
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
                use_container_width=True,
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

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Batch simulation ──────────────────────────────────────────────
        with st.expander("BATCH PREDICTION SIMULATOR"):
            st.caption("Upload a CSV file containing columns: home_team, away_team, neutral (bool), tournament (str)")
            uploaded = st.file_uploader("Upload fixtures CSV", type=["csv"], label_visibility="collapsed")
            if uploaded:
                fixtures = pd.read_csv(uploaded)
                predictions = predictor.predict_batch(fixtures)
                st.dataframe(predictions, use_container_width=True)
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
