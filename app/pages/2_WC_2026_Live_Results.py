"""
2_WC_2026_Live_Results.py
--------------------------
Streamlit page: FIFA World Cup 2026 Group Stage predictions tracker.
Displays a tactical match schedule with live scores, win probabilities, and accuracy stats.
"""

import sys
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import json
import copy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="2026 Live Predictions Tracker",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared Theme & Flag CDN ───────────────────────────────────────────────────
import importlib
import app.shared_theme
importlib.reload(app.shared_theme)
from app.shared_theme import inject_theme, get_flag_url
inject_theme()

# ── Simulator mappings & imports ──────────────────────────────────────────────
from src.group_simulator import WC2026_GROUPS, load_real_played_matches, load_real_shootouts

# Custom styles for the match tracker and bracket
st.markdown("""
<style>
/* Summary telemetry grid */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.telemetry-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.telemetry-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #10b981;
    margin-top: 0.25rem;
}
.telemetry-lbl {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
}

/* Match card layout */
.match-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 180px;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.match-card:hover {
    transform: translateY(-2px);
    border-color: #10b981;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
}
.match-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}
.match-group-badge {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
}
.match-date-lbl {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
}
.match-teams-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    gap: 0.5rem;
}
.match-team {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex: 1;
    min-width: 0;
}
.match-team.home {
    justify-content: flex-start;
}
.match-team.away {
    flex-direction: row-reverse;
    justify-content: flex-start;
    text-align: right;
}
.team-mini-flag {
    width: 32px;
    height: auto;
    border-radius: 3px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.team-mini-flag-placeholder {
    width: 32px;
    height: 20px;
    background: var(--card-border);
    border-radius: 3px;
    opacity: 0.5;
}
.team-label-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--title-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 110px;
}
.match-score-display {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--title-color);
    background: var(--background-color);
    border: 1px solid var(--card-border);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    min-width: 65px;
    text-align: center;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
}
.match-score-vs {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-muted);
}

/* Win Probabilities Bar */
.match-probs-bar {
    display: flex;
    height: 18px;
    border-radius: 4px;
    overflow: hidden;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    color: #0f1414;
    margin-bottom: 0.75rem;
    background: var(--background-color);
    border: 1px solid var(--card-border);
}
.prob-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    white-space: nowrap;
    transition: width 0.3s ease;
}
.prob-segment.home {
    background: var(--home-color);
}
.prob-segment.draw {
    background: var(--draw-color);
}
.prob-segment.away {
    background: var(--away-color);
}

/* Card footer details */
.match-card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    border-top: 1px solid var(--card-border);
    padding-top: 0.6rem;
    margin-top: 0.25rem;
}
.prediction-pred-lbl {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: var(--text-muted);
}
.prediction-pred-lbl strong {
    color: var(--title-color);
}
.accuracy-badge {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    letter-spacing: 0.02em;
}
.accuracy-badge.correct {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.accuracy-badge.incorrect {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.25);
}
.accuracy-badge.scheduled {
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-muted);
    border: 1px solid var(--card-border);
}

/* ── Visual Bracket Tree Styling ───────────────────────────── */
.bracket-wrapper {
    display: flex;
    gap: 1.5rem;
    overflow-x: auto;
    padding: 1.5rem 1rem;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    margin-top: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    min-width: 1200px;
}
.bracket-column {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    height: 1000px;
    flex: 1;
    min-width: 230px;
}
.bracket-match {
    background: color-mix(in srgb, var(--card-bg), #fff 2%);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.6rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.bracket-match:hover {
    border-color: #10b981;
    transform: translateY(-1px);
}
.bracket-match-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 0.25rem;
    margin-bottom: 0.2rem;
    display: flex;
    justify-content: space-between;
}
.bracket-team {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    height: 28px;
}
.bracket-team.winner {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.15);
}
.bracket-team.winner .bracket-team-name {
    color: #10b981;
    font-weight: 700;
}
.bracket-team.winner .bracket-team-prob {
    color: #10b981;
    font-weight: 700;
}
.bracket-team-flag {
    width: 20px;
    height: auto;
    border-radius: 2px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.bracket-team-flag-placeholder {
    width: 20px;
    height: 12px;
    background: var(--card-border);
    border-radius: 2px;
    opacity: 0.4;
}
.bracket-team-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: var(--title-color);
    flex-grow: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bracket-team-prob, .bracket-team-score {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 600;
}
.bracket-team-score {
    color: var(--title-color);
    font-weight: 800;
    margin-left: auto;
}
.bracket-team.winner .bracket-team-score {
    color: #10b981;
}
.bracket-probs-bar {
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--background-color);
    margin-top: 0.2rem;
    border: 1px solid var(--card-border);
}

/* Champion Box */
.champion-box {
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.1) 0%, rgba(133, 77, 14, 0.15) 100%);
    border: 2px solid #eab308;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(234, 179, 8, 0.15);
    margin-top: 1.5rem;
    animation: pulse-glow 2s infinite;
}
@keyframes pulse-glow {
    0% { box-shadow: 0 4px 15px rgba(234, 179, 8, 0.15); }
    50% { box-shadow: 0 4px 25px rgba(234, 179, 8, 0.35); }
    100% { box-shadow: 0 4px 15px rgba(234, 179, 8, 0.15); }
}
.champion-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    color: #eab308;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.champion-flag {
    width: 48px;
    height: auto;
    border-radius: 4px;
    margin-bottom: 0.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.champion-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    color: var(--title-color);
}
</style>
""", unsafe_allow_html=True)


# ── Load Predictor Engine ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running predictive engine and calculating Elo ratings...")
def load_predictor():
    try:
        import importlib
        import src.predict
        importlib.reload(src.predict)
        from src.predict import Predictor
        return Predictor()
    except Exception as e:
        return f"Error loading model: {e}"

# ── Load and Clean Match Data ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_wc_schedule():
    df = pd.read_csv('data/results.csv')
    wc_2026 = df[(df['tournament'] == 'FIFA World Cup') & (df['date'].astype(str).str.startswith('2026'))].copy()
    return wc_2026

@st.cache_data(show_spinner=False)
def load_knockout_bracket_structure():
    with open('data/knockout_bracket.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_name_for_mapping(team_name):
    t_clean = str(team_name).strip()
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

def is_placeholder(team_name):
    name = str(team_name).strip()
    if name.upper() == "TBD":
        return True
    prefixes = ["RD32 ", "RD16 ", "QF ", "SF ", "Winner ", "Loser "]
    return any(name.startswith(p) for p in prefixes)

# ELO-based bracket path progression forecasting
def resolve_bracket_state(bracket_structure, predictor, forecast_mode=True):
    # Load actual played knockout results
    knockout_played = load_real_played_matches(knockout=True)
    real_shootouts = load_real_shootouts()
    
    bracket = copy.deepcopy(bracket_structure)
    
    # Helper to clean names
    def clean(name):
        return clean_name_for_mapping(name)

    # Dictionary of match winners resolved so far
    winners = {}
    losers = {}
    
    # 1. Round of 32
    for m in bracket["r32"]:
        t1 = m["team1"]
        t2 = m["team2"]
        m["team1_actual"] = t1
        m["team2_actual"] = t2
        
        # Check if played in real life
        res = knockout_played.get((clean(t1), clean(t2))) or knockout_played.get((clean(t2), clean(t1)))
        if res is not None:
            hs, as_ = res
            m["home_score_actual"] = hs
            m["away_score_actual"] = as_
            if hs > as_:
                winner = t1
            elif as_ > hs:
                winner = t2
            else:
                sho_win = real_shootouts.get((clean(t1), clean(t2))) or real_shootouts.get((clean(t2), clean(t1)))
                winner = sho_win if sho_win else t1
            loser = t2 if winner == t1 else t1
        else:
            if forecast_mode:
                pred = predictor.predict(t1, t2, neutral=True, tournament="FIFA World Cup", match_date=m["date"])
                winner = t1 if pred["home_win_prob"] >= pred["away_win_prob"] else t2
                m["prob1"] = pred["home_win_prob"]
                m["prob2"] = pred["away_win_prob"]
                m["prob_draw"] = pred["draw_prob"]
                m["predicted_outcome"] = pred["predicted_outcome"]
                loser = t2 if winner == t1 else t1
            else:
                winner = "TBD"
                loser = "TBD"
        m["winner"] = winner
        winners[m["match_number"]] = winner
        losers[m["match_number"]] = loser

    # Helper to resolve slot labels
    def get_rd32_winner(label):
        if is_placeholder(label):
            idx = int(label.replace("RD32 W", ""))
            match_num = f"Match {72 + idx}"
            return winners.get(match_num, label)
        return label

    # 2. Round of 16
    for m in bracket["r16"]:
        t1 = get_rd32_winner(m["team1"])
        t2 = get_rd32_winner(m["team2"])
        m["team1_actual"] = t1
        m["team2_actual"] = t2
        
        if is_placeholder(t1) or is_placeholder(t2) or t1 == "TBD" or t2 == "TBD":
            m["winner"] = "TBD"
            winners[m["match_number"]] = "TBD"
            losers[m["match_number"]] = "TBD"
            continue
            
        res = knockout_played.get((clean(t1), clean(t2))) or knockout_played.get((clean(t2), clean(t1)))
        if res is not None:
            hs, as_ = res
            m["home_score_actual"] = hs
            m["away_score_actual"] = as_
            if hs > as_:
                winner = t1
            elif as_ > hs:
                winner = t2
            else:
                sho_win = real_shootouts.get((clean(t1), clean(t2))) or real_shootouts.get((clean(t2), clean(t1)))
                winner = sho_win if sho_win else t1
            loser = t2 if winner == t1 else t1
        else:
            if forecast_mode:
                pred = predictor.predict(t1, t2, neutral=True, tournament="FIFA World Cup", match_date=m["date"])
                winner = t1 if pred["home_win_prob"] >= pred["away_win_prob"] else t2
                m["prob1"] = pred["home_win_prob"]
                m["prob2"] = pred["away_win_prob"]
                m["prob_draw"] = pred["draw_prob"]
                m["predicted_outcome"] = pred["predicted_outcome"]
                loser = t2 if winner == t1 else t1
            else:
                winner = "TBD"
                loser = "TBD"
        m["winner"] = winner
        winners[m["match_number"]] = winner
        losers[m["match_number"]] = loser

    def get_rd16_winner(label):
        if is_placeholder(label):
            idx = int(label.replace("RD16 W", ""))
            match_num = f"Match {88 + idx}"
            return winners.get(match_num, label)
        return label

    # 3. Quarterfinals
    for m in bracket["qf"]:
        t1 = get_rd16_winner(m["team1"])
        t2 = get_rd16_winner(m["team2"])
        m["team1_actual"] = t1
        m["team2_actual"] = t2
        
        if is_placeholder(t1) or is_placeholder(t2) or t1 == "TBD" or t2 == "TBD":
            m["winner"] = "TBD"
            winners[m["match_number"]] = "TBD"
            losers[m["match_number"]] = "TBD"
            continue
            
        res = knockout_played.get((clean(t1), clean(t2))) or knockout_played.get((clean(t2), clean(t1)))
        if res is not None:
            hs, as_ = res
            m["home_score_actual"] = hs
            m["away_score_actual"] = as_
            if hs > as_:
                winner = t1
            elif as_ > hs:
                winner = t2
            else:
                sho_win = real_shootouts.get((clean(t1), clean(t2))) or real_shootouts.get((clean(t2), clean(t1)))
                winner = sho_win if sho_win else t1
            loser = t2 if winner == t1 else t1
        else:
            if forecast_mode:
                pred = predictor.predict(t1, t2, neutral=True, tournament="FIFA World Cup", match_date=m["date"])
                winner = t1 if pred["home_win_prob"] >= pred["away_win_prob"] else t2
                m["prob1"] = pred["home_win_prob"]
                m["prob2"] = pred["away_win_prob"]
                m["prob_draw"] = pred["draw_prob"]
                m["predicted_outcome"] = pred["predicted_outcome"]
                loser = t2 if winner == t1 else t1
            else:
                winner = "TBD"
                loser = "TBD"
        m["winner"] = winner
        winners[m["match_number"]] = winner
        losers[m["match_number"]] = loser

    def get_qf_winner(label):
        if is_placeholder(label):
            idx = int(label.replace("QF W", ""))
            match_num = f"Match {96 + idx}"
            return winners.get(match_num, label)
        return label

    # 4. Semifinals
    for m in bracket["sf"]:
        t1 = get_qf_winner(m["team1"])
        t2 = get_qf_winner(m["team2"])
        m["team1_actual"] = t1
        m["team2_actual"] = t2
        
        if is_placeholder(t1) or is_placeholder(t2) or t1 == "TBD" or t2 == "TBD":
            m["winner"] = "TBD"
            m["loser"] = "TBD"
            winners[m["match_number"]] = "TBD"
            losers[m["match_number"]] = "TBD"
            continue
            
        res = knockout_played.get((clean(t1), clean(t2))) or knockout_played.get((clean(t2), clean(t1)))
        if res is not None:
            hs, as_ = res
            m["home_score_actual"] = hs
            m["away_score_actual"] = as_
            if hs > as_:
                winner = t1
            elif as_ > hs:
                winner = t2
            else:
                sho_win = real_shootouts.get((clean(t1), clean(t2))) or real_shootouts.get((clean(t2), clean(t1)))
                winner = sho_win if sho_win else t1
            loser = t2 if winner == t1 else t1
        else:
            if forecast_mode:
                pred = predictor.predict(t1, t2, neutral=True, tournament="FIFA World Cup", match_date=m["date"])
                winner = t1 if pred["home_win_prob"] >= pred["away_win_prob"] else t2
                m["prob1"] = pred["home_win_prob"]
                m["prob2"] = pred["away_win_prob"]
                m["prob_draw"] = pred["draw_prob"]
                m["predicted_outcome"] = pred["predicted_outcome"]
                loser = t2 if winner == t1 else t1
            else:
                winner = "TBD"
                loser = "TBD"
        m["winner"] = winner
        m["loser"] = loser
        winners[m["match_number"]] = winner
        losers[m["match_number"]] = loser

    # 5. Third Place and Final
    t1_3rd = losers.get("Match 101", "SF L1")
    t2_3rd = losers.get("Match 102", "SF L2")
    m_3rd = bracket["third_place"]
    m_3rd["team1_actual"] = t1_3rd
    m_3rd["team2_actual"] = t2_3rd
    
    if is_placeholder(t1_3rd) or is_placeholder(t2_3rd) or t1_3rd == "TBD" or t2_3rd == "TBD":
        m_3rd["winner"] = "TBD"
    else:
        res = knockout_played.get((clean(t1_3rd), clean(t2_3rd))) or knockout_played.get((clean(t2_3rd), clean(t1_3rd)))
        if res is not None:
            hs, as_ = res
            m_3rd["home_score_actual"] = hs
            m_3rd["away_score_actual"] = as_
            if hs > as_:
                winner = t1_3rd
            elif as_ > hs:
                winner = t2_3rd
            else:
                sho_win = real_shootouts.get((clean(t1_3rd), clean(t2_3rd))) or real_shootouts.get((clean(t2_3rd), clean(t1_3rd)))
                winner = sho_win if sho_win else t1_3rd
            m_3rd["winner"] = winner
        else:
            if forecast_mode:
                pred_3rd = predictor.predict(t1_3rd, t2_3rd, neutral=True, tournament="FIFA World Cup", match_date=m_3rd["date"])
                m_3rd["winner"] = t1_3rd if pred_3rd["home_win_prob"] >= pred_3rd["away_win_prob"] else t2_3rd
                m_3rd["prob1"] = pred_3rd["home_win_prob"]
                m_3rd["prob2"] = pred_3rd["away_win_prob"]
                m_3rd["prob_draw"] = pred_3rd["draw_prob"]
                m_3rd["predicted_outcome"] = pred_3rd["predicted_outcome"]
            else:
                m_3rd["winner"] = "TBD"

    t1_final = winners.get("Match 101", "SF W1")
    t2_final = winners.get("Match 102", "SF W2")
    m_final = bracket["final"]
    m_final["team1_actual"] = t1_final
    m_final["team2_actual"] = t2_final
    
    if is_placeholder(t1_final) or is_placeholder(t2_final) or t1_final == "TBD" or t2_final == "TBD":
        m_final["winner"] = "TBD"
    else:
        res = knockout_played.get((clean(t1_final), clean(t2_final))) or knockout_played.get((clean(t2_final), clean(t1_final)))
        if res is not None:
            hs, as_ = res
            m_final["home_score_actual"] = hs
            m_final["away_score_actual"] = as_
            if hs > as_:
                winner = t1_final
            elif as_ > hs:
                winner = t2_final
            else:
                sho_win = real_shootouts.get((clean(t1_final), clean(t2_final))) or real_shootouts.get((clean(t2_final), clean(t1_final)))
                winner = sho_win if sho_win else t1_final
            m_final["winner"] = winner
        else:
            if forecast_mode:
                pred_final = predictor.predict(t1_final, t2_final, neutral=True, tournament="FIFA World Cup", match_date=m_final["date"])
                m_final["winner"] = t1_final if pred_final["home_win_prob"] >= pred_final["away_win_prob"] else t2_final
                m_final["prob1"] = pred_final["home_win_prob"]
                m_final["prob2"] = pred_final["away_win_prob"]
                m_final["prob_draw"] = pred_final["draw_prob"]
                m_final["predicted_outcome"] = pred_final["predicted_outcome"]
            else:
                m_final["winner"] = "TBD"

    return bracket

def render_bracket_match_html(m, match_title):
    t1 = m.get("team1_actual", m["team1"])
    t2 = m.get("team2_actual", m["team2"])
    
    t1_clean = clean_name_for_mapping(t1)
    t2_clean = clean_name_for_mapping(t2)
    
    flag1 = get_flag_url(t1_clean) if not is_placeholder(t1) else ""
    flag2 = get_flag_url(t2_clean) if not is_placeholder(t2) else ""
    
    date_str = m["date"]
    
    # Check if prediction is available
    has_pred = "prob1" in m and "prob2" in m
    
    # Deciding scores and winner highlight
    right_lbl1 = ""
    right_lbl2 = ""
    win1 = ""
    win2 = ""
    
    # If played in real life, show actual score (and highlight shootout winner if draw)
    if "home_score_actual" in m and "away_score_actual" in m:
        hs = int(m["home_score_actual"])
        as_ = int(m["away_score_actual"])
        
        real_shootouts = load_real_shootouts()
        sho_win = real_shootouts.get((t1_clean, t2_clean)) or real_shootouts.get((t2_clean, t1_clean))
        
        s1_suffix = " (p)" if (hs == as_ and sho_win == t1) else ""
        s2_suffix = " (p)" if (hs == as_ and sho_win == t2) else ""
        
        right_lbl1 = f'<span class="bracket-team-score">{hs}{s1_suffix}</span>'
        right_lbl2 = f'<span class="bracket-team-score">{as_}{s2_suffix}</span>'
        
        actual_winner = t1 if (hs > as_ or (hs == as_ and sho_win == t1)) else t2
        win1 = "winner" if actual_winner == t1 else ""
        win2 = "winner" if actual_winner == t2 else ""
    else:
        # Not played yet, show prediction percentages if available
        if has_pred:
            right_lbl1 = f'<span class="bracket-team-prob">{m["prob1"]*100:.0f}%</span>'
            right_lbl2 = f'<span class="bracket-team-prob">{m["prob2"]*100:.0f}%</span>'
            
            w = m.get("winner", "")
            win1 = "winner" if w == t1 else ""
            win2 = "winner" if w == t2 else ""
        else:
            right_lbl1 = ''
            right_lbl2 = ''
            win1 = ""
            win2 = ""
            
    # Flag tags
    flag1_html = f'<img class="bracket-team-flag" src="{flag1}" crossorigin="anonymous" />' if flag1 else '<div class="bracket-team-flag-placeholder"></div>'
    flag2_html = f'<img class="bracket-team-flag" src="{flag2}" crossorigin="anonymous" />' if flag2 else '<div class="bracket-team-flag-placeholder"></div>'
    
    probs_bar_html = ""
    if has_pred and "home_score_actual" not in m:
        h_prob = m["prob1"] * 100
        d_prob = m["prob_draw"] * 100
        a_prob = m["prob2"] * 100
        probs_bar_html = f"""
        <div class="bracket-probs-bar" title="Home: {h_prob:.1f}% | Draw: {d_prob:.1f}% | Away: {a_prob:.1f}%">
            <div class="prob-segment home" style="width: {h_prob}%;"></div>
            <div class="prob-segment draw" style="width: {d_prob}%;"></div>
            <div class="prob-segment away" style="width: {a_prob}%;"></div>
        </div>
        """
        
    return f"""
    <div class="bracket-match">
        <div class="bracket-match-header">
            <span>{match_title}</span>
            <span>📅 {date_str[5:]}</span>
        </div>
        <div class="bracket-team {win1}">
            {flag1_html}
            <span class="bracket-team-name" title="{t1}">{t1}</span>
            {right_lbl1}
        </div>
        <div class="bracket-team {win2}">
            {flag2_html}
            <span class="bracket-team-name" title="{t2}">{t2}</span>
            {right_lbl2}
        </div>
        {probs_bar_html}
    </div>
    """

def main():
    # Header Billboard
    st.markdown(
        """<div class="tactical-header">
<div class="telemetry-badge">MATCH DECISION TELEMETRY v1.5</div>
<h1 class="tactical-title">
<svg class="header-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none">
<defs>
<radialGradient id="sphereShadow" cx="35%" cy="30%" r="70%">
<stop offset="0%" stop-color="#ffffff" />
<stop offset="65%" stop-color="#f8fafc" />
<stop offset="85%" stop-color="#e2e8f0" />
<stop offset="100%" stop-color="#cbd5e1" />
</radialGradient>
<linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#fef08a" />
<stop offset="40%" stop-color="#eab308" />
<stop offset="75%" stop-color="#ca8a04" />
<stop offset="100%" stop-color="#854d0e" />
</linearGradient>
<linearGradient id="redGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#f87171" />
<stop offset="100%" stop-color="#dc2626" />
</linearGradient>
<linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#34d399" />
<stop offset="100%" stop-color="#059669" />
</linearGradient>
<linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#60a5fa" />
<stop offset="100%" stop-color="#2563eb" />
</linearGradient>
</defs>
<circle cx="50" cy="50" r="46" fill="url(#sphereShadow)" stroke="#94a3b8" stroke-width="1.2" />
<circle cx="50" cy="50" r="41" fill="none" stroke="#e2e8f0" stroke-width="0.8" stroke-dasharray="10,4,3,4" opacity="0.9" />
<circle cx="50" cy="50" r="46" fill="none" stroke="#e2e8f0" stroke-width="0.5" opacity="0.7" />
<path d="M 17 33 A 46 46 0 0 1 83 33" fill="none" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="4,4" opacity="0.6" />
<path d="M 17 67 A 46 46 0 0 0 83 67" fill="none" stroke="#cbd5e1" stroke-width="0.8" stroke-dasharray="4,4" opacity="0.6" />
<path d="M 36.14 58.0 A 28 28 0 0 1 50.0 34.0 C 40.0 22.0, 28.0 18.0, 18.0 30.0 C 8.0 42.0, 18.0 54.0, 36.14 58.0 Z" fill="url(#redGrad)" stroke="url(#goldGrad)" stroke-width="1" stroke-linejoin="round" />
<path d="M 50.0 34.0 A 28 28 0 0 1 63.86 58.0 C 79.25 55.34, 88.71 46.95, 83.32 32.29 C 77.93 17.63, 62.54 20.29, 50.0 34.0 Z" fill="url(#blueGrad)" stroke="url(#goldGrad)" stroke-width="1" stroke-linejoin="round" />
<path d="M 63.86 58.0 A 28 28 0 0 1 36.14 58.0 C 30.75 72.66, 33.29 85.05, 48.68 87.71 C 64.07 90.37, 69.46 75.71, 63.86 58.0 Z" fill="url(#greenGrad)" stroke="url(#goldGrad)" stroke-width="1" stroke-linejoin="round" />
<path d="M 50.0 34.0 A 28 28 0 0 1 63.86 58.0 A 28 28 0 0 1 36.14 58.0 A 28 28 0 0 1 50.0 34.0 Z" fill="url(#goldGrad)" stroke="#1e293b" stroke-width="1.2" stroke-linejoin="round" />
<path d="M 50 34 L 50 42 M 63.86 58 L 56.93 54 M 36.14 58 L 43.07 54" fill="none" stroke="#854d0e" stroke-width="0.8" opacity="0.5" />
<g transform="translate(0, 1.5)">
<rect x="47.5" y="52.0" width="5.0" height="0.8" rx="0.2" fill="#166534" />
<rect x="47.0" y="52.8" width="6.0" height="0.8" rx="0.2" fill="url(#goldGrad)" />
<rect x="47.5" y="53.6" width="5.0" height="0.8" rx="0.2" fill="#166534" />
<path d="M 48.2 52.0 C 48.2 48.0, 46.8 45.0, 48.8 42.0 C 49.3 41.2, 50.7 41.2, 51.2 42.0 C 53.2 45.0, 51.8 48.0, 51.8 52.0 Z" fill="url(#goldGrad)" stroke="#854d0e" stroke-width="0.3" />
<circle cx="50.0" cy="40.8" r="1.8" fill="url(#goldGrad)" stroke="#854d0e" stroke-width="0.3" />
<circle cx="50.0" cy="40.8" r="1.4" fill="#fef08a" opacity="0.8" />
</g>
<g transform="translate(32.5, 38.0) scale(0.24)" opacity="0.95">
<path d="M 0 -22 L 3 -11 L 11 -13 L 9 -4 L 18 -1 L 8 4 L 9 10 L 3 7 L 1.2 16 L -1.2 16 L -3 7 L -9 10 L -8 4 L -18 -1 L -9 -4 L -11 -13 L -3 -11 Z" fill="#ffffff" stroke="#854d0e" stroke-width="0.8" stroke-linejoin="round" />
</g>
<g transform="translate(67.5, 38.0) scale(0.24)" opacity="0.95">
<path d="M 0 -20 L 5.5 -5 L 20 -5 L 8 4 L 13 18 L 0 9 L -13 18 L -8 4 L -20 -5 L -5.5 -5 Z" fill="#ffffff" stroke="#854d0e" stroke-width="0.8" stroke-linejoin="round" />
</g>
<g transform="translate(50.0, 71.0) scale(0.24)" opacity="0.95">
<path d="M -15 12 C -12 2, -8 -12, 2 -14 C 8 -15, 14 -11, 18 -5 C 21 -2, 23 3, 18 7 C 14 10, 10 6, 8 3 C 5 1, 2 4, 0 7 C -4 11, -10 12, -15 12 Z" fill="#ffffff" stroke="#854d0e" stroke-width="0.8" stroke-linejoin="round" />
<circle cx="3" cy="-4" r="1.5" fill="#854d0e" />
</g>
</svg>
2026 WORLD CUP PREDICTIONS
</h1>
<p class="tactical-subtitle">Real-time match outcomes tracked against pre-game model probabilities</p>
</div>""",
        unsafe_allow_html=True
    )

    predictor = load_predictor()
    if isinstance(predictor, str):
        st.error(f"Failed to load model: {predictor}")
        st.stop()

    wc_schedule = load_wc_schedule()
    bracket_structure = load_knockout_bracket_structure()

    # Resolve the actual real-world bracket state (no ELO forecast path)
    # and automatically update the results.csv file with qualified teams
    try:
        resolved_actual = resolve_bracket_state(bracket_structure, predictor, forecast_mode=False)
        
        resolved_matches = []
        for m in resolved_actual["r32"]:
            resolved_matches.append((m["team1_actual"], m["team2_actual"]))
        for m in resolved_actual["r16"]:
            resolved_matches.append((m["team1_actual"], m["team2_actual"]))
        for m in resolved_actual["qf"]:
            resolved_matches.append((m["team1_actual"], m["team2_actual"]))
        for m in resolved_actual["sf"]:
            resolved_matches.append((m["team1_actual"], m["team2_actual"]))
        m_3rd = resolved_actual["third_place"]
        resolved_matches.append((m_3rd["team1_actual"], m_3rd["team2_actual"]))
        m_final = resolved_actual["final"]
        resolved_matches.append((m_final["team1_actual"], m_final["team2_actual"]))

        csv_path = 'data/results.csv'
        entire_df = pd.read_csv(csv_path)
        updated = False
        for idx, (t1_res, t2_res) in enumerate(resolved_matches):
            row_idx = len(entire_df) - 32 + idx
            current_t1 = str(entire_df.loc[row_idx, 'home_team'])
            current_t2 = str(entire_df.loc[row_idx, 'away_team'])
            
            if t1_res and t1_res != "TBD" and not is_placeholder(t1_res) and current_t1 != t1_res:
                entire_df.loc[row_idx, 'home_team'] = t1_res
                updated = True
            if t2_res and t2_res != "TBD" and not is_placeholder(t2_res) and current_t2 != t2_res:
                entire_df.loc[row_idx, 'away_team'] = t2_res
                updated = True
                
        if updated:
            entire_df.to_csv(csv_path, index=False)
            st.cache_data.clear()
            if "wc_predictions_df" in st.session_state:
                del st.session_state.wc_predictions_df
            st.rerun()
    except Exception as e:
        st.error(f"Error auto-updating results.csv: {e}")

    # Pre-calculate group mappings
    team_to_group = {}
    for g, teams in WC2026_GROUPS.items():
        for t in teams:
            team_to_group[t] = g

    # Build pairing to match info lookup for knockout stages using resolved team names
    pairing_to_info = {}
    stages = {
        "r32": "Round of 32",
        "r16": "Round of 16",
        "qf": "Quarterfinal",
        "sf": "Semifinal",
        "third_place": "3rd-Place Match",
        "final": "World Cup Final"
    }
    for key, val in resolved_actual.items():
        stage_name = stages[key]
        if isinstance(val, list):
            for m in val:
                t1 = m.get("team1_actual", m["team1"])
                t2 = m.get("team2_actual", m["team2"])
                pairing_to_info[(t1, t2)] = (m["match_number"], stage_name)
        else:
            if val:
                t1 = val.get("team1_actual", val["team1"])
                t2 = val.get("team2_actual", val["team2"])
                pairing_to_info[(t1, t2)] = (val["match_number"], stage_name)

    # Build predictions dataframe
    if "wc_predictions_df" in st.session_state and "round_name" not in st.session_state.wc_predictions_df.columns:
        del st.session_state.wc_predictions_df

    if "wc_predictions_df" not in st.session_state:
        with st.spinner("Computing match outcome probabilities..."):
            pred_rows = []
            for _, row in wc_schedule.iterrows():
                home = row['home_team']
                away = row['away_team']
                neutral = bool(row['neutral'])
                date = row['date']
                home_score = row['home_score']
                away_score = row['away_score']
                
                # Determine stage
                stage = "Knockout" if date >= "2026-06-29" else "Group"

                # Determine round_name
                if stage == "Group":
                    round_name = "Group Stages"
                else:
                    match_info = pairing_to_info.get((home, away), None)
                    round_name = match_info[1] if match_info else "Knockout"

                if is_placeholder(home) or is_placeholder(away):
                    # Placeholder match prediction placeholder ELO
                    pred_rows.append({
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "neutral": neutral,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_prob": 0.33,
                        "draw_prob": 0.34,
                        "away_prob": 0.33,
                        "predicted_outcome": "TBD",
                        "stage": stage,
                        "round_name": round_name
                    })
                else:
                    pred = predictor.predict(
                        home, 
                        away, 
                        neutral=neutral, 
                        tournament="FIFA World Cup",
                        match_date=date
                    )
                    
                    pred_rows.append({
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "neutral": neutral,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_prob": pred['home_win_prob'],
                        "draw_prob": pred['draw_prob'],
                        "away_prob": pred['away_win_prob'],
                        "predicted_outcome": pred['predicted_outcome'],
                        "stage": stage,
                        "round_name": round_name
                    })
            st.session_state.wc_predictions_df = pd.DataFrame(pred_rows)

    df_preds = st.session_state.wc_predictions_df.copy()

    # Calculate actual outcomes & prediction correctness
    df_preds['actual_outcome'] = "-"
    df_preds['correct'] = "Pending"
    
    played_idx = df_preds['home_score'].notna() & df_preds['away_score'].notna()
    
    def get_actual_outcome(row):
        hs, as_ = int(row['home_score']), int(row['away_score'])
        if hs > as_: return "Home Win"
        if hs < as_: return "Away Win"
        return "Draw"
        
    if played_idx.any():
        df_preds.loc[played_idx, 'actual_outcome'] = df_preds[played_idx].apply(get_actual_outcome, axis=1)
        df_preds.loc[played_idx, 'correct'] = np.where(
            df_preds[played_idx]['predicted_outcome'] == df_preds[played_idx]['actual_outcome'],
            "Correct", "Incorrect"
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Reload Predictions
        if st.button("RELOAD PREDICTIONS", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            if "wc_predictions_df" in st.session_state:
                del st.session_state.wc_predictions_df
            st.rerun()

    # Base match list is all predictions
    filtered_df = df_preds.copy()

    # Sort matches chronologically
    filtered_df = filtered_df.sort_values(by=['date', 'home_team'])

    # Determine active stage from session state (or default to All Matches)
    active_stage = st.session_state.get("stage_filter_segmented_control", "All Matches")
    
    # Filter telemetry_df for stats computation based on segmented stage control
    telemetry_df = df_preds.copy()
    if active_stage == "Group Stages":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "Group Stages"]
    elif active_stage == "Round of 32":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "Round of 32"]
    elif active_stage == "Round of 16":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "Round of 16"]
    elif active_stage == "Quarterfinals":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "Quarterfinal"]
    elif active_stage == "Semifinals":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "Semifinal"]
    elif active_stage == "3rd place":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "3rd-Place Match"]
    elif active_stage == "final":
        telemetry_df = telemetry_df[telemetry_df['round_name'] == "World Cup Final"]

    # ── Telemetry Summary Cards (Based on Selected stage) ────────────────────
    total_wc = len(telemetry_df)
    total_played = telemetry_df['home_score'].notna().sum()
    correct_preds = (telemetry_df['correct'] == "Correct").sum()
    
    accuracy_pct = 0.0
    if total_played > 0:
        accuracy_pct = (correct_preds / total_played) * 100.0

    st.html(f"""
    <div class="telemetry-grid">
        <div class="telemetry-card">
            <div class="telemetry-lbl">Total matches</div>
            <div class="telemetry-val" style="color: var(--title-color);">{total_wc}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-lbl">Matches Played</div>
            <div class="telemetry-val" style="color: var(--draw-color);">{total_played} / {total_wc}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-lbl">Correct Predictions</div>
            <div class="telemetry-val" style="color: #10b981;">{correct_preds} / {total_played}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-lbl">Model Accuracy</div>
            <div class="telemetry-val" style="color: #10b981;">{accuracy_pct:.1f}%</div>
        </div>
    </div>
    """)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 LIVE MATCH LIST", "🌳 INTERACTIVE KNOCKOUT BRACKET"])

    with tab1:
        real_shootouts = load_real_shootouts()
        selected_stage = st.segmented_control(
            "Stage Filter",
            options=[
                "All Matches", 
                "Group Stages", 
                "Round of 32", 
                "Round of 16", 
                "Quarterfinals", 
                "Semifinals", 
                "3rd place", 
                "final"
            ],
            default="All Matches",
            key="stage_filter_segmented_control",
            label_visibility="collapsed"
        )
        
        # Apply tab-level filter
        tab_filtered_df = filtered_df.copy()
        if selected_stage == "Group Stages":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "Group Stages"]
        elif selected_stage == "Round of 32":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "Round of 32"]
        elif selected_stage == "Round of 16":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "Round of 16"]
        elif selected_stage == "Quarterfinals":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "Quarterfinal"]
        elif selected_stage == "Semifinals":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "Semifinal"]
        elif selected_stage == "3rd place":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "3rd-Place Match"]
        elif selected_stage == "final":
            tab_filtered_df = tab_filtered_df[tab_filtered_df['round_name'] == "World Cup Final"]

        if len(tab_filtered_df) == 0:
            st.info("No matches found matching the current stage filter.")
        else:
            st.subheader(f"Matches ({len(tab_filtered_df)})")

            # Render matches in 2-column grid
            cols = st.columns(2)
            for idx, (_, row) in enumerate(tab_filtered_df.iterrows()):
                col = cols[idx % 2]
                
                home = row['home_team']
                away = row['away_team']
                date = row['date']
                h_prob = row['home_prob'] * 100
                d_prob = row['draw_prob'] * 100
                a_prob = row['away_prob'] * 100
                pred_outcome = row['predicted_outcome']
                stage = row['stage']
                
                home_clean = clean_name_for_mapping(home)
                away_clean = clean_name_for_mapping(away)
                
                home_flag = get_flag_url(home_clean) if not is_placeholder(home) else ""
                away_flag = get_flag_url(away_clean) if not is_placeholder(away) else ""
                
                # Determine stage/group label for badge
                if stage == "Group":
                    group_letter = team_to_group.get(home_clean, team_to_group.get(away_clean, "?"))
                    stage_lbl = f"Group {group_letter}"
                else:
                    match_info = pairing_to_info.get((home, away), None)
                    stage_lbl = match_info[1] if match_info else "Knockout"

                # Flag tags
                flag_home_tag = f'<img class="team-mini-flag" src="{home_flag}" crossorigin="anonymous" />' if home_flag else '<div class="team-mini-flag-placeholder"></div>'
                flag_away_tag = f'<img class="team-mini-flag" src="{away_flag}" crossorigin="anonymous" />' if away_flag else '<div class="team-mini-flag-placeholder"></div>'

                # Format score
                shootout_winner_html = ""
                need_shootout_input = False
                
                if not pd.isna(row['home_score']) and not pd.isna(row['away_score']):
                    hs = int(row['home_score'])
                    as_ = int(row['away_score'])
                    score_str = f"{hs} - {as_}"
                    correct_status = row['correct']
                    if correct_status == "Correct":
                        accuracy_badge_html = '<span class="accuracy-badge correct">Correct ✅</span>'
                    else:
                        accuracy_badge_html = '<span class="accuracy-badge incorrect">Incorrect ❌</span>'
                        
                    # Check for shootout
                    if hs == as_ and stage == "Knockout":
                        sho_winner = real_shootouts.get((clean_name_for_mapping(home), clean_name_for_mapping(away))) or \
                                     real_shootouts.get((clean_name_for_mapping(away), clean_name_for_mapping(home)))
                        if sho_winner:
                            shootout_winner_html = f"""
                            <div style="text-align: center; font-size: 0.75rem; color: #f59e0b; font-weight: 700; margin-top: -0.4rem; margin-bottom: 0.4rem;">
                                🏆 {sho_winner} won on penalties
                            </div>
                            """
                        else:
                            need_shootout_input = True
                else:
                    score_str = '<span class="match-score-vs">VS</span>'
                    accuracy_badge_html = '<span class="accuracy-badge scheduled">Scheduled ⏳</span>'

                # Forecast label
                if is_placeholder(home) or is_placeholder(away):
                    pred_desc = "<strong>TBD</strong> (Calculated when teams qualify)"
                    probs_bar_style = "display: none;"
                else:
                    probs_bar_style = ""
                    if pred_outcome == "Home Win":
                        pred_desc = f"<strong>{home} Win</strong> ({h_prob:.0f}%)"
                    elif pred_outcome == "Away Win":
                        pred_desc = f"<strong>{away} Win</strong> ({a_prob:.0f}%)"
                    else:
                        pred_desc = f"<strong>Draw</strong> ({d_prob:.0f}%)"

                match_card_html = f"""
                <div class="match-card">
                    <div class="match-card-header">
                        <span class="match-date-lbl">📅 {date}</span>
                        <span class="match-group-badge">{stage_lbl}</span>
                    </div>
                    <div class="match-teams-row">
                        <div class="match-team home">
                            {flag_home_tag}
                            <span class="team-label-name" title="{home}">{home}</span>
                        </div>
                        <div class="match-score-display">{score_str}</div>
                        <div class="match-team away">
                            {flag_away_tag}
                            <span class="team-label-name" title="{away}">{away}</span>
                        </div>
                    </div>
                    
                    {shootout_winner_html}
                    
                    <div class="match-probs-bar" style="{probs_bar_style}" title="Home: {h_prob:.1f}% | Draw: {d_prob:.1f}% | Away: {a_prob:.1f}%">
                        <div class="prob-segment home" style="width: {h_prob}%;"></div>
                        <div class="prob-segment draw" style="width: {d_prob}%;"></div>
                        <div class="prob-segment away" style="width: {a_prob}%;"></div>
                    </div>
                    
                    <div class="match-card-footer">
                        <span class="prediction-pred-lbl">Forecast: {pred_desc}</span>
                        {accuracy_badge_html}
                    </div>
                </div>
                """
                with col:
                    st.html(match_card_html)
                    if need_shootout_input:
                        sho_win_select = st.selectbox(
                            f"Shootout Winner:",
                            options=["-", home, away],
                            key=f"sho_select_{home}_{away}_{date}"
                        )
                        if sho_win_select != "-":
                            try:
                                with open('data/shootouts.csv', 'r+', encoding='utf-8') as f_sho:
                                    content = f_sho.read()
                                    if content and not content.endswith('\n'):
                                        f_sho.write('\n')
                                    f_sho.write(f"{date},{home},{away},{sho_win_select},")
                                st.success(f"Recorded {sho_win_select} as shootout winner!")
                                st.cache_data.clear()
                                if "wc_predictions_df" in st.session_state:
                                    del st.session_state.wc_predictions_df
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving shootout: {e}")

    with tab2:
        st.subheader("Tournament Knockout Tree")
        
        # Controls for forecast
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            forecast_path = st.toggle(
                "🔮 RUN MODEL ELO FORECAST PATH", 
                value=True,
                help="Recursively predict the winners of all knockout matches and display the model's complete path to the champion."
            )
        
        # Calculate active bracket state
        active_bracket = resolve_bracket_state(bracket_structure, predictor, forecast_mode=forecast_path)
            
        # Compile columns
        # Col 1: R32
        r32_html = '<div class="bracket-column">'
        for idx, m in enumerate(active_bracket["r32"], 1):
            r32_html += render_bracket_match_html(m, f"R32 Match {idx}")
        r32_html += '</div>'
        
        # Col 2: R16
        r16_html = '<div class="bracket-column">'
        for idx, m in enumerate(active_bracket["r16"], 1):
            r16_html += render_bracket_match_html(m, f"R16 Match {idx}")
        r16_html += '</div>'
        
        # Col 3: QF
        qf_html = '<div class="bracket-column">'
        for idx, m in enumerate(active_bracket["qf"], 1):
            qf_html += render_bracket_match_html(m, f"Quarterfinal {idx}")
        qf_html += '</div>'
        
        # Col 4: SF
        sf_html = '<div class="bracket-column">'
        for idx, m in enumerate(active_bracket["sf"], 1):
            sf_html += render_bracket_match_html(m, f"Semifinal {idx}")
        sf_html += '</div>'
        
        # Col 5: Final / 3rd-Place & Champion
        final_html = '<div class="bracket-column">'
        final_html += render_bracket_match_html(active_bracket["final"], "World Cup Final")
        final_html += render_bracket_match_html(active_bracket["third_place"], "3rd-Place Match")
        
        # Champion Display
        if forecast_path:
            champ = active_bracket["final"].get("winner", "TBD")
            champ_clean = clean_name_for_mapping(champ)
            champ_flag = get_flag_url(champ_clean)
            
            final_html += f"""
            <div class="champion-box">
                <div class="champion-title">🏆 PREDICTED CHAMPION 🏆</div>
                <img class="champion-flag" src="{champ_flag}" crossorigin="anonymous" />
                <div class="champion-name">{champ}</div>
            </div>
            """
        else:
            final_html += f"""
            <div class="champion-box" style="border-color: var(--card-border); background: rgba(255,255,255,0.03);">
                <div class="champion-title" style="color: var(--text-muted);">🏆 CHAMPION 🏆</div>
                <div class="champion-name" style="color: var(--text-muted); font-size: 1rem;">TBD</div>
            </div>
            """
        final_html += '</div>'
        
        # Combine all columns
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

    # ── CSV Export ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Export Prediction Data")
    
    # Format the df for download
    download_df = df_preds.copy()
    download_df['home_win_probability'] = download_df['home_prob'].round(4)
    download_df['draw_probability'] = download_df['draw_prob'].round(4)
    download_df['away_win_probability'] = download_df['away_prob'].round(4)
    
    columns_to_keep = [
        "date", "home_team", "away_team", "neutral", "home_score", "away_score",
        "home_win_probability", "draw_probability", "away_win_probability",
        "predicted_outcome", "actual_outcome", "correct", "stage"
    ]
    csv_data = download_df[columns_to_keep].to_csv(index=False)
    
    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="wc_2026_predictions.csv",
        mime="text/csv",
        use_container_width=False
    )

if __name__ == '__main__':
    main()

