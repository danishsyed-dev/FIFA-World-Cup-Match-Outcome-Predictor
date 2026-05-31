"""
streamlit_app.py
----------------
Streamlit dashboard for the FIFA Match Outcome Predictor.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0e1a; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .prob-card {
        background: linear-gradient(135deg, #1e2535 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }

    .prob-card:hover { transform: translateY(-4px); }

    .prob-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.3rem 0;
    }

    .prob-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .winner-banner {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1rem 2rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #6ee7b7;
        margin: 1rem 0;
    }

    .metric-box {
        background: #1e2535;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #3b82f6;
        margin: 0.4rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load predictor (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and historical data…")
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


def probability_gauge(prob: float, label: str, color: str):
    """Return a Plotly gauge figure."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": color}},
            title={"text": label, "font": {"size": 14, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": color},
                "bgcolor": "#1e2535",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 33], "color": "#1e293b"},
                    {"range": [33, 66], "color": "#1e2535"},
                    {"range": [66, 100], "color": "#172032"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        height=200,
    )
    return fig


def probability_bar_chart(home_team, away_team, hw, dr, aw):
    """Horizontal stacked bar chart for probabilities."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[hw * 100],
        name=f"{home_team} Win",
        orientation="h",
        marker_color="#3b82f6",
        text=f"{hw*100:.1f}%",
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[dr * 100],
        name="Draw",
        orientation="h",
        marker_color="#f59e0b",
        text=f"{dr*100:.1f}%",
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[aw * 100],
        name=f"{away_team} Win",
        orientation="h",
        marker_color="#ef4444",
        text=f"{aw*100:.1f}%",
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=40, b=10, l=10, r=10),
        height=130,
        yaxis=dict(visible=False),
        xaxis=dict(visible=False, range=[0, 100]),
    )
    return fig


def elo_comparison_chart(home_team, away_team, home_elo, away_elo):
    """Bar chart comparing Elo ratings."""
    fig = go.Figure(go.Bar(
        x=[home_team, away_team],
        y=[home_elo, away_elo],
        marker_color=["#3b82f6", "#ef4444"],
        text=[f"{home_elo:.0f}", f"{away_elo:.0f}"],
        textposition="outside",
        textfont=dict(size=14, color="#e2e8f0"),
    ))
    fig.update_layout(
        title="Elo Rating Comparison",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis=dict(gridcolor="#1e2535", showgrid=True, range=[min(home_elo, away_elo) - 200, max(home_elo, away_elo) + 200]),
        xaxis=dict(tickfont=dict(size=13)),
        margin=dict(t=40, b=10, l=10, r=10),
        height=260,
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    # Hero header
    st.markdown('<div class="hero-title">⚽ FIFA Match Outcome Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Machine Learning powered predictions using historical international football data</div>',
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    if isinstance(predictor, str):
        st.error(f"⚠️ Model not found. Please run training first.\n\n```\npython src/train.py\n```\n\n{predictor}")
        st.stop()

    all_teams = get_all_teams(predictor)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎮 Match Setup")

        default_home = all_teams.index("Brazil") if "Brazil" in all_teams else 0
        default_away = all_teams.index("Argentina") if "Argentina" in all_teams else 1

        home_team = st.selectbox("🏠 Home Team", all_teams, index=default_home)
        away_team = st.selectbox("✈️ Away Team", all_teams, index=default_away)

        tournament = st.selectbox(
            "🏆 Tournament Type",
            ["Friendly", "Qualifier", "Continental Cup", "FIFA World Cup"],
            index=3,
        )
        neutral = st.toggle("⚖️ Neutral Venue", value=False)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Outcome", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📖 About")
        st.caption(
            "This model uses Elo ratings, recent form, goals scored/conceded, "
            "head-to-head records, and tournament importance to predict match outcomes. "
            "Target accuracy: 55–70%."
        )

    # ── Main content ──────────────────────────────────────────────────────────
    if home_team == away_team:
        st.warning("⚠️ Please select two different teams.")
        st.stop()

    # Auto-predict or on button click
    if predict_btn or True:
        with st.spinner("Computing prediction…"):
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

        # ── Match banner ──────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center; font-size:2rem; font-weight:800; 
                    color:#e2e8f0; padding:0.5rem 0; margin:0.5rem 0;">
            🏴 {home_team} &nbsp;⚔️&nbsp; {away_team} 🏴
        </div>
        """, unsafe_allow_html=True)

        # Tournament & venue badge
        venue_text = "⚖️ Neutral Venue" if neutral else "🏠 Home Advantage"
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <span style="background:#1e2535;border:1px solid #334155;border-radius:20px;
                         padding:0.3rem 1rem;color:#94a3b8;font-size:0.9rem;">
                🏆 {tournament} &nbsp;|&nbsp; {venue_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability cards ─────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-label">{home_team} Win</div>
                <div class="prob-value" style="color:#3b82f6;">{hw*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-label">Draw</div>
                <div class="prob-value" style="color:#f59e0b;">{dr*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="prob-card">
                <div class="prob-label">{away_team} Win</div>
                <div class="prob-value" style="color:#ef4444;">{aw*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability bar ───────────────────────────────────────────────
        st.plotly_chart(
            probability_bar_chart(home_team, away_team, hw, dr, aw),
            use_container_width=True,
        )

        # ── Winner & confidence ───────────────────────────────────────────
        st.markdown(
            f'<div class="winner-banner">🏆 Predicted: {predicted} &nbsp;|&nbsp; Confidence: {confidence*100:.1f}%</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Details section ───────────────────────────────────────────────
        left, right = st.columns(2)

        with left:
            st.subheader("🔢 Elo Ratings")
            st.plotly_chart(
                elo_comparison_chart(home_team, away_team, home_elo, away_elo),
                use_container_width=True,
            )

        with right:
            st.subheader("📊 Match Stats")

            # Compute supporting stats
            today = pd.Timestamp.now()
            from src.feature_engineering import _recent_stats, _h2h_stats

            hf, hgs, hgc = _recent_stats(predictor.df, home_team, today)
            af, ags, agc = _recent_stats(predictor.df, away_team, today)
            h2h_hw, h2h_draws = _h2h_stats(predictor.df, home_team, away_team, today)

            stats = [
                ("Elo Difference", f"{home_elo - away_elo:+.0f}", "🔢"),
                (f"{home_team} Recent Form (last 5)", f"{hf:.1f} pts", "📈"),
                (f"{away_team} Recent Form (last 5)", f"{af:.1f} pts", "📈"),
                (f"{home_team} Avg Goals Scored", f"{hgs:.2f}", "⚽"),
                (f"{away_team} Avg Goals Scored", f"{ags:.2f}", "⚽"),
                (f"{home_team} H2H Wins (last 10)", str(h2h_hw), "🏆"),
                ("H2H Draws (last 10)", str(h2h_draws), "🤝"),
            ]
            for label, val, icon in stats:
                st.markdown(
                    f'<div class="metric-box">{icon} <strong>{label}:</strong> {val}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Batch simulation ──────────────────────────────────────────────
        with st.expander("🔄 Batch Predict Multiple Fixtures"):
            st.caption("Upload a CSV with columns: home_team, away_team, neutral (bool), tournament (str)")
            uploaded = st.file_uploader("Upload fixtures CSV", type=["csv"])
            if uploaded:
                fixtures = pd.read_csv(uploaded)
                predictions = predictor.predict_batch(fixtures)
                st.dataframe(predictions, use_container_width=True)
                csv_bytes = predictions.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions", csv_bytes, "predictions.csv", "text/csv")

        # ── Model info ────────────────────────────────────────────────────
        with st.expander("ℹ️ Model Information"):
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
