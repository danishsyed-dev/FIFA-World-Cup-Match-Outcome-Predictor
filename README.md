# FIFA Match Outcome Predictor

A machine learning system that predicts the outcome of any international football match — **Win / Draw / Loss** — using historical data, Elo ratings, and recent form metrics.

> **"Given Team A vs Team B, predict Win / Draw / Loss probability using historical international football data."**

---

## 🏆 Features

| Feature | Description |
|---------|-------------|
| **Elo Rating System** | Built from scratch using the World Football Elo formula |
| **Recent Form** | Last 5 matches — Win=3, Draw=1, Loss=0 |
| **Goals Scored / Conceded** | Rolling average over last 5 matches |
| **Head-to-Head Record** | Last 10 meetings between the two teams |
| **Home Advantage** | Encoded as 1 (home) or 0 (neutral venue) |
| **Tournament Weight** | Friendly → Qualifier → Continental → World Cup (0–4) |

---

## 📊 Models

| Model | Role |
|-------|------|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| XGBoost | Primary |
| LightGBM | Primary |

**Target accuracy: 55–70%** (expert forecasters achieve ~55–60%)

---

## 🗂️ Project Structure

```
match-outcome-predictor/
├── data/
│   ├── results.csv          ← Download from Kaggle (see below)
│   └── elo.csv              ← Optional Elo dataset
├── notebooks/
│   └── analysis.ipynb       ← EDA + feature exploration
├── src/
│   ├── data_loader.py       ← Load & clean datasets
│   ├── elo_calculator.py    ← Custom Elo rating system
│   ├── feature_engineering.py ← All feature creation
│   ├── train.py             ← Train & evaluate models
│   └── predict.py           ← Prediction interface
├── models/                  ← Saved .pkl model files
├── app/
│   └── streamlit_app.py     ← Interactive web dashboard
├── requirements.txt
├── PRD.md
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Download these datasets from Kaggle and place CSVs in `data/`:

- **results.csv** → [International Football Results 1872–2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **elo.csv** (optional) → [International Football Elo Ratings](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)

### 3. Train Models

```bash
python src/train.py
```

This will:
- Load and clean data
- Calculate Elo ratings from scratch
- Engineer all features
- Train Logistic Regression, Random Forest, XGBoost, LightGBM
- Save models to `models/`
- Generate evaluation plots

### 4. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 5. CLI Prediction

```bash
python src/predict.py --team-a "Brazil" --team-b "Argentina" --tournament "FIFA World Cup"
```

**Output:**
```
=====================================================
  ⚽  Brazil  vs  Argentina
=====================================================
  Brazil Win : 47.3%  ████████████░░░░░░░░░░░░░
  Draw       : 24.1%  ██████░░░░░░░░░░░░░░░░░░░
  Argentina  : 28.6%  ███████░░░░░░░░░░░░░░░░░░

  🏆 Predicted: Home Win
  📊 Confidence: 47.3%
  🔢 Elo  — Brazil: 2078  |  Argentina: 2044
=====================================================
```

### 6. Batch Predictions

```bash
python src/predict.py --input fixtures.csv --output predictions.csv
```

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Log Loss (probability calibration)
- ROC-AUC (macro, one-vs-rest)
- Confusion Matrix per model

---

## 📚 Datasets Used

| Dataset | Source | Matches |
|---------|--------|---------|
| International Football Results | Kaggle (martj42) | 49,000+ |
| Elo Ratings | Kaggle (saifalnimri) | Historical |
| FIFA World Cup Matches | Kaggle (abecklas) | WC only |

---

## 🚀 Resume-Worthy Extensions

After the base project, add:

1. ☐ Random Forest + XGBoost models *(done)*
2. ☐ Elo Calculator from Scratch *(done)*
3. ☐ Feature Importance Dashboard
4. ☐ World Cup Tournament Bracket Simulator
5. ☐ Monte Carlo Simulation
6. ☐ Live Match Prediction API (FastAPI)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data | Pandas, NumPy |
| ML | Scikit-Learn, XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Model Storage | Joblib (Pickle) |

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.

---

*Based on PRD: [Match Outcome Predictor using International Football Data](PRD.md)*
