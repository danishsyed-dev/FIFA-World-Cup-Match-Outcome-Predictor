# Product Requirements Document (PRD)

# Match Outcome Predictor using International Football Data

> This is a strong ML project because it combines **data engineering + feature engineering + machine learning + sports analytics + deployment**.
>
> You are basically building:
> **"Given Team A vs Team B, predict Win / Draw / Loss probability using historical international football data."**

---

## Project Overview

Build a machine learning system that predicts the outcome of an international football match using historical match data, team strength metrics, and recent performance statistics.

The system should allow a user to select any two national teams and receive:

- Win probability for Team A
- Draw probability
- Win probability for Team B
- Predicted winner
- Confidence score

---

## Problem Statement

Football match outcomes are influenced by many factors:

- Team strength
- Recent form
- Historical performance
- Home advantage
- Tournament importance

Fans and analysts often rely on intuition.

This project aims to build a data-driven prediction model using historical international football data.

---

## Goals

### Primary Goal

Predict match outcomes using machine learning.

### Secondary Goals

- Build Elo rating system
- Generate team form metrics
- Analyze important prediction features
- Create prediction dashboard

---

## Success Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Log Loss
- ROC-AUC

**Target Accuracy: 55% – 70%**

---

## Input

User selects:

- Team A
- Team B
- Match location
- Tournament type

---

## Output

Example:

```
Brazil vs Argentina

Brazil Win:      47%
Draw:            24%
Argentina Win:   29%

Predicted Winner: Brazil
Confidence:       47%
```

---

## Machine Learning Approach

**Baseline Model:**
- Logistic Regression

**Advanced Models:**
- Random Forest
- XGBoost
- LightGBM

---

## Deployment

| Layer     | Technology  |
|-----------|-------------|
| Frontend  | Streamlit   |
| Backend   | Python      |
| Model     | Scikit-Learn|
| Database  | CSV / SQLite|

---

## Deliverables

1. Data Collection Pipeline
2. Feature Engineering Pipeline
3. Elo Rating Calculator
4. Logistic Regression Model
5. Evaluation Report
6. Streamlit Web App
7. Documentation
8. GitHub Repository

---

## Architecture

```
Historical Match Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ├── Elo Rating
        ├── Recent Form
        ├── Goals Scored
        ├── Goals Conceded
        ├── Home Advantage
        ├── Tournament Type
        │
        ▼
Training Dataset
        │
        ▼
Logistic Regression
        │
        ▼
Evaluation
        │
        ▼
Prediction API
        │
        ▼
Streamlit Dashboard
```

---

## Datasets

### 1. International Football Results Dataset (MAIN)

Primary dataset — 49,000+ matches.

[Kaggle: International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Contains: Match Date · Home Team · Away Team · Home Score · Away Score · Tournament · Country · Neutral Venue

### 2. Elo Ratings Dataset

[Kaggle: International Football Elo Ratings](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)

Contains: Team · Date · Elo Rating (historical from 1872 onwards)

### 3. FIFA Rankings Dataset (Optional)

[Kaggle: FIFA Rankings](https://www.kaggle.com/datasets?search=fifa+rankings)

Contains: FIFA Rank · Team Points · Ranking Date

### 4. World Cup Matches Dataset

[Kaggle: FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)

Contains: World Cup matches · Tournament details · Historical winners

---

## Features to Engineer

| Feature | Description |
|---------|-------------|
| `elo_difference` | `home_elo - away_elo` — current team strength proxy |
| `recent_form_score` | Last 5 matches: Win=3, Draw=1, Loss=0 |
| `avg_goals_scored` | Avg goals scored in last 5 matches |
| `avg_goals_conceded` | Avg goals conceded in last 5 matches |
| `head_to_head_wins` | Win count in last 10 H2H meetings |
| `home_advantage` | home=1, neutral=0 |
| `tournament_importance` | Friendly / Qualifier / Continental / World Cup (encoded) |

---

## Model Training

**Target Classes:**
```
0 = Away Win
1 = Draw
2 = Home Win
```

**Training Steps:**
1. Start with `LogisticRegression()` as baseline
2. Add Random Forest, XGBoost, LightGBM
3. Evaluate all models and select best

---

## Evaluation

```python
accuracy_score()
confusion_matrix()
classification_report()
log_loss()
roc_auc_score()
```

---

## Folder Structure

```
match-outcome-predictor/
├── data/
│   ├── results.csv
│   └── elo.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── elo_calculator.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── logistic_model.pkl
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component     | Technology        |
|---------------|-------------------|
| Language      | Python            |
| Data          | Pandas            |
| ML            | Scikit-Learn      |
| Visualization | Matplotlib        |
| Dashboard     | Streamlit         |
| Model Storage | Pickle            |
| Data Source   | Kaggle            |
| API (Optional)| Football-Data.org |

---

## Resume-Worthy Extensions (Future Scope)

After Logistic Regression, add:

1. Random Forest
2. XGBoost
3. Elo Rating Calculator from Scratch
4. Feature Importance Dashboard
5. Live Match Prediction API
6. World Cup Tournament Simulator
7. Monte Carlo Simulation
8. Predict Entire FIFA World Cup Bracket

---

*Document created from: https://chatgpt.com/s/t_6a1c63b67f2c8191a32c361facdd4a19*
