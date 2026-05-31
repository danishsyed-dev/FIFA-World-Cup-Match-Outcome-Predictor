# Data Directory

Place your Kaggle CSV files here before running the pipeline.

## Required Files

### results.csv (REQUIRED)
**Download:** https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Contains 49,000+ international football matches from 1872 to 2026.

Columns:
- `date` — Match date
- `home_team` — Home team name
- `away_team` — Away team name
- `home_score` — Goals scored by home team
- `away_score` — Goals scored by away team
- `tournament` — Tournament name
- `city` — City where match was played
- `country` — Country where match was played
- `neutral` — True/False (neutral venue)

---

### elo.csv (OPTIONAL but recommended)
**Download:** https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings

If not provided, the EloSystem in `src/elo_calculator.py` will calculate
Elo ratings from scratch using the `results.csv` data.

Columns:
- `team` — Team name
- `date` — Date of rating
- `elo` — Elo rating value

---

## Quick Kaggle Download (requires kaggle CLI)

```bash
pip install kaggle
kaggle datasets download martj42/international-football-results-from-1872-to-2017 -p data/ --unzip
kaggle datasets download saifalnimri/international-football-elo-ratings -p data/ --unzip
```
