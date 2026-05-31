"""
train.py
--------
Trains and evaluates multiple classification models, then saves the best one.

Models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM

Usage:
    python src/train.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.data_loader import load_all
from src.feature_engineering import build_features, get_feature_columns
from src.elo_calculator import EloSystem

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = get_feature_columns()
TARGET_COL = "outcome"
LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}
RANDOM_STATE = 42


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return models


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute and print evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)

    # ROC-AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Log Loss : {ll:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values())))

    return {"name": name, "model": model, "accuracy": acc, "log_loss": ll, "roc_auc": auc}


def plot_confusion_matrix(name: str, model, X_test, y_test, save_dir: Path) -> None:
    """Save a confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(LABEL_MAP.values()),
        yticklabels=list(LABEL_MAP.values()),
        ax=ax,
    )
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    filename = save_dir / f"confusion_{name.lower().replace(' ', '_')}.png"
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  [train] Confusion matrix saved → {filename}")


def plot_feature_importance(name: str, model, feature_cols: list, save_dir: Path) -> None:
    """Save a feature importance bar chart (tree models only)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(feature_cols)), importances[idx], color="steelblue")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_title(f"{name} — Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    filename = save_dir / f"importance_{name.lower().replace(' ', '_')}.png"
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  [train] Feature importance saved → {filename}")


def plot_model_comparison(results: list, save_dir: Path) -> None:
    """Bar chart comparing all models by accuracy."""
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color=["#4472C4", "#ED7D31", "#A9D18E", "#FF0000"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Accuracy")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    filename = save_dir / "model_comparison.png"
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  [train] Model comparison chart saved → {filename}")


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  FIFA Match Outcome Predictor — Model Training")
    print("="*60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_all()

    # 2. Build Elo ratings from scratch (overwrite data_loader Elo)
    print("\n[2/5] Calculating Elo ratings from scratch...")
    elo_system = EloSystem()
    df = elo_system.calculate(df)

    # 3. Feature engineering
    print("\n[3/5] Engineering features...")
    df = build_features(df, verbose=True)

    # Drop rows with NaN features
    df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
    print(f"  Dataset size after dropping NaN: {len(df_model):,} rows")
    print(f"  Class distribution:\n{df_model[TARGET_COL].value_counts().to_string()}")

    X = df_model[FEATURE_COLS].values
    y = df_model[TARGET_COL].values

    # 4. Train/Test split (80/20, NO shuffle to respect time ordering)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, shuffle=False
    )
    print(f"\n  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # 5. Train models
    print("\n[4/5] Training models...")
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\n  → Training {name}...")
        model.fit(X_train, y_train)

        # Save model
        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        print(f"    Model saved → {model_path}")

        # Evaluate
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        # Plots
        plot_confusion_matrix(name, model, X_test, y_test, MODELS_DIR)
        plot_feature_importance(name, model, FEATURE_COLS, MODELS_DIR)

    # 6. Compare & select best model
    print("\n[5/5] Selecting best model...")
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n  ★ Best Model: {best['name']}  (Accuracy: {best['accuracy']:.4f})")

    # Save best model separately for easy loading
    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best["model"], best_path)
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")
    print(f"  Best model saved → {best_path}")

    plot_model_comparison(results, MODELS_DIR)

    # Summary table
    print("\n" + "="*55)
    print("  Model Summary")
    print("="*55)
    print(f"  {'Model':<22} {'Accuracy':>10} {'Log Loss':>10} {'ROC-AUC':>10}")
    print("  " + "-"*52)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"  {r['name']:<22} {r['accuracy']:>10.4f} {r['log_loss']:>10.4f} {r['roc_auc']:>10.4f}")

    print("\n  Training complete.\n")


if __name__ == "__main__":
    main()
