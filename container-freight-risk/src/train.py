"""
Train two models on the engineered feature set using strict temporal splits.
  Train:      departures in 2019-2022
  Validation: departures in 2023
  Test:       departures in 2024

Run: python src/train.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LIGHTGBM = False

from features import FEATURE_COLS, TARGET_COL

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
OUT_DIR    = Path(__file__).parent.parent / "outputs"


def temporal_split(df: pd.DataFrame):
    """Strict temporal split. No shuffling."""
    train = df[df["year"].between(2019, 2022)].copy()
    val   = df[df["year"] == 2023].copy()
    test  = df[df["year"] == 2024].copy()
    return train, val, test


def print_split_summary(train, val, test):
    print("\nTemporal split summary:")
    print(f"{'Split':<10} {'N':>8}  {'Delay%':>7}  {'Years'}")
    for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
        yr_range = f"{split['departure_date'].dt.year.min()}-{split['departure_date'].dt.year.max()}"
        print(f"  {name:<10} {len(split):>8,}  {split[TARGET_COL].mean():>7.1%}  {yr_range}")


def build_logreg(X_train, y_train) -> Pipeline:
    """Logistic Regression with standard scaling and balanced class weights."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            C=0.5,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def build_gbm(X_train, y_train, X_val, y_val):
    """Gradient-boosted trees. Prefers LightGBM, falls back to HistGBM."""
    if USE_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=15,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            verbose=-1,
            metric="auc",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(period=-1)],
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.03,
            max_depth=5,
            min_samples_leaf=60,
            l2_regularization=0.5,
            class_weight="balanced",
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(X_train, y_train)

    return model


def quick_metrics(model, X, y, label="") -> dict:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.decision_function(X)

    roc  = roc_auc_score(y, proba)
    pr   = average_precision_score(y, proba)
    return {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "label": label}


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading feature set...")
    df = pd.read_csv(
        DATA_DIR / "features.csv",
        parse_dates=["departure_date", "scheduled_arrival", "actual_arrival"],
    )

    train, val, test = temporal_split(df)
    print_split_summary(train, val, test)

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_COL].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET_COL].values

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    logreg = build_logreg(X_train, y_train)
    lr_val  = quick_metrics(logreg, X_val,  y_val,  "LR val")
    lr_test = quick_metrics(logreg, X_test, y_test, "LR test")
    print(f"  Val  ROC-AUC={lr_val['roc_auc']:.4f}  PR-AUC={lr_val['pr_auc']:.4f}")
    print(f"  Test ROC-AUC={lr_test['roc_auc']:.4f}  PR-AUC={lr_test['pr_auc']:.4f}")
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")

    # --- Gradient-Boosted Trees ---
    backend = "LightGBM" if USE_LIGHTGBM else "HistGradientBoosting (sklearn)"
    print(f"\nTraining GBM ({backend})...")
    gbm = build_gbm(X_train, y_train, X_val, y_val)
    gbm_val  = quick_metrics(gbm, X_val,  y_val,  "GBM val")
    gbm_test = quick_metrics(gbm, X_test, y_test, "GBM test")
    print(f"  Val  ROC-AUC={gbm_val['roc_auc']:.4f}  PR-AUC={gbm_val['pr_auc']:.4f}")
    print(f"  Test ROC-AUC={gbm_test['roc_auc']:.4f}  PR-AUC={gbm_test['pr_auc']:.4f}")
    joblib.dump(gbm, MODELS_DIR / "gbm.joblib")

    # Save split metadata for downstream scripts
    split_meta = {
        "train_size": len(train),
        "val_size":   len(val),
        "test_size":  len(test),
        "train_delay_rate": float(y_train.mean()),
        "val_delay_rate":   float(y_val.mean()),
        "test_delay_rate":  float(y_test.mean()),
        "use_lightgbm":     USE_LIGHTGBM,
        "feature_cols":     FEATURE_COLS,
        "logreg_val":       lr_val,
        "logreg_test":      lr_test,
        "gbm_val":          gbm_val,
        "gbm_test":         gbm_test,
    }
    with open(OUT_DIR / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    print(f"\nModels saved: {MODELS_DIR}/logreg.joblib, {MODELS_DIR}/gbm.joblib")
    print(f"Metadata saved: {OUT_DIR}/split_meta.json")
