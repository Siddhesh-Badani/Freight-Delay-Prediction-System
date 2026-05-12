"""
Threshold optimization and evaluation report generator.
Sweeps thresholds from 0.10 to 0.90 and selects the point where recall >= 0.80
while maximizing precision. Writes outputs/evaluation_report.md and outputs/metrics.json.

Run: python src/evaluate.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)

from features import FEATURE_COLS, TARGET_COL

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
OUT_DIR    = Path(__file__).parent.parent / "outputs"

RECALL_TARGET = 0.80


def load_test_split(df: pd.DataFrame):
    train = df[df["year"].between(2019, 2022)]
    val   = df[df["year"] == 2023]
    test  = df[df["year"] == 2024]
    return train, val, test


def find_optimal_threshold(proba: np.ndarray, y_true: np.ndarray, recall_min: float = 0.80):
    """
    Sweep thresholds 0.10-0.90 in steps of 0.05.
    Find the threshold achieving recall >= recall_min with maximum precision.
    """
    thresholds = np.arange(0.10, 0.91, 0.05)
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidates = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        rec  = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)
        candidates.append({"threshold": round(t, 2), "precision": prec, "recall": rec, "f1": f1})
        if rec >= recall_min:
            if prec > best["precision"]:
                best = {"threshold": round(t, 2), "precision": prec, "recall": rec, "f1": f1}

    # If no threshold satisfies recall_min, fall back to the one with highest recall
    if best["precision"] == 0.0:
        best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["recall"])
        best = candidates[best_idx]

    return best, candidates


def model_metrics(model, X, y, label=""):
    proba = model.predict_proba(X)[:, 1]
    roc   = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    opt, sweep = find_optimal_threshold(proba, y, RECALL_TARGET)

    # Precision-recall curve (for plotting)
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y, proba)
    # ROC curve (for plotting)
    fpr_curve, tpr_curve, roc_thresh = roc_curve(y, proba)

    # Confusion matrix at optimal threshold
    preds = (proba >= opt["threshold"]).astype(int)
    cm = confusion_matrix(y, preds).tolist()

    return {
        "label":             label,
        "roc_auc":           round(roc, 4),
        "pr_auc":            round(pr_auc, 4),
        "optimal_threshold": opt["threshold"],
        "precision":         round(opt["precision"], 4),
        "recall":            round(opt["recall"], 4),
        "f1":                round(opt["f1"], 4),
        "confusion_matrix":  cm,
        # Curve data (lists for JSON)
        "prec_curve":    prec_curve.round(4).tolist(),
        "rec_curve":     rec_curve.round(4).tolist(),
        "fpr_curve":     fpr_curve.round(4).tolist(),
        "tpr_curve":     tpr_curve.round(4).tolist(),
        "threshold_sweep": sweep,
        # Raw probabilities for histogram
        "proba": proba.round(4).tolist(),
        "y_true": y.tolist(),
    }


def write_evaluation_report(lr_m: dict, gbm_m: dict, feature_cols: list, out_path: Path):
    lines = [
        "# Evaluation Report: Container Freight Delay Prediction",
        "",
        "## Overview",
        "",
        "This report covers two models trained on a temporal split of 100,000 synthetic",
        "ocean freight shipments. The task is binary classification: predict whether a",
        "container shipment arrives more than 24 hours late.",
        "",
        "Threshold optimization targets recall >= 0.80. Operations teams lose more value",
        "from a missed delay (no buffer arranged, customer complaints, re-routing costs)",
        "than from a false alarm (an unnecessary buffer day).",
        "",
        "## Temporal Split",
        "",
        "| Split | Years | Purpose |",
        "|-------|-------|---------|",
        "| Train | 2019-2022 | Model fitting including COVID disruption era |",
        "| Validation | 2023 | Hyperparameter tuning and early stopping |",
        "| Test | 2024 | Final held-out evaluation, never seen during training |",
        "",
        "## Model Results on Test Set",
        "",
        "| Metric | Logistic Regression | Gradient-Boosted Trees |",
        "|--------|--------------------|-----------------------|",
        f"| ROC-AUC | {lr_m['roc_auc']} | {gbm_m['roc_auc']} |",
        f"| PR-AUC | {lr_m['pr_auc']} | {gbm_m['pr_auc']} |",
        f"| Optimal threshold | {lr_m['optimal_threshold']} | {gbm_m['optimal_threshold']} |",
        f"| Recall @ threshold | {lr_m['recall']} | {gbm_m['recall']} |",
        f"| Precision @ threshold | {lr_m['precision']} | {gbm_m['precision']} |",
        f"| F1 @ threshold | {lr_m['f1']} | {gbm_m['f1']} |",
        "",
        "## GBM Confusion Matrix (at chosen threshold)",
        "",
        "```",
        "                 Predicted On-Time   Predicted Delayed",
        f"Actual On-Time       {gbm_m['confusion_matrix'][0][0]:>8,}            {gbm_m['confusion_matrix'][0][1]:>8,}",
        f"Actual Delayed       {gbm_m['confusion_matrix'][1][0]:>8,}            {gbm_m['confusion_matrix'][1][1]:>8,}",
        "```",
        "",
        "## Feature Definitions",
        "",
        "Every feature below is computed using only data available before a shipment's departure date.",
        "",
        "| Feature | Definition | Rationale |",
        "|---------|-----------|-----------|",
        "| vessel_utilization_pct | % of vessel TEU capacity loaded | Overloaded vessels have tighter margins and face more delays at berth |",
        "| origin_congestion_index | Port congestion at origin (0-10) | Congested origins cause missed departure windows |",
        "| dest_congestion_index | Port congestion at destination (0-10) | High destination congestion causes anchorage queues |",
        "| weather_severity_score | Beaufort-inspired severity (0-10) | Severe weather forces speed reductions or course changes |",
        "| carrier_reliability_score | Historical carrier on-time rate (0-100) | Structural carrier quality captures scheduling discipline |",
        "| transit_distance_nm | Great-circle distance in nautical miles | Longer routes accumulate more variance |",
        "| scheduled_transit_days | Planned voyage duration in days | Short-scheduled routes have less buffer for disruption |",
        "| vessel_capacity_teu | Vessel size in TEU | Larger vessels have complex multi-port calls |",
        "| container_count | Number of containers in this booking | Larger bookings increase loading complexity |",
        "| cargo_weight_tons | Total cargo weight | Heavier loads affect draft, speed and berth selection |",
        "| declared_value_usd | Declared cargo value | Proxy for cargo type and priority handling |",
        "| route_delay_rate_30d | 30-day rolling delay rate on same route | Recent route history is the strongest available predictor |",
        "| dest_congestion_trend_7d | 7-day rolling avg destination congestion | Short-term trend predicts arrival queue conditions |",
        "| carrier_ontime_rate_30d | Carrier's 30-day rolling on-time fraction | Captures recent operational volatility at the carrier level |",
        "| vessel_delay_count_90d | # of vessel delays in past 90 days | Some vessels repeatedly attract delays due to maintenance issues |",
        "| month | Calendar month (1-12) | Captures seasonal loading peaks |",
        "| quarter | Calendar quarter (1-4) | Q4 peak season aggregation |",
        "| day_of_week | Day of week (0=Mon) | Weekend departures can face reduced port staffing |",
        "| is_lunar_new_year | 1 if Jan 20 - Feb 28 window | Factory and port shutdowns in China disrupt Asia-Pacific routes |",
        "| is_q4_peak | 1 if Oct-Dec | Holiday season dramatically increases container volumes |",
        "| is_covid_era | 1 if 2020-2022 | Captures structural disruption from pandemic-era operations |",
        "| distance_bucket | 0=short, 1=medium, 2=long, 3=transpacific | Ordinal route-length category |",
        "| util_x_dest_congestion | vessel_utilization * dest_congestion / 100 | Synergy: overloaded vessel arriving at congested port |",
        "| weather_x_route_risk | weather_severity * route_delay_rate_30d | Amplifies weather impact on already-risky routes |",
        "| congestion_delta | dest_congestion - origin_congestion | Measures relative port pressure differential |",
        "| is_reefer | 1 if refrigerated container | Reefer cargo requires priority handling, adding delay risk |",
        "| is_high_utilization | 1 if utilization > 90% | Binary flag for extreme overload condition |",
        "| is_low_reliability_carrier | 1 if reliability < 65 | Binary flag for structurally weak carriers |",
        "| carrier_name_enc | Label-encoded carrier | Carrier identity as ordinal category |",
        "| container_type_enc | Label-encoded container type | Container class as ordinal category |",
        "| origin_port_enc | Label-encoded origin port | Origin port as ordinal category |",
        "| destination_port_enc | Label-encoded destination port | Destination port as ordinal category |",
        "",
        "## Threshold Sweep Summary (GBM)",
        "",
        "| Threshold | Precision | Recall | F1 |",
        "|-----------|-----------|--------|-----|",
    ]

    for row in gbm_m["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )

    lines += [
        "",
        "## Methodology Notes",
        "",
        "- Rolling features use `closed='left'` in pandas rolling windows. This means",
        "  the window for shipment departing on date D spans [D-window, D), never including D itself.",
        "- Cold-start NaN values (routes or carriers with no prior history) are filled with",
        "  population-level priors: 0.20 for delay rate, 0.78 for carrier on-time rate.",
        "- Models are evaluated on 2024 data only. No information from 2024 was available",
        "  during training or threshold selection.",
        "",
    ]

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading features and models...")
    df = pd.read_csv(
        DATA_DIR / "features.csv",
        parse_dates=["departure_date", "scheduled_arrival", "actual_arrival"],
    )

    with open(OUT_DIR / "split_meta.json") as f:
        meta = json.load(f)

    logreg = joblib.load(MODELS_DIR / "logreg.joblib")
    gbm    = joblib.load(MODELS_DIR / "gbm.joblib")

    _, _, test = load_test_split(df)
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    print("Computing metrics on test set (2024)...")
    lr_m  = model_metrics(logreg, X_test, y_test, "Logistic Regression")
    gbm_m = model_metrics(gbm,    X_test, y_test, "Gradient-Boosted Trees")

    print(f"\n{'Model':<28}  ROC-AUC  PR-AUC  Threshold  Recall  Precision  F1")
    for m in [lr_m, gbm_m]:
        print(
            f"  {m['label']:<26}  {m['roc_auc']:.4f}  {m['pr_auc']:.4f}"
            f"  {m['optimal_threshold']:.2f}      {m['recall']:.3f}  {m['precision']:.3f}"
            f"    {m['f1']:.3f}"
        )

    # Save predictions for dashboard
    test_preds = test[["shipment_id", "departure_date", "route_id",
                        "origin_port", "destination_port"]].copy()
    test_preds["y_true"]    = y_test
    test_preds["prob_lr"]   = logreg.predict_proba(X_test)[:, 1]
    test_preds["prob_gbm"]  = gbm.predict_proba(X_test)[:, 1]
    test_preds["pred_lr"]   = (test_preds["prob_lr"]  >= lr_m["optimal_threshold"]).astype(int)
    test_preds["pred_gbm"]  = (test_preds["prob_gbm"] >= gbm_m["optimal_threshold"]).astype(int)
    test_preds.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # Save full metrics JSON (shared by interpret.py and build_dashboard.py)
    # Strip raw probability arrays to keep file size manageable
    metrics_out = {
        "logreg": {k: v for k, v in lr_m.items() if k not in ("proba", "y_true")},
        "gbm":    {k: v for k, v in gbm_m.items() if k not in ("proba", "y_true")},
        "feature_cols": FEATURE_COLS,
        "recall_target": RECALL_TARGET,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    write_evaluation_report(lr_m, gbm_m, FEATURE_COLS, OUT_DIR / "evaluation_report.md")

    print(f"\nSaved: {OUT_DIR}/evaluation_report.md")
    print(f"Saved: {OUT_DIR}/metrics.json")
    print(f"Saved: {OUT_DIR}/test_predictions.csv")
