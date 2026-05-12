"""
End-to-end pipeline runner.
Executes all steps in order and prints a final summary.

Run from project root: python run_pipeline.py
"""

import subprocess
import sys
import json
import time
from pathlib import Path

ROOT    = Path(__file__).parent
SRC     = ROOT / "src"
OUT_DIR = ROOT / "outputs"


def run_step(label: str, script: Path):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SRC),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFATAL: {label} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  Completed in {elapsed:.1f}s")


def print_final_summary():
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}\n")

    # Load metrics
    metrics_path = OUT_DIR / "metrics.json"
    split_path   = OUT_DIR / "split_meta.json"
    shap_path    = OUT_DIR / "shap_importance.csv"

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(split_path) as f:
            meta = json.load(f)

        print(f"  Records generated:    100,000")
        print(f"  Train (2019-2022):    {meta['train_size']:>8,}  ({meta['train_delay_rate']:.1%} delayed)")
        print(f"  Validation (2023):    {meta['val_size']:>8,}  ({meta['val_delay_rate']:.1%} delayed)")
        print(f"  Test (2024):          {meta['test_size']:>8,}  ({meta['test_delay_rate']:.1%} delayed)")

        gbm = metrics["gbm"]
        lr  = metrics["logreg"]
        print(f"\n  Best model:           Gradient-Boosted Trees")
        print(f"  Chosen threshold:     {gbm['optimal_threshold']}")
        print(f"  Recall  @ threshold:  {gbm['recall']:.3f}")
        print(f"  Precision @ thresh:   {gbm['precision']:.3f}")
        print(f"  F1 @ threshold:       {gbm['f1']:.3f}")
        print(f"  ROC-AUC (GBM):        {gbm['roc_auc']:.4f}")
        print(f"  ROC-AUC (LR):         {lr['roc_auc']:.4f}")

    if shap_path.exists():
        import pandas as pd
        imp = pd.read_csv(shap_path)
        print(f"\n  Top 5 features by SHAP importance:")
        for _, row in imp.head(5).iterrows():
            print(f"    {row['feature']:<35}  {row['mean_abs_shap']:.4f}")

    print(f"\n  Generated artifacts:")
    artifacts = [
        ("data/raw_shipments.csv",          "Synthetic ocean freight dataset"),
        ("data/features.csv",               "Engineered feature set"),
        ("models/logreg.joblib",            "Logistic Regression model"),
        ("models/gbm.joblib",               "Gradient-Boosted Trees model"),
        ("outputs/evaluation_report.md",    "Full evaluation report"),
        ("outputs/recommendations.md",      "Business recommendations"),
        ("outputs/shap_summary.png",        "SHAP beeswarm summary plot"),
        ("outputs/shap_bar.png",            "SHAP feature importance bar chart"),
        ("outputs/dashboard.html",          "Interactive Plotly dashboard"),
        ("outputs/story.html",              "Scroll narrative story page"),
    ]
    for rel_path, desc in artifacts:
        full = ROOT / rel_path
        size = f"{full.stat().st_size / 1024:.0f} KB" if full.exists() else "MISSING"
        print(f"    {rel_path:<40}  {size:<8}  {desc}")

    print()


if __name__ == "__main__":
    steps = [
        ("1/7  Generate synthetic data",    SRC / "generate_data.py"),
        ("2/7  Engineer features",          SRC / "features.py"),
        ("3/7  Train models",               SRC / "train.py"),
        ("4/7  Evaluate and optimize",      SRC / "evaluate.py"),
        ("5/7  SHAP interpretation",        SRC / "interpret.py"),
        ("6/7  Build dashboard",            SRC / "build_dashboard.py"),
        ("7/7  Build story page",           SRC / "build_story.py"),
    ]

    total_t0 = time.time()
    for label, script in steps:
        run_step(label, script)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s")

    print_final_summary()
