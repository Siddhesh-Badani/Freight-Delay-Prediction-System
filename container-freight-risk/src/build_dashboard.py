"""
Builds outputs/dashboard.html: a self-contained Plotly dashboard with six panels.
Requires: outputs/metrics.json, outputs/test_predictions.csv,
          outputs/shap_importance.csv, data/features.csv

Run: python src/build_dashboard.py
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs"

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#D97706"
GRAY   = "#6B7280"
LIGHT  = "#F1F5F9"


def load_inputs():
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["departure_date"])
    preds = pd.read_csv(OUT_DIR / "test_predictions.csv", parse_dates=["departure_date"])
    importance = pd.read_csv(OUT_DIR / "shap_importance.csv")
    with open(OUT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return df, preds, importance, metrics


def fig_delay_over_time(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.assign(month=df["departure_date"].dt.to_period("M"))
        .groupby("month", as_index=False)["delayed_24h"]
        .mean()
    )
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly["delay_pct"] = monthly["delayed_24h"] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month_dt"], y=monthly["delay_pct"],
        mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(size=4),
        name="Delay rate %",
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    ))

    # COVID annotation band
    fig.add_vrect(
        x0="2020-01-01", x1="2022-12-31",
        fillcolor="rgba(220,38,38,0.07)",
        line_width=0,
        annotation_text="COVID-era disruption",
        annotation_position="top left",
        annotation_font_color=RED,
    )
    fig.update_layout(
        title="Monthly Delay Rate (2019-2024)",
        xaxis_title="Month",
        yaxis_title="Delay rate (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=60, b=50),
    )
    return fig


def fig_delay_by_route(df: pd.DataFrame) -> go.Figure:
    route_stats = (
        df.groupby("route_id", as_index=False)
        .agg(delay_rate=("delayed_24h", "mean"), n=("delayed_24h", "count"))
        .query("n >= 50")
        .nlargest(20, "delay_rate")
        .sort_values("delay_rate")
    )

    fig = go.Figure(go.Bar(
        x=route_stats["delay_rate"] * 100,
        y=route_stats["route_id"],
        orientation="h",
        marker_color=BLUE,
        text=route_stats["delay_rate"].apply(lambda v: f"{v:.1%}"),
        textposition="outside",
        hovertemplate="Route: %{y}<br>Delay rate: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 Routes by Delay Rate",
        xaxis_title="Delay rate (%)",
        yaxis_title="Route",
        template="plotly_white",
        margin=dict(t=60, b=50, l=120),
    )
    return fig


def fig_shap_importance(importance: pd.DataFrame) -> go.Figure:
    top = importance.head(15).sort_values("mean_abs_shap")

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature"],
        orientation="h",
        marker_color=BLUE,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Importance (mean |SHAP value|, GBM)",
        xaxis_title="Mean |SHAP|",
        yaxis_title="Feature",
        template="plotly_white",
        margin=dict(t=60, b=50, l=200),
    )
    return fig


def fig_pr_curve(metrics: dict) -> go.Figure:
    fig = go.Figure()

    for model_key, color, dash in [("logreg", ORANGE, "dash"), ("gbm", BLUE, "solid")]:
        m = metrics[model_key]
        label = "Logistic Regression" if model_key == "logreg" else "Gradient-Boosted Trees"
        fig.add_trace(go.Scatter(
            x=m["rec_curve"],
            y=m["prec_curve"],
            mode="lines",
            name=f"{label} (PR-AUC={m['pr_auc']:.3f})",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="Recall=%{x:.3f}, Precision=%{y:.3f}<extra></extra>",
        ))
        # Mark optimal threshold point
        opt_t = m["optimal_threshold"]
        fig.add_trace(go.Scatter(
            x=[m["recall"]], y=[m["precision"]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="star"),
            name=f"{label} threshold={opt_t}",
            showlegend=True,
            hovertemplate=f"Optimal threshold={opt_t}<br>Recall=%{{x:.3f}}, Precision=%{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Precision-Recall Curves (Test Set, 2024)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend=dict(x=0.55, y=0.95),
        margin=dict(t=60, b=50),
    )
    return fig


def fig_confusion_matrix(metrics: dict) -> go.Figure:
    cm = np.array(metrics["gbm"]["confusion_matrix"])
    total = cm.sum()
    z_pct = cm / total * 100

    labels = ["On-Time", "Delayed"]
    text = [
        [f"TN<br>{cm[0,0]:,}<br>({z_pct[0,0]:.1f}%)",   f"FP<br>{cm[0,1]:,}<br>({z_pct[0,1]:.1f}%)"],
        [f"FN<br>{cm[1,0]:,}<br>({z_pct[1,0]:.1f}%)",   f"TP<br>{cm[1,1]:,}<br>({z_pct[1,1]:.1f}%)"],
    ]

    colorscale = [[0, "#F8FAFC"], [1, BLUE]]
    fig = go.Figure(go.Heatmap(
        z=z_pct,
        x=["Predicted On-Time", "Predicted Delayed"],
        y=["Actual On-Time", "Actual Delayed"],
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count: %{text}<extra></extra>",
    ))
    opt_t = metrics["gbm"]["optimal_threshold"]
    fig.update_layout(
        title=f"Confusion Matrix (GBM, threshold={opt_t})",
        template="plotly_white",
        margin=dict(t=60, b=50),
    )
    return fig


def fig_risk_distribution(preds: pd.DataFrame, metrics: dict) -> go.Figure:
    threshold = metrics["gbm"]["optimal_threshold"]

    delayed    = preds[preds["y_true"] == 1]["prob_gbm"]
    on_time    = preds[preds["y_true"] == 0]["prob_gbm"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=on_time, name="On-Time (actual)",
        marker_color=GREEN, opacity=0.65,
        xbins=dict(size=0.025),
        hovertemplate="P(delay)=%{x:.2f}, Count=%{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=delayed, name="Delayed (actual)",
        marker_color=RED, opacity=0.65,
        xbins=dict(size=0.025),
        hovertemplate="P(delay)=%{x:.2f}, Count=%{y}<extra></extra>",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="black",
        annotation_text=f"Threshold={threshold}",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title="Risk Score Distribution (Test Set, 2024)",
        xaxis_title="Predicted delay probability",
        yaxis_title="Count",
        template="plotly_white",
        legend=dict(x=0.55, y=0.95),
        margin=dict(t=60, b=50),
    )
    return fig


CAPTIONS = {
    "delay_over_time": (
        "Monthly delay rate across all routes from 2019 to 2024. "
        "The COVID disruption era (shaded) drove delay rates well above the baseline, "
        "peaking in 2021 when port congestion and equipment shortages compounded. "
        "Rates normalised through 2023 but the tail effects persisted into early 2024."
    ),
    "delay_by_route": (
        "Top 20 routes ranked by historical delay rate. "
        "Routes calling at Los Angeles, Long Beach or Shanghai consistently appear at the top "
        "due to chronic berth congestion and high vessel utilisation on these lanes. "
        "Routes with fewer than 50 historical shipments are excluded to avoid small-sample noise."
    ),
    "shap_importance": (
        "Mean absolute SHAP value per feature from the gradient-boosted model, computed on the 2024 test set. "
        "Higher values indicate features that shift the predicted delay probability further from the base rate. "
        "Rolling historical features dominate static route attributes, confirming that recent operational signals "
        "carry more predictive weight than long-run route characteristics."
    ),
    "pr_curve": (
        "Precision-recall curves for both models on the held-out 2024 test set. "
        "Stars mark the recall-optimised threshold for each model. "
        "The gradient-boosted model achieves a substantially higher PR-AUC, "
        "meaning it maintains better precision at every level of recall compared to logistic regression."
    ),
    "confusion_matrix": (
        "Confusion matrix for the gradient-boosted model at its chosen threshold. "
        "False negatives (bottom-left) are missed delays that reach port without a proactive alert. "
        "False positives (top-right) are unnecessary interventions, which cost less than missed delays. "
        "The threshold was chosen to keep false negatives low."
    ),
    "risk_distribution": (
        "Predicted delay probability histograms split by actual outcome. "
        "Good separation between the green (on-time) and red (delayed) distributions "
        "indicates the model assigns meaningfully higher risk scores to shipments that actually arrive late. "
        "The dashed vertical line shows the recall-optimised threshold."
    ),
}


def build_dashboard(df, preds, importance, metrics) -> str:
    """Assemble all figures into a single-page HTML dashboard."""
    figs = [
        ("delay_over_time", fig_delay_over_time(df)),
        ("delay_by_route",  fig_delay_by_route(df)),
        ("shap_importance", fig_shap_importance(importance)),
        ("pr_curve",        fig_pr_curve(metrics)),
        ("confusion_matrix", fig_confusion_matrix(metrics)),
        ("risk_distribution", fig_risk_distribution(preds, metrics)),
    ]

    divs = []
    for i, (key, fig) in enumerate(figs):
        chart_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if i == 0 else False,
            config={"displayModeBar": False},
        )
        caption = CAPTIONS[key]
        divs.append(f"""
        <div class="panel">
            {chart_html}
            <p class="caption">{caption}</p>
        </div>
        """)

    panels_html = "\n".join(divs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Container Freight Delay Risk</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #F8FAFC;
    color: #1E293B;
  }}
  header {{
    background: #1E3A5F;
    color: #fff;
    padding: 24px 32px;
  }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
  header p  {{ font-size: 0.9rem; color: #94A3B8; margin-top: 4px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
    gap: 20px;
    padding: 24px 32px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .panel {{
    background: #fff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }}
  .caption {{
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 12px;
    line-height: 1.6;
  }}
  footer {{
    text-align: center;
    padding: 24px;
    font-size: 0.8rem;
    color: #94A3B8;
  }}
</style>
</head>
<body>
<header>
  <h1>Container Freight Delay Risk: Predictive Modeling System</h1>
  <p>Machine learning pipeline for predicting shipment arrivals more than 24 hours late. Test set: 2024 held-out data.</p>
</header>
<div class="grid">
{panels_html}
</div>
<footer>
  Built by Siddhesh Badani &mdash; siddhesh.org/projects
</footer>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("Loading inputs...")
    df, preds, importance, metrics = load_inputs()

    print("Building dashboard...")
    html = build_dashboard(df, preds, importance, metrics)

    out_path = OUT_DIR / "dashboard.html"
    out_path.write_text(html)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
