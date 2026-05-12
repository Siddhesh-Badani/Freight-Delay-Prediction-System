"""
Builds outputs/story.html: a scroll-based narrative page with five chapters.
Requires: outputs/metrics.json, outputs/shap_importance.csv,
          outputs/shap_summary.png, data/features.csv

Run: python src/build_story.py
"""

import base64
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs"

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#D97706"


def load_inputs():
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=["departure_date"])
    importance = pd.read_csv(OUT_DIR / "shap_importance.csv")
    with open(OUT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(OUT_DIR / "split_meta.json") as f:
        split_meta = json.load(f)
    return df, importance, metrics, split_meta


def embed_image(path: Path) -> str:
    """Return a base64-encoded <img> tag for embedding in HTML."""
    if not path.exists():
        return f'<div class="img-placeholder">Image not found: {path.name}</div>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" style="width:100%;border-radius:8px;" alt="{path.stem}">'


def chart_to_div(fig: go.Figure, include_js: bool = False) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False},
    )


def chapter1_fig(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.assign(period=df["departure_date"].dt.to_period("M"))
        .groupby("period", as_index=False)["delayed_24h"]
        .mean()
    )
    monthly["dt"] = monthly["period"].dt.to_timestamp()
    monthly["pct"] = monthly["delayed_24h"] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["dt"], y=monthly["pct"],
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.10)",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    ))
    fig.add_vrect(
        x0="2020-01-01", x1="2022-12-31",
        fillcolor="rgba(220,38,38,0.07)", line_width=0,
        annotation_text="COVID era", annotation_font_color=RED,
        annotation_position="top left",
    )
    fig.update_layout(
        title="Delay rate by month, all routes 2019-2024",
        xaxis_title="", yaxis_title="Shipments delayed > 24 h (%)",
        template="plotly_white", margin=dict(t=50, b=40),
    )
    return fig


def chapter2_fig(df: pd.DataFrame) -> go.Figure:
    carrier_delay = (
        df.groupby("carrier_name", as_index=False)["delayed_24h"]
        .agg(delay_rate="mean", n="count")
        .query("n >= 200")
        .sort_values("delay_rate", ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=carrier_delay["delay_rate"] * 100,
        y=carrier_delay["carrier_name"],
        orientation="h",
        marker_color=BLUE,
        text=carrier_delay["delay_rate"].apply(lambda v: f"{v:.1%}"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Delay rate by carrier (all years)",
        xaxis_title="Delay rate (%)", yaxis_title="",
        template="plotly_white", margin=dict(t=50, b=40, l=110),
    )
    return fig


def chapter3_fig(split_meta: dict) -> go.Figure:
    splits = [
        ("Train (2019-2022)", split_meta["train_size"], split_meta["train_delay_rate"]),
        ("Validation (2023)", split_meta["val_size"],   split_meta["val_delay_rate"]),
        ("Test (2024)",       split_meta["test_size"],  split_meta["test_delay_rate"]),
    ]
    labels  = [s[0] for s in splits]
    sizes   = [s[1] for s in splits]
    rates   = [s[2] * 100 for s in splits]
    colors  = [BLUE, ORANGE, RED]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Shipments",
        x=labels, y=sizes,
        marker_color=colors,
        yaxis="y",
        text=[f"{s:,}" for s in sizes],
        textposition="auto",
    ))
    fig.add_trace(go.Scatter(
        name="Delay rate %",
        x=labels, y=rates,
        mode="markers+text",
        marker=dict(color="black", size=12, symbol="diamond"),
        text=[f"{r:.1f}%" for r in rates],
        textposition="top center",
        yaxis="y2",
    ))
    fig.update_layout(
        title="Train / Validation / Test split",
        yaxis=dict(title="Shipment count"),
        yaxis2=dict(title="Delay rate (%)", overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(x=0.5, y=1.12, orientation="h"),
        margin=dict(t=70, b=40),
    )
    return fig


def chapter4_fig(metrics: dict) -> go.Figure:
    gbm = metrics["gbm"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gbm["rec_curve"], y=gbm["prec_curve"],
        mode="lines", name="GBM",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="Recall=%{x:.3f}, Precision=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=metrics["logreg"]["rec_curve"], y=metrics["logreg"]["prec_curve"],
        mode="lines", name="Logistic Regression",
        line=dict(color=ORANGE, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=[gbm["recall"]], y=[gbm["precision"]],
        mode="markers",
        marker=dict(color=RED, size=12, symbol="star"),
        name=f"Chosen threshold {gbm['optimal_threshold']}",
        hovertemplate=f"Threshold={gbm['optimal_threshold']}<br>Recall={gbm['recall']:.3f}, Precision={gbm['precision']:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Precision-Recall curve, GBM vs Logistic Regression",
        xaxis_title="Recall", yaxis_title="Precision",
        template="plotly_white",
        legend=dict(x=0.5, y=0.1),
        margin=dict(t=50, b=40),
    )
    return fig


def chapter5_fig(importance: pd.DataFrame) -> go.Figure:
    top = importance.head(12).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature"],
        orientation="h",
        marker_color=BLUE,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 12 delay drivers (mean |SHAP value|)",
        xaxis_title="Mean |SHAP|", yaxis_title="",
        template="plotly_white",
        margin=dict(t=50, b=40, l=200),
    )
    return fig


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Georgia, serif;
  background: #FAFAFA;
  color: #1A1A2E;
  line-height: 1.75;
}
.hero {
  background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
  color: #fff;
  padding: 72px 24px;
  text-align: center;
}
.hero h1 {
  font-size: clamp(1.8rem, 5vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.2;
  max-width: 700px;
  margin: 0 auto 16px;
}
.hero p {
  font-size: 1.05rem;
  opacity: 0.85;
  max-width: 560px;
  margin: 0 auto;
}
.chapter {
  max-width: 820px;
  margin: 0 auto;
  padding: 64px 24px;
  border-bottom: 1px solid #E8ECF0;
}
.chapter:last-of-type { border-bottom: none; }
.chapter-label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #2563EB;
  margin-bottom: 10px;
}
.chapter h2 {
  font-size: clamp(1.4rem, 3vw, 2rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 20px;
  line-height: 1.3;
}
.chapter p { font-size: 1rem; color: #334155; margin-bottom: 16px; }
.stat-box {
  background: #EFF6FF;
  border-left: 4px solid #2563EB;
  border-radius: 4px;
  padding: 16px 20px;
  margin: 24px 0;
}
.stat-box .stat-number {
  font-size: 2rem;
  font-weight: 800;
  color: #1D4ED8;
  line-height: 1;
}
.stat-box .stat-label {
  font-size: 0.88rem;
  color: #475569;
  margin-top: 4px;
}
.chart-wrap {
  margin: 32px 0;
  background: #fff;
  border-radius: 10px;
  padding: 16px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
footer {
  text-align: center;
  padding: 40px 24px;
  font-size: 0.82rem;
  color: #94A3B8;
  background: #F1F5F9;
}
footer a { color: #2563EB; text-decoration: none; }
"""


def build_story(df, importance, metrics, split_meta) -> str:
    gbm = metrics["gbm"]
    lr  = metrics["logreg"]
    delay_pct = round(df["delayed_24h"].mean() * 100, 1)
    n_routes = df["route_id"].nunique()

    # Build figures
    f1 = chapter1_fig(df)
    f2 = chapter2_fig(df)
    f3 = chapter3_fig(split_meta)
    f4 = chapter4_fig(metrics)
    f5 = chapter5_fig(importance)

    c1_div  = chart_to_div(f1, include_js=True)   # first chart loads plotly.js via CDN
    c2_div  = chart_to_div(f2)
    c3_div  = chart_to_div(f3)
    c4_div  = chart_to_div(f4)
    c5_div  = chart_to_div(f5)

    shap_img = embed_image(OUT_DIR / "shap_summary.png")

    top_feature = importance.iloc[0]["feature"] if len(importance) > 0 else "route_delay_rate_30d"
    top_feature_label = top_feature.replace("_", " ")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Container Freight Delay Risk: A Data Story</title>
<style>{CSS}</style>
</head>
<body>

<div class="hero">
  <h1>When a Container Does Not Arrive on Time</h1>
  <p>A machine learning story about predicting ocean freight delays, from raw port data to actionable risk scores.</p>
</div>

<!-- CHAPTER 1 -->
<section class="chapter">
  <div class="chapter-label">Chapter 1</div>
  <h2>The Cost of a Late Container</h2>

  <p>A single container arriving more than 24 hours late can cascade into production line shutdowns,
  missed retail deadlines and expensive spot-air freight orders to compensate. Ocean freight moves
  roughly 90% of global trade by volume. Even a modest delay rate of 20% across 800 million TEUs
  shipped annually represents an enormous economic drag, measured in billions of lost productivity
  and emergency logistics costs.</p>

  <p>Port congestion, weather events, vessel reliability and carrier scheduling discipline all
  contribute. No single factor dominates in isolation. What makes this problem tractable for machine
  learning is that the combination of these signals, observed before a shipment departs, carries
  real predictive power about whether that shipment will arrive late.</p>

  <p>This project builds a full prediction system: a synthetic but realistic ocean freight dataset,
  engineered features that mirror what is observable before departure, two trained models and
  a threshold tuned to catch delays rather than minimize overall error.</p>

  <div class="stat-box">
    <div class="stat-number">{delay_pct}%</div>
    <div class="stat-label">of shipments in this dataset arrived more than 24 hours late</div>
  </div>

  <div class="chart-wrap">
    {c1_div}
  </div>
</section>

<!-- CHAPTER 2 -->
<section class="chapter">
  <div class="chapter-label">Chapter 2</div>
  <h2>Building a Realistic Dataset</h2>

  <p>Because real ocean freight data is commercially sensitive and rarely public, this project
  generates 100,000 synthetic shipment records that mirror the statistical structure of actual
  operations. The generator uses known patterns from maritime research: the COVID disruption effect
  on 2020-2022 operations, Lunar New Year congestion on Asia-Pacific routes, Q4 peak-season
  volume spikes and the relationship between carrier reliability scores and on-time performance.</p>

  <p>Each record includes vessel utilisation, port congestion indexes at both origin and destination,
  weather severity, carrier identity and cargo characteristics. Delay probability is modelled as
  a function of these inputs with realistic interaction effects, not sampled uniformly. The result
  is a dataset where the model has genuine signal to learn from, not random noise dressed up as data.</p>

  <p>The 13 ports covered include the world's highest-throughput container terminals: Shanghai,
  Singapore, Ningbo, Rotterdam and Los Angeles among them. Routes between high-congestion port
  pairs receive an additional delay weight, reflecting chronic infrastructure constraints at
  those locations.</p>

  <div class="stat-box">
    <div class="stat-number">{n_routes}</div>
    <div class="stat-label">unique route pairs across 13 major global ports</div>
  </div>

  <div class="chart-wrap">
    {c2_div}
  </div>
</section>

<!-- CHAPTER 3 -->
<section class="chapter">
  <div class="chapter-label">Chapter 3</div>
  <h2>Time Is Everything</h2>

  <p>The most common way ML models fail on temporal data is data leakage: using information from
  the future to predict the past. A model trained this way will look impressive on paper and fail
  the moment it meets real operational data. This project prevents leakage at two levels.</p>

  <p>First, all rolling features, the 30-day route delay rate, 7-day destination congestion trend
  and 90-day vessel delay count, are computed using a left-closed rolling window. For any shipment
  departing on date D, the window covers [D minus window, D). The shipment itself is never in
  its own training signal. Second, the dataset is split strictly by year. The model trains on
  2019-2022 data, validates on 2023 and is evaluated only on 2024. No shuffling. No sampling across
  time boundaries.</p>

  <p>This mirrors how the system would actually be deployed. A model trained today would never
  have access to tomorrow's port congestion indexes. The temporal split enforces that discipline
  from the beginning.</p>

  <div class="stat-box">
    <div class="stat-number">{split_meta['train_size']:,}</div>
    <div class="stat-label">training shipments from 2019-2022, including the full COVID disruption period</div>
  </div>

  <div class="chart-wrap">
    {c3_div}
  </div>
</section>

<!-- CHAPTER 4 -->
<section class="chapter">
  <div class="chapter-label">Chapter 4</div>
  <h2>From Numbers to Decisions</h2>

  <p>A default classification threshold of 0.50 assumes that missing a delay and raising a false
  alarm carry equal cost. For port operations, they do not. A false alarm means an unnecessary
  buffer day booked; a missed delay means a customer gets no warning, ground logistics scramble
  and the shipment arrives unmanaged. The asymmetric cost justifies a lower threshold that
  biases toward catching delays.</p>

  <p>The threshold sweep finds the lowest threshold value that achieves recall of at least 80%
  while maximising precision. At the chosen threshold of {gbm['optimal_threshold']}, the
  gradient-boosted model catches {gbm['recall']:.0%} of actual delays with a precision of
  {gbm['precision']:.0%}. That means roughly {gbm['precision']:.0%} of shipments flagged for
  intervention actually arrive late, and {gbm['recall']:.0%} of all late arrivals are
  flagged in advance.</p>

  <p>The gradient-boosted model outperforms logistic regression on PR-AUC
  ({gbm['pr_auc']:.3f} vs {lr['pr_auc']:.3f}) because it captures the non-linear interaction
  effects between congestion, utilisation and carrier reliability that logistic regression
  models only approximately.</p>

  <div class="stat-box">
    <div class="stat-number">{gbm['recall']:.0%}</div>
    <div class="stat-label">recall at the chosen threshold: {gbm['recall']:.0%} of actual delays flagged before arrival</div>
  </div>

  <div class="chart-wrap">
    {c4_div}
  </div>
</section>

<!-- CHAPTER 5 -->
<section class="chapter">
  <div class="chapter-label">Chapter 5</div>
  <h2>Reading the Model's Mind</h2>

  <p>SHAP (SHapley Additive exPlanations) assigns each prediction a contribution from each feature,
  grounded in cooperative game theory. For a given shipment, SHAP shows exactly which signals
  pushed the predicted delay probability up and which pulled it down. This makes the model
  auditable and actionable.</p>

  <p>The top delay driver across the test set is <strong>{top_feature_label}</strong>. When a
  route has been consistently late in the prior 30 days, the model pushes probability significantly
  upward, regardless of the current vessel or weather conditions. This makes operational sense:
  chronic route problems reflect infrastructure or scheduling constraints that persist
  beyond individual voyages.</p>

  <p>Destination port congestion and carrier recent on-time rate rank close behind. Static features
  like transit distance and declared cargo value contribute, but their SHAP values are smaller
  on average. The model has learned that dynamic, real-time signals matter more than
  long-run route characteristics for predicting any specific shipment's outcome.</p>

  <div class="stat-box">
    <div class="stat-number">#{importance.iloc[0]['feature']}</div>
    <div class="stat-label">top SHAP feature: the single strongest predictor of delay risk</div>
  </div>

  <div class="chart-wrap">
    {c5_div}
  </div>

  <div class="chart-wrap">
    <p style="font-size:0.85rem;color:#475569;margin-bottom:12px;">
      SHAP beeswarm plot: each dot is one shipment in the 2024 test set. Red dots show high feature values,
      blue dots show low values. Position on the x-axis shows whether that feature increased or decreased
      the predicted delay probability.
    </p>
    {shap_img}
  </div>
</section>

<footer>
  Built by <a href="https://siddhesh.org/projects">Siddhesh Badani</a>.
  Synthetic dataset, gradient-boosted model with SHAP interpretation.
  Full code at siddhesh.org/projects.
</footer>

</body>
</html>"""
    return html


if __name__ == "__main__":
    print("Loading inputs...")
    df, importance, metrics, split_meta = load_inputs()

    print("Building story page...")
    html = build_story(df, importance, metrics, split_meta)

    out_path = OUT_DIR / "story.html"
    out_path.write_text(html)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
