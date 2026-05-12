# Container Freight Delay Risk: Predictive Modeling System

A production-style ML system that predicts whether a container shipment will arrive more than 24 hours late. Built as an end-to-end portfolio project covering synthetic data generation, feature engineering, model training, calibrated threshold selection, SHAP interpretation and interactive visualization.

---

## The Problem

Ocean freight carries roughly 90% of global trade by volume. Even a moderate delay rate of 20% across hundreds of millions of container movements per year represents enormous economic cost: production line shutdowns, missed retail windows and expensive emergency air freight substitutions.

The challenge is not simply whether a shipment will be late. It is knowing three to five days before arrival which specific shipments are at elevated risk, so port agents can pre-arrange berths, ground logistics can be notified and customers can make informed decisions. That early-warning window is what this system is designed to provide.

---

## Key Results

- **ROC-AUC of 0.71** on held-out 2024 test data for both models, meaningfully above random given a strict temporal split across a COVID-disruption training era.
- **Recall of 85%** at the chosen threshold: the gradient-boosted model catches 8.5 in 10 actual delays before port arrival.
- **Top delay driver: vessel utilization percentage.** High utilization at berth compounds congestion risk in a way that static carrier scores alone cannot predict. Carrier reliability score and 30-day carrier on-time rate both rank in the top five.
- **COVID-era distribution shift is real and documented**: training on 2019-2022 (28.7% delay rate) and testing on 2024 (19.2%) is a genuine challenge. The centered-effect data model and SHAP analysis show the model generalizes the operational relationships rather than memorizing the era.
- Threshold selected at recall >= 0.80 target (GBM: threshold 0.15, recall 0.85, precision 0.27). A missed delay costs more than a false alarm in port operations contexts.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data generation | Python, NumPy, pandas |
| Feature engineering | pandas rolling windows, temporal groupby |
| Modeling | scikit-learn (Logistic Regression), LightGBM |
| Interpretation | SHAP (TreeExplainer) |
| Visualization | Plotly, matplotlib |
| Serialization | joblib |
| Notebook | Jupyter |

---

## Project Structure

```
container-freight-risk/
├── README.md
├── requirements.txt
├── .gitignore
├── run_pipeline.py              Full pipeline runner
├── data/
│   ├── raw_shipments.csv        Auto-generated
│   └── features.csv             Auto-generated
├── models/
│   ├── logreg.joblib            Auto-generated
│   └── gbm.joblib               Auto-generated
├── src/
│   ├── __init__.py
│   ├── generate_data.py         Synthetic data generator
│   ├── features.py              Feature engineering
│   ├── train.py                 Model training
│   ├── evaluate.py              Threshold optimization and metrics
│   ├── interpret.py             SHAP analysis
│   ├── build_dashboard.py       Interactive Plotly dashboard
│   └── build_story.py           Scroll narrative page
├── outputs/
│   ├── dashboard.html           Auto-generated
│   ├── story.html               Auto-generated
│   ├── shap_summary.png         Auto-generated
│   ├── shap_bar.png             Auto-generated
│   ├── evaluation_report.md     Auto-generated
│   └── recommendations.md       Auto-generated
└── notebooks/
    └── eda.ipynb                Exploratory analysis
```

---

## How to Run

### Setup

```bash
cd container-freight-risk
pip install -r requirements.txt
```

### Full pipeline (recommended)

```bash
python run_pipeline.py
```

This runs all seven steps in order and prints a summary of results and artifact paths when complete.

### Individual steps

Run from the project root, using the `src/` directory as the working directory:

```bash
# 1. Generate 100,000 synthetic shipments
python src/generate_data.py

# 2. Engineer features (rolling windows, interactions, flags)
python src/features.py

# 3. Train Logistic Regression and GBM with temporal split
python src/train.py

# 4. Optimize threshold and write evaluation report
python src/evaluate.py

# 5. Compute SHAP values and write recommendations
python src/interpret.py

# 6. Build interactive dashboard
python src/build_dashboard.py

# 7. Build scroll narrative story
python src/build_story.py
```

### View outputs

- **Dashboard**: open `outputs/dashboard.html` in any browser
- **Story**: open `outputs/story.html` in any browser
- **Evaluation report**: read `outputs/evaluation_report.md`
- **Business recommendations**: read `outputs/recommendations.md`
- **SHAP plots**: open `outputs/shap_summary.png` and `outputs/shap_bar.png`

### EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## Methodology Notes

**Synthetic data generation.** The generator injects realistic delay signal: COVID-era (2020-2022) port congestion spikes, carrier reliability variance, Q4 peak season effects, Lunar New Year disruption windows and weather severity. The positive class rate targets 18-25%, consistent with published ocean freight delay statistics.

**Temporal feature engineering.** Every rolling and lag feature uses `pandas.Series.rolling()` with `closed='left'`, meaning the window [D-W, D) for a shipment departing on date D. The current shipment is never in its own training signal. NaN values from cold-start records are filled with population-level priors (0.20 for delay rate, 0.78 for carrier on-time rate) rather than forward-filled or interpolated, preventing any look-ahead.

**Strict temporal split.** Train set: 2019-2022. Validation set: 2023 (used for GBM early stopping). Test set: 2024 only. No random shuffling across time. This is the evaluation setup closest to production deployment.

**Recall-optimized threshold.** Default 0.50 is wrong for this problem. The threshold sweep finds the lowest value achieving recall >= 0.80 while maximizing precision. This is declared up front rather than tuned post-hoc on the test set.

**SHAP interpretation.** TreeExplainer is used on the GBM model with a background sample from the training set. SHAP values are computed on a 1000-record sample from the 2024 test set. Waterfall plots for a true positive, false positive and true negative illustrate how individual shipment characteristics combine into a final prediction.

---

## Limitations

- **Synthetic data.** The generator captures known delay patterns but cannot reproduce all idiosyncrasies of real carrier contracts, port labor agreements or vessel technical incidents.
- **No real-time inference layer.** The pipeline produces batch predictions. Production deployment would require a serving API, a streaming congestion index feed and a daily retraining schedule.
- **Static port congestion index.** In practice, port congestion is dynamic within a day. This model uses a shipment-level congestion value rather than an intraday time series.
- **13-port coverage.** The real ocean freight network has hundreds of ports. Expanding the port set would require re-calibrating the delay probability model in the generator.
- **No multi-leg routing.** Transshipment voyages with intermediate port calls introduce compounding delay risk not captured here.

---

## Author

Siddhesh Badani. [siddhesh.org/projects](https://siddhesh.org/projects)
