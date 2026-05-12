"""
Feature engineering for container freight delay prediction.
All rolling and lag features use ONLY data available before a shipment's
departure_date. No future leakage.

Run: python src/features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def load_raw(path: Path = DATA_DIR / "raw_shipments.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["departure_date", "scheduled_arrival", "actual_arrival"])
    return df


def _rolling_stat_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: str,
    stat: str = "mean",
    min_periods: int = 3,
) -> pd.Series:
    """
    Compute a time-windowed rolling statistic per group.
    Uses closed='left' so each row excludes its own departure date.
    df must be sorted by departure_date with a DatetimeIndex set to departure_date.
    Returns a Series aligned to df's index.
    """
    def _roll(group):
        rolled = group.rolling(window, closed="left", min_periods=min_periods)
        if stat == "mean":
            return rolled.mean()
        if stat == "sum":
            return rolled.sum()
        if stat == "count":
            return rolled.count()
        raise ValueError(f"Unknown stat: {stat}")

    return df.groupby(group_col)[value_col].transform(_roll)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all features. Returns a new DataFrame with the target column included."""
    df = df.copy()
    df = df.sort_values("departure_date").reset_index(drop=True)

    # Set DatetimeIndex for time-aware rolling operations
    df_dt = df.set_index("departure_date")

    # --- Rolling route delay rate (30-day window) ---
    # "What fraction of shipments on this route were delayed in the past 30 days?"
    route_delay_30d = _rolling_stat_by_group(
        df_dt, "route_id", "delayed_24h", "30D", stat="mean", min_periods=3
    )
    df["route_delay_rate_30d"] = route_delay_30d.values
    # Fill cold-start NaN with the overall dataset mean (computed from training data later;
    # here we use a prior of 0.20 which sits in the realistic range for ocean freight).
    df["route_delay_rate_30d"] = df["route_delay_rate_30d"].fillna(0.20)

    # --- Rolling destination port congestion trend (7-day window) ---
    # Captures whether the destination port has been getting more or less congested recently.
    dest_cong_7d = _rolling_stat_by_group(
        df_dt, "destination_port", "dest_congestion_index", "7D", stat="mean", min_periods=2
    )
    df["dest_congestion_trend_7d"] = dest_cong_7d.values
    df["dest_congestion_trend_7d"] = df["dest_congestion_trend_7d"].fillna(df["dest_congestion_index"])

    # --- Rolling carrier on-time rate (30-day window) ---
    # Carrier's recent fraction of on-time deliveries (1 - delay_rate).
    carrier_ontime_30d = _rolling_stat_by_group(
        df_dt, "carrier_name", "delayed_24h", "30D", stat="mean", min_periods=5
    )
    df["carrier_ontime_rate_30d"] = (1 - carrier_ontime_30d).values
    df["carrier_ontime_rate_30d"] = df["carrier_ontime_rate_30d"].fillna(0.78)

    # --- Vessel delay count (90-day window) ---
    # Number of times this vessel has been delayed in the past 90 days.
    vessel_delays_90d = _rolling_stat_by_group(
        df_dt, "vessel_id", "delayed_24h", "90D", stat="sum", min_periods=1
    )
    df["vessel_delay_count_90d"] = vessel_delays_90d.values.clip(0)
    df["vessel_delay_count_90d"] = df["vessel_delay_count_90d"].fillna(0)

    # --- Temporal features ---
    df["month"]   = df["departure_date"].dt.month
    df["quarter"] = df["departure_date"].dt.quarter
    df["year"]    = df["departure_date"].dt.year
    df["day_of_week"] = df["departure_date"].dt.dayofweek

    # Lunar New Year window: Jan 20 - Feb 28 (approximate; shifts yearly but this is close enough)
    df["is_lunar_new_year"] = (
        ((df["month"] == 1) & (df["departure_date"].dt.day >= 20)) |
        (df["month"] == 2)
    ).astype(int)

    df["is_q4_peak"] = (df["month"] >= 10).astype(int)

    # COVID-era indicator (operationally meaningful; captures systemic disruption)
    df["is_covid_era"] = ((df["year"] >= 2020) & (df["year"] <= 2022)).astype(int)

    # --- Distance bucket ---
    # Thresholds in nautical miles: short (<3000), medium (3000-6000), long (6000-10000), transpacific (>10000)
    distance_bins  = [0, 3_000, 6_000, 10_000, np.inf]
    distance_labels = [0, 1, 2, 3]  # ordinal encoding
    df["distance_bucket"] = pd.cut(
        df["transit_distance_nm"],
        bins=distance_bins,
        labels=distance_labels,
        right=True,
    ).astype(int)

    # --- Interaction features ---
    # High-utilization vessels calling at congested ports are disproportionately at risk.
    df["util_x_dest_congestion"] = (
        df["vessel_utilization_pct"] * df["dest_congestion_index"] / 100.0
    )

    # Severe weather compounds existing route delay risk.
    df["weather_x_route_risk"] = (
        df["weather_severity_score"] * df["route_delay_rate_30d"]
    )

    # Congestion delta: how much worse is the destination vs origin right now?
    df["congestion_delta"] = df["dest_congestion_index"] - df["origin_congestion_index"]

    # Reefer flag (refrigerated containers require stricter handling, more delay-sensitive)
    df["is_reefer"] = (df["container_type"] == "reefer").astype(int)

    # High-utilization flag (>90%)
    df["is_high_utilization"] = (df["vessel_utilization_pct"] > 90.0).astype(int)

    # Low carrier reliability flag (<65)
    df["is_low_reliability_carrier"] = (df["carrier_reliability_score"] < 65.0).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode string categorical columns for model consumption."""
    df = df.copy()
    for col in ["carrier_name", "container_type", "origin_port", "destination_port"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes
    return df


FEATURE_COLS = [
    # Raw operational features
    "vessel_utilization_pct",
    "origin_congestion_index",
    "dest_congestion_index",
    "weather_severity_score",
    "carrier_reliability_score",
    "transit_distance_nm",
    "scheduled_transit_days",
    "vessel_capacity_teu",
    "container_count",
    "cargo_weight_tons",
    "declared_value_usd",
    # Rolling / lagged features
    "route_delay_rate_30d",
    "dest_congestion_trend_7d",
    "carrier_ontime_rate_30d",
    "vessel_delay_count_90d",
    # Temporal features
    "month",
    "quarter",
    "day_of_week",
    "is_lunar_new_year",
    "is_q4_peak",
    # Engineered features
    "distance_bucket",
    "util_x_dest_congestion",
    "weather_x_route_risk",
    "congestion_delta",
    "is_reefer",
    "is_high_utilization",
    "is_low_reliability_carrier",
    # Encoded categoricals
    "carrier_name_enc",
    "container_type_enc",
    "origin_port_enc",
    "destination_port_enc",
]

TARGET_COL = "delayed_24h"


if __name__ == "__main__":
    print("Loading raw shipments...")
    raw = load_raw()
    print(f"  {len(raw):,} records loaded")

    print("Engineering features...")
    df = build_features(raw)
    df = encode_categoricals(df)

    out_path = DATA_DIR / "features.csv"
    df.to_csv(out_path, index=False)

    print(f"\nFeature set dimensions: {df[FEATURE_COLS].shape}")
    print(f"Target balance: {df[TARGET_COL].mean():.1%} delayed")
    print(f"\nRolling feature coverage (non-null rate):")
    for col in ["route_delay_rate_30d", "dest_congestion_trend_7d",
                "carrier_ontime_rate_30d", "vessel_delay_count_90d"]:
        nn = df[col].notna().mean()
        print(f"  {col}: {nn:.1%} non-null")
    print(f"\nSaved: {out_path}")
