"""
Synthetic ocean freight dataset generator.
Produces 100,000 container shipment records spanning 2019-2024.
Run: python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_RECORDS = 100_000

PORTS = [
    "Shanghai", "Singapore", "Ningbo", "Shenzhen", "Rotterdam",
    "Los Angeles", "Long Beach", "Hamburg", "Antwerp", "Busan",
    "Dubai", "New York", "Savannah",
]

# (carrier_name, base_reliability_score)
# Wide range intentionally: some carriers are excellent, some are poor.
# This creates the variance needed for the model to learn carrier quality.
CARRIERS = [
    ("Maersk",       87),
    ("MSC",          77),
    ("CMA CGM",      74),
    ("COSCO",        68),
    ("Hapag-Lloyd",  84),
    ("Evergreen",    60),
    ("Yang Ming",    50),   # structurally poor
    ("ONE",          71),
    ("HMM",          63),
    ("ZIM",          40),   # structurally poor
]

CONTAINER_TYPES = ["20ft", "40ft", "40ft HC", "reefer"]
CONTAINER_TYPE_PROBS = [0.30, 0.35, 0.25, 0.10]

HIGH_CONGESTION_PORTS = {"Los Angeles", "Long Beach", "Shanghai", "Shenzhen"}

# Approximate (lat, lon) for haversine distance calculation
PORT_COORDS = {
    "Shanghai":    (31.23,  121.47),
    "Singapore":   ( 1.29,  103.85),
    "Ningbo":      (29.87,  121.55),
    "Shenzhen":    (22.54,  114.06),
    "Rotterdam":   (51.92,    4.48),
    "Los Angeles": (33.74, -118.25),
    "Long Beach":  (33.77, -118.19),
    "Hamburg":     (53.55,   10.00),
    "Antwerp":     (51.23,    4.40),
    "Busan":       (35.10,  129.04),
    "Dubai":       (25.27,   55.33),
    "New York":    (40.66,  -74.04),
    "Savannah":    (32.08,  -81.10),
}


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    r_km = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    dist_km = r_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return dist_km * 0.539957


def generate_data(n: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- IDs ---
    shipment_ids = [f"SHP{i:07d}" for i in range(n)]
    # 500 unique vessels cycling through the fleet
    vessel_ids = [f"V{rng.integers(1, 501):04d}" for _ in range(n)]

    # --- Dates: 2019-01-01 through 2024-12-31 ---
    start = pd.Timestamp("2019-01-01")
    total_days = (pd.Timestamp("2024-12-31") - start).days
    dep_offsets = rng.integers(0, total_days + 1, size=n)
    departure_dates = pd.to_datetime([start + pd.Timedelta(days=int(d)) for d in dep_offsets])

    # --- Port pairs (origin != destination) ---
    n_ports = len(PORTS)
    orig_idx = rng.integers(0, n_ports, size=n)
    dest_idx = rng.integers(0, n_ports, size=n)
    same = orig_idx == dest_idx
    while same.any():
        dest_idx[same] = rng.integers(0, n_ports, size=int(same.sum()))
        same = orig_idx == dest_idx

    origins = [PORTS[i] for i in orig_idx]
    destinations = [PORTS[i] for i in dest_idx]
    route_ids = [f"{o[:3].upper()}_{d[:3].upper()}" for o, d in zip(origins, destinations)]

    # --- Cargo ---
    container_types = rng.choice(CONTAINER_TYPES, size=n, p=CONTAINER_TYPE_PROBS)
    container_counts = rng.integers(1, 51, size=n)
    cargo_weights = rng.uniform(10.0, 2000.0, size=n).round(2)
    declared_values = rng.uniform(5_000.0, 5_000_000.0, size=n).round(2)

    # --- Vessel specs ---
    capacity_choices = [2_000, 4_000, 8_000, 14_000, 20_000, 24_000]
    capacity_probs   = [0.10,  0.20,  0.25,   0.25,   0.15,   0.05]
    vessel_capacities = rng.choice(capacity_choices, size=n, p=capacity_probs)
    vessel_utilization = rng.uniform(40.0, 100.0, size=n)

    # --- Carriers ---
    carrier_indices = rng.integers(0, len(CARRIERS), size=n)
    carrier_names = [CARRIERS[i][0] for i in carrier_indices]
    base_reliabilities = np.array([CARRIERS[i][1] for i in carrier_indices], dtype=float)
    carrier_reliabilities = np.clip(
        base_reliabilities + rng.normal(0.0, 5.0, size=n), 30.0, 100.0
    ).round(1)

    # --- Port congestion (full 0-10 scale for realistic variance) ---
    origin_congestion = rng.uniform(0.0, 10.0, size=n)
    dest_congestion   = rng.uniform(0.0, 10.0, size=n)

    # --- Weather severity (0-10 Beaufort-inspired, full range) ---
    weather_severity = rng.uniform(0.0, 10.0, size=n)

    # --- Transit distance and scheduled days ---
    distances_nm = np.array([
        haversine_nm(*PORT_COORDS[o], *PORT_COORDS[d])
        for o, d in zip(origins, destinations)
    ]).round(0).astype(int)
    # Average vessel speed ~14 knots, plus ~1 day port overhead
    sched_days = np.maximum(2, np.round(distances_nm / (14.0 * 24.0) + 1.0)).astype(int)
    scheduled_arrivals = pd.to_datetime([
        dep + pd.Timedelta(days=int(d)) for dep, d in zip(departure_dates, sched_days)
    ])

    # --- Delay probability model ---
    years  = departure_dates.year.values
    months = departure_dates.month.values

    # --- Delay probability model (signed, centered effects) ---
    # Using signed additive effects centered on mean feature values.
    # Good conditions reduce p below the base rate; bad conditions increase it.
    # This creates wide variance in p while preserving the target mean rate.
    # Non-COVID mean ~19%, COVID mean ~28%, overall ~22%.

    covid      = (years >= 2020) & (years <= 2022)
    peak_covid = (years == 2021)

    # Base delay rate per era
    base = np.where(covid, 0.25, 0.15)
    base = np.where(peak_covid, 0.29, base).astype(float)
    delay_prob = base.copy()

    # --- Carrier reliability (dominant signal) ---
    # Mean carrier reliability ≈ 67.4 across the ten carriers.
    # Effect is signed: bad carriers (r<67) increase risk; good ones decrease it.
    r = carrier_reliabilities
    R_MEAN = 67.4
    delay_prob += 0.22 * (R_MEAN - r) / 50.0
    # r=40 → +0.120; r=67 → 0; r=87 → −0.088

    # --- Destination port congestion (signed around midpoint 5.0) ---
    dc = dest_congestion
    delay_prob += 0.15 * (dc - 5.0) / 10.0
    # dc=10 → +0.075; dc=5 → 0; dc=0 → −0.075

    # --- Weather severity (signed around midpoint 5.0) ---
    w = weather_severity
    delay_prob += 0.12 * (w - 5.0) / 10.0
    # w=10 → +0.060; w=5 → 0; w=0 → −0.060

    # --- Vessel utilization (signed around 70%) ---
    u = vessel_utilization
    delay_prob += 0.15 * (u - 70.0) / 30.0
    # u=100 → +0.150; u=70 → 0; u=40 → −0.150

    # --- Origin port congestion (signed, secondary effect) ---
    oc = origin_congestion
    delay_prob += 0.06 * (oc - 5.0) / 10.0
    # oc=10 → +0.030; oc=5 → 0; oc=0 → −0.030

    # --- High-congestion port pairs (always positive: structural constraint) ---
    high_cong_flag = np.array([
        1 if (o in HIGH_CONGESTION_PORTS or d in HIGH_CONGESTION_PORTS) else 0
        for o, d in zip(origins, destinations)
    ])
    delay_prob += high_cong_flag * 0.04
    delay_prob += high_cong_flag * covid * 0.04

    # --- Seasonal effects (always positive for affected windows) ---
    q4_peak    = months >= 10
    lny_window = months == 2
    pre_lny    = months == 1
    delay_prob[q4_peak]    += 0.03
    delay_prob[lny_window] += 0.05
    delay_prob[pre_lny]    += 0.02

    delay_prob = np.clip(delay_prob, 0.02, 0.90)

    # --- Sample delay outcome ---
    delayed_24h = rng.binomial(1, delay_prob).astype(int)

    # --- Actual arrival: delayed shipments get > 24h extra ---
    delay_hours = np.zeros(n, dtype=float)

    delayed_mask = delayed_24h == 1
    n_delayed = int(delayed_mask.sum())
    # Exponential delay distribution skewed toward 25-72 h, long tail
    raw_delay = rng.exponential(42.0, size=n_delayed) + 25.0
    # COVID era: longer tail
    raw_delay[covid[delayed_mask]] += rng.exponential(20.0, size=int(covid[delayed_mask].sum()))
    delay_hours[delayed_mask] = raw_delay

    # On-time shipments: small jitter, some arrive slightly early
    n_ontime = int((~delayed_mask).sum())
    delay_hours[~delayed_mask] = np.clip(rng.normal(0.0, 7.0, size=n_ontime), -36.0, 23.9)

    actual_arrivals = pd.to_datetime([
        sched + pd.Timedelta(hours=float(h))
        for sched, h in zip(scheduled_arrivals, delay_hours)
    ])

    # --- COVID-era port congestion boost (post-assignment, for realism) ---
    origin_congestion = np.array(origin_congestion)
    dest_congestion   = np.array(dest_congestion)
    origin_congestion[covid] = np.clip(
        origin_congestion[covid] + rng.uniform(0.5, 2.5, size=int(covid.sum())), 0.0, 10.0
    )
    dest_congestion[covid] = np.clip(
        dest_congestion[covid] + rng.uniform(1.0, 3.5, size=int(covid.sum())), 0.0, 10.0
    )

    df = pd.DataFrame({
        "shipment_id":             shipment_ids,
        "vessel_id":               vessel_ids,
        "route_id":                route_ids,
        "origin_port":             origins,
        "destination_port":        destinations,
        "departure_date":          departure_dates,
        "scheduled_arrival":       scheduled_arrivals,
        "actual_arrival":          actual_arrivals,
        "container_type":          container_types,
        "container_count":         container_counts,
        "cargo_weight_tons":       cargo_weights,
        "declared_value_usd":      declared_values,
        "vessel_capacity_teu":     vessel_capacities,
        "vessel_utilization_pct":  vessel_utilization.round(1),
        "origin_congestion_index": origin_congestion.round(2),
        "dest_congestion_index":   dest_congestion.round(2),
        "weather_severity_score":  weather_severity.round(2),
        "carrier_name":            carrier_names,
        "carrier_reliability_score": carrier_reliabilities,
        "transit_distance_nm":     distances_nm,
        "scheduled_transit_days":  sched_days,
        "delayed_24h":             delayed_24h,
    })

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("Generating synthetic ocean freight dataset...")
    df = generate_data()

    out_path = out_dir / "raw_shipments.csv"
    df.to_csv(out_path, index=False)

    delay_rate = df["delayed_24h"].mean()
    year_dist  = df.groupby(df["departure_date"].dt.year)["delayed_24h"].mean()

    print(f"\nRecords generated:          {len(df):,}")
    print(f"Delay rate (>24 h):         {delay_rate:.1%}")
    print(f"Date range:                 {df['departure_date'].min().date()} to {df['departure_date'].max().date()}")
    print(f"\nDelay rate by year:")
    for yr, rate in year_dist.items():
        print(f"  {yr}: {rate:.1%}")
    print(f"\nSaved: {out_path}")
