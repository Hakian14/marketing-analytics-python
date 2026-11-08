"""
Synthetic marketing datasets for the notebooks (seeded, reproducible).

    python data/generate_data.py      -> data/raw/*.csv

1. campaign_daily.csv      daily spend / impressions / clicks / conversions / revenue
                           per paid channel, with seasonality, diminishing returns
                           and ad-stock carry-over.  -> EDA + marketing-mix regression
2. ab_test_checkout.csv    user-level A/B test of a checkout redesign.
                           -> A/B test notebook
3. orders_promotions.csv   order-level data with promotion type, region, channel.
                           -> hypothesis-testing notebook
No real customer or company data is used.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 2025
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

CHANNELS = {
    # name: (daily spend mean VND, CPC VND, CTR, CVR at typical spend, AOV VND, elasticity, adstock decay)
    "Facebook Ads": (18_000_000, 4_000, 0.018, 0.021, 850_000, 0.70, 0.45),
    "Google Ads":   (14_000_000, 9_000, 0.045, 0.034, 1_050_000, 0.80, 0.25),
    "TikTok Ads":   (9_000_000, 3_000, 0.012, 0.012, 620_000, 0.60, 0.55),
    "Zalo OA":      (3_500_000, 3_500, 0.030, 0.028, 780_000, 0.65, 0.35),
    "Email":        (1_200_000, 4_000, 0.060, 0.041, 920_000, 0.50, 0.20),
    "Affiliate":    (5_000_000, 5_000, 0.025, 0.038, 700_000, 0.75, 0.15),
}
# share of conversions that are first-time customers (prospecting vs. retention channels)
NEW_SHARE = {"Facebook Ads": 0.62, "Google Ads": 0.57, "TikTok Ads": 0.74, "Zalo OA": 0.31, "Email": 0.06, "Affiliate": 0.49}
MONTH = {1: 1.30, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.10, 7: 0.95, 8: 0.95, 9: 1.00, 10: 1.05, 11: 1.45, 12: 1.30}


def gen_campaign_daily(rng) -> pd.DataFrame:
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    rows = []
    for ch, (spend_mu, cpc, ctr, cvr, aov, elas, decay) in CHANNELS.items():
        adstock = 0.0
        ref_clicks = spend_mu / cpc                               # clicks at a typical day's spend
        for d in days:
            season = MONTH[d.month] * (1.15 if d.weekday() >= 5 else 1.0)
            # marketers push spend in peak months; occasional campaign bursts
            spend = max(0.0, rng.normal(spend_mu * season, spend_mu * 0.25))
            if (d.month, d.day) in [(11, 11), (12, 12), (6, 6)]:
                spend *= 2.5
            if ch == "Email" and d.weekday() not in (1, 4):      # email sends Tue/Fri
                spend *= 0.3
            adstock = spend + decay * adstock
            clicks = int(adstock / cpc * rng.uniform(0.85, 1.15))
            impressions = int(clicks / (ctr * rng.uniform(0.9, 1.1))) if clicks else 0
            # diminishing returns: conversion rate falls as spend rises above the typical level
            eff_cvr = cvr * (max(clicks, 1) / ref_clicks) ** (elas - 1) * (1 + 0.15 * (MONTH[d.month] - 1))
            conversions = int(rng.binomial(clicks, min(eff_cvr, 1.0))) if clicks else 0
            revenue = int(sum(rng.lognormal(np.log(aov), 0.45, size=conversions))) if conversions else 0
            rows.append({
                "date": d.date().isoformat(), "channel": ch, "spend": int(round(spend / 1000) * 1000),
                "impressions": impressions, "clicks": clicks, "conversions": conversions, "revenue": revenue,
                "new_customers": int(rng.binomial(conversions, NEW_SHARE[ch])) if conversions else 0,
            })
    return pd.DataFrame(rows)


def gen_ab_test(rng, n=40_000) -> pd.DataFrame:
    user_type = rng.choice(["new", "returning"], size=n, p=[0.58, 0.42])
    device = rng.choice(["mobile", "desktop"], size=n, p=[0.72, 0.28])
    variant = rng.choice(["A", "B"], size=n, p=[0.5, 0.5])
    day = rng.integers(0, 21, size=n)
    # baseline conversion by segment; treatment lifts returning users clearly,
    # new users only slightly (the redesign removed the guest-checkout friction
    # mostly felt by returning customers who had saved addresses)
    base = np.where(user_type == "new", 0.026, 0.040) * np.where(device == "mobile", 0.9, 1.25)
    lift = np.where(variant == "B", np.where(user_type == "new", 1.04, 1.22), 1.0)
    # mild novelty effect in week 1 for variant B
    lift = lift * np.where((variant == "B") & (day < 7), 1.06, 1.0)
    p = np.clip(base * lift, 0, 1)
    converted = rng.random(n) < p
    order_value = np.where(converted, np.round(rng.lognormal(np.log(780_000), 0.5, size=n) / 1000) * 1000, 0).astype(int)
    # variant B nudges a slightly bigger basket via "add-on" suggestions
    order_value = np.where(converted & (variant == "B"), (order_value * 1.03).astype(int) // 1000 * 1000, order_value)
    start = date(2025, 9, 1)
    return pd.DataFrame({
        "user_id": [f"U{500000 + i}" for i in range(n)],
        "exposure_date": [(start + timedelta(days=int(x))).isoformat() for x in day],
        "variant": variant, "user_type": user_type, "device": device,
        "converted": converted.astype(int), "order_value": order_value,
        "sessions": rng.choice([1, 2, 3, 4], size=n, p=[0.6, 0.25, 0.1, 0.05]),
    })


def gen_orders_promotions(rng, n=15_000) -> pd.DataFrame:
    region = rng.choice(["North", "Central", "South"], size=n, p=[0.42, 0.18, 0.40])
    channel = rng.choice(["web", "mobile_app", "marketplace"], size=n, p=[0.38, 0.37, 0.25])
    promo = rng.choice(["none", "free_shipping", "10_percent_off"], size=n, p=[0.55, 0.25, 0.20])
    # AOV: free shipping lifts basket (people add items to clear the threshold);
    # 10% off lowers paid value; regions differ slightly
    base = 820_000 * np.where(region == "South", 1.06, np.where(region == "Central", 0.95, 1.0))
    mult = np.where(promo == "free_shipping", 1.12, np.where(promo == "10_percent_off", 0.96, 1.0))
    value = np.round(rng.lognormal(np.log(base * mult), 0.48, size=n) / 1000) * 1000
    items = np.clip(np.round(value / 350_000 + rng.normal(0, 0.6, size=n)), 1, 8).astype(int)
    # repeat purchase within 90 days: promo type matters a little, channel more
    p_rep = 0.30 + np.where(channel == "mobile_app", 0.08, np.where(channel == "marketplace", -0.06, 0.0)) \
        + np.where(promo == "free_shipping", 0.04, np.where(promo == "10_percent_off", -0.02, 0.0))
    repeat = rng.random(n) < p_rep
    delivery_days = np.clip(rng.normal(np.where(region == "Central", 5.2, 4.1), 1.4, size=n).round(), 1, 12).astype(int)
    csat = np.clip(np.round(rng.normal(4.3 - 0.18 * np.clip(delivery_days - 4, 0, None), 0.7, size=n)), 1, 5).astype(int)
    start = date(2025, 1, 1)
    return pd.DataFrame({
        "order_id": [f"O{700000 + i}" for i in range(n)],
        "order_date": [(start + timedelta(days=int(x))).isoformat() for x in rng.integers(0, 365, size=n)],
        "region": region, "channel": channel, "promo_type": promo,
        "order_value": value.astype(int), "items": items, "delivery_days": delivery_days,
        "csat_score": csat, "repeat_within_90d": repeat.astype(int),
    })


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    for name, df in [("campaign_daily", gen_campaign_daily(rng)), ("ab_test_checkout", gen_ab_test(rng)),
                     ("orders_promotions", gen_orders_promotions(rng))]:
        path = os.path.join(OUT, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name:20s} {len(df):>7,} rows -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
