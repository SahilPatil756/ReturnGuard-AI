"""Generate a documented SYNTHETIC order dataset. Not real merchant data."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.config import (
    CATEGORIES,
    DATA_RAW,
    DATASET_VERSION,
    PAYMENTS,
    RANDOM_SEED,
    ROOT,
    TIMEZONE,
    ZONES,
)

N_ORDERS = 12000
START = datetime(2024, 6, 1)


def generate(n: int = N_ORDERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order_times = START + pd.to_timedelta(rng.integers(0, 400, size=n), unit="D")
    order_times = order_times + pd.to_timedelta(rng.integers(0, 24 * 60, size=n), unit="m")

    category = rng.choice(
        CATEGORIES,
        size=n,
        p=[0.34, 0.18, 0.16, 0.12, 0.10, 0.10],
    )
    zone = rng.choice(ZONES, size=n, p=[0.28, 0.32, 0.25, 0.15])
    payment = rng.choice(PAYMENTS, size=n, p=[0.42, 0.22, 0.18, 0.10, 0.08])

    account_age_days = np.clip(rng.lognormal(mean=4.6, sigma=1.15, size=n).astype(int), 1, 2400)
    historical_orders = np.clip(rng.poisson(lam=np.clip(account_age_days / 45, 0.3, 40), size=n), 0, 80)
    # Prior return rate known before this order (customer history only)
    base_return = rng.beta(1.4, 8.5, size=n)
    serial = rng.random(n) < 0.07
    base_return = np.where(serial, np.clip(base_return + rng.uniform(0.35, 0.7, n), 0, 0.95), base_return)
    historical_returns = np.minimum(historical_orders, np.round(historical_orders * base_return).astype(int))
    historical_return_rate = np.where(historical_orders > 0, historical_returns / historical_orders, 0.0)

    cat_price = {
        "apparel": (899, 0.55),
        "electronics": (12999, 0.65),
        "home": (2499, 0.5),
        "beauty": (799, 0.45),
        "sports": (1899, 0.5),
        "accessories": (699, 0.5),
    }
    price_mean = np.array([cat_price[c][0] for c in category])
    price_sigma = np.array([cat_price[c][1] for c in category])
    product_price = np.round(np.exp(np.log(price_mean) + rng.normal(0, price_sigma)), -1)
    product_price = np.clip(product_price, 199, 89999)
    quantity = rng.choice([1, 2, 3, 4], size=n, p=[0.72, 0.18, 0.07, 0.03])
    discount_pct = np.clip(rng.beta(1.6, 4.2, size=n) * 70, 0, 70)
    # flash-sale spike
    flash = rng.random(n) < 0.08
    discount_pct = np.where(flash, np.clip(discount_pct + rng.uniform(15, 35, n), 0, 75), discount_pct)
    order_value = np.round(product_price * quantity * (1 - discount_pct / 100), 2)

    promised_delivery_days = rng.choice([1, 2, 3, 5, 7, 10], size=n, p=[0.08, 0.18, 0.32, 0.22, 0.14, 0.06])
    promised_delivery_days = np.where(zone == "Z4", promised_delivery_days + 2, promised_delivery_days)
    prior_delay_rate = np.clip(rng.beta(1.2, 6.0, size=n) + (zone == "Z4") * 0.08, 0, 0.8)
    address_changes_90d = rng.poisson(0.25, size=n)
    address_changes_90d = np.where(rng.random(n) < 0.04, address_changes_90d + rng.integers(2, 5, n), address_changes_90d)
    unique_addresses = np.clip(1 + address_changes_90d + rng.integers(0, 2, n), 1, 8)
    orders_30d = np.clip(rng.poisson(np.clip(historical_orders / 8, 0.2, 6), size=n), 0, 20)
    refunds_30d = np.minimum(orders_30d, rng.binomial(np.maximum(orders_30d, 1), np.clip(historical_return_rate, 0.02, 0.8)))

    # Latent return probability (decision-time signals only)
    cat_lift = {
        "apparel": 0.22,
        "electronics": 0.08,
        "home": 0.10,
        "beauty": 0.14,
        "sports": 0.11,
        "accessories": 0.16,
    }
    logit = (
        -2.55
        + 2.4 * historical_return_rate
        + 0.018 * discount_pct
        + np.array([cat_lift[c] for c in category]) * 3.1
        + 0.000012 * order_value
        + 0.35 * (account_age_days < 21).astype(float)
        + 0.28 * (historical_orders <= 1).astype(float)
        + 0.22 * np.clip(address_changes_90d / 3, 0, 1.5)
        + 0.18 * (zone == "Z4").astype(float)
        + 0.12 * prior_delay_rate
        + 0.15 * (payment == "cod").astype(float)
        + 0.20 * np.clip(refunds_30d / 3, 0, 1.2)
        + 0.08 * (promised_delivery_days >= 7).astype(float)
        + rng.normal(0, 0.45, n)
    )
    p_return = 1 / (1 + np.exp(-logit))
    returned_within_window = rng.random(n) < p_return

    # Problematic/costly return vs ordinary return (size exchange, change of mind)
    abuse_logit = (
        -1.1
        + 3.2 * historical_return_rate
        + 0.9 * (account_age_days < 14).astype(float)
        + 0.000018 * order_value
        + 0.55 * np.clip(address_changes_90d / 2, 0, 2)
        + 0.4 * np.clip(refunds_30d / 2, 0, 2)
        + 0.35 * (category == "electronics").astype(float)
        + 0.25 * flash.astype(float)
    )
    p_problem_given_return = 1 / (1 + np.exp(-abuse_logit))
    problematic_return = returned_within_window & (rng.random(n) < p_problem_given_return)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{100000 + i}" for i in range(n)],
            "order_time": order_times,
            "timezone": TIMEZONE,
            "product_category": category,
            "product_price": product_price,
            "quantity": quantity,
            "discount_pct": np.round(discount_pct, 2),
            "order_value": order_value,
            "shipping_zone": zone,
            "promised_delivery_days": promised_delivery_days,
            "payment_method": payment,
            "account_age_days": account_age_days,
            "historical_orders": historical_orders,
            "historical_returns": historical_returns,
            "historical_return_rate": np.round(historical_return_rate, 4),
            "prior_delivery_delay_rate": np.round(prior_delay_rate, 4),
            "address_changes_90d": address_changes_90d,
            "unique_addresses": unique_addresses,
            "orders_30d": orders_30d,
            "refunds_30d": refunds_30d,
            "returned_within_window": returned_within_window.astype(int),
            "problematic_return": problematic_return.astype(int),
        }
    )
    df = df.sort_values("order_time").reset_index(drop=True)
    df["order_id"] = [f"ORD-{100000 + i}" for i in range(len(df))]
    return df


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df = generate()
    path = DATA_RAW / "orders_synthetic.csv"
    df.to_csv(path, index=False)
    meta = {
        "dataset_version": DATASET_VERSION,
        "synthetic": True,
        "disclaimer": "This is synthetic data generated for demo/evaluation. It is not real merchant data.",
        "n_rows": int(len(df)),
        "target": "problematic_return",
        "target_definition": (
            "1 if the order was returned within 30 days AND the return was loss-driving "
            "(serial/high-cost/abuse-like pattern), else 0. Ordinary returns are labeled 0."
        ),
        "observation_window_days": 30,
        "timezone": TIMEZONE,
        "date_min": str(df["order_time"].min()),
        "date_max": str(df["order_time"].max()),
        "positive_rate": float(df["problematic_return"].mean()),
        "ordinary_return_rate": float(df["returned_within_window"].mean()),
    }
    (DATA_RAW / "dataset_card.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {path} rows={len(df)} positive_rate={meta['positive_rate']:.3f}")


if __name__ == "__main__":
    main()
