from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "product_price",
    "quantity",
    "discount_pct",
    "order_value",
    "promised_delivery_days",
    "account_age_days",
    "historical_orders",
    "historical_returns",
    "historical_return_rate",
    "prior_delivery_delay_rate",
    "address_changes_90d",
    "orders_30d",
    "refunds_30d",
    "return_rate",
    "refund_intensity_30d",
    "high_value_flag",
    "new_account_flag",
    "thin_history_flag",
    "deep_discount_flag",
    "log_order_value",
]

CATEGORICAL_FEATURES = [
    "product_category",
    "shipping_zone",
    "payment_method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    orders = out["historical_orders"].clip(lower=0)
    out["return_rate"] = np.where(orders > 0, out["historical_returns"] / orders, 0.0)
    out["refund_intensity_30d"] = np.where(
        out["orders_30d"] > 0, out["refunds_30d"] / out["orders_30d"].clip(lower=1), 0.0
    )
    out["high_value_flag"] = (out["order_value"] >= 8000).astype(int)
    out["new_account_flag"] = (out["account_age_days"] < 21).astype(int)
    out["thin_history_flag"] = (out["historical_orders"] <= 1).astype(int)
    out["deep_discount_flag"] = (out["discount_pct"] >= 30).astype(int)
    out["log_order_value"] = np.log1p(out["order_value"].clip(lower=0))
    out["product_category"] = out["product_category"].fillna("unknown").astype(str)
    out["shipping_zone"] = out["shipping_zone"].fillna("unknown").astype(str)
    out["payment_method"] = out["payment_method"].fillna("unknown").astype(str)
    return out


def temporal_split(df: pd.DataFrame, train=0.70, valid=0.15):
    d = df.sort_values("order_time").reset_index(drop=True)
    n = len(d)
    i_train = int(n * train)
    i_valid = int(n * (train + valid))
    return d.iloc[:i_train].copy(), d.iloc[i_train:i_valid].copy(), d.iloc[i_valid:].copy()
