from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "order_time",
    "product_category",
    "product_price",
    "quantity",
    "discount_pct",
    "order_value",
    "shipping_zone",
    "promised_delivery_days",
    "payment_method",
    "account_age_days",
    "historical_orders",
    "historical_returns",
    "historical_return_rate",
    "prior_delivery_delay_rate",
    "address_changes_90d",
    "orders_30d",
    "refunds_30d",
]

LEAKAGE_CANDIDATES = [
    "refund_issued",
    "return_reason",
    "delivered_at",
    "post_delivery_status",
    "chargeback_flag",
    "csat_after_return",
]


@dataclass
class QualityReport:
    n_rows: int
    n_duplicates_order_id: int
    missing: dict
    invalid_counts: dict
    target_balance: dict
    leakage_columns_present: list
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_duplicates_order_id": self.n_duplicates_order_id,
            "missing": self.missing,
            "invalid_counts": self.invalid_counts,
            "target_balance": self.target_balance,
            "leakage_columns_present": self.leakage_columns_present,
            "warnings": self.warnings,
        }


def profile(df: pd.DataFrame, target: str = "problematic_return") -> QualityReport:
    warnings: list[str] = []
    missing = {c: int(df[c].isna().sum()) for c in df.columns if int(df[c].isna().sum()) > 0}
    invalid = {
        "negative_price": int((df["product_price"] < 0).sum()) if "product_price" in df else 0,
        "negative_qty": int((df["quantity"] < 0).sum()) if "quantity" in df else 0,
        "discount_out_of_range": int(((df["discount_pct"] < 0) | (df["discount_pct"] > 100)).sum())
        if "discount_pct" in df
        else 0,
        "return_rate_out_of_range": int(
            ((df["historical_return_rate"] < 0) | (df["historical_return_rate"] > 1)).sum()
        )
        if "historical_return_rate" in df
        else 0,
    }
    dup = int(df["order_id"].duplicated().sum()) if "order_id" in df else 0
    if dup:
        warnings.append("Duplicate order_id values detected; they are flagged, not silently dropped.")
    leakage = [c for c in LEAKAGE_CANDIDATES if c in df.columns]
    if leakage:
        warnings.append("Potential leakage columns present; they must not be used as decision-time features.")
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        warnings.append(f"Missing required columns: {missing_required}")
    balance = {}
    if target in df.columns:
        vc = df[target].value_counts(dropna=False)
        balance = {str(k): int(v) for k, v in vc.items()}
        pos = float(df[target].mean())
        if pos < 0.01:
            warnings.append("Positive class is extremely rare; accuracy will be misleading.")
    return QualityReport(
        n_rows=len(df),
        n_duplicates_order_id=dup,
        missing=missing,
        invalid_counts=invalid,
        target_balance=balance,
        leakage_columns_present=leakage,
        warnings=warnings,
    )
