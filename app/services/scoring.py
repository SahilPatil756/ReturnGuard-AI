from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import ALLOWED_ACTIONS, MODEL_VERSION, POLICY_VERSION, PREDICTION_WINDOW_DAYS
from ml.features.pipeline import engineer
from ml.models.explain import explain


class ModelUnavailable(RuntimeError):
    pass


def band_for(prob: float, high_cut: float, review_cut: float | None = None) -> str:
    review_cut = review_cut if review_cut is not None else max(0.25, high_cut - 0.20)
    if prob >= high_cut:
        return "HIGH"
    if prob >= review_cut:
        return "REVIEW"
    return "LOW"


def policy_for(band: str) -> dict[str, Any]:
    if band == "HIGH":
        return {
            "recommended_action": "manual_review",
            "allowed_actions": ["manual_review", "address_confirmation", "confirm_delivery_preference"],
        }
    if band == "REVIEW":
        return {
            "recommended_action": "send_fit_reminder",
            "allowed_actions": ["send_fit_reminder", "confirm_delivery_preference", "address_confirmation", "approve_normally"],
        }
    return {
        "recommended_action": "approve_normally",
        "allowed_actions": ["approve_normally", "send_fit_reminder"],
    }


def request_to_frame(payload: dict) -> pd.DataFrame:
    row = dict(payload)
    if row.get("order_value") is None:
        row["order_value"] = float(row["product_price"]) * int(row["quantity"]) * (1 - float(row["discount_pct"]) / 100)
    if row.get("historical_return_rate") is None:
        ho = int(row.get("historical_orders") or 0)
        hr = int(row.get("historical_returns") or 0)
        row["historical_return_rate"] = (hr / ho) if ho else 0.0
    row.setdefault("unique_addresses", 1)
    return pd.DataFrame([row])


def score_frame(bundle: dict, df: pd.DataFrame) -> dict:
    if bundle is None:
        raise ModelUnavailable("Model artifact is not loaded")
    feat = engineer(df)
    cols = bundle["feature_columns"]
    proba = float(bundle["model"].predict_proba(feat[cols])[0, 1])
    threshold = float(bundle["threshold"])
    band = band_for(proba, threshold)
    policy = policy_for(band)
    reasons = explain(feat.iloc[0])
    pred = "potential_costly_return" if band != "LOW" else "routine_fulfillment"
    return {
        "order_id": str(df.iloc[0].get("order_id", "ORD-MANUAL")),
        "risk_probability": round(proba, 4),
        "risk_score": round(proba * 100, 1),
        "risk_band": band,
        "prediction": pred,
        "recommended_action": policy["recommended_action"],
        "allowed_actions": policy["allowed_actions"],
        "reason_codes": reasons,
        "model_version": bundle.get("model_version", MODEL_VERSION),
        "policy_version": bundle.get("policy_version", POLICY_VERSION),
        "prediction_window_days": PREDICTION_WINDOW_DAYS,
        "disclaimer": "This is a model signal, not proof of abuse or customer intent.",
        "model_note": "Risk is probabilistic and can be wrong. Operators make the final decision.",
        "allowed_action_catalog": ALLOWED_ACTIONS,
    }
