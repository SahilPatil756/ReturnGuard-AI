from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    order_id: str = Field(default="ORD-MANUAL")
    product_category: str
    product_price: float = Field(gt=0)
    quantity: int = Field(ge=1, le=20)
    discount_pct: float = Field(ge=0, le=90)
    order_value: float | None = None
    shipping_zone: str
    promised_delivery_days: int = Field(ge=1, le=21)
    payment_method: str
    account_age_days: int = Field(ge=0)
    historical_orders: int = Field(ge=0)
    historical_returns: int = Field(ge=0)
    historical_return_rate: float | None = None
    prior_delivery_delay_rate: float = Field(ge=0, le=1, default=0.08)
    address_changes_90d: int = Field(ge=0, default=0)
    unique_addresses: int = Field(ge=1, default=1)
    orders_30d: int = Field(ge=0, default=0)
    refunds_30d: int = Field(ge=0, default=0)


class Reason(BaseModel):
    code: str
    detail: str


class ScoreResponse(BaseModel):
    order_id: str
    risk_probability: float
    risk_score: float
    risk_band: Literal["LOW", "REVIEW", "HIGH"]
    prediction: str
    recommended_action: str
    allowed_actions: list[str]
    reason_codes: list[Reason]
    model_version: str
    policy_version: str
    prediction_window_days: int
    disclaimer: str
    model_note: str


class ReviewDecision(BaseModel):
    order_id: str
    decision: Literal["approve_normally", "confirm_delivery_preference", "send_fit_reminder", "address_confirmation", "manual_review", "hold_for_review"]
    operator: str = "demo.operator"
    note: str = ""
    actual_label: int | None = None
