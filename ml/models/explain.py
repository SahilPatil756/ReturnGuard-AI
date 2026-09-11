from __future__ import annotations

from ml.models.train import reason_code_rules

REASON_COPY = {
    "HISTORICAL_RETURN_RATE_HIGH": "Prior return rate is high relative to typical customers.",
    "DISCOUNT_HIGH": "Discount is large versus category norms, which often lifts return probability.",
    "CATEGORY_RETURN_RATE_HIGH": "This category historically sees more fit-related returns.",
    "ZONE_RETURN_RATE_ELEVATED": "Delivery zone has elevated historical return and delay rates.",
    "NEW_ACCOUNT": "Account is new, so there is little reliable purchase history.",
    "HIGH_VALUE_ORDER": "Order value is high, so a return would be costly.",
    "ADDRESS_CHURN": "Shipping address has changed repeatedly in 90 days.",
    "RECENT_REFUND_ACTIVITY": "Several refunds already occurred in the last 30 days.",
    "THIN_PURCHASE_HISTORY": "Very few prior orders, so behavior is harder to trust statistically.",
    "COMBINED_BEHAVIORAL_SIGNALS": "No single extreme signal; combined features raise estimated risk.",
}


def explain(row) -> list[dict]:
    codes = reason_code_rules(row)
    return [{"code": c, "detail": REASON_COPY.get(c, c)} for c in codes]
