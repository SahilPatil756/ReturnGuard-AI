# Data dictionary

All fields below are available at **decision time** except the two label columns, which are outcomes after the 30-day window.

| Field | Type | Role |
| --- | --- | --- |
| order_id | string | Pseudonymous key |
| order_time | datetime | Temporal split; Asia/Kolkata |
| product_category | enum | Feature |
| product_price | float INR | Feature |
| quantity | int | Feature |
| discount_pct | float 0–75 | Feature |
| order_value | float INR | Feature |
| shipping_zone | Z1–Z4 | Feature |
| promised_delivery_days | int | Feature |
| payment_method | category | Feature (not PAN/credentials) |
| account_age_days | int | Feature |
| historical_orders | int | Pre-order history |
| historical_returns | int | Pre-order history |
| historical_return_rate | 0–1 | Pre-order history |
| prior_delivery_delay_rate | 0–1 | Pre-order history |
| address_changes_90d | int | Feature |
| unique_addresses | int | Feature |
| orders_30d | int | Feature |
| refunds_30d | int | Feature |
| returned_within_window | 0/1 | Outcome (not a live feature) |
| problematic_return | 0/1 | **Modeling target** |

## Engineered (still decision-time)

`return_rate`, `refund_intensity_30d`, `high_value_flag`, `new_account_flag`, `thin_history_flag`, `deep_discount_flag`, `log_order_value`.
