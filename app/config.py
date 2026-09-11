from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FRONTEND_DIR = ROOT / "frontend"

MODEL_VERSION = "rrs-1.4.0"
POLICY_VERSION = "policy-1.2"
DATASET_VERSION = "synthetic-return-risk-v1"
RANDOM_SEED = 42
PREDICTION_WINDOW_DAYS = 30
TIMEZONE = "Asia/Kolkata"

# Cost assumptions (INR). Documented in reports/cost_assumptions.md
FP_COST = 120  # unnecessary review + customer friction
INTERVENTION_COST = 80  # benign confirmation / fit reminder
FN_COST_FIXED = 220  # reverse logistics overhead
FN_COST_MARGIN = 0.28  # lost margin share of order value on missed costly return
TP_SAVE_RATE = 0.55  # share of return cost avoided via timely intervention

CATEGORIES = ["apparel", "electronics", "home", "beauty", "sports", "accessories"]
ZONES = ["Z1", "Z2", "Z3", "Z4"]
PAYMENTS = ["upi", "card", "cod", "wallet", "netbanking"]
ALLOWED_ACTIONS = [
    "confirm_delivery_preference",
    "send_fit_reminder",
    "address_confirmation",
    "manual_review",
    "approve_normally",
]

PROHIBITED_FIELDS = {
    "name",
    "email",
    "phone",
    "exact_address",
    "card_number",
    "cvv",
    "password",
    "gender",
    "religion",
    "caste",
    "ethnicity",
    "age",
    "post_delivery_status",
    "return_reason_text",
    "refund_issued",
}
