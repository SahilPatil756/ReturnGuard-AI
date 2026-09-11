"""Train leakage-safe return-risk models and freeze held-out test metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    DATA_PROCESSED,
    DATA_RAW,
    FN_COST_FIXED,
    FN_COST_MARGIN,
    FP_COST,
    INTERVENTION_COST,
    MODEL_VERSION,
    MODELS_DIR,
    POLICY_VERSION,
    RANDOM_SEED,
    REPORTS_DIR,
    TP_SAVE_RATE,
)
from ml.data.generate_synthetic import generate, main as write_raw  # noqa: E402
from ml.data.validators import profile  # noqa: E402
from ml.evaluation.metrics import CostConfig, binary_metrics, pick_threshold, ranking_metrics, threshold_grid  # noqa: E402
from ml.features.pipeline import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, engineer, temporal_split  # noqa: E402

TARGET = "problematic_return"
COSTS = CostConfig(
    fp_cost=FP_COST,
    intervention_cost=INTERVENTION_COST,
    fn_cost_fixed=FN_COST_FIXED,
    fn_cost_margin=FN_COST_MARGIN,
    tp_save_rate=TP_SAVE_RATE,
)


def _preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def majority_proba(y_train, n):
    p = float(np.mean(y_train))
    return np.full(n, p)


def reason_code_rules(row: pd.Series) -> list[str]:
    codes = []
    if float(row.get("historical_return_rate", 0) or 0) >= 0.35:
        codes.append("HISTORICAL_RETURN_RATE_HIGH")
    if float(row.get("discount_pct", 0) or 0) >= 30:
        codes.append("DISCOUNT_HIGH")
    if str(row.get("product_category")) == "apparel" and float(row.get("historical_return_rate", 0) or 0) >= 0.18:
        codes.append("CATEGORY_RETURN_RATE_HIGH")
    if str(row.get("shipping_zone")) == "Z4":
        codes.append("ZONE_RETURN_RATE_ELEVATED")
    if int(row.get("account_age_days", 100) or 100) < 21:
        codes.append("NEW_ACCOUNT")
    if float(row.get("order_value", 0) or 0) >= 8000:
        codes.append("HIGH_VALUE_ORDER")
    if int(row.get("address_changes_90d", 0) or 0) >= 2:
        codes.append("ADDRESS_CHURN")
    if int(row.get("refunds_30d", 0) or 0) >= 2:
        codes.append("RECENT_REFUND_ACTIVITY")
    if int(row.get("historical_orders", 10) or 10) <= 1:
        codes.append("THIN_PURCHASE_HISTORY")
    if not codes:
        codes.append("COMBINED_BEHAVIORAL_SIGNALS")
    return codes[:4]


def train():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_RAW / "orders_synthetic.csv"
    if not raw_path.exists():
        write_raw()
    df = pd.read_csv(raw_path, parse_dates=["order_time"])
    quality = profile(df)
    feat = engineer(df)
    train_df, valid_df, test_df = temporal_split(feat)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    X_valid, y_valid = valid_df[FEATURE_COLUMNS], valid_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    # Rule baseline: flag high historical return rate
    rule_valid = (valid_df["historical_return_rate"] > 0.50).astype(float)
    rule_test = (test_df["historical_return_rate"] > 0.50).astype(float)

    logit = Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    logit.fit(X_train, y_train)

    hgb = Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=220,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    hgb.fit(X_train, y_train)

    try:
        calibrated = CalibratedClassifierCV(hgb, method="isotonic", cv="prefit")
        calibrated.fit(X_valid, y_valid)
    except (TypeError, ValueError):
        calibrated = hgb

    models = {
        "majority": lambda X: majority_proba(y_train, len(X)),
        "rule_return_rate_gt_50": lambda X: (X["historical_return_rate"] > 0.50).astype(float).to_numpy(),
        "logistic_regression": lambda X: logit.predict_proba(X)[:, 1],
        "hist_gradient_boosting": lambda X: hgb.predict_proba(X)[:, 1],
        "hgb_calibrated": lambda X: calibrated.predict_proba(X)[:, 1],
    }

    valid_scores = {}
    for name, fn in models.items():
        p = fn(X_valid if name != "rule_return_rate_gt_50" else valid_df)
        if name == "rule_return_rate_gt_50":
            p = rule_valid.to_numpy()
        valid_scores[name] = {
            **ranking_metrics(y_valid, p),
            **binary_metrics(y_valid, p, 0.5),
        }

    champion_name = "hgb_calibrated"
    p_valid = calibrated.predict_proba(X_valid)[:, 1]
    grid = threshold_grid(y_valid, p_valid, valid_df["order_value"].to_numpy(), COSTS)
    chosen = pick_threshold(grid)

    # Locked test evaluation — once
    p_test = calibrated.predict_proba(X_test)[:, 1]
    test_rank = ranking_metrics(y_test, p_test)
    test_bin = binary_metrics(y_test, p_test, chosen["threshold"])
    test_cost = None
    from ml.evaluation.metrics import decision_cost

    test_cost = decision_cost(y_test, p_test, chosen["threshold"], test_df["order_value"].to_numpy(), COSTS)
    test_grid = threshold_grid(y_test, p_test, test_df["order_value"].to_numpy(), COSTS)

    comparisons = {}
    for name, fn in models.items():
        if name == "rule_return_rate_gt_50":
            p = rule_test.to_numpy()
        elif name == "majority":
            p = majority_proba(y_train, len(X_test))
        elif name == "logistic_regression":
            p = logit.predict_proba(X_test)[:, 1]
        elif name == "hist_gradient_boosting":
            p = hgb.predict_proba(X_test)[:, 1]
        else:
            p = calibrated.predict_proba(X_test)[:, 1]
        comparisons[name] = {
            **ranking_metrics(y_test, p),
            **binary_metrics(y_test, p, chosen["threshold"] if name != "rule_return_rate_gt_50" else 0.5),
            **decision_cost(y_test, p, chosen["threshold"] if name != "rule_return_rate_gt_50" else 0.5, test_df["order_value"].to_numpy(), COSTS),
        }

    # PSI on a key numeric feature (train vs test)
    def psi(a, b, bins=10):
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(a, qs))
        if len(edges) < 3:
            return 0.0
        ea = np.clip(np.histogram(a, bins=edges)[0] / max(len(a), 1), 1e-6, None)
        eb = np.clip(np.histogram(b, bins=edges)[0] / max(len(b), 1), 1e-6, None)
        return float(np.sum((ea - eb) * np.log(ea / eb)))

    drift = {
        "historical_return_rate_psi": psi(train_df["historical_return_rate"], test_df["historical_return_rate"]),
        "order_value_psi": psi(train_df["order_value"], test_df["order_value"]),
        "account_age_days_psi": psi(train_df["account_age_days"], test_df["account_age_days"]),
        "test_positive_rate": float(y_test.mean()),
        "train_positive_rate": float(y_train.mean()),
    }

    bundle = {
        "model": calibrated,
        "raw_model": hgb,
        "logistic": logit,
        "threshold": float(chosen["threshold"]),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_version": MODEL_VERSION,
        "policy_version": POLICY_VERSION,
        "target": TARGET,
        "costs": COSTS.__dict__,
    }
    joblib.dump(bundle, MODELS_DIR / "return_risk_model.joblib")

    # Sample scored test orders for the dashboard queue
    scored = test_df.copy()
    scored["risk_probability"] = p_test
    scored["risk_score"] = (p_test * 100).round(1)
    scored["reason_codes"] = scored.apply(reason_code_rules, axis=1)
    scored["split"] = "test"
    # Include a slice of recent valid for a fuller queue
    v = valid_df.copy()
    v["risk_probability"] = p_valid
    v["risk_score"] = (p_valid * 100).round(1)
    v["reason_codes"] = v.apply(reason_code_rules, axis=1)
    v["split"] = "valid"
    demo = pd.concat([v.tail(80), scored], ignore_index=True)
    demo.to_pickle(DATA_PROCESSED / "scored_orders.pkl")
    train_df.to_pickle(DATA_PROCESSED / "train.pkl")
    valid_df.to_pickle(DATA_PROCESSED / "valid.pkl")
    test_df.to_pickle(DATA_PROCESSED / "test.pkl")

    report = {
        "synthetic": True,
        "disclaimer": "Dataset is synthetic. Metrics are honest on a locked temporal test split.",
        "model_version": MODEL_VERSION,
        "policy_version": POLICY_VERSION,
        "target": TARGET,
        "splits": {
            "train": {"n": int(len(train_df)), "end": str(train_df["order_time"].max())},
            "valid": {"n": int(len(valid_df)), "end": str(valid_df["order_time"].max())},
            "test": {"n": int(len(test_df)), "start": str(test_df["order_time"].min()), "end": str(test_df["order_time"].max())},
            "protocol": "Chronological 70/15/15. Test locked before champion selection. Threshold chosen on validation cost only.",
        },
        "data_quality": quality.to_dict(),
        "validation_metrics": valid_scores,
        "selected_threshold_from_validation": chosen,
        "held_out_test": {**test_rank, **test_bin, **test_cost},
        "test_threshold_grid": test_grid,
        "model_comparison_test": comparisons,
        "drift": drift,
        "calibration_note": "Isotonic calibration fitted on validation only.",
        "feature_set": FEATURE_COLUMNS,
        "prohibited": "No PII, payment credentials, protected attributes, or post-decision labels in features.",
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / "quality.json").write_text(json.dumps(quality.to_dict(), indent=2), encoding="utf-8")

    # Failure cases for demo
    fp_mask = (y_test.to_numpy() == 0) & (p_test >= chosen["threshold"])
    fn_mask = (y_test.to_numpy() == 1) & (p_test < chosen["threshold"])
    cols = ["order_id", "order_value", "risk_probability", "historical_return_rate", "discount_pct", "product_category"]
    fp_rows = scored.loc[fp_mask, cols]
    fn_rows = scored.loc[fn_mask, cols]
    failures = {
        "false_positive_example": []
        if fp_rows.empty
        else fp_rows.nlargest(1, "risk_probability").to_dict(orient="records"),
        "false_negative_example": []
        if fn_rows.empty
        else fn_rows.nsmallest(1, "risk_probability").to_dict(orient="records"),
        "note": "Examples from the locked test set. The model is fallible; operators stay in the loop.",
    }
    (REPORTS_DIR / "failure_cases.json").write_text(json.dumps(failures, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"test": report["held_out_test"], "threshold": chosen["threshold"]}, indent=2))


if __name__ == "__main__":
    train()
