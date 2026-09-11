from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR


def ensure_trained():
    model_path = MODELS_DIR / "return_risk_model.joblib"
    if model_path.exists() and (REPORTS_DIR / "metrics.json").exists():
        return
    from ml.models.train import train

    train()


@lru_cache(maxsize=1)
def load_bundle():
    ensure_trained()
    return joblib.load(MODELS_DIR / "return_risk_model.joblib")


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    ensure_trained()
    return json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_quality() -> dict:
    ensure_trained()
    path = REPORTS_DIR / "quality.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_failures() -> dict:
    ensure_trained()
    path = REPORTS_DIR / "failure_cases.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_scored() -> pd.DataFrame:
    ensure_trained()
    return pd.read_pickle(DATA_PROCESSED / "scored_orders.pkl")
