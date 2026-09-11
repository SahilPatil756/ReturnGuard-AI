from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class CostConfig:
    fp_cost: float
    intervention_cost: float
    fn_cost_fixed: float
    fn_cost_margin: float
    tp_save_rate: float


def binary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "review_rate": float((tp + fp) / max(len(y_true), 1)),
        "support_positive": int(y_true.sum()),
        "n": int(len(y_true)),
    }


def ranking_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def decision_cost(y_true, y_prob, threshold: float, order_value: np.ndarray, costs: CostConfig) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    values = np.asarray(order_value, dtype=float)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    tp = (y_true == 1) & (y_pred == 1)
    fn_loss = np.sum(costs.fn_cost_fixed + costs.fn_cost_margin * values[fn])
    fp_loss = np.sum(np.full(fp.sum(), costs.fp_cost))
    tp_gross = np.sum(costs.tp_save_rate * (costs.fn_cost_fixed + costs.fn_cost_margin * values[tp]))
    tp_net = tp_gross - costs.intervention_cost * tp.sum()
    total = float(fp_loss + fn_loss - tp_net)
    per_1k = total / max(len(y_true), 1) * 1000
    return {
        "fp_cost_total": float(fp_loss),
        "fn_cost_total": float(fn_loss),
        "tp_saved_net": float(tp_net),
        "expected_decision_cost": total,
        "cost_per_1000_orders": float(per_1k),
        "estimated_avoidable_loss": float(max(-total, 0)),
    }


def threshold_grid(y_true, y_prob, order_value, costs: CostConfig, thresholds=None) -> list[dict]:
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.linspace(0.10, 0.80, 15)]
    rows = []
    for t in thresholds:
        m = binary_metrics(y_true, y_prob, t)
        c = decision_cost(y_true, y_prob, t, order_value, costs)
        rows.append({**m, **c})
    return rows


def pick_threshold(grid: list[dict]) -> dict:
    return min(grid, key=lambda r: r["expected_decision_cost"])
