"""Simple PSI helper used by training reports and the dashboard."""

from __future__ import annotations

import numpy as np


def population_stability_index(expected, actual, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.clip(np.histogram(expected, bins=edges)[0] / max(len(expected), 1), 1e-6, None)
    a = np.clip(np.histogram(actual, bins=edges)[0] / max(len(actual), 1), 1e-6, None)
    return float(np.sum((e - a) * np.log(e / a)))
