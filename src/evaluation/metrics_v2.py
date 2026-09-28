"""
Predictive metrics for failure prediction (protocol v1.0, section 8).
Positive class is always FAILURE (y = 1).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on [0, 1]; p = 1.0 falls in the top bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(total / len(y))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Dict[str, list]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = {"bin_lo": [], "bin_hi": [], "n": [], "mean_p": [], "frac_fail": []}
    for b in range(n_bins):
        m = bins == b
        rows["bin_lo"].append(b / n_bins)
        rows["bin_hi"].append((b + 1) / n_bins)
        rows["n"].append(int(m.sum()))
        rows["mean_p"].append(float(p[m].mean()) if m.any() else None)
        rows["frac_fail"].append(float(y[m].mean()) if m.any() else None)
    return rows


def predictive_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Dict[str, Optional[float]]:
    y, p = np.asarray(y, int), np.asarray(p, float)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_fail": int(y.sum()),
        "prevalence_fail": float(y.mean()) if len(y) else None,
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc_fail": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "ece": ece(y, p, n_bins) if len(y) else None,
        "ece_bins": n_bins,
    }


def bootstrap_delta_auroc(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired (grouped) bootstrap of AUROC(a) - AUROC(b). Groups are resampled whole."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    deltas = np.array(deltas)
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return {
        "delta_auroc": float(point),
        "ci95_lo": float(np.percentile(deltas, 2.5)),
        "ci95_hi": float(np.percentile(deltas, 97.5)),
        "n_boot_used": int(len(deltas)),
    }


def bootstrap_auroc_ci(y, p, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return {"ci95_lo": float(np.percentile(vals, 2.5)), "ci95_hi": float(np.percentile(vals, 97.5))}
