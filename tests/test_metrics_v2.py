import numpy as np

from src.evaluation.metrics_v2 import bootstrap_delta_auroc, ece, predictive_metrics


def test_ece_counts_p_equal_one():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert ece(y, p) == 0.0
    # p = 1 with y = 0 must contribute (legacy digitize bug dropped it)
    assert ece(np.array([0]), np.array([1.0])) == 1.0


def test_auprc_uses_failure_as_positive():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.3])
    m = predictive_metrics(y, p)
    assert m["auprc_fail"] == 1.0 and m["prevalence_fail"] == 0.25


def test_delta_auroc_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    d = bootstrap_delta_auroc(y, p, p, n_boot=200)
    assert d["delta_auroc"] == 0.0 and d["ci95_lo"] == 0.0 and d["ci95_hi"] == 0.0
