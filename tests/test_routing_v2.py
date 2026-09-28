import numpy as np

from src.evaluation.routing_v2 import (
    cp_upper,
    random_baseline,
    route,
    routing_summary,
    select_tau_high,
    select_tau_low,
)


def test_cp_upper_zero_failures_matches_closed_form():
    for n in (59, 299):
        assert abs(cp_upper(0, n, 0.05) - (1 - 0.05 ** (1 / n))) < 1e-9
    assert cp_upper(0, 299, 0.05) <= 0.01 < cp_upper(0, 298, 0.05)


def test_perfect_separation_routes_everything_without_simulator():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])
    lo = select_tau_low(p, y, alpha=0.0, rule="point")
    hi = select_tau_high(p, y, beta=0.0, rule="point")
    r = route(p, lo, hi)
    assert list(r) == ["ALLOW"] * 3 + ["DISALLOW"] * 3


def test_thresholds_respect_point_targets_on_selection_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    lo = select_tau_low(p, y, alpha=0.1, rule="point")
    hi = select_tau_high(p, y, beta=0.1, rule="point")
    s = routing_summary(route(p, lo, hi), y)
    if s["n_allow"]:
        assert s["allow_failure_rate"] <= 0.1
    if s["n_disallow"]:
        assert s["disallow_valid_rate"] <= 0.1


def test_ucb_rule_is_more_conservative_than_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    assert select_tau_low(p, y, 0.1, "ucb") <= select_tau_low(p, y, 0.1, "point")
    assert select_tau_high(p, y, 0.1, "ucb") >= select_tau_high(p, y, 0.1, "point")


def test_no_qualifying_threshold_means_verify_everything():
    y = np.ones(10, dtype=int)
    p = np.linspace(0, 1, 10)
    assert select_tau_low(p, y, 0.05, "point") == 0.0
    s = routing_summary(route(p, 0.0, 1.0), y)
    assert s["n_verify"] == 10 and s["allow_failure_rate"] is None


def test_ties_are_routed_together():
    p = np.array([0.2, 0.2, 0.2, 0.9])
    y = np.array([0, 0, 1, 1])
    lo = select_tau_low(p, y, alpha=0.0, rule="point")
    assert lo == 0.0  # cannot split the tied block to exclude the failure


def test_random_baseline_expectation():
    y = np.array([1] * 50 + [0] * 50)
    rb = random_baseline(y, n_allow=20, n_disallow=20, n_rep=500)
    assert abs(rb["escaped_failures_mean"] - 10) < 1.0
    assert abs(rb["lost_viable_mean"] - 10) < 1.0


def test_joint_allow_beats_single_when_errors_differ():
    from src.evaluation.routing_v2 import route_joint, select_joint_allow

    # probe a misses failure at index 0, probe b misses failure at index 1
    y = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    p_a = np.array([0.05, 0.9, 0.1, 0.2, 0.3, 0.4, 0.8, 0.95])
    p_b = np.array([0.9, 0.05, 0.1, 0.2, 0.3, 0.4, 0.85, 0.95])
    ta, tb, n = select_joint_allow(p_a, p_b, y, alpha=0.0)
    assert n == 4
    assert select_tau_low(p_a, y, 0.0, "point") == 0.0  # single probe a cannot allow anything
    r = route_joint(p_a, p_b, ta, tb, p_a, 1.0)
    assert (r == "ALLOW").sum() == 4 and not ((r == "ALLOW") & (y == 1)).any()


def test_joint_contradiction_goes_to_verify():
    from src.evaluation.routing_v2 import route_joint

    r = route_joint(np.array([0.1]), np.array([0.1]), 0.5, 0.5, np.array([0.9]), 0.8)
    assert r[0] == "VERIFY"
