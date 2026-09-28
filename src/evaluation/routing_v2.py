"""
Three-way selective-verification routing (protocol v1.0, sections 3 and 8).

    p < tau_low             -> ALLOW     execute without the simulator
    tau_low <= p <= tau_high -> VERIFY    call the simulator
    p > tau_high            -> DISALLOW  reject without the simulator

p is the predicted FAILURE probability. Thresholds are selected on a development
partition and then frozen; evaluation partitions only apply them.

Error types:
    escaped failure  = ALLOW   and y = 1   (unsafe bypass)
    lost viable      = DISALLOW and y = 0   (a valid action thrown away)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import beta as beta_dist


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound for a binomial rate."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))


def _ok(k: int, n: int, target: float, rule: str, delta: float) -> bool:
    if n == 0:
        return False
    if rule == "point":
        return k / n <= target
    if rule == "ucb":
        return cp_upper(k, n, delta) <= target
    raise ValueError(rule)


def select_tau_low(p: np.ndarray, y: np.ndarray, alpha: float, rule: str = "ucb", delta: float = 0.05) -> float:
    """
    Largest tau_low such that the ALLOW set {p < tau_low} meets the escaped-failure
    criterion (failure rate <= alpha: point estimate, or exact upper bound). Returns 0.0
    (allow nothing) if no threshold qualifies.
    """
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    best = 0.0
    fails = np.cumsum(ys)
    for i in range(len(ps)):
        # ALLOW set = first i+1 items; tau just above ps[i], and ties must be included together.
        if i + 1 < len(ps) and ps[i + 1] == ps[i]:
            continue
        if _ok(int(fails[i]), i + 1, alpha, rule, delta):
            best = float(np.nextafter(ps[i], np.inf))
    return best


def select_tau_high(p: np.ndarray, y: np.ndarray, beta: float, rule: str = "ucb", delta: float = 0.05) -> float:
    """
    Smallest tau_high such that the DISALLOW set {p > tau_high} meets the lost-viable
    criterion (valid fraction <= beta). Returns 1.0 (disallow nothing) if none qualifies.
    """
    order = np.argsort(-p, kind="stable")
    ps, valid = p[order], 1 - y[order]
    best = 1.0
    lost = np.cumsum(valid)
    for i in range(len(ps)):
        if i + 1 < len(ps) and ps[i + 1] == ps[i]:
            continue
        if _ok(int(lost[i]), i + 1, beta, rule, delta):
            best = float(np.nextafter(ps[i], -np.inf))
    return best


def select_joint_allow(
    p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray, alpha: float = 0.0, rule: str = "point", delta: float = 0.05
) -> tuple[float, float, int]:
    """
    Agreement rule: ALLOW iff p_a < tau_a AND p_b < tau_b. Searches all (tau_a, tau_b)
    cut points on the selection data and returns the pair that maximises the ALLOW count
    subject to the escaped-failure criterion. Ties: the smaller tau_a + tau_b (more
    conservative) wins. Returns (tau_a, tau_b, n_allow); (0, 0, 0) if none qualifies.
    """
    ua = np.unique(p_a)
    best = (0.0, 0.0, 0)
    for ta_val in ua:
        in_a = p_a <= ta_val
        pb, yb = p_b[in_a], y[in_a]
        order = np.argsort(pb, kind="stable")
        pbs, ybs = pb[order], yb[order]
        fails = np.cumsum(ybs)
        for i in range(len(pbs)):
            if i + 1 < len(pbs) and pbs[i + 1] == pbs[i]:
                continue
            if not _ok(int(fails[i]), i + 1, alpha, rule, delta):
                continue
            n = i + 1
            ta, tb = float(np.nextafter(ta_val, np.inf)), float(np.nextafter(pbs[i], np.inf))
            if n > best[2] or (n == best[2] and n > 0 and ta + tb < best[0] + best[1]):
                best = (ta, tb, n)
    return best


def route_joint(
    p_a: np.ndarray, p_b: np.ndarray, tau_a: float, tau_b: float, p_dis: np.ndarray, tau_high: float
) -> np.ndarray:
    """
    ALLOW on joint agreement; DISALLOW from a single probe p_dis > tau_high; else VERIFY.
    A candidate that satisfies both (contradictory probes) is sent to VERIFY.
    """
    allow = (p_a < tau_a) & (p_b < tau_b)
    dis = p_dis > tau_high
    out = np.full(len(p_a), "VERIFY", dtype=object)
    out[allow & ~dis] = "ALLOW"
    out[dis & ~allow] = "DISALLOW"
    return out


def route(p: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    tau_high = max(tau_high, tau_low)
    out = np.full(len(p), "VERIFY", dtype=object)
    out[p < tau_low] = "ALLOW"
    out[p > tau_high] = "DISALLOW"
    return out


def routing_summary(routes: np.ndarray, y: np.ndarray, delta: float = 0.05) -> Dict[str, Optional[float]]:
    n = len(y)
    a, v, d = routes == "ALLOW", routes == "VERIFY", routes == "DISALLOW"
    n_a, n_v, n_d = int(a.sum()), int(v.sum()), int(d.sum())
    esc, lost = int((a & (y == 1)).sum()), int((d & (y == 0)).sum())
    return {
        "n": n,
        "n_allow": n_a,
        "n_verify": n_v,
        "n_disallow": n_d,
        "frac_allow": n_a / n,
        "frac_verify": n_v / n,
        "frac_disallow": n_d / n,
        "simulator_calls_saved_frac": (n_a + n_d) / n,
        "escaped_failures": esc,
        "allow_failure_rate": esc / n_a if n_a else None,
        "allow_failure_rate_cp95_upper": cp_upper(esc, n_a, delta) if n_a else None,
        "escaped_failure_rate_all": esc / n,
        "lost_viable": lost,
        "disallow_valid_rate": lost / n_d if n_d else None,
        "disallow_valid_rate_cp95_upper": cp_upper(lost, n_d, delta) if n_d else None,
        "lost_viable_rate_all": lost / n,
        "verify_failure_rate": float(y[v].mean()) if n_v else None,
    }


def random_baseline(y: np.ndarray, n_allow: int, n_disallow: int, n_rep: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Random routing with the same ALLOW/DISALLOW counts; distribution of errors."""
    rng = np.random.default_rng(seed)
    n = len(y)
    esc, lost = [], []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        a, d = perm[:n_allow], perm[n_allow : n_allow + n_disallow]
        esc.append(int(y[a].sum()))
        lost.append(int((1 - y[d]).sum()))
    esc, lost = np.array(esc), np.array(lost)
    return {
        "escaped_failures_mean": float(esc.mean()),
        "escaped_failures_p2.5": float(np.percentile(esc, 2.5)),
        "escaped_failures_p97.5": float(np.percentile(esc, 97.5)),
        "lost_viable_mean": float(lost.mean()),
        "lost_viable_p2.5": float(np.percentile(lost, 2.5)),
        "lost_viable_p97.5": float(np.percentile(lost, 97.5)),
    }
