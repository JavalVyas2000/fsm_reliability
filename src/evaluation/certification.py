"""
Finite-sample certification of a frozen routing policy (protocol v1.0, section 8; Stage 3).

For a policy frozen before the certification data is seen, with k failures among the
n candidates it ALLOWs on an independent certification sample:

    U = exact one-sided (Clopper-Pearson) upper bound at confidence 1 - delta_i
    certified at level alpha  <=>  n > 0 and U <= alpha

With delta_i = delta / m (Bonferroni over the m declared policies), with probability at
least 1 - delta over the certification data, no declared policy is certified at a level
below its true ALLOW failure rate. A single bound per policy covers every alpha level at
once, so multiplicity is over policies, not levels.

This is the standard fixed-policy binomial argument (cf. Learn then Test,
arXiv:2110.01052); no new theorem is claimed. It holds for the declared IID population
(fixed prompt/model/decoding, instances from the declared distribution) and says
nothing about OOD instances or individual actions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

from src.evaluation.routing_v2 import cp_upper


def zero_failure_bound(n: int, delta: float) -> float:
    """U for k = 0: 1 - delta**(1/n)."""
    return 1.0 - delta ** (1.0 / n) if n > 0 else 1.0


def required_n(alpha: float, delta: float, k: int = 0) -> int:
    """Smallest n with cp_upper(k, n, delta) <= alpha."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    n = max(k + 1, int(math.ceil(math.log(delta) / math.log(1 - alpha))) if k == 0 else k + 1)
    while cp_upper(k, n, delta) > alpha:
        n += 1
    return n


def required_cert_size(alpha: float, delta: float, allow_coverage: float, k_planned: int = 2, margin: float = 1.5) -> int:
    """Protocol cert-size rule: n_required(k_planned) / expected ALLOW coverage * margin."""
    if allow_coverage <= 0:
        raise ValueError("allow_coverage must be > 0")
    return int(math.ceil(required_n(alpha, delta, k_planned) / allow_coverage * margin))


def certify(k: int, n: int, alphas: Sequence[float], delta_i: float) -> Dict:
    u = cp_upper(k, n, delta_i) if n > 0 else None
    return {
        "k_fail": k,
        "n_allow": n,
        "empirical_rate": k / n if n else None,
        "upper_bound": u,
        "delta_per_policy": delta_i,
        "certified": {str(a): bool(n > 0 and u is not None and u <= a) for a in alphas},
    }


def bonferroni(delta: float, m: int) -> float:
    return delta / m


def sample_size_table(alphas: Sequence[float], delta: float, ks: Sequence[int] = (0, 1, 2, 5)) -> List[Dict]:
    return [{"alpha": a, "delta": delta, **{f"k{k}": required_n(a, delta, k) for k in ks}} for a in alphas]
