import numpy as np
import pytest
from scipy.stats import binom

from src.evaluation.certification import (
    bonferroni,
    certify,
    required_cert_size,
    required_n,
    zero_failure_bound,
)
from src.evaluation.routing_v2 import cp_upper


def test_protocol_sample_size_table():
    # values recorded in protocol section 8 (delta = 0.05)
    expected = {0.05: (59, 93, 124, 208), 0.02: (149, 236, 313, 523), 0.01: (299, 473, 628, 1049)}
    for a, ns in expected.items():
        assert tuple(required_n(a, 0.05, k) for k in (0, 1, 2, 5)) == ns


def test_zero_failure_closed_form():
    for n in (1, 59, 299, 1000):
        assert abs(zero_failure_bound(n, 0.05) - cp_upper(0, n, 0.05)) < 1e-9


def test_bound_definition_is_exact():
    # U is the rate at which observing <= k failures has probability delta
    for k, n in [(0, 100), (2, 300), (5, 1000)]:
        u = cp_upper(k, n, 0.05)
        assert abs(binom.cdf(k, n, u) - 0.05) < 1e-6


def test_certify_requires_nonempty_allow_set():
    r = certify(0, 0, [0.05], 0.05)
    assert r["certified"]["0.05"] is False and r["upper_bound"] is None


def test_certify_levels_share_one_bound():
    r = certify(0, 300, [0.05, 0.02, 0.01], 0.05)
    assert r["certified"] == {"0.05": True, "0.02": True, "0.01": True}
    r = certify(0, 150, [0.05, 0.02, 0.01], 0.05)
    assert r["certified"] == {"0.05": True, "0.02": True, "0.01": False}


def test_bonferroni_and_cert_size():
    assert bonferroni(0.05, 4) == pytest.approx(0.0125)
    n = required_cert_size(0.05, 0.0125, allow_coverage=0.1, k_planned=0, margin=1.0)
    assert n == int(np.ceil(required_n(0.05, 0.0125, 0) / 0.1))


def test_coverage_of_procedure_by_simulation():
    # true rate exactly at alpha: false certification must happen with prob <= delta
    rng = np.random.default_rng(0)
    alpha, delta, n = 0.05, 0.05, 200
    k = rng.binomial(n, alpha, size=20000)
    false_cert = np.mean([cp_upper(int(x), n, delta) <= alpha for x in k])
    assert false_cert <= delta + 0.005
