"""
CSTR adapter checks (protocol section 7.2). Uses the real ctrl-alt-recover simulator;
each rollout takes a few seconds. Skipped if the CAR repository is not present.
"""
import copy

import numpy as np
import pytest

from src.cstr.car_bridge import CSTR_DIR

pytestmark = pytest.mark.skipif(not CSTR_DIR.exists(), reason="ctrl-alt-recover not available")

from src.cstr import episodes as ep  # noqa: E402

KG = "@prefix : <http://example.org/test#> .\n:a :b :c ."
FOUL = ep.EpisodeSpec("test_foul", "fouling", {"fouling_max": 0.7, "fouling_tau": 2000.0}, 2000.0, 11)
NOMINAL = dict(ep.NOMINAL)
HALF_FEED = dict(ep.NOMINAL, Fin_sp=0.0167)


@pytest.fixture(scope="module")
def snap():
    s = ep.run_to_trigger(FOUL)
    assert s.triggered
    return s


def test_replay_is_deterministic(snap):
    a, b = ep.verify(snap, HALF_FEED), ep.verify(snap, HALF_FEED)
    assert a["verifier_pass"] == b["verifier_pass"] and a["metrics"] == b["metrics"]


def test_order_independence(snap):
    x1, y1 = ep.verify(snap, NOMINAL), ep.verify(snap, HALF_FEED)
    y2, x2 = ep.verify(snap, HALF_FEED), ep.verify(snap, NOMINAL)
    assert x1["metrics"] == x2["metrics"] and y1["metrics"] == y2["metrics"]


def test_verify_has_no_side_effects(snap):
    rng_before = np.random.get_state()
    plant_t, plant_x = snap.plant.t, snap.plant.x.copy()
    snap_rng = copy.deepcopy(snap.np_rng_state)
    ep.verify(snap, HALF_FEED)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1]) and rng_before[2] == rng_after[2]
    assert snap.plant.t == plant_t and np.array_equal(snap.plant.x, plant_x)
    assert np.array_equal(snap.np_rng_state[1], snap_rng[1])


def test_known_witnesses(snap):
    # severe fouling: keeping nominal feed does not recover; halving the feed does
    assert ep.verify(snap, NOMINAL)["verifier_pass"] is False
    assert ep.verify(snap, HALF_FEED)["verifier_pass"] is True


def test_simulator_error_is_unknown_not_pass(snap, monkeypatch):
    m = ep.load_cstr_case()

    def boom(**kwargs):
        raise FloatingPointError("forced")

    monkeypatch.setattr(m, "rollout_validate_setpoints", boom)
    r = ep.verify(snap, HALF_FEED)
    assert r["label_status"] == "simulator_error" and r["verifier_pass"] is None


def test_admissibility_bounds():
    assert ep.admissible(NOMINAL) == (True, None)
    assert ep.admissible(dict(NOMINAL, T_sp=float("nan")))[0] is False
    assert ep.admissible(dict(NOMINAL, L_sp=12.0)) == (False, "out_of_bounds_L_sp")
    assert ep.admissible(dict(NOMINAL, Fin_sp="0.02"))[0] is False


def test_episode_determinism_and_prompt_diversity():
    ep.set_kg_context(KG)
    a1, a2 = ep.run_to_trigger(FOUL), ep.run_to_trigger(FOUL)
    other = ep.run_to_trigger(ep.EpisodeSpec("test_foul_b", "fouling", {"fouling_max": 0.7, "fouling_tau": 2000.0}, 2000.0, 12))
    m1, m2, m3 = ep.build_messages(a1), ep.build_messages(a2), ep.build_messages(other)
    assert a1.t_trigger == a2.t_trigger and m1 == m2  # same spec -> same snapshot and prompt
    assert m1[1]["content"] != m3[1]["content"]  # different noise seed -> different snapshot values in the prompt
    assert KG in m1[0]["content"] and "CURRENT SNAPSHOT" in m1[1]["content"]
