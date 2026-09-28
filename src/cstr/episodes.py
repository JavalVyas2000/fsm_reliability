"""
CSTR episodes, snapshots and the verifier adapter (protocol v1.0, sections 5.2 and 7).

Everything physical (simulator, monitoring/trigger logic, prompt text, rollout validator)
is the unmodified ctrl-alt-recover code, driven here node by node instead of through the
LangGraph application:

    initializing -> [plant -> monitoring]* until should_act  => Snapshot

Differences from the historical LangGraph runs (documented deviations):
  - fault onset and severity come from the episode spec (no unseeded `random` draw);
  - each episode has its own simulator noise seed (the constructor otherwise reseeds 42);
  - a plain dict carries the state, so `symptoms` and the cooling-limited detector buffers
    persist between steps as the code intends (LangGraph dropped these undeclared keys).

Verifier determinism: each rollout runs with numpy's global RNG set to the state captured
at the snapshot (i.e. the noise the real plant would see next) and restores the caller's
RNG afterwards, so verdicts are a function of (snapshot, action) and independent of
evaluation order.
"""
from __future__ import annotations

import contextlib
import copy
import io
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.cstr.car_bridge import load_cstr_case

FAMILIES = ("fouling", "pump_degrade", "cool_stuck_closed", "outlet_block", "leak")
NOMINAL = {"T_sp": 310.0, "L_sp": 10.0, "Fin_sp": 2.0 / 60.0}
T_END = 8000.0  # CLI default of the historical runs
NORMAL_END = 7000.0  # start of SHUTDOWN phase
# Cheap admissibility bounds for proposed setpoints (outside -> REJECT, no rollout).
# T: coolant inlet (290 K) to well above the 313 K safety limit; L: the checker's safe
# band [2, 11]; Fin: 0 to 3x nominal.
ACTION_BOUNDS = {"T_sp": (290.0, 330.0), "L_sp": (2.0, 11.0), "Fin_sp": (0.0, 0.1)}
VERIFIER_KNOBS = dict(grace_s=60.0, safe_persist_s=60.0, deadline_s=600.0, unsafe_fraction_max=0.20,
                      enforce_only_normal_phase=True)


# Nominal-plant parameters that may be varied per episode (CSTRSimulation constructor
# arguments; CAR's code is not modified). T_sp stays at 310 K because CAR's safety and
# validation logic uses absolute thresholds anchored at 310 / 313 K.
PLANT_PARAM_KEYS = ("UA", "Tin_base", "Tc_base", "CA_in_base", "CD_in_base", "Fc_max", "L_sp", "Fin_sp")


@dataclass
class EpisodeSpec:
    episode_id: str
    family: str
    params: Dict[str, float]
    onset_s: float
    noise_seed: int
    # Optional per-episode nominal plant (empty = CAR's defaults). Specs pickled before this
    # field existed lack it; always read it with getattr(spec, "plant", {}).
    plant: Dict[str, float] = field(default_factory=dict)


@dataclass
class Snapshot:
    spec: EpisodeSpec
    triggered: bool
    t_trigger: Optional[float]
    plant: Any = None  # CSTRSimulation deepcopy at the trigger
    np_rng_state: Any = None  # numpy global RNG state at the trigger (plant's future noise)
    state: Dict[str, Any] = field(default_factory=dict)  # prompt/validator fields only
    wall_s: float = 0.0


# ----------------------------------------------------------------------------- faults

_SFM_CLS = None


def __getattr__(name):  # PEP 562: lets pickle resolve the lazily created class
    if name == "SeverityFaultManager":
        return _severity_fault_manager_cls()
    raise AttributeError(name)


def _severity_fault_manager_cls():
    global _SFM_CLS
    if _SFM_CLS is not None:
        return _SFM_CLS
    load_cstr_case()
    import cstr_digital_twin as dt  # importable once the bridge has set sys.path

    class SeverityFaultManager(dt.FaultManager):
        """FaultManager with a parameterised stuck-closed cooling opening (hard-coded 0.3 upstream)."""

        def __init__(self, cfg, UA_nominal: float, cool_stuck_closed_opening: float):
            super().__init__(cfg, UA_nominal)
            self.cool_stuck_closed_opening = float(cool_stuck_closed_opening)

        def actuator_cool_opening(self, t: float, u_cmd: float) -> float:
            c = self.cfg
            if c.enable and c.cool_stuck_closed and not c.cool_stuck_open and t >= c.cool_stuck_closed_t:
                return self.cool_stuck_closed_opening
            return super().actuator_cool_opening(t, u_cmd)

    SeverityFaultManager.__module__ = __name__
    SeverityFaultManager.__qualname__ = "SeverityFaultManager"
    _SFM_CLS = SeverityFaultManager
    return SeverityFaultManager


def build_fault_cfg(spec: EpisodeSpec):
    """Upstream fault_cfg_from_name(family), then deterministic onset and severity."""
    m = load_cstr_case()
    py_state = random.getstate()
    try:
        cfg = m.fault_cfg_from_name(spec.family)  # consumes python RNG for onset; overwritten below
    finally:
        random.setstate(py_state)
    p, on = spec.params, float(spec.onset_s)
    if spec.family == "fouling":
        cfg.fouling_t_start = on
        cfg.fouling_max = float(p["fouling_max"])
        cfg.fouling_tau = float(p["fouling_tau"])
    elif spec.family == "pump_degrade":
        cfg.pump_degrade_t = on
        cfg.pump_degrade_factor = float(p["pump_degrade_factor"])
    elif spec.family == "cool_stuck_closed":
        cfg.cool_stuck_closed_t = on
    elif spec.family == "outlet_block":
        cfg.outlet_block_t = on
        cfg.outlet_block_factor = float(p["outlet_block_factor"])
    elif spec.family == "leak":
        cfg.leak_t = on
        cfg.leak_k = float(p["leak_k"])
    else:
        raise ValueError(spec.family)
    return cfg


# ----------------------------------------------------------------------------- episodes

PROMPT_STATE_KEYS = (
    "T_end", "mode", "sim_last", "T_sp", "L_sp", "Fin_sp", "anomaly_ratio", "violated_params",
    "control_zone", "control_reasons", "symptoms", "safe_hold_seconds", "action_calls",
    "fault_name",
)


def run_to_trigger(spec: EpisodeSpec, quiet: bool = True) -> Snapshot:
    """Simulate from t = 0 with the real monitoring loop until the first action trigger."""
    m = load_cstr_case()
    t_start = time.perf_counter()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        plant_p = dict(getattr(spec, "plant", {}) or {})
        state: Dict[str, Any] = {"T_end": T_END, "fault_name": "normal"}
        if "L_sp" in plant_p:
            state["L_sp"] = float(plant_p["L_sp"])
        if "Fin_sp" in plant_p:
            state["Fin_sp"] = float(plant_p["Fin_sp"])
        m.initializing(state)
        cfg = build_fault_cfg(spec)
        state["fault_name"] = spec.family
        state["fault_cfg"] = cfg
        phys = {k: float(plant_p[k]) for k in ("UA", "Tin_base", "Tc_base", "CA_in_base", "CD_in_base") if k in plant_p}
        if "Fc_max" in plant_p:  # coolant supply capacity: base flow and valve capacity scale together
            phys["Fc_max"] = float(plant_p["Fc_max"])
            phys["Fc_in_base"] = float(plant_p["Fc_max"])
        plant = m.CSTRSimulation(
            dt=1.0, T_sp=state["T_sp"], L_sp=state["L_sp"], Fin_sp_normal=state["Fin_sp_normal"],
            Fin_sp_startup=state["Fin_sp_startup"], fault_cfg=cfg, seed=int(spec.noise_seed), **phys,
        )
        if spec.family == "cool_stuck_closed":
            plant.faults = _severity_fault_manager_cls()(cfg, plant.UA, spec.params["stuck_opening"])
        state["_plant"] = plant

        triggered = False
        while True:
            m.plant(state)
            m.monitoring(state)
            if state.get("should_act"):
                triggered = True
                break
            if float(state["sim_last"].get("t", 0.0)) >= NORMAL_END:
                break

    t_trig = float(state["sim_last"]["t"]) if triggered else None
    snap = Snapshot(spec=spec, triggered=triggered, t_trigger=t_trig, wall_s=time.perf_counter() - t_start)
    # A trigger before the fault onset means the nominal plant itself is not healthy under
    # CAR's monitor (a false alarm); such episodes are flagged and excluded upstream.
    snap.pre_fault_trigger = bool(triggered and t_trig is not None and t_trig < float(spec.onset_s))
    if triggered:
        snap.plant = copy.deepcopy(state["_plant"])
        snap.np_rng_state = np.random.get_state()
        snap.state = {k: copy.deepcopy(state.get(k)) for k in PROMPT_STATE_KEYS}
        snap.state["action_calls"] = 0
    return snap


# ----------------------------------------------------------------------------- prompt

class _Captured(Exception):
    pass


class _CaptureLLM:
    """Captures the messages action_propose sends; never produces an answer."""

    def __init__(self):
        self.messages = None

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        raise _Captured()


def set_kg_context(ttl: str) -> None:
    """Freeze the KG context used by the prompt (module globals of cstr_case)."""
    m = load_cstr_case()
    m._KG_ACTION_TTL = ttl
    m._KG_LOADED = bool(ttl)


def fetch_kg_context_live() -> str:
    m = load_cstr_case()
    log = io.StringIO()  # upstream prints emoji that a cp1252 console cannot encode
    with contextlib.redirect_stdout(log):
        ok = m.load_kg_context()
    if not ok:
        raise RuntimeError(f"GraphDB KG context could not be loaded: {log.getvalue()}")
    return m._KG_ACTION_TTL


def build_messages(snapshot: Snapshot) -> List[Dict[str, str]]:
    """Exact first-proposal prompt of the upstream action_propose node for this snapshot."""
    m = load_cstr_case()
    cap = _CaptureLLM()
    state = copy.deepcopy(snapshot.state)
    state.update(reprompt_count=0, eval_feedback="", proposed_sp={}, llm_model="hf-local", llm_timeout=1.0)
    original = m._make_llm
    m._make_llm = lambda *a, **k: cap
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m.action_propose(state)
    except m.LLMActionError:
        pass
    finally:
        m._make_llm = original
    if cap.messages is None:
        raise RuntimeError("Prompt capture failed")
    return [{"role": role, "content": content} for role, content in cap.messages]


# ----------------------------------------------------------------------------- verifier

def admissible(action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    for k, (lo, hi) in ACTION_BOUNDS.items():
        v = action.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return False, f"non_finite_{k}"
        if not (lo <= float(v) <= hi):
            return False, f"out_of_bounds_{k}"
    return True, None


def verify(snapshot: Snapshot, action: Dict[str, float]) -> Dict[str, Any]:
    """Legacy rollout_validate_setpoints under deterministic, isolated RNG."""
    m = load_cstr_case()
    saved_np, saved_py = np.random.get_state(), random.getstate()
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        np.random.set_state(snapshot.np_rng_state)
        with contextlib.redirect_stdout(io.StringIO()):
            ok, summary, reason, metrics, traj = m.rollout_validate_setpoints(
                base_plant=snapshot.plant, state=copy.deepcopy(snapshot.state),
                proposed={k: float(action[k]) for k in ("T_sp", "L_sp", "Fin_sp")}, **VERIFIER_KNOBS,
            )
        status, n_steps = "known", len(traj)
    except Exception as exc:  # unknown physical outcome, never a pass
        ok, summary, reason, metrics, status, n_steps = None, f"simulator_error: {exc!r}", "simulator_error", {}, "simulator_error", 0
    finally:
        np.random.set_state(saved_np)
        random.setstate(saved_py)
    return {
        "label_status": status,
        "verifier_pass": None if ok is None else bool(ok),
        "fail_reason": reason,
        "summary": summary,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v) for k, v in metrics.items()},
        "n_steps": n_steps,
        "wall_s": time.perf_counter() - t0,
        # single-threaded rollout: process CPU time is the contention-robust cost measure
        "cpu_s": time.process_time() - c0,
    }


def current_setpoints(snapshot: Snapshot) -> Dict[str, float]:
    """The episode's own setpoints at the trigger (the 'no change' action)."""
    return {k: float(snapshot.state.get(k, NOMINAL[k])) for k in ("T_sp", "L_sp", "Fin_sp")}


def verify_job(snapshot: Snapshot, action: Dict[str, float], with_nochange: bool) -> Dict[str, Any]:
    """Process-pool job: verify a proposal (and optionally the no-change action, offline only)."""
    out = {"result": verify(snapshot, action)}
    if with_nochange:
        out["nochange"] = verify(snapshot, current_setpoints(snapshot))
    return out


def reprompt_feedback(snapshot: Snapshot, verify_result: Dict[str, Any], reprompt_count: int) -> str:
    """Feedback text of the upstream `reprompting` node for a failed verification."""
    m = load_cstr_case()
    state = copy.deepcopy(snapshot.state)
    state.update(reprompt_count=reprompt_count - 1, reprompt_max=5, eval_summary=verify_result.get("summary", ""),
                 eval_fail_reason=verify_result.get("fail_reason", ""), eval_metrics=verify_result.get("metrics", {}))
    with contextlib.redirect_stdout(io.StringIO()):
        m.reprompting(state)
    return state["eval_feedback"]


def spec_to_dict(spec: EpisodeSpec) -> Dict[str, Any]:
    return asdict(spec)
