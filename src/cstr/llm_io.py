"""
CSTR prompt regions, answer parsing and observable context/action features
(protocol v1.0, sections 2 and 6).
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from src.cstr.episodes import ACTION_BOUNDS, admissible
from src.evaluation.fsm_labels_v2 import find_first_json_object

PROMPT_VERSION = "car_action_propose_v1"
ACTION_KEYS = ("T_sp", "L_sp", "Fin_sp")
ZONE_CODE = {"SAFE": 0.0, "WARNING": 1.0, "UNSAFE": 2.0}

CSTR_CONTEXT_ACTION = [
    "time", "T_meas", "L_meas", "u_valve", "u_pump", "u_cool", "anomaly_ratio", "control_zone_code",
    "viol_temp", "viol_level", "viol_unsafe", "viol_cool_limited",
    "dT_sp", "dL_sp", "dFin_sp", "dFin_sp_rel",
]
# Observable reprompt context (present in the prompt of reprompt rounds; 0 for first proposals).
CSTR_ROUND_FEATURES = ["round", "prev_fail_no_recovery", "prev_fail_slow_recovery", "prev_fail_unsafe_fraction"]
# Current setpoints as shown in the prompt: constant for the fixed nominal plant, varying when
# the nominal plant varies per episode. Hidden plant parameters (UA, temperatures,
# concentrations) are privileged and never features.
CSTR_SETPOINT_FEATURES = ["L_sp_current", "Fin_sp_current"]


def region_char_spans(rendered: str, messages: List[Dict[str, str]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Regions in the rendered chat prompt:
      system_instruction : system prompt text before the KG section
      kg_context         : the KG section (header + Turtle block)
      snapshot           : 'CURRENT SNAPSHOT:' up to 'EVALUATION FEEDBACK'
      user_instruction   : the rest of the user message
    Uncovered characters are template tokens.
    """
    system, user = messages[0]["content"], messages[1]["content"]
    kg_marker = "## KNOWLEDGE GRAPH DATA"
    s_at = rendered.find(system.strip()[:200])
    if s_at < 0:
        raise ValueError("system prompt not found in rendered prompt")
    kg_at = rendered.find(kg_marker, s_at)
    sys_end = s_at + len(system.strip())
    spans: Dict[str, List[Tuple[int, int]]] = {}
    if kg_at >= 0:
        spans["system_instruction"] = [(s_at, kg_at)]
        spans["kg_context"] = [(kg_at, sys_end)]
    else:
        spans["system_instruction"] = [(s_at, sys_end)]
    u_core = user.strip()
    u_at = rendered.find(u_core[:200], sys_end)
    if u_at < 0:
        raise ValueError("user prompt not found in rendered prompt")
    u_end = u_at + len(u_core)
    snap_at = rendered.find("CURRENT SNAPSHOT:", u_at)
    fb_at = rendered.find("EVALUATION FEEDBACK", snap_at)
    if snap_at < 0 or fb_at < 0:
        raise ValueError("snapshot section not found")
    spans["snapshot"] = [(snap_at, fb_at)]
    spans["user_instruction"] = [(u_at, snap_at), (fb_at, u_end)]
    return spans


def parse_action(text: str) -> Dict[str, Any]:
    """Strict: first JSON object; T_sp/L_sp/Fin_sp must be JSON numbers; then admissibility bounds."""
    rec: Dict[str, Any] = {"parse_success": 0, "schema_valid": 0, "action": None, "reasoning": None,
                           "format_failure_reason": None}
    span = find_first_json_object(text)
    if span is None:
        rec["format_failure_reason"] = "no_complete_json_object"
        return rec
    try:
        obj = json.loads(text[span[0]:span[1]])
    except json.JSONDecodeError:
        rec["format_failure_reason"] = "json_decode_error"
        return rec
    if not isinstance(obj, dict) or any(k not in obj for k in ACTION_KEYS):
        rec["format_failure_reason"] = "missing_setpoint_key"
        return rec
    rec["parse_success"] = 1
    rec["reasoning"] = obj.get("reasoning")
    action = {k: obj[k] for k in ACTION_KEYS}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in action.values()):
        rec["format_failure_reason"] = "non_numeric_setpoint"
        return rec
    action = {k: float(v) for k, v in action.items()}
    rec["action"] = action
    ok, reason = admissible(action)
    if not ok:
        rec["format_failure_reason"] = reason
        return rec
    rec["schema_valid"] = 1
    return rec


def context_action(snapshot_state: Dict[str, Any], action: Optional[Dict[str, float]],
                   round_idx: int = 0, prev_fail_reason: Optional[str] = None) -> Dict[str, float]:
    """Observable plant context shown in the prompt, the proposed setpoint changes, and reprompt context."""
    out = snapshot_state.get("sim_last", {}) or {}
    viol = str(snapshot_state.get("violated_params") or "")
    f = {
        "time": float(out.get("t", math.nan)),
        "T_meas": float(out.get("T_meas", math.nan)),
        "L_meas": float(out.get("L_meas", math.nan)),
        "u_valve": float(out.get("u_valve", math.nan)),
        "u_pump": float(out.get("u_pump", math.nan)),
        "u_cool": float(out.get("u_cool", math.nan)),
        "anomaly_ratio": float(snapshot_state.get("anomaly_ratio") or 0.0),
        "control_zone_code": ZONE_CODE.get(str(snapshot_state.get("control_zone")), math.nan),
        "viol_temp": float("T_meas_error" in viol),
        "viol_level": float("level_error_large" in viol),
        "viol_unsafe": float("entered_UNSAFE" in viol),
        "viol_cool_limited": float("cool_limited" in viol),
    }
    if action is None:
        f.update(dT_sp=math.nan, dL_sp=math.nan, dFin_sp=math.nan, dFin_sp_rel=math.nan)
    else:
        cur = {k: float(snapshot_state.get(k)) for k in ACTION_KEYS}
        f.update(
            dT_sp=action["T_sp"] - cur["T_sp"],
            dL_sp=action["L_sp"] - cur["L_sp"],
            dFin_sp=action["Fin_sp"] - cur["Fin_sp"],
            dFin_sp_rel=(action["Fin_sp"] - cur["Fin_sp"]) / cur["Fin_sp"] if cur["Fin_sp"] else math.nan,
        )
    f.update(
        L_sp_current=float(snapshot_state.get("L_sp", math.nan)),
        Fin_sp_current=float(snapshot_state.get("Fin_sp", math.nan)),
        round=float(round_idx),
        prev_fail_no_recovery=float(prev_fail_reason == "no_recovery"),
        prev_fail_slow_recovery=float(prev_fail_reason == "slow_recovery"),
        prev_fail_unsafe_fraction=float(prev_fail_reason == "unsafe_fraction"),
    )
    return f
