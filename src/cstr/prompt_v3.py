"""
CSTR prompt v3: neutral about the action, explicit about what makes an action achievable.

Motivation (2026-09-28): v2.1 still tells the model which lever to pull and in which direction
("typically by reducing Fin_sp", "prefer reducing Fin_sp before changing T_sp", "and/or reduce
Fin_sp"), and the upstream reprompt feedback adds direction and size hints ("Fin_sp down",
"lower Fin_sp more", "milder setpoints"). v3 removes all of these, so which setpoint to change,
in which direction and by how much must be derived from the KG and the snapshot.

What v3 states instead are facts about the plant and the acceptance test, so that the model
can tell an achievable proposal from an unachievable one:
  - actuator commands are fractions of capacity and saturate at about 0.95;
  - the fault persists during the rollout and the current setpoints are not necessarily right;
  - the SAFE thresholds are absolute (313 K / 310 K, level 2-11) and do not move with T_sp;
  - a proposal is achievable only if every controller can hold its setpoint without saturating.

Other changes from v2.1:
  - the KG reference values of the default plant (L_sp 10, T_sp 310, Fin_sp 0.0333) are removed:
    with varied nominal plants they are not this plant's values;
  - reprompt feedback = the verifier's own summary, fail reason and measured rollout metrics,
    without the upstream hint;
  - the snapshot block, goal and allowed ranges are unchanged (cut from the upstream prompt).
No privileged information (fault family, severity, onset) is added.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.cstr.car_bridge import load_cstr_case
from src.cstr.episodes import ACTION_BOUNDS, Snapshot, build_messages
from src.cstr.prompt_v2 import USER_TAIL_V2, _snapshot_block

PROMPT_VERSION = "cstr_prompt_v3"

# v3.0 pilot (20 episodes, 2026-09-28): the model stopped anchoring on small feed cuts but guessed
# the direction (Fin_sp raised in 55% of proposals), copied the SAFE limits as setpoints (T_sp 313,
# L_sp 2) and passed 1/20 against 5/20 for v2.1.
SYSTEM_V3 = """You are the **Corrective Setpoint Action Agent** for a Continuous Stirred Tank Reactor (CSTR).

A fault has occurred and the plant monitor has flagged it. You propose values for these setpoints ONLY:
- T_sp (temperature setpoint, K)
- L_sp (level setpoint)
- Fin_sp (inlet flow setpoint, L/s)

# The plant
The Knowledge Graph (KG) below describes how the plant works: which controller each setpoint feeds,
which actuator each controller drives, and the balance equations linking flows, level, temperature
and cooling. Use the KG and the snapshot numbers to decide which setpoints to change, in which
direction, and by how much. Keeping a setpoint unchanged is also a decision that must be justified.

Facts about the snapshot:
- u_valve, u_pump and u_cool are actuator commands as a fraction of full capacity (0 = closed or
  stopped, 1 = fully open or full speed). An actuator at 0.95 or above is saturated: its controller
  has no capacity left and cannot hold its setpoint if more is needed.
- "current setpoints" are the values in force now. The snapshot readings show what the faulty plant
  does with them; they are not necessarily the right values.
- The fault is still present and stays present during the rollout. Setpoints are the only thing you
  can change.
- The SAFE thresholds are fixed and do not move with T_sp: measured temperature at most 313 K, and
  close to 310 K whenever an actuator is saturated; level between 2 and 11.

# What makes a proposal achievable
A proposal passes only if the plant, with the fault still present, can actually settle at it:
1) every setpoint is within its allowed range;
2) at the new operating point every controller can hold its setpoint without its actuator saturating;
3) the resulting temperature and level meet the SAFE thresholds.
Check your proposal against the KG equations and the snapshot numbers before answering. A setpoint the
actuators cannot reach fails, however it was chosen.

# Output format (STRICT)
Return JSON ONLY, with the keys in this order (reason first, then decide the numbers):
{
"reasoning": "<brief: the KG relations used; for each setpoint, keep or change, the direction, and how the size follows from the snapshot>",
"T_sp": <float>,
"L_sp": <float>,
"Fin_sp": <float>
}"""

# v3.1: still no advice on which setpoint to change, in which direction or by how much. Adds the
# plant physics in words (a restatement of the KG equations, stated symmetrically), describes the
# SAFE limits without repeating their numbers (they stay in the unchanged GOAL text), and asks for
# brief reasoning.
SYSTEM_V3_1 = """You are the **Corrective Setpoint Action Agent** for a Continuous Stirred Tank Reactor (CSTR).

A fault has occurred and the plant monitor has flagged it. You propose values for these setpoints ONLY:
- T_sp (temperature setpoint, K)
- L_sp (level setpoint)
- Fin_sp (inlet flow setpoint, L/s)
Use the Knowledge Graph (KG) below and the snapshot numbers to decide which setpoints to change, in
which direction, and by how much. Keeping a setpoint unchanged is also a decision that must be justified.

# How each setpoint acts (KG)
- Fin_sp: the flow controller (PO8) moves the inlet valve (u_valve) so that the feed Fin equals Fin_sp.
  The feed carries the reactants (CA_in, CD_in) into the reactor at temperature Tin.
- L_sp: the level controller (PO6) sets the outlet pump speed (u_pump) so that the level equals L_sp.
  At a steady level the outflow equals the inflow (PO5, dV/dt).
- T_sp: the temperature controller (PO7) opens the cooling valve (u_cool) further when T_meas is above
  T_sp and closes it when T_meas is below. T_sp is the controller's target, not a limit.

# Plant physics (KG, PO5 dT/dt)
- The reactions release heat (-DH1 x R1 - DH2 x R2). Their rate depends on the reactant supplied by the
  feed, so the heat released rises when Fin rises and falls when Fin falls. The feed also brings heat,
  because Tin is above the reactor temperature.
- The coolant removes heat at the rate UA_scaled x (T - Tc). It cannot remove more than the fully open
  cooling valve (u_cool = 1) allows.
- The temperature settles where the heat added equals the heat removed.
- The outlet pump can remove only as much liquid as its capacity allows. If it cannot match the
  inflow, the level rises.

# Facts about the snapshot
- u_valve, u_pump and u_cool are actuator commands as a fraction of full capacity (0 = closed or stopped,
  1 = fully open or full speed). At 0.95 or above an actuator is saturated: its controller has no capacity
  left and cannot hold its setpoint if more is needed.
- "current setpoints" are the values in force now; the readings show what the faulty plant does with them.
- The fault stays present during the rollout. Setpoints are the only thing you can change.

# What makes a proposal achievable
A proposal passes only if the plant, with the fault still present, can settle at it:
1) every setpoint is within its allowed range (see ALLOWED SETPOINT RANGES);
2) at the new operating point every controller can hold its setpoint without its actuator saturating;
3) the measured temperature and level then meet the SAFE conditions in GOAL. These are fixed limits on
   the measurements; they do not move with the setpoints.

# Output format (STRICT)
Return JSON ONLY, with the keys in this order (reason first, then decide the numbers):
{
"reasoning": "<at most 80 words: the KG relations used; for each setpoint, keep or change, the direction, and how the size follows from the snapshot>",
"T_sp": <float>,
"L_sp": <float>,
"Fin_sp": <float>
}"""

SYSTEMS = {"cstr_prompt_v3": SYSTEM_V3, "cstr_prompt_v3.1": SYSTEM_V3_1}

# Default-plant reference values in the KG (`:information_X_sp owl:sameAs :ref_X_sp .` and
# `:ref_X_sp DIN17359:referenceValue v .`). Each is a complete Turtle statement on its own line.
_REF_LINE = re.compile(r"^:(information_\w+_sp owl:sameAs :ref_\w+|ref_\w+ DIN17359:referenceValue \S+) \.\n(\n)?",
                       re.MULTILINE)

FEEDBACK_METRICS = ("worst_zone", "time_to_safe", "unsafe_fraction", "peak_T_meas", "peak_u_cool",
                    "peak_u_valve", "peak_u_pump", "any_invalid_actuators")


def strip_reference_values(kg_section: str) -> str:
    out, n = _REF_LINE.subn("", kg_section)
    if n and n != 6:
        raise ValueError(f"expected 6 reference-value statements in the KG, removed {n}")
    return out


def neutral_feedback(verify_result: Dict[str, Any]) -> str:
    """Verifier outcome only: summary, fail reason, measured rollout metrics (no advice)."""
    m = verify_result.get("metrics", {}) or {}

    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    measured = ", ".join(f"{k}={fmt(m[k])}" for k in FEEDBACK_METRICS if k in m)
    return f"{verify_result.get('summary', '')}; fail_reason={verify_result.get('fail_reason', '')}; measured: {measured}"


def build_messages_v3(snapshot: Snapshot, feedback: Optional[str] = None,
                      previous: Optional[Dict[str, float]] = None,
                      version: str = PROMPT_VERSION) -> List[Dict[str, str]]:
    """As build_messages_v2, with a v3 system prompt (`version`, see SYSTEMS) and the KG reference
    values removed. `feedback` should come from neutral_feedback(). Requires the KG context to be set."""
    m = load_cstr_case()
    snap = _snapshot_block(build_messages(snapshot)[1]["content"])
    (T_lo, T_hi), (L_lo, L_hi), (F_lo, F_hi) = ACTION_BOUNDS["T_sp"], ACTION_BOUNDS["L_sp"], ACTION_BOUNDS["Fin_sp"]
    if feedback is None:
        fb = "EVALUATION FEEDBACK (if any):\n(none)\n"
    else:
        fb = (f"EVALUATION FEEDBACK (if any):\n{feedback}\n"
              f"Previous proposal: {previous}\n"
              "The previous proposal failed validation: you MUST propose a DIFFERENT set of setpoints.\n")
    user = snap + fb + USER_TAIL_V2.format(T_lo=T_lo, T_hi=T_hi, L_lo=L_lo, L_hi=L_hi, F_lo=F_lo, F_hi=F_hi)
    system = SYSTEMS[version] + "\n\n" + strip_reference_values(m.get_action_kg_context() or "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
