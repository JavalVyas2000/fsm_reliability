"""
CSTR prompt v2 (user decision 2026-09-24).

Motivation: with the reference prompt (car_action_propose_v1) Qwen2.5-3B and Llama-3.2-3B
first proposals stay at or next to the current setpoints (0/30 greedy, 0/40 sampled pass).
v2 tests whether the wording that anchors to tiny moves is responsible.

Kept from v1 (taken from the upstream code, not retyped where possible):
  - agent role, the KG usage rules and causal hints (PO5-PO8), JSON output format;
  - the KG section (upstream get_action_kg_context());
  - the CURRENT SNAPSHOT block, cut verbatim from the captured v1 user prompt.
Removed: "Prefer SMALL changes ..." and the 0.05 / 0.005 grid-step rule.
Added: a plain statement of the validation goal and the admissible setpoint ranges
(the same bounds the cheap admissibility check enforces).
v2.1: "reasoning" is requested BEFORE the setpoints. In v2.0 the model emitted the numbers
first (copying the current setpoints in 30/30 cases) and only then reasoned that Fin_sp
should be reduced.
No privileged information (fault family, severity, onset) is added.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.cstr.car_bridge import load_cstr_case
from src.cstr.episodes import ACTION_BOUNDS, Snapshot, build_messages

PROMPT_VERSION = "cstr_prompt_v2.1"

SYSTEM_V2 = """You are the **Corrective Setpoint Action Agent** for a Continuous Stirred Tank Reactor (CSTR).

You must propose ONLY these setpoints:
- T_sp (temperature setpoint)
- L_sp (level setpoint)
- Fin_sp (inlet flow setpoint)

You will be given a Knowledge Graph (KG) Turtle snippet. You MUST use it.

# How to use the KG (MANDATORY)
From the KG, infer these causal links (examples from the KG you will see):
- PO7_TempPID: reverse-acting temperature control:
"e = T_meas - T_sp (reverse-acting)" → if T_meas > T_sp then controller increases cooling command u_cool_cmd.
- PO8_FlowPID: Fin_sp affects u_valve_cmd → changes Fin through PO1_InletFlow.
- PO6_LevelPID: L_sp affects u_pump_cmd → changes Fout through PO2_OutletFlow.
- PO5_StateDerivatives: dV/dt = Fin - Fout - F_leak - F_overflow and dT/dt depends on Fin and cooling.

So:
- If cooling is saturated (u_cool near 1.0) and temperature is high, you cannot "cool harder".
You must reduce heat input / load, typically by reducing Fin_sp (via PO8 -> u_valve_cmd -> Fin).
- If level is drifting out of safe bounds, adjust L_sp (via PO6 -> u_pump_cmd -> Fout) and/or reduce Fin_sp.

# Output format (STRICT)
Return JSON ONLY, with the keys in this order (reason first, then decide the numbers):
{
"reasoning": "<brief kg-grounded reasoning, including how much each setpoint should change>",
"T_sp": <float>,
"L_sp": <float>,
"Fin_sp": <float>
}

# Rules
1) The setpoints you return are held constant while a digital twin of the plant is simulated.
2) Choose the size of each change from the snapshot: the change must be large enough to remove the cause of the fault symptoms.
3) If u_cool/u_valve/u_pump is near saturation (>=0.95), avoid pushing it further.
4) If u_cool is saturated, prefer reducing Fin_sp before changing T_sp.
5) State which KG relationship justifies each change (cite PO6/PO7/PO8 in reasoning)."""

USER_TAIL_V2 = """
GOAL:
In the digital-twin rollout the plant must reach the SAFE state within 600 seconds and then stay SAFE for at least 60 consecutive seconds. SAFE means measured temperature at most 313 K, level between 2 and 11, no overflow, and no actuator saturated while temperature is away from 310 K. Afterwards the plant must remain safe for most of the remaining run.

ALLOWED SETPOINT RANGES:
- T_sp: {T_lo} to {T_hi} K
- L_sp: {L_lo} to {L_hi}
- Fin_sp: {F_lo} to {F_hi} L/s

Now propose setpoints. Return ONLY JSON (no markdown).
"""


def _snapshot_block(v1_user: str) -> str:
    start = v1_user.index("CURRENT SNAPSHOT:")
    end = v1_user.index("EVALUATION FEEDBACK")
    return v1_user[start:end].rstrip() + "\n"


def build_messages_v2(snapshot: Snapshot, feedback: Optional[str] = None,
                      previous: Optional[Dict[str, float]] = None) -> List[Dict[str, str]]:
    """
    First proposal: feedback=None. Reprompt rounds: `feedback` is the upstream reprompting
    text for the failed verification and `previous` the previous proposal; as in v1, the
    model must then propose a different setpoint triple.
    Requires the KG context to be set (episodes.set_kg_context).
    """
    m = load_cstr_case()
    v1 = build_messages(snapshot)
    snap = _snapshot_block(v1[1]["content"])
    (T_lo, T_hi), (L_lo, L_hi), (F_lo, F_hi) = ACTION_BOUNDS["T_sp"], ACTION_BOUNDS["L_sp"], ACTION_BOUNDS["Fin_sp"]
    if feedback is None:
        fb = "EVALUATION FEEDBACK (if any):\n(none)\n"
    else:
        fb = (f"EVALUATION FEEDBACK (if any):\n{feedback}\n"
              f"Previous proposal: {previous}\n"
              "The previous proposal failed validation: you MUST propose a DIFFERENT set of setpoints.\n")
    user = (snap + fb
            + USER_TAIL_V2.format(T_lo=T_lo, T_hi=T_hi, L_lo=L_lo, L_hi=L_hi, F_lo=F_lo, F_hi=F_hi))
    system = SYSTEM_V2 + "\n\n" + (m.get_action_kg_context() or "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
