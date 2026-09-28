import math

from src.cstr.llm_io import context_action, parse_action, region_char_spans

SYSTEM = "You are the agent.\nRules here.\n\n\n## KNOWLEDGE GRAPH DATA (x)\n```turtle\n:a :b :c .\n```"
USER = "\n        CURRENT SNAPSHOT:\n        - time: 2000.0\n        EVALUATION FEEDBACK (if any):\n\n        Now propose. Return ONLY JSON (no markdown).\n        "


def render(msgs):
    return "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n"


def test_regions_cover_sections():
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    r = render(msgs)
    sp = region_char_spans(r, msgs)
    assert r[slice(*sp["kg_context"][0])].startswith("## KNOWLEDGE GRAPH DATA")
    assert r[slice(*sp["system_instruction"][0])].startswith("You are the agent.")
    assert r[slice(*sp["snapshot"][0])].startswith("CURRENT SNAPSHOT:")
    assert "Return ONLY JSON" in r[slice(*sp["user_instruction"][1])]


def test_parse_action_strict():
    ok = parse_action('{"T_sp": 309.5, "L_sp": 10, "Fin_sp": 0.025, "reasoning": "PO8"}')
    assert ok["schema_valid"] == 1 and ok["action"]["L_sp"] == 10.0
    assert parse_action('{"T_sp": "309", "L_sp": 10, "Fin_sp": 0.02}')["format_failure_reason"] == "non_numeric_setpoint"
    assert parse_action('{"T_sp": 309, "L_sp": 10}')["format_failure_reason"] == "missing_setpoint_key"
    assert parse_action('{"T_sp": 309, "L_sp": 12.5, "Fin_sp": 0.02}')["format_failure_reason"] == "out_of_bounds_L_sp"
    assert parse_action('{"T_sp": 309, "L_sp": 10')["format_failure_reason"] == "no_complete_json_object"


def test_context_action_deltas_and_flags():
    st = {"sim_last": {"t": 2100.0, "T_meas": 312.5, "L_meas": 10.1, "u_valve": 0.4, "u_pump": 0.5, "u_cool": 1.0},
          "anomaly_ratio": 0.7, "control_zone": "UNSAFE", "violated_params": "T_meas_error>2.0; entered_UNSAFE(control_zone)",
          "T_sp": 310.0, "L_sp": 10.0, "Fin_sp": 2 / 60}
    f = context_action(st, {"T_sp": 309.0, "L_sp": 10.0, "Fin_sp": 1 / 60})
    assert f["control_zone_code"] == 2.0 and f["viol_temp"] == 1.0 and f["viol_unsafe"] == 1.0 and f["viol_level"] == 0.0
    assert f["dT_sp"] == -1.0 and abs(f["dFin_sp_rel"] + 0.5) < 1e-12
    assert math.isnan(context_action(st, None)["dT_sp"])
