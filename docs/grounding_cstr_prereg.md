# Pre-registration: field grounding as a trust signal (CSTR)

**Frozen:** 2026-09-26, before any CSTR grounding value was computed. The FSM analogue
(`docs/grounding_attention_prereg.md`) was positive.

**Idea.** When the model writes the number for a setpoint, does it attend to the parts
of the prompt that should decide that number?

**Data.** First proposals (round 0) of the main CSTR run
(`outputs/cstr_collect/20260924_212031_qwen25-3b-instruct_main3000`). The cert partition
stays sealed until a policy is frozen. The stored answer tokens are re-run teacher-forced;
the prompt-length match is checked.

## Regions (character spans in the rendered prompt)

**Snapshot field lines.** Each `- <field>: ...` line of the CURRENT SNAPSHOT block
(time, phase, T_meas, L_meas, u_valve, u_pump, u_cool, anomaly_ratio, violated_params,
control_zone, control_reasons, current setpoints, symptoms).

**KG relation lines.** Every line of the system prompt or KG Turtle block that mentions:
- PO6 (the level loop);
- PO7 (the temperature loop);
- PO8 or PO1 (the flow loop).

**Decisive regions per setpoint** (fixed in advance from the process structure stated in
the prompt itself):

| Setpoint written | Decisive regions |
|---|---|
| Fin_sp | u_cool, T_meas, current setpoints, PO8/PO1 lines |
| L_sp | L_meas, u_pump, current setpoints, PO6 lines |
| T_sp | T_meas, u_cool, current setpoints, PO7 lines |

## Measurements

**Decision rows.** For each setpoint, the decision rows of its number tokens in the
answer JSON (as in FSM).

**`share`.** Attention mass on the decisive regions ÷ attention mass on all snapshot
lines + all KG-relation lines + the rest of the system/KG text. Template tokens and
previously generated tokens are excluded from the denominator. It is averaged over rows
and heads, and over layers at relative depth 0.25, 0.5, 0.75 and 1.0.

**`snap_share`.** The same quantity for the whole snapshot block.

## Candidate features (prefix `fld_`)

- `fld_Fin_share`, `fld_L_share`, `fld_T_share`;
- `fld_min_share`, the minimum of the three (**primary**);
- `fld_Fin_snap_share`, `fld_L_snap_share`, `fld_T_snap_share`;
- per-layer `fld_L{025,050,075,100}_Fin_share`.

## Hypotheses (direction fixed)

- **H1c.** Lower `fld_min_share` → higher failure probability.
  - Test: AUROC of −fld_min_share on train + dev first proposals.
- **H2c.** Adding the `field_grounding` family to context_action raises AUROC on first proposals.
  - Test: paired ΔAUROC with grouped bootstrap on dev_thr first, then test_iid once.

**Expectation, stated in advance.** CSTR failures are mostly "correct lever, too small a
change". Grounding may therefore help less than in FSM. A null result is informative:
it bounds where grounding works.

## Limitations

- The decisive regions are domain-informed choices, taken from the causal hints the prompt gives.
- Attention is not a full explanation.
