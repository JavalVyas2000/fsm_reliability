# CSTR field grounding results (pre-registered: `docs/grounding_cstr_prereg.md`)

**Date:** 2026-09-26.

**Data:** first proposals of the main CSTR run (Qwen2.5-3B, prompt v2.1, fixed plant,
ranges v2): train 1400 / dev_cal 300 / dev_thr 300 / test_iid 300. The cert partition is
sealed.

**Computation:** teacher-forced re-run; 2300 / 2300 aligned; token agreement 98.6%
(`outputs/cstr_grounding/20260926_181659_...`).

**Probes:** `outputs/fsm_baseline/20260926_184405_cstr_main_fieldgrounding_r0`, trained
on first proposals only, so all families share the same training data.

## H1c: lower grounding → more failures (train + dev first proposals)

| Feature | AUROC of −feature | Mean share (pass / fail) |
|---|---|---|
| `fld_min_share` (primary) | **0.624** | 0.273 / 0.264 |
| `fld_L_share` | 0.629 | 0.295 / 0.280 |
| `fld_Fin_share` | 0.552 | 0.276 / 0.274 |
| `fld_T_share` | 0.470 | 0.319 / 0.321 |

**Weakly supported.** The effect is in the predicted direction but small. It is far smaller than FSM's step-level effect (0.64–0.81, with shares nearly halved on invalid steps).

## H2c: incremental value over the observable probe

AUROC on first proposals:

| Family | dev_thr | test_iid [95% CI] |
|---|---|---|
| context_action | 0.890 | 0.887 [0.847, 0.923] |
| field_grounding | 0.611 | 0.638 [0.575, 0.697] |
| all_internal | 0.698 | 0.677 |
| context_action + field_grounding | 0.890 | 0.881 |
| context_action + all_internal + field_grounding | 0.904 | 0.856 |

**Paired ΔAUROC** (grouped bootstrap):

| Comparison | dev_thr | test_iid |
|---|---|---|
| ctx + field_grounding − ctx | −0.000 [−0.019, +0.018] | −0.007 [−0.027, +0.012] |
| ctx + all internals + field_grounding − ctx | +0.014 [−0.018, +0.043] | −0.031 [−0.065, +0.003] |

**Not supported.** Field grounding adds nothing beyond observable plant and action features on CSTR.

## Why: an interpretable account

When the model writes Fin_sp, this is where its prompt attention goes (layer-mean
share, train first proposals):

| Region | Passing answers | Failing answers |
|---|---|---|
| current setpoints line | 0.204 | 0.201 |
| symptoms line | 0.083 | 0.083 |
| u_valve / u_cool / u_pump | 0.034 / 0.029 / 0.028 | 0.034 / 0.031 / 0.028 |
| KG PO8 (flow loop) | 0.022 | 0.022 |
| T_meas | 0.021 | 0.021 |

- **The model anchors on the current setpoint value in every case.** The current-setpoints line is its most attended region by far.
- **Its attention pattern is the same whether the proposal later passes or fails.** The outcome depends on the plant instance (fault severity) and on whether the model's habitual small correction happens to be enough. Where the model looks cannot tell those cases apart.

## Conclusion across both tasks

| Task | Failure mechanism | Does grounding separate failures? | Added value over observables |
|---|---|---|---|
| FSM (4 models) | consulting the wrong part of the prompt (the wrong adjacency line) | yes: step-level AUROC 0.64–0.81 | **+0.03 to +0.05 AUROC, all 4 models, dev and test** |
| CSTR (Qwen2.5-3B) | consulting the right fields but under-correcting (anchoring on the current value) | barely: 0.62 | none |

**A principled boundary for the paper.** Attention grounding is a useful trust signal
when failures are *retrieval errors*, where the decisive information is in the prompt
and the model did not look at it. It is not useful when failures are *magnitude or
policy errors* made while looking at the right place. The CSTR attention pattern
itself documents the anchoring behaviour behind the failures.
