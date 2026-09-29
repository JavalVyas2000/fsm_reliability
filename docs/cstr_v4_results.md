# CSTR v4 results (pre-registered: `docs/cstr_v4_prereg.md`)

## Part A: development data only

Written 2026-09-29, before test_iid or cert were examined.

## Data and runs

| Item | Location / value |
|---|---|
| Episodes | `data/cstr/episodes_v4` (5000). 2961 non-triggering draws were redrawn within family, mostly fouling (61% of first draws unusable) and cool_stuck_closed (46%). |
| Collection | `outputs/cstr_collect/20260928_201301_qwen25-3b-instruct_v4_v31_r0`: 5000 first proposals, 24.6 h, 55 format failures (1.1%), internals extracted for all |
| Grounding | `outputs/cstr_grounding/20260929_204745_...`: 4447 eligible (cert excluded), teacher-forced agreement 98.7% |
| Probes | `outputs/fsm_baseline/20260929_212511_cstr_v4_v31_dev`, run with `--seal_test`, so no test_iid metric or prediction was computed |
| H2 / H3 | `dev_analysis_h2_h3.json` in the same folder (`scripts/37`) |

Outcome rates on the open partitions (train + dev_cal + dev_thr):

| Quantity | Value |
|---|---|
| First-proposal pass rate | 29.7% (train 30.2%, dev_cal 27.2%, dev_thr 29.6%) |
| No-change passes | 34% of episodes |
| Model pass where no-change fails (n = 2611) | 10.4% |
| Model pass where no-change passes (n = 1343) | 67.3% |

## H1 (primary): do internals add to the observable probe? Not supported on dev_thr

AUROC for failure, dev_thr, n = 493:

| Probe | AUROC [95% CI] |
|---|---|
| context_action (baseline) | **0.881** [0.848, 0.912] |
| context_only_no_action | 0.768 |
| token_confidence | 0.565 |
| attention | 0.729 |
| hidden | 0.715 |
| all_internal | 0.742 |
| grounding_v31 | 0.686 |
| grounding_v31_primary_only (g31_min_share) | 0.490 |
| all_internal + grounding_v31 (P4) | 0.751 |
| context_action + grounding_v31 (P3) | 0.879 |
| **context_action + all_internal + grounding_v31 (P2, primary)** | **0.879** |

Paired ΔAUROC against context_action, grouped bootstrap:

| Comparison | ΔAUROC [95% CI] |
|---|---|
| **P2 (primary)** | **−0.002 [−0.019, +0.015]** |
| context_action + token_confidence | +0.002 [−0.002, +0.007] |
| context_action + attention | +0.008 [−0.002, +0.018] |
| context_action + hidden | +0.000 [−0.015, +0.016] |
| context_action + all_internal | +0.001 [−0.017, +0.018] |
| context_action + grounding_v31 | −0.001 [−0.007, +0.005] |

The internals carry real signal on their own: 0.72–0.75 for attention, hidden and all_internal, far above chance. But that signal is already contained in the observable plant readings plus the proposed action.

## H2 (mechanism): no stratum shows a gain

Paired ΔAUROC against context_action, dev_thr:

| Stratum | n | Failure rate | AUROC ctx | P2 − ctx | ctx+hidden − ctx | ctx+grounding − ctx |
|---|---|---|---|---|---|---|
| no-change fails | 326 | 0.91 | 0.812 | +0.015 [−0.035, +0.059] | +0.014 [−0.031, +0.057] | +0.000 [−0.014, +0.014] |
| no-change passes | 167 | 0.29 | **0.929** | +0.007 [−0.016, +0.033] | +0.003 [−0.019, +0.026] | +0.003 [−0.010, +0.017] |

The predicted larger gain where no-change passes did not appear. In that stratum the observable probe is already very strong (0.93): whether the model's own change breaks a recovering plant is largely readable from the change itself together with the plant state.

## H3 (grounding): opposite to the prediction

Proposals that raise Fin_sp while u_cool ≥ 0.95 and T_meas > 310 K:
- 339 of 3954 open proposals;
- 5.9% of them pass.

| Population | AUROC of −g31_Fin_physics_share | Mean physics share (against / other) |
|---|---|---|
| All proposals (pre-registered) | **0.310** | 0.0169 / 0.0157 |
| Failing proposals only (sensitivity) | 0.316 | 0.0169 / 0.0158 |

The prediction was that these proposals would show *less* attention to the physics section. They show slightly *more*: AUROC below 0.5 means the effect runs in the opposite direction. The model attends to the relevant physics and still moves the feed the wrong way. As in the v2.1 CSTR result, where the model looked leads to where it should, and the error is in the reasoning, not the retrieval.

## Interpretation (development data)

- Neutral prompt v3.1 removed the prompt-induced anchoring, but the CSTR picture did not change. In distribution, internals are informative but redundant given observables, because the relevant information is in the plant readings and the proposed action.
- This is consistent with the FSM–CSTR boundary in `docs/grounding_cstr_results.md`:
  - attention grounding helps when failures are retrieval errors (FSM);
  - it does not help when the model reads the right place and reasons wrongly (CSTR).

  H3 adds direct evidence for the second case with a neutral prompt.
- The earlier plant-shift result (`docs/cstr_varied_and_shift_results.md`) remains the setting where internals beat observables.

## Next (per the pre-registration)

1. **test_iid, examined once:** H1–H3 with the probes already fitted.
2. **Freeze P1–P4** on dev_thr (α = 0 threshold), then certify on cert (H4).
