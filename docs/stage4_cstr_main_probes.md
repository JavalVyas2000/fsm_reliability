# Stage 4 — CSTR main dataset: probe results (exploratory; certification partition sealed)

**Date:** 2026-09-25.

**Data:** `outputs/cstr_collect/20260924_212031_qwen25-3b-instruct_main3000`.
- 3000 episodes, 6961 labelled candidates (all rounds).
- Qwen2.5-3B, prompt v2.1, severity ranges v2, CAR reprompt loop.

**Probe runs:**
- `outputs/fsm_baseline/20260925_152013_cstr_main_evalall` (evaluated on all rounds)
- `outputs/fsm_baseline/20260925_152416_cstr_main_evalr0` (evaluated on first proposals)

**Setup:**
- Probes are fit on train, all rounds (3245 candidates from 1400 episodes).
- Calibration uses dev_cal, thresholds dev_thr, reporting test_iid.
- The cert partition (700 episodes) is excluded from fitting and all outputs.

## Headline (first proposals = the deployment and certification population)

Test: 300 episodes, failure prevalence 0.65.

| Family | AUROC dev_thr | AUROC test [95% CI] |
|---|---|---|
| context_action (observable plant state + proposed change) | 0.888 | **0.884 [0.843, 0.919]** |
| context_only_no_action | 0.858 | 0.848 |
| token_confidence | 0.552 | 0.571 [0.504, 0.639] |
| attention | 0.538 | 0.641 [0.576, 0.705] |
| hidden | 0.608 | 0.666 [0.602, 0.725] |
| all_internal | 0.646 | 0.656 [0.594, 0.716] |
| context_action + token_confidence | 0.891 | 0.884 |
| context_action + hidden | 0.896 | 0.874 |
| context_action + all_internal | 0.901 | 0.855 |
| untrained mean log-prob / entropy | 0.52 / 0.52 | 0.57 / 0.57 |

**Paired ΔAUROC vs context_action, test** (grouped bootstrap over episodes):

| Comparison | ΔAUROC [95% CI] |
|---|---|
| context_action + all_internal | −0.029 [−0.059, −0.000] |
| context_action + attention | −0.023 [−0.043, −0.003] |
| context_action + hidden | −0.010 [−0.038, +0.020] |
| context_action + token_confidence | −0.001 [−0.012, +0.011] |

## Findings

1. **More data gives internals a weak signal.** With 10× the pilot data, hidden states and attention now carry a weak but real signal on first proposals (AUROC 0.64–0.67, CIs above 0.5); in the pilot they were at chance.
2. **That signal is redundant with observable features.** Adding internal signals to context_action never helps, and for attention and all_internal it slightly hurts on test. The CIs are tight, so this is a confident negative result on CSTR.
3. **The observable context/action probe is strong** (AUROC 0.88 on first proposals). Selective verification is therefore well-posed: the simulator can be skipped for many candidates, and the decision needs no model internals.
4. **All-rounds evaluation is confounded by the round.** On all rounds, internals reach 0.85–0.87 AUROC, but this is almost entirely round detection:
   - 98% of reprompt candidates fail (100% of repeats) against 65% of first proposals;
   - "is a reprompt" alone gives AUROC 0.79;
   - the internal probes identify reprompts with AUROC 0.98–0.996.

   Within reprompts, the pass class is too rare (about 2%) for AUROC to be reliable. All-rounds numbers must not be used as evidence for internals.
5. **The reprompt loop adds little recovery.** Only 59 of about 2000 first-proposal failures were solved by reprompting (1063 solved episodes against 1004 first-try passes). 1922 episodes ended on an exact repeat.

## Consistency with earlier stages

| Stage | Internals vs observable context/action |
|---|---|
| FSM (4 models) | add value only for Qwen2.5-3B (+0.03 to +0.04 AUROC) |
| CSTR pilot (n = 150 train) | at chance |
| CSTR main (n = 1400 train episodes) | weak alone; no incremental value |

## Next steps (not yet done)

- Routing table with measured time saved on test first proposals, for context_action and the internal families.
- Freeze a declared policy set.
- Certify on the sealed cert partition (700 first proposals).
