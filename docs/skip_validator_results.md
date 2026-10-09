# How often can we skip the validator? Results per signal (FSM, 4 models; CSTR, Qwen2.5-3B)

**Date:** 2026-09-29.
**Pre-registration:** `docs/cstr_v4_prereg.md`, Amendment 2. The same rules are applied to FSM.
**Summary table:** `outputs/cross_model/20260929_220058_skip_validator_summary/` (`skip_validator_summary.md`, `skip_validator_all.csv`).

## Method

Per model:
- **Probes:** fitted on train, Platt calibration on dev_cal (`scripts/12` pipeline).
- **Rules** (thresholds frozen on dev_cal + dev_thr pooled; `scripts/39`):
  - ACCEPT (execute without validation): the largest risk threshold whose dev accepted set has at most X failures, with **X = 10% (primary)** and 5%.
  - REJECT (send back to the model without validation): the smallest threshold whose dev rejected set is at least 95% failures.
  - Everything else is validated.
- **Evaluation:** once on test_iid and once on cert (`scripts/40`). The certified bound is a one-sided Clopper–Pearson bound on the failure rate among ACCEPTs, Bonferroni over the signals.

**Status of the evaluation sets:**
- **CSTR:** test_iid and cert were sealed until these runs.
- **FSM:** test_iid and cert had been used in earlier FSM analyses (`docs/stage3_certification_report.md`, `docs/grounding_results.md`). The FSM numbers here are therefore descriptive, not confirmatory.

## Share of validator calls skipped (cert, X = 10%)

Each cell is calls skipped (failures executed unchecked / accepted). * = certified at 10%.

| Signal | FSM Qwen2.5-3B | FSM Llama-3.2-3B | FSM Qwen2.5-1.5B | FSM SmolLM2-1.7B | CSTR Qwen2.5-3B |
|---|---|---|---|---|---|
| **Observables** (task context + proposed action) | 23% (42/417) | 43% (39/353) | 42% (23/322) | 85% (28/204) | **56%** (11/76) |
| Token confidence | 19% | 19% | 17% | 57% | 0% |
| Attention by prompt region | 25% | 27% | 38% | 73% | 15% |
| Region-based attention grounding | 19% | 37% | 36%* | 81% | 10% |
| Hidden states | 25% | 34% | 34% | 74% | 13% |
| All internals | 32% | 38% | 30% | 75% | 15% |
| All internals + grounding | 31% | 41% | 41%* | 80% | 12% |
| **Observables + all internals + grounding** | **41%** (93/814) | **66%** (67/763) | **44%*** (19/381) | **95%** (28/273) | 54% (10/65) |

Change in calls skipped against observables, test_iid (bootstrap 95% CI, points):

| Signal | FSM Qwen2.5-3B | FSM Llama-3.2-3B | FSM Qwen2.5-1.5B | FSM SmolLM2-1.7B | CSTR Qwen2.5-3B |
|---|---|---|---|---|---|
| Observables + all internals + grounding | **+15.4** [+11.4, +19.4] | **+22.4** [+17.6, +27.2] | +0.8 [−3.4, +4.8] | **+8.9** [+5.9, +12.0] | −0.6 [−3.9, +3.0] |
| All internals (alone) | +7.8 [+3.6, +12.0] | −6.8 | −12.9 | −8.7 | −41.8 |
| Region-based grounding (alone) | −4.6 | −4.0 | −5.8 | −2.8 | −44.8 |
| Token confidence (alone) | −0.6 | −24.6 | −25.9 | −28.6 | −56.2 |

At X = 5% the ranking is the same. The only certified signal at 5% is observables + internals + grounding for FSM Qwen2.5-1.5B (40% skipped). Full tables are in the summary file.

## Time, not just calls (CSTR)

Measured per proposal, Qwen2.5-3B, 8 GB laptop GPU:

| Item | Cost |
|---|---|
| Generation | 17.2 s mean |
| Validator (rollout) | 10.1 s median wall time, 4 parallel workers |
| Internals pass | 0.41 s mean |
| Grounding pass | ≈ 0.47 s |
| Probe scoring | milliseconds |

**Net time saved per proposal = (share skipped × 10.1 s) − signal cost** (cert, X = 10%):

| Signal | Net time saved per proposal | Share of verification time |
|---|---|---|
| Observables | **+5.7 s** | 56% |
| Observables + all internals + grounding | +4.6 s | 46% |
| All internals | +1.1 s | 11% |
| Region-based grounding | +0.6 s | 5% |

On CSTR the internals cost more time than they save beyond observables.

**FSM:** the validator is a path check costing microseconds, so no signal saves time. FSM shows only which signal routes better, as the research plan anticipates.

## Verdict per signal

- **Token confidence:** the weakest signal everywhere. It saves nothing on CSTR, and 17–19% on FSM for three models (57% for SmolLM2, where most proposals fail).
- **Coarse attention, hidden states, all internals (alone):** at or below observables on FSM (except Qwen2.5-3B, where all internals is +8 points). On CSTR they save only 10–15%, by rejecting obvious failures. They never identify proposals that are safe to execute.
- **Region-based attention grounding (alone):** below observables in every setting.
- **Internals and grounding added to observables:**
  - **FSM:** the best signal for 3 of 4 models, +9 to +22 points of calls skipped on test_iid (41–95% on cert).
  - **CSTR:** no gain.

  This matches the mechanism result. In FSM, failures are retrieval errors: the model did not attend to the adjacency line that decides the step. In CSTR the model attends to the right information and reasons wrongly; the pre-registered H3 was reversed.
- **Safety of accepting unchecked:**
  - The dev-chosen thresholds are slightly too loose. On cert the accepted sets often fail at 10–15%, above the 10% target.
  - Only FSM Qwen2.5-1.5B certifies.

  A bound-based threshold rule would make the accept decision certifiable at the cost of fewer accepts: choose the largest threshold whose *upper confidence bound* on dev is ≤ X, not its point estimate. It must be tested on fresh data (next CSTR model, or a new FSM certification set).

## Limitations and open items

- **One model on CSTR:** Llama-3.2-3B is being collected on the same episodes.
- **Isolating grounding's contribution:** "observables + grounding" and "observables + all internals" are not in this table. They separate grounding's contribution from the other internals within the combined probe. The earlier FSM study had them (`docs/grounding_results.md`: grounding added +7 to +15 points certified for Llama and Qwen2.5-1.5B).
- **Rejecting without validation** saves most calls on CSTR. Its effect on recovery (the extra retries) needs the closed-loop experiment.
- **Validator timing** is wall time with 4 parallel workers. A sequential re-timing earlier gave 6.95 s per call.

## Update 2026-10-02: confirmatory FSM (fresh cert2) and more CSTR models

Rules as in `docs/cstr_v4_prereg.md` Amendment 3: upper-bound ACCEPT rule (δ_sel = 0.05), 10 FSM / 11 CSTR signals, Bonferroni.

### FSM, fresh certification set

- **Data:** `data/v2/fsm_cert2_seed20260930`, 3000 graphs per model, never used before.
- **Probes and rules:** frozen on the FSM pilot train/dev.
- **Runs:** `outputs/certification/*_fsm_*_skip_ucb_cert2`.

Calls skipped at X = 10%. Bracketed: Δ against observables in points, paired bootstrap 95% CI. Accepted counts are failures/accepted. CERT = certified at 10%.

| Signal | Qwen2.5-3B | Llama-3.2-3B | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|---|---|
| **Observables (context + path)** | 21% (15/335 CERT) | 43% (26/333) | 40% (7/284 CERT) | 85% (15/166) |
| Observables + all internals + grounding | **34%** +14.0 [+12.4, +15.5] CERT | **58%** +15.0 [+13.2, +16.9] CERT | 40% −0.4 [−2.0, +1.3] CERT | **91%** +6.2 [+4.8, +7.5] |
| Observables + grounding | 21% +0.2 [−0.8, +1.3] CERT | **59%** +15.5 [+13.8, +17.3] CERT | **46%** +5.1 [+3.7, +6.6] CERT (1/228) | 85% +0.6 |
| Observables + all internals | 32% +11.2 [+9.7, +12.6] CERT | 47% +3.7 [+2.2, +5.3] CERT | 37% −3.7 | 79% −5.6 |
| Internals alone (best of hidden / attention / all) | 21–25% | 17–36% | 22–38% | 74–79% |
| Token confidence | 18% | 14% | 13% | 56% |

At X = 5% the pattern holds:
- Llama, observables + internals + grounding: +13.1 points, CERT, 3/329 failures among accepts.
- Qwen2.5-1.5B, observables + grounding: +7.3 points, CERT, 0/204.

**Confirmed on fresh data.** In FSM, internals added to observables skip more validator calls at a certified failure rate for 3 of 4 models, by +6 to +15 points:
- **Region-based grounding** carries the whole gain for Llama-3.2-3B and Qwen2.5-1.5B.
- **Hidden states and attention** carry it for Qwen2.5-3B.
- **SmolLM2:** the gain is not certifiable, because too few of its proposals pass to build a large enough accept set.

### CSTR, further models (test_iid evaluated once; cert where noted)

| Model (pass rate) | Observables | + all internals + grounding | + all internals | + grounding | Internals alone |
|---|---|---|---|---|---|
| Qwen2.5-3B (30%), exploratory UCB re-analysis | 41–44%, 0 accepts | 50–52%, 43–49 accepts, 8–12% failures | 51–54% | 39–40% | 12–15% |
| Llama-3.2-3B (17%), cert | 70.7%, 16/79 good rejected | 74.1%, 10/79 good rejected | 71.5%, **8/79** (Δ −8 [−15, −2]) | 74.9%, 16/79 | 33–53% |
| Qwen2.5-1.5B (34%), test_iid | **63.8%**, 108 accepts, 8% failures | 59.9% (−3.9 [−6.9, −0.6]) | 62.1% | 64.6% | 28–32% |

**CSTR across 3 models:**
- Grounding never helps.
- Internals help on one side of the decision for some models:
  - Qwen2.5-3B, accepting: internals make unchecked accepts possible at all;
  - Llama, rejecting: half as many good proposals rejected.
- For Qwen2.5-1.5B observables are sufficient. Even the plant readings alone, without the proposed change, skip 63%.

**Still to come:** Qwen2.5-7B (collecting), Qwen2.5-1.5B cert (needs a GPU gap for cert grounding), the closed loop.

## Update 2026-10-03: all CSTR models complete

CSTR, calls skipped at X = 10%. Cert unless noted; Δ against observables, paired bootstrap 95% CI.

| Model (failure rate) | Observables | + all internals + grounding | + all internals | + grounding | Internals alone | Token confidence |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (66%) | 62.8% (14/121 accepted) | 60.6% (−2.2 [−5.5, +1.0]) | 61.4% (−1.4) | 60.0% (−2.8 [−5.3, −0.4]) | 27–30% | 4% |
| Qwen2.5-3B (67%), exploratory UCB | 41–44% (no accepts) | 50–52% (+8 to +10) | 51–54% | 39–40% | 12–15% | 0% |
| Llama-3.2-3B (83%) | 70.7% | 74.1%; good rejected 16 → 10 | 71.5%; good rejected 16 → **8** [−15, −2] | 74.9% | 33–53% | 34% |
| Qwen2.5-7B, 4-bit (80%) | **85.5%** (2/54 accepted; test_iid 0/58, certified) | 79.9% (−5.6 [−8.6, −2.6]) | 82.7% (−2.8) | 85.7% (+0.2) | 47–50% | 17% |

**CSTR across 4 models:**
- Region-based grounding never adds to the observables.
- Internals add for 2 of 4 models, on one side of the decision each:
  - Qwen2.5-3B: they make unchecked accepts possible;
  - Llama: they halve the good proposals wrongly rejected.
- For Qwen2.5-1.5B and Qwen2.5-7B the observables are as good or better.

**FSM (fresh cert2):** internals + grounding add +6 to +15 points for 3 of 4 models, certified. Grounding alone carries the gain for Llama and Qwen2.5-1.5B.

The cross-domain boundary therefore holds across all models: grounding helps where failures are retrieval errors, and not where they are reasoning errors.

## Update 2026-10-09: SmolLM2-1.7B on CSTR (prereg Amendment 4)

The pilot gate was met (38% pass, 0 format failures). The full collection survived two simulator-worker crashes through automatic resume, with no records lost. 5000 proposals, 35% pass. Freeze `outputs/certification/20261009_153848_cstr_v4_smollm2-17b_skip_ucb`; test_iid and cert each evaluated once.

Calls skipped at X = 10%; Δ against observables with paired bootstrap 95% CI:

| Signal | test_iid | cert |
|---|---|---|
| Observables | 61.5% | 61.1% (14/128 accepted were failures) |
| Obs. + all internals | 66.7% (+5.2 [+2.0, +8.6]) | 65.9% (+4.8 [+1.4, +8.3]) |
| Obs. + grounding | 67.1% (+5.6 [+3.4, +7.8]) | 64.9% (+3.8 [+1.4, +6.2]) |
| Obs. + internals + grounding | 65.9% (+4.4 [+1.2, +7.8]) | 65.7% (+4.6 [+1.2, +8.3]) |
| All internals alone | 41.7% | 39.9% (1/37 accepted failed) |
| Region grounding alone | — | 24.4% (0 good rejected) |

**This revises the CSTR summary.** SmolLM2 is the one CSTR model where internals, and grounding alone, add significantly to the observables. CSTR now stands at: grounding adds for **1 of 5** models (SmolLM2); internals add for **3 of 5** (Qwen2.5-3B accept side, Llama reject side, SmolLM2 overall).

The cross-domain contrast still holds in degree, not absolutely:
- **FSM:** internals and grounding add for 4 of 5 models (all but Qwen2.5-7B), +6 to +15 points;
- **CSTR:** they add +4 to +6 points for one model and little or nothing for the rest.
