# Stage 4 — CSTR snapshot pilot (exploratory)

**Date:** 2026-09-24. **Status:** pilot complete. A decision is needed before
scaling (§8).

All numbers are exploratory. The pilot has 300 episodes: train 150,
dev_cal 50, dev_thr 50, test_iid 50.

## 1. What was built (`src/cstr/`, `scripts/20`–`24`)

**Bridge to the CAR code (`car_bridge.py`).** Imports the **unmodified**
ctrl-alt-recover CSTR code: simulator, monitoring and trigger logic, the
action-prompt construction, and the legacy rollout validator.
- No packages were installed for this.
- The LangChain/LangGraph imports are stood in by stubs that raise an error if used.
- The CAR file sha256 hashes are recorded.

**Episodes (`episodes.py`).** Each episode draws its own fault family, continuous
severity, onset time and simulator noise seed. The code then runs the real
monitoring loop up to the first action trigger and snapshots:
- the plant;
- the RNG state;
- the prompt fields.

**Verifier.** The legacy `rollout_validate_setpoints` under deterministic,
isolated RNG. It is tested for:
- replay determinism;
- order independence;
- no side effects;
- known witnesses;
- simulator-error handling (the result becomes unknown, never a pass).

**Prompt capture.** The exact upstream `action_propose` prompt is captured
without calling any LLM. The KG context comes live from GraphDB and is frozen
per dataset; it is identical to the historical trace subgraph.

**Chunked prefill.** The 4.8k-token prompt is prefilled in chunks.
- A single SDPA prefill on this machine took 10 GB and 61 s.
- Chunked prefill takes 6.4 GB and 1.5 s.
- Tested equal to plain generation.

## 2. Severity calibration (no LLM; `outputs/cstr_calibration/20260924_092150_severity_grid`)

Five fault families × 5–7 severities × 3 seeds × 16 grid actions. "Actionable"
means:
- the no-change action fails; and
- some grid action passes.

| Family | Actionable range (v1) |
|---|---|
| fouling_max | 0.6–0.9 |
| pump factor | 0.35–0.2 |
| stuck-closed cooling opening | 0.4–0.1 |
| outlet-block factor | 0.45–0.2 |
| leak | never recoverable on the grid; excluded |

## 3. Proposal behaviour of local models (user question: "is the first report the same via the prompt?")

**The prompts differ in every episode.** Every snapshot produces a distinct
user prompt; only the fixed system prompt is shared.

**The first proposals barely differ.** Results on 30 training snapshots with v1 ranges:

| Model / prompt | Distinct proposals | Pass |
|---|---|---|
| Qwen2.5-3B, reference prompt v1 (greedy) | 8 | 0/30 |
| Llama-3.2-3B, v1 | 8 | 0/30 |
| Qwen2.5-3B, v1, sampled (T = 0.8) | varied | 0/40 |
| Qwen2.5-3B, prompt v2.0 (anchoring wording removed) | 1 (all unchanged) | 0/30 |
| Qwen2.5-3B, **prompt v2.1 (reasoning before numbers)** | 9 | 2/30 |
| Qwen2.5-3B, v2.1, sampled | varied | 2/48 |
| Qwen2.5-7B 4-bit, v2.1 | 14 | 0/30 |

Two findings:
- **v2.0 emitted the numbers before the reasoning.** It copied the current setpoints and only afterwards argued for a feed reduction. v2.1 fixed that ordering.
- **The v1 ranges were too severe for these models.** A no-LLM boundary check showed that the typical LLM move (Fin_sp −10%) passes only at the mild edge of the v1 ranges. The ranges were therefore moved to straddle that boundary (`configs/cstr_severity_ranges_v2.json`; user decision). This is a declared, milder population; the v1 regime is reported separately as one where local models fail.

## 4. Pilot data (v2 ranges, Qwen2.5-3B, prompt v2.1, greedy)

- **Episodes and prompts.** 420 drawn, 415 triggered (5 mild stuck-cooling episodes did not), 300 used. All 300 user prompts are distinct.
- **Proposals.** 300/300 are parseable and admissible, with 35 distinct proposals. 59% are the same move (310, 10, 0.03).
- **Verifier pass rate by partition:**

  | Partition | Pass rate |
  |---|---|
  | train | 0.26 |
  | dev_cal | 0.40 |
  | dev_thr | 0.50 |
  | test_iid | 0.44 |

- **Verifier pass rate by family:**

  | Family | Pass rate |
  |---|---|
  | pump_degrade | 0.17 |
  | fouling | 0.25 |
  | cool_stuck_closed | 0.43 |
  | outlet_block | 0.51 |

- **The proposal matters.** In 63 episodes the proposal passes where no-change would fail; in 4 it is the other way round.
- **Costs:**

  | Item | Cost |
  |---|---|
  | Generation | 10.2 s per proposal (prompt ≈ 4800 tokens, answer ≈ 108 tokens) |
  | Feature pass | 0.27–0.30 s |
  | Verifier, sequential on an idle machine (test_iid, n = 50) | mean 7.6 s, median 8.2 s |

  - The sequential verifier re-timing gave 0 verdict mismatches against the labels.
  - Parallel labelling inflates wall time to about 13.8 s per call.

## 5. Failure prediction (test_iid, n = 50; dev_thr, n = 50)

| Family | AUROC dev_thr | AUROC test [95% CI] |
|---|---|---|
| context_action (observable plant state + proposed deltas) | 0.790 | 0.753 [0.60, 0.88] |
| context_only_no_action | 0.595 | 0.718 |
| token_confidence | 0.566 | 0.498 |
| attention | 0.435 | 0.542 |
| hidden | 0.520 | 0.498 |
| attention+token_confidence | 0.477 | 0.490 |
| context_action + all internal | 0.520 | 0.516 |
| untrained mean log-prob / entropy | 0.63 / 0.65 | 0.57 / 0.58 |

- **Internal signals are at chance on CSTR.** They are significantly worse than context_action on both splits.
- **Adding them to context_action degrades it.** With 150 training rows the extra dimensions cause overfitting.

**Diagnosis.** For the modal proposal (59% of episodes, 45% of which pass), the
internal features of passing and failing cases are indistinguishable:

| Feature | Passing mean | Failing mean | SD across cases |
|---|---|---|---|
| mean token log-prob | −0.352 | −0.364 | 0.045 |
| attention to snapshot, L050 | 0.097 | 0.094 | 0.009 |

The outcome is set by the plant instance, which the observable snapshot
reflects. The model's internal state while writing the answer does not carry
that information.

## 6. Routing and measured time saved (test_iid, n = 50; `outputs/fsm_time_savings/20260924_192419_cstr_pilot`)

Thresholds are chosen on dev_thr and applied to test. The point rule is used.

| Probe, setting | ALLOW / VERIFY / DISALLOW | Calls saved | Failures let through | Valid rejected | Time saved (measured) |
|---|---|---|---|---|---|
| context_action, α = 0, β = 0.10 | 7 / 32 / 11 | 36% | **0** | 2 | **+135 s (2.7 s per candidate, 36% of verification time)** |
| context_action, α = 0.10 | 15 / 24 / 11 | 52% | 5 | 2 | +199 s |
| context_action + all internal, α = 0 | 10 / 40 / 0 | 20% | 6 | 0 | +65 s |
| attention (any setting) | 0 / 50 / 0 | 0% | 0 | 0 | −13 s (overhead only) |

- **The conservative (UCB) rule allows nothing.** 50 dev samples are far too few.
- **Time saving is positive and substantial when the probe is good.** Verification costs 7.6 s against a 0.27 s feature overhead (0.001 s for context_action). But the probe that achieves it uses no model internals.

## 7. Conclusions for the paper

1. **The routing + time accounting idea works on CSTR with real, measured costs.** At zero failures let through on this test set, 36% of 7.6-second simulator calls are skipped.
2. **On CSTR, as for 3 of the 4 FSM models, the value comes from observable context and action features, not from internal signals.** Internals are at chance on this pilot.
3. **The pilot is small.** Train has 150 rows and test has 50. The hidden-state probe (PCA 64) cannot be judged at this size; token and attention probes are already at chance on dev.

## 8. Decision needed

- **(a) Scale CSTR** to about 1500 train / 400 / 400 / 800 episodes (about 5 h GPU plus labelling). This tests whether internals improve with data, and gives certification-sized sets for the context_action policy.
- **(b) Reframe the paper** around certified selective verification with a measured cost saving, where internal signals are one candidate family. The negative finding is reported: they add little beyond cheap observable features on these tasks and models.
- **(c) Do both:** run (a) overnight, then decide the framing from its result.
