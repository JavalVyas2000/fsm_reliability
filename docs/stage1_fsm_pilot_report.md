# Stage 1 — FSM pilot report (exploratory)

**Date:** 2026-09-23. **Protocol:** v1.0 (`docs/selective_verification_protocol.md`).

All numbers are from the pilot. The pilot `test_iid` split is **exploratory** and
is never reused as confirmatory data.

## Artefacts

| What | Path |
|---|---|
| Dataset (graph-disjoint, root seed 20260923) | `data/v2/fsm_pilot_seed20260923/` (+ `dataset_manifest.json`) |
| Inference records, hidden vectors, manifest | `outputs/fsm_inference/20260923_205804_qwen25-3b-instruct_pilot3000/` |
| Probe fits, metrics, predictions, audits, plots | `outputs/fsm_baseline/20260923_221755_qwen25_3b_pilot/` |
| Timing subset (200) and token-only timing (50) | `outputs/fsm_inference/*timing200`, `*timing50_noint` |
| Smoke run (5) | `outputs/fsm_inference_smoke/` |

**Code:**

| Module | Purpose |
|---|---|
| `src/data/fsm_dataset_v2.py` | dataset generator |
| `src/prompts/fsm_prompts_v2.py` | prompt v2 |
| `src/evaluation/fsm_labels_v2.py` | answer labelling |
| `src/models/inference_v2.py` | inference and internals extraction |
| `src/features/context_action.py` | context/action features |
| `src/evaluation/metrics_v2.py` | metrics |
| `src/utils/manifest.py` | run manifests |
| `scripts/10_generate_fsm_v2.py`, `scripts/11_run_fsm_inference_v2.py`, `scripts/12_fit_fsm_baseline.py` | entry points |

`tests/` holds 29 tests, all passing.

**Commands:**
```
python -m scripts.10_generate_fsm_v2 --root_seed 20260923 --tag fsm_pilot --sizes train=1500 dev_cal=500 dev_thr=500 test_iid=500
python -m scripts.11_run_fsm_inference_v2 --dataset_dir data/v2/fsm_pilot_seed20260923 --tag pilot3000 --local_files_only
python -m scripts.12_fit_fsm_baseline --inference_dir outputs/fsm_inference/20260923_205804_qwen25-3b-instruct_pilot3000 --tag qwen25_3b_pilot
python -m pytest tests
```

**Run setup:**
- Model: Qwen2.5-3B-Instruct in bf16 on an RTX 4060 Laptop GPU (8 GB).
- Decoding: greedy, `repetition_penalty=1.0`, stop at the first JSON object, `max_new_tokens=96`.

## Checks performed

**Split integrity:**
- Pairwise graph overlap between partitions: 0.
- Overlap with the 12,989 graphs in existing `data/raw*`: 0.
- Instance ids are unique.

**Determinism:**
- The 200 instances shared with the timing run gave identical generations and identical features (max diff 0.0).

**Alignment and extraction** (test suite, fp32, Qwen2.5-1.5B):
- The answer span ends exactly at the token that closes the JSON object.
- Decision-pass attention matches a full eager pass (≤ 1e-3).
- Region masses sum to 1.
- Greedy decoding picks the argmax of the raw logits for 100% of answer tokens.

**bf16 caveat:**
- Generation-time and teacher-forced selected log-probs differ by median 0.073 (p95 0.29).
- Features use the teacher-forced pass consistently.

## Data

| Partition | n | Format failures | Eligible | Failure prevalence | Suboptimal-valid |
|---|---|---|---|---|---|
| train | 1500 | 3 (unclosed JSON) | 1497 | 0.524 | 0.214 |
| dev_cal | 500 | 3 | 497 | 0.547 | 0.221 |
| dev_thr | 500 | 0 | 500 | 0.536 | 0.212 |
| test_iid | 500 | 0 | 500 | 0.486 | 0.240 |

Failure prevalence rises with graph size. Train: 5 nodes 0.27, 10 nodes 0.43, 15 nodes 0.68, 20 nodes 0.72.

No feature column has missing values on eligible rows.

## Cost (per candidate, n = 3000)

| Step | Time |
|---|---|
| Generation | mean 1.45 s (median 1.27 s, p95 2.85 s) |
| Feature pass (token + attention + hidden) | mean 0.090 s (p95 0.11 s) |
| Token-only feature pass (50-instance timing run) | 0.070 s |

- The incremental cost of attention plus hidden states is about 0.02 s.
- Peak GPU memory: 6.67 GB.
- The FSM verifier cost is negligible, so no FSM runtime saving is claimed (protocol section 1).

## Predictive results

**Target and setup:**
- Target: `candidate_invalid`; positive class = failure.
- Model: logistic regression fit on train. Hidden states pass through train-only PCA(64).
- Calibration: Platt, fit on dev_cal.

**Per-family metrics:**

| Family | AUROC dev_thr | AUROC test_iid [95% CI] | AUPRC test | Brier test (Platt) | ECE test (Platt, 15 bins) |
|---|---|---|---|---|---|
| context_action | 0.800 | 0.811 [0.771, 0.846] | 0.813 | 0.182 | 0.072 |
| context_action_noBFS | 0.802 | 0.812 | 0.813 | 0.181 | 0.076 |
| token_confidence | 0.762 | 0.793 [0.753, 0.830] | 0.787 | 0.187 | 0.079 |
| attention | 0.807 | 0.828 [0.793, 0.860] | 0.821 | 0.174 | 0.085 |
| hidden | 0.819 | 0.817 [0.780, 0.851] | 0.803 | 0.179 | 0.075 |
| attention+token_confidence | 0.818 | 0.843 [0.810, 0.875] | 0.838 | 0.166 | 0.072 |
| all_internal | 0.835 | 0.840 [0.805, 0.871] | 0.814 | 0.167 | 0.079 |
| context_action+all_internal | 0.838 | 0.849 [0.816, 0.880] | 0.838 | 0.164 | 0.074 |

**Untrained single scores (test AUROC):**
- negative mean selected log-prob: 0.748
- mean token entropy: 0.758

**Paired ΔAUROC on test_iid** (bootstrap, 2000 resamples, grouped by graph):

| Comparison | ΔAUROC [95% CI] |
|---|---|
| attention − context_action | +0.017 [−0.002, +0.037] |
| attention+token_confidence − context_action | **+0.032 [+0.010, +0.056]** |
| all_internal − context_action | **+0.029 [+0.002, +0.056]** |
| context_action+attention − context_action | **+0.023 [+0.006, +0.042]** |
| context_action+all_internal − context_action | **+0.038 [+0.013, +0.065]** (dev_thr: +0.037 [+0.014, +0.062]) |
| attention − token_confidence | **+0.035 [+0.004, +0.068]** |

## Stage 1 gate

**Passed, with a modest margin.** Internal families improve over the
`context_action` baseline, and the grouped 95% CIs exclude 0 on test_iid:
- attention+token_confidence;
- all_internal;
- context_action + attention;
- context_action + all_internal.

The context_action + all_internal gain replicates on dev_thr.

Other findings:
- Attention beats token confidence (+0.035).
- Attention alone is not significantly better than context_action.
- `shortest_length` (the BFS feature) adds nothing beyond the other context features: noBFS − full = +0.001.

## Implication for certification (descriptive, dev_thr, in-sample threshold)

At ~53% failure prevalence, the largest low-risk prefix of dev_thr sorted by predicted risk covers:

| Family | risk ≤ 0.05 | risk ≤ 0.10 | risk ≤ 0.20 |
|---|---|---|---|
| context_action | 0.12 | 0.19 | 0.30 |
| attention+token_confidence | 0.14 | 0.17 | 0.31 |
| context_action+all_internal | 0.17 | 0.25 | 0.37 |

- These figures are optimistic, because the threshold is chosen in-sample.
- Certification margins would reduce them further.
- At this prevalence, α = 0.05 would leave little bypass coverage, and certifying it would require a large cert set.

## Limitations

- One model, one prompt, one pilot sample. The effect sizes are small (+0.02 to +0.04 AUROC).
- Probe hyperparameters were fixed (C = 1, PCA 64) and not tuned.
- ECE on n = 500 with 15 bins has a noise floor of a few hundredths.
- bf16 numerics (see Checks performed).
- The format-failure reprompt was not run (offline pilot; recorded in the deviations log).

## Decision needed before Stage 2

The protocol permits **one** adjustment of the FSM task distribution, using pilot data, towards failure prevalence of 10–40%. Options:

- **(a) Keep {5, 10, 15, 20}.** Prevalence stays around 0.5.
- **(b) Drop 20 nodes, or replace it with 12 nodes.** Using pilot train rates, {5, 10, 15} gives about 0.46, and {5, 8, 10, 12} gives roughly 0.35–0.40. The 8- and 12-node rates are not measured and would be interpolated.
- **(c) Keep node counts but raise edge probability.** This makes shorter paths more likely; its effect is not measured.
