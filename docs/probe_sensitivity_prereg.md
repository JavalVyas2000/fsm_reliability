# Pre-registration: sensitivity of the skip-the-validator results to the probe class and to validator-derived observables

**Frozen:** 2026-10-29, before any probe in this analysis was fitted. Written at the author's request (paper revision item 3a).

**Status of the data.** All test and certification sets used here were already examined once in the confirmatory
analyses (`docs/skip_validator_results.md`, paper Tables 4-5). This is therefore a **sensitivity analysis**,
reported as such. It cannot replace the confirmatory results; it tests whether they depend on two design choices
a reviewer can attack.

## Questions

- **Q1 (probe flexibility).** All confirmatory screens use an l2-logistic probe. Does the gain from adding
  internals to observables survive when every signal set gets a flexible nonlinear probe? (Objection: "a linear
  probe handicaps the low-dimensional observables, so the internals' gain is an artefact.")
- **Q2 (validator-derived observables, path planning only).** The path-planning observables include
  `shortest_length` (a BFS on the prompt's graph) and `answer_endpoints_match` (part of the validator's own check).
  How much of the observable screen rests on these, and does the internals' added value change without them?

## Design

**Data.** Exactly the partitions of the confirmatory analysis, per model and domain:
- fit on `train`, Platt calibration on `dev_cal`, thresholds on `dev_cal + dev_thr` pooled;
- evaluate on the path-planning fresh certification set (`cert2`) and on the CSTR `cert` partition;
- first proposals only; eligibility as in `scripts/39`/`40`.

The data directories are those recorded in each confirmatory freeze (`outputs/certification/*_skip_ucb*`). Qwen2.5-3B CSTR stays exploratory.

**Probe classes.**
- `logistic`: the confirmatory pipeline (`scripts/12.make_pipeline`). It must reproduce the confirmatory
  certification results exactly; this is the analysis's sanity check.
- `gbm`: the same preprocessing (median imputation and standardisation for scalars; standardisation and PCA-64 for
  hidden states, fitted on train), followed by `sklearn.ensemble.HistGradientBoostingClassifier` with fixed
  hyperparameters chosen before any fit and **not tuned**:
  `max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
  early_stopping=False, random_state=0`.

**Signal sets.**

| Domain | Set | Features |
|---|---|---|
| both | observables | `context_action` (confirmatory baseline) |
| both | internals only | all internals + grounding |
| both | combined | observables + all internals + grounding |
| path planning | observables without BFS | observables minus `shortest_length` |
| path planning | validator-free observables | observables minus `shortest_length` and `answer_endpoints_match` |
| path planning | validator-free combined | validator-free observables + all internals + grounding |

Each set is fitted with both probe classes. That gives 6 screens in the CSTR and 12 in path planning per model.

**Rules (unchanged from the confirmatory analysis).**
- ACCEPT threshold: bound-based (one-sided Clopper-Pearson, delta_sel = 0.05) at X = 10% (primary) and 5%.
- REJECT threshold: rejected development set >= 95% failures.
- Certification: Bonferroni over the screens of this analysis (K = 6 in the CSTR, 12 in path planning).

**Comparisons** (calls skipped at X = 10%, paired bootstrap over graphs/episodes, 2000 resamples, 95% CI):
- C1: combined − observables, **within** each probe class (Q1; the key comparison is within `gbm`);
- C2: `gbm` − `logistic` for observables and for combined (how much flexibility helps each);
- C3 (path planning): validator-free observables − observables, and observables without BFS − observables (Q2);
- C4 (path planning): validator-free combined − validator-free observables (Q2).

## Expectations (stated in advance; not hypotheses of the confirmatory study)

- E1. In path planning, the internals' gain (C1) stays significantly positive under `gbm` for at least three of
  the five models.
- E2. In the CSTR, `gbm` changes the picture little: observables remain as good as or better than the combination
  for most models.
- E3. Removing the validator-derived features reduces the calls skipped by the path-planning observable screen
  (C3 < 0) and increases the internals' added value (C4 > C1 for logistic).

## Reporting

All screens, models and comparisons are reported whatever the outcome, in a supplementary table and in the
paper's robustness subsection. If E1 fails, the paper's path-planning complementarity claim is weakened
accordingly. Script: `scripts/51_probe_sensitivity.py`. Outputs: `outputs/sensitivity/<ts>_probe_class/`.
