# Pre-registration: skip-the-validator under plant and fault shift (CSTR)

**Frozen:** 2026-10-03, before any shift result was computed.

**Question.** When the probes are trained on one part of the plant/fault space and deployed on another, with no recalibration on the new part:
- Which signal still skips validator calls safely?
- Does the observable probe degrade more than the model-internal signals?

The motivating earlier result (`docs/cstr_varied_and_shift_results.md`, prompt v2.1, one model): under a fixed-plant → varied-plant shift the observable probe's AUROC fell from 0.89 to 0.45, while token confidence and hidden states fell only to about 0.59.

## Data

- **Records:** the existing v4 first proposals with features and grounding, 4 models: Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B (4-bit), Llama-3.2-3B. All 5000 episodes per model: train, dev_cal, dev_thr, test_iid and cert, the cert grounding coming from the `_withcert` runs.
- **Status of those sets:** test_iid and cert were each used once for the in-distribution claims. This experiment re-partitions all episodes by plant and fault properties. It is a separate analysis and makes no claim about the earlier sealed sets.
- **Plant and fault properties per episode:** `data/cstr/episodes_v4/cstr_episodes.xlsx`.
- **Unit:** one proposal per episode, so splits are disjoint by episode.

## Shifts

Each defines a *source* (fitting and thresholds) and a *target* (evaluation only):

| Shift | Source | Target | Axis |
|---|---|---|---|
| **feed** | nominal Fin_sp ≤ median | > median | seen in the prompt as the current Fin_sp |
| **cooling** | UA ≤ median | > median | not shown in the prompt |
| **family-k** (×4) | the other 3 fault families | held-out family k | — |
| **control** | random half | the other half | same sizes, no shift |

- **Within the source:** a seeded random split by episode, 60% train, 20% dev_cal (Platt), 20% dev_thr. Thresholds come from dev_cal + dev_thr pooled.
- **Rules:**
  - ACCEPT: the upper-bound rule, δ_sel = 0.05, X = 10% (primary) and 5%.
  - REJECT: the dev rejected set is ≥ 95% failures.
  - Probes as in `scripts/12`: logistic regression, PCA-64 for hidden states, Platt calibration.
- **Signals:** the same 11 as Amendment 3 of `docs/cstr_v4_prereg.md`.

## Outcomes on the target, per model × shift × signal

- AUROC for failure;
- share of validator calls skipped;
- failures among unchecked accepts (count and rate);
- good proposals rejected unchecked;
- paired Δ(calls skipped) against observables, bootstrap 95% CI.

The family shift is reported per family and averaged over the 4 families.

## Hypotheses (directions fixed)

- **S1.** Under shift, the observable probe degrades more than it does in the control split, either in AUROC or in failures among unchecked accepts exceeding X.
- **S2.** On the target, all internals (+ grounding) loses less AUROC than observables, measured as the shift-minus-control difference.
- **S3.** Observables + internals keeps a lower failure rate among unchecked accepts than observables alone on the target, at an equal or larger share of calls skipped.

All results are reported whatever the outcome, and nothing is tuned on target data.
