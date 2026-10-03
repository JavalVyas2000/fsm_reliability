# CSTR under plant / fault shift: results (pre-registered: `docs/cstr_shift_prereg.md`)

**Date:** 2026-10-03.
**Run:** `outputs/cstr_shift_v4/20261003_180425_shift` (`scripts/45`).
**Tables:** `paper_outputs/tables/shift.md`, `shift_all.csv`.
**Figure:** `paper_outputs/figures/shift_auroc_change`.

**Setup:**
- 4 models, about 4950 first proposals each.
- Probes and thresholds fitted on the source part only, evaluated once on the target.
- Shifts:
  - **feed:** low → high nominal feed;
  - **cooling:** low → high UA;
  - **family:** each fault family held out in turn;
  - **control:** a random half, for comparison.

## AUROC for failure (target)

| Model | Signal | Control | Feed | Cooling | Family (pooled) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | Observables | 0.927 | 0.705 | 0.912 | 0.722 |
| | Obs. + all internals | 0.919 | 0.724 | 0.901 | 0.740 |
| | All internals | 0.802 | 0.644 | 0.787 | 0.659 |
| Qwen2.5-3B | Observables | 0.870 | 0.725 | 0.860 | 0.795 |
| | Obs. + all internals | 0.851 | 0.741 | 0.851 | 0.738 |
| | All internals | 0.714 | 0.603 | 0.717 | 0.617 |
| Qwen2.5-7B (4-bit) | Observables | 0.934 | 0.657 | 0.924 | 0.773 |
| | Obs. + all internals | 0.923 | 0.645 | 0.909 | 0.707 |
| | All internals | 0.803 | 0.522 | 0.770 | 0.547 |
| Llama-3.2-3B | Observables | 0.900 | 0.861 | 0.911 | 0.801 |
| | Obs. + all internals | 0.924 | 0.905 | 0.933 | 0.820 |
| | All internals | 0.845 | 0.829 | 0.859 | 0.707 |

Token confidence: 0.47–0.66 everywhere.

## Hypotheses

- **S1, observables degrade under shift: supported.**
  - Under the feed shift, the observable probe loses 0.04–0.28 AUROC; under the fault-family shift, 0.08–0.21.
  - The cooling shift (UA, not shown in the prompt) hardly matters for any signal: |Δ| ≤ 0.03.
- **S2, internals degrade less: not supported.**
  - Internals alone lose about as much as observables under shift, and they start lower. For Qwen2.5-7B they fall to 0.52–0.55.
  - On the target, the observable probe keeps a higher AUROC than internals alone for every model and shift.
- **S3, observables + internals accept more safely on the target: not supported.**
  - Qwen2.5-1.5B under the family shift: observables + all internals let 349 of 1284 accepted proposals through as failures (27%), against 28 of 874 (3%) for observables. The internal features make the accept decision less safe on unseen fault types.
  - For Llama-3.2-3B, observables + internals stay 0.02–0.04 AUROC above observables in every shift. That's the same small advantage as in distribution, not a robustness gain.

## Interpretation

With the neutral prompt and a varied-plant training population, model internals are **not** more robust to plant or fault shift than observable features, contrary to the earlier v2.1 result (`docs/cstr_varied_and_shift_results.md`). That shift was much larger: a single fixed training plant evaluated on a varied population.

**For the paper:** observables + proposed action are a strong and comparatively robust baseline in CSTR. When the deployment population moves along an axis the operator can see (feed rate) or to a new fault type, every probe degrades, so recalibration on the target population is required. The internal signals do not remove that requirement.
