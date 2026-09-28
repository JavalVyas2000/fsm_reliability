# Pre-registration: attention grounding as a trust signal (FSM)

**Frozen:** 2026-09-26, before any grounding value was computed.

**Idea (user's hypothesis).** If, while writing an answer, the model attends to the
parts of the prompt that actually decide that answer, the answer can be trusted. If
it attends elsewhere, it should not be.

## Why FSM

An FSM answer is a path `[s, v1, v2, ..., g]`. A step u → v is legal only if v appears
in the adjacency line `u: [...]` of the graph block in the prompt. That line is
therefore the part of the prompt that decides whether the step is valid.

## Definitions

**Decision rows.** For answer token a_i, the attention row of the position that
predicts a_i (the "decision position", as in `src/models/inference_v2.py`). For a
path node v written as tokens a_j..a_m, the node's decision rows are rows j..m.

**Graph-line regions.** Each adjacency line `u: [...]` of the instance graph in the
prompt is one region. A prompt token belongs to the line with the largest character
overlap. The worked-example graph is excluded.

**Measurements for a step u → v.** At each selected layer (relative depth 0.25,
0.5, 0.75, 1.0):
- `share` = attention mass on line u ÷ attention mass on all instance-graph lines. It is averaged over v's decision rows and over heads (head-mean).
- `share_hmax` = the same quantity, taking the maximum over heads of each head's share.
- `top1` = 1 if line u receives the most attention among all instance-graph lines (head-mean mass).

## Candidate-level features (family `grounding`, prefix `grd_`)

For each answer with at least one step:
- `grd_min_share`: **primary**. The minimum over steps of `share`, averaged over the 4 layers.
- `grd_mean_share`: the mean over steps of `share`, averaged over layers.
- `grd_top1_frac`: the fraction of steps with `top1 = 1`, averaged over layers.
- `grd_first_share`: `share` of the first step, averaged over layers.
- Per layer: `grd_L{025,050,075,100}_min_share` and `grd_L..._top1_frac`.
- `grd_min_share_hmax`: the minimum over steps of `share_hmax`, averaged over layers.

Answers without a complete step (fewer than two nodes) get NaN, handled by median
imputation as for other features.

## Hypotheses and direction (fixed in advance)

- **H1 (step level).** Invalid steps (v not in adj(u)) have lower `share` than valid steps.
  - Test: AUROC of −share for step invalidity.
  - Uncertainty: bootstrap grouped by instance.
- **H2 (answer level).** Lower `grd_min_share` → higher probability that the answer is invalid.
  - Test: AUROC of −grd_min_share on its own.
  - Then ΔAUROC when the `grounding` family is added to:
    - (a) token_confidence;
    - (b) context_action;
    - (c) context_action + all existing internals.
  - Uncertainty: grouped bootstrap over graphs.
- **H3 (routing).** Adding `grounding` to the probe reduces failures let through, or increases calls saved, at the same α = 0 / β = 0.10 routing rule.

## Evaluation protocol

- **Data:** the existing FSM pilot inference runs for all four models. The same partitions are used: fit on train, calibrate on dev_cal, thresholds on dev_thr. test_iid is examined **once, at the end**, after the dev results are written down.
- **Computation:** teacher-forced re-run of the stored generated tokens. There is no regeneration.
  - Alignment check: the teacher-forced argmax is compared with the stored tokens and the agreement rate reported.
- **Model settings:** the same models, dtype, chat template and date as the original runs.
- **Reporting:** all four models, positive and negative results alike. The primary feature is `grd_min_share`. The others are secondary and labelled as such.

## Known limitations

- Attention weights are an imperfect explanation of model computation. The test asks only whether grounding is a *useful warning signal*.
- Head-mean attention can hide individual "retrieval" heads; `share_hmax` is a secondary check for that.
