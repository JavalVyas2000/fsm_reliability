# CSTR with varied nominal plants, and the plant-shift test

**Date:** 2026-09-28.

**Data:**

| Dataset | Content | Run |
|---|---|---|
| Fixed plant | CAR nominal plant, onset 1800–2200 s, ranges v2 | `outputs/cstr_collect/20260924_212031_qwen25-3b-instruct_main3000` |
| Varied plants | per episode: UA ±15%, inlet T 345–355 K, coolant T 285–295 K, inlet concentrations ±10%, coolant capacity ±10%, nominal L_sp 8–10, nominal feed 0.0267–0.04; onset 1500–5000 s; ranges v3; T_sp fixed at 310 K | `outputs/cstr_collect/20260926_200509_qwen25-3b-instruct_varied3000_r0` |

Both use Qwen2.5-3B with prompt v2.1 and first proposals only. The certification
partitions are sealed.

## 1. Varied-plant collection

**Episodes.** 4800 drawn, 4223 triggered; fouling and mild stuck-cooling episodes
often do not trigger on some plants. 6 pre-fault false alarms were excluded. 3000
episodes are used, and all prompts are distinct.

**First-proposal pass rate: 21.5%** (fixed plant: 33.5%).

| Family | Passed |
|---|---|
| outlet_block | 32% |
| pump_degrade | 25% |
| cool_stuck_closed | 17% |
| fouling | 5% |

**The model adapts its proposal to each plant, but under-corrects.**
- The proposed feed correlates 0.89 with the plant's own nominal feed.
- It almost never increases the feed (1%).
- The median feed change is −4% (about −10% on the fixed plant).

## 2. In-distribution on varied plants (`outputs/fsm_baseline/20260928_110744_cstr_varied_r0`)

| Family | AUROC dev_thr | AUROC test [95% CI] |
|---|---|---|
| context_action | 0.861 | **0.891 [0.834, 0.938]** |
| context_only_no_action | 0.836 | 0.862 |
| token_confidence | 0.597 | 0.714 |
| attention | 0.759 | 0.708 |
| hidden | 0.752 | 0.723 |
| all_internal | 0.776 | 0.766 |
| context_action + all_internal | 0.856 | 0.872 |

- Internals alone are stronger than on the fixed plant (0.71–0.77 against 0.64–0.67).
- Adding them to context_action still gives no gain (test: −0.026 to +0.002).
- In distribution, the observable probe remains the best.

## 3. Shift test: train on the fixed plant, test on varied plants (`outputs/cstr_shift/..._fixed_to_varied`)

| Family | fixed → fixed | fixed → varied [95% CI] | varied → varied | Drop under shift |
|---|---|---|---|---|
| context_action | 0.887 | **0.453 [0.374, 0.528]** | 0.889 | **−0.435** |
| context_only_no_action | 0.843 | 0.438 | 0.862 | −0.405 |
| token_confidence | 0.627 | **0.589 [0.516, 0.662]** | 0.714 | −0.037 |
| hidden | 0.695 | **0.582 [0.506, 0.653]** | 0.723 | −0.113 |
| attention | 0.672 | 0.476 | 0.708 | −0.195 |
| all_internal | 0.677 | 0.460 | 0.766 | −0.217 |
| context_action + all_internal | 0.849 | 0.407 | 0.869 | −0.442 |

**Paired, on the varied test episodes, for probes trained on the fixed plant:**

| Comparison | ΔAUROC [95% CI] |
|---|---|
| token_confidence − context_action | **+0.137 [+0.031, +0.237]** |
| hidden − context_action | **+0.129 [+0.022, +0.233]** |
| attention − context_action | +0.023 [−0.087, +0.126] |
| context_action + all internals − context_action | −0.046 [−0.075, −0.019] |

**Why the observable probe breaks.** Relationships between plant readings and failure
change with the plant. Single-feature AUROC for failure:

| Feature | Fixed-plant test | Varied-plant test |
|---|---|---|
| u_valve | 0.508 | 0.832 |
| u_pump | 0.362 | 0.549 (direction flips) |
| u_cool | 0.552 | 0.708 |
| level-error alarm | 0.474 | 0.369 |

A probe that learned one plant's readings mis-ranks the others, falling to chance or
below. The model's own confidence and hidden state transfer much better: they drop
only 0.04–0.11.

## 4. Interpretation for the paper

- **In distribution** (same plant, or the same family of plants seen in training), cheap observable features are the best trust signal, and internals are redundant.
- **Under plant shift** (training data from one nominal plant, deployment on others), observable features fail badly. Token confidence and hidden states retain a significant, if modest, signal and beat the observable probe by about 0.13 AUROC.
- **Neither is good enough on its own under shift** for safe bypass (AUROC around 0.59). The practical conclusion is that the probe must be retrained or recalibrated on the target plant population. Certification is population-specific by construction, and this experiment shows why that matters.
- **Attention grounding:**
  - FSM, with retrieval failures: grounding adds value in distribution.
  - CSTR: field grounding adds nothing in distribution (magnitude failures), and coarse attention does not transfer across plants.

## Caveats

- **The shift is large:** a single training plant against a wide plant distribution. A smaller shift, for example training on part of the varied range and testing outside it, would likely degrade the observable probe less.
- **One model** (Qwen2.5-3B).
- **The varied test set has high failure prevalence (0.80),** so AUROC CIs are wider.
