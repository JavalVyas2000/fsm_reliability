# Attention grounding results (FSM, 4 models)

**Pre-registration:** `docs/grounding_attention_prereg.md`, frozen before computation.

**Data:** the FSM pilot of each model (3000 answers; train 1500 / dev_cal 500 /
dev_thr 500 / test_iid 500).

**Computation:** teacher-forced re-run of the stored answers (`scripts/27`).

| Model | Answers with a computable step | Teacher-forced argmax agreement |
|---|---|---|
| Qwen2.5-3B | 2994 | 99.3% |
| Llama-3.2-3B | 2999 | 99.6% |
| Qwen2.5-1.5B | 2970 | 92.8% |
| SmolLM2-1.7B | 2909 | 99.4% |

Lower agreement means the bf16 re-run diverges more from generation. For
Qwen2.5-1.5B this is a caveat.

## Part A — development data only (written before test_iid was examined; 2026-09-26)

### H1: step level (train + dev_cal + dev_thr)

For each step, `share` is the share of graph attention on the source node's
adjacency line, averaged over the 4 layers.

| Model | Steps (invalid) | Share valid / invalid | AUROC of −share for invalid step [95% CI] | Correct line is top-attended (valid / invalid) |
|---|---|---|---|---|
| Qwen2.5-3B | 9797 (2243) | 0.114 / 0.076 | 0.644 [0.631, 0.656] | 15% / 7% |
| Llama-3.2-3B | 9299 (2354) | 0.223 / 0.125 | **0.813 [0.804, 0.823]** | 51% / 20% |
| Qwen2.5-1.5B | 9515 (3378) | 0.188 / 0.102 | 0.757 [0.747, 0.767] | 32% / 13% |
| SmolLM2-1.7B | 10192 (5505) | 0.138 / 0.077 | 0.746 [0.736, 0.756] | 21% / 7% |

**H1 is supported for all four models.** When a model writes an illegal step, it
attends much less to the adjacency line that decides that step.

### H2: answer level (dev_thr, n = 500; AUROC for answer invalid)

| Model | context_action | token_conf. | coarse attention | grounding (all grd_) | grd_min_share only | ctx + grounding | ctx + all coarse internals | ctx + all internals + grounding |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 0.800 | 0.762 | 0.807 | 0.798 | 0.705 | 0.836 | 0.838 | **0.858** |
| Llama-3.2-3B | 0.857 | 0.784 | 0.830 | 0.857 | 0.818 | **0.912** | 0.877 | **0.912** |
| Qwen2.5-1.5B | 0.833 | 0.746 | 0.825 | 0.837 | 0.819 | **0.874** | 0.827 | 0.852 |
| SmolLM2-1.7B | 0.914 | 0.795 | 0.877 | 0.911 | 0.858 | **0.955** | 0.917 | 0.939 |

**Paired ΔAUROC on dev_thr** (grouped bootstrap, 95% CI):

| Model | ctx+grounding − ctx | grounding − token_conf. | ctx+coarse internals − ctx (for comparison) |
|---|---|---|---|
| Qwen2.5-3B | **+0.036 [+0.018, +0.055]** | +0.036 [−0.002, +0.071] | +0.037 [+0.014, +0.062] |
| Llama-3.2-3B | **+0.055 [+0.032, +0.080]** | **+0.073 [+0.031, +0.119]** | +0.020 [−0.003, +0.044] |
| Qwen2.5-1.5B | **+0.041 [+0.018, +0.065]** | **+0.091 [+0.049, +0.130]** | −0.006 [−0.031, +0.020] |
| SmolLM2-1.7B | **+0.041 [+0.022, +0.060]** | **+0.116 [+0.073, +0.162]** | +0.003 [−0.018, +0.024] |

On development data:
- **Grounding adds information beyond the observable context/action features for all four models.**
- The coarse attention and confidence features did so only for Qwen2.5-3B.
- Grounding alone is about as good as the observable probe (Δ ≈ 0), and clearly better than token confidence.
- The pre-registered primary feature on its own (`grd_min_share`) is weaker than the full grounding family. It is still informative (0.71–0.86).

## Part B — test_iid (examined once, after Part A was written; 2026-09-26)

### H1: step level, test_iid only

| Model | Steps (invalid) | AUROC of −share |
|---|---|---|
| Qwen2.5-3B | 1951 (416) | 0.672 |
| Llama-3.2-3B | 1828 (488) | 0.813 |
| Qwen2.5-1.5B | 1918 (685) | 0.761 |
| SmolLM2-1.7B | 2148 (1199) | 0.741 |

### H2: answer level, test_iid (n = 500)

| Model | context_action | token_conf. | coarse attention | grounding | ctx + grounding | ctx + coarse internals | ctx + all internals + grounding |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 0.811 | 0.793 | 0.828 | 0.812 | 0.852 | 0.849 | **0.876** |
| Llama-3.2-3B | 0.868 | 0.812 | 0.851 | 0.877 | **0.919** | 0.875 | **0.919** |
| Qwen2.5-1.5B | 0.859 | 0.815 | 0.858 | 0.883 | **0.910** | 0.846 | 0.883 |
| SmolLM2-1.7B | 0.938 | 0.859 | 0.909 | 0.934 | **0.970** | 0.941 | 0.957 |

**Paired ΔAUROC on test_iid** (grouped bootstrap, 95% CI):

| Model | ctx+grounding − ctx | grounding − token_conf. | ctx+coarse internals − ctx |
|---|---|---|---|
| Qwen2.5-3B | **+0.041 [+0.020, +0.063]** | +0.019 [−0.019, +0.060] | +0.038 [+0.013, +0.065] |
| Llama-3.2-3B | **+0.051 [+0.028, +0.075]** | **+0.065 [+0.032, +0.102]** | +0.006 [−0.016, +0.031] |
| Qwen2.5-1.5B | **+0.050 [+0.029, +0.073]** | **+0.068 [+0.029, +0.108]** | −0.013 [−0.037, +0.011] |
| SmolLM2-1.7B | **+0.032 [+0.019, +0.046]** | **+0.075 [+0.037, +0.114]** | +0.003 [−0.012, +0.018] |

**The test set replicates the development findings for every model.**
- Grounding adds +0.03 to +0.05 AUROC beyond the observable context/action probe, with all four CIs above 0.
- The coarse internal features did so only for Qwen2.5-3B.

## Part C — H3: routing and certification on the independent certification sets (2026-09-26)

**Policies.** Frozen per model in `outputs/certification/*_grounding_frozen` before grounding was computed on the cert data:

| Policy | Probe |
|---|---|
| G1 | context_action + grounding |
| G2 | context_action + all internals + grounding |
| G3 | grounding only |
| G4 | context_action baseline |

**Rules.** ALLOW threshold at α = 0 on pilot dev_thr; DISALLOW at β = 0.10. Bonferroni over the 4 policies (δ_i = 0.0125). Certified means the ALLOW failure rate upper bound is at most 0.05.

**Disclosure.** The certification sets' labels were used before, to certify policy set P1–P4. The grounding set was selected on pilot data only.

**G1 (primary) against G4 (baseline):**

| Model | G1 calls saved | G1 failures / ALLOW [UB] | G1 certified? | G4 calls saved | G4 failures / ALLOW [UB] | G4 certified? |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | **63.0%** | 2 / 160 [0.050] | yes | 47.7% | 3 / 198 [0.048] | yes |
| Qwen2.5-1.5B | **65.9%** | 1 / 231 [0.027] | yes | 58.4% | 0 / 194 [0.022] | yes |
| SmolLM2-1.7B | 92.6% | 0 / 56 [0.075] | no (too few ALLOWs) | 90.3% | 0 / 125 [0.034] | yes |
| Qwen2.5-3B | 21.0% | 8 / 290 [0.058] | no | 25.7% | 5 / 274 [0.046] | yes |

**Other certified grounding policies:** G2 for Llama (65.5% saved, 2/260, UB 0.031) and for Qwen2.5-1.5B (56.3% saved, 1/170, UB 0.037). G3 (grounding only) certifies for no model.

- **Where grounding helps, it certifies and saves more calls:** Llama +15 points, Qwen2.5-1.5B +7.5 points. The extra saving comes mostly from DISALLOWing more failing answers.
- **Cost of the extra DISALLOWs:** more valid answers are rejected. For Llama, G1 rejects 172 valid answers against G4's 132.
- **Where grounding does not help:**
  - Qwen2.5-3B (weakest step-level signal, AUROC 0.67): G1 lets through more failures and does not certify.
  - SmolLM2: G1 allows too few answers for a certifiable bound. It is not less safe (0 failures let through) but the evidence is too thin.
- **Grounding alone is not enough**, and the observable features remain essential. Grounding is a complement to them.

**Interpretable accounts.** `scripts/29_explain_flags.py` writes `explanations.md` / `.jsonl` in each certification run. For every routed answer it gives:
- the decision and risk;
- a step-by-step table: attention share on the deciding adjacency line, which line the model looked at most, and legality (offline truth);
- a bar chart of attention over graph lines for the weakest-grounded step;
- the probe's per-feature contributions to the risk score.

## Interpretation

"Did the model look at the part of the prompt that decides the answer?" is a useful
trust signal. Coarse attention ("how much did it look at the graph overall?") and
output confidence ("how sure was it?") are not. The signal is:

- **mechanistic and interpretable:** it points to the specific step and the line the model failed to consult;
- **consistent across three model families and two sizes;**
- **complementary to observable features:** it does not replace them, and it improves them.

## Caveats

- **Test history.** The pilot test_iid had been used in earlier analyses of other feature families. The grounding definitions and hypotheses were fixed before any grounding value was computed, and test was examined once after dev. An independent confirmation on the sealed certification sets is the next step (H3 routing plus certification).
- **Qwen2.5-1.5B** has lower teacher-forced agreement (92.8%).
- **Attention weights are not a full explanation** of model computation. The claim is predictive usefulness.
- **FSM only so far.** In CSTR, failures are mostly "right lever, wrong magnitude", where grounding may help less; that remains to be tested.

## Next

- **H3:** routing and time accounting with the grounding family; frozen grounding policies certified on the independent cert sets (the cert sets' grounding features need computing first, about 15 min per model).
- **CSTR field grounding.**
