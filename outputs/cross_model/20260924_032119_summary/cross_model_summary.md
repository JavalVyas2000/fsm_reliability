# Cross-model summary: selective verification on FSM (Qwen/Llama/SmolLM families)

All models: prompt fsm_v2.0, greedy decoding, same pilot dataset (seed 20260923) and certification dataset (seed 20260924, 3000 graphs). Per-model policies frozen before that model's certification inference.
Pilot test = exploratory; certification = independent, Bonferroni over 4 policies per model (δ_i = 0.0125).
What-if column = routing share × CSTR legacy verifier cost (6.95 s/call) − measured FSM overhead; not a CSTR result.

## 1. Failure prediction (pilot test_iid AUROC; failure = positive)

| model | test_prevalence_fail | gen_latency_mean_s | feat_latency_mean_s | auroc_context_action | auroc_token_confidence | auroc_attention | auroc_attention+token_confidence | auroc_all_internal | auroc_context_action+all_internal | delta[attention+token_confidence - context_action] | delta[context_action+all_internal - context_action] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | 0.486 | 1.454 | 0.090 | 0.811 | 0.793 | 0.828 | 0.843 | 0.840 | 0.849 | +0.032 [+0.010, +0.056] | +0.038 [+0.013, +0.065] |
| meta-llama/Llama-3.2-3B-Instruct | 0.630 | 0.841 | 0.059 | 0.868 | 0.812 | 0.851 | 0.859 | 0.845 | 0.875 | -0.009 [-0.032, +0.016] | +0.006 [-0.016, +0.031] |
| Qwen/Qwen2.5-1.5B-Instruct | 0.689 | 0.861 | 0.059 | 0.859 | 0.815 | 0.858 | 0.862 | 0.836 | 0.846 | +0.002 [-0.019, +0.025] | -0.013 [-0.037, +0.011] |

## 2. Zero-failure routing on pilot test (α = 0 on dev_thr, β = 0.10)

| model | family | allow | verify | disallow | calls_saved | failures_let_through | lost_valid | whatif_saved_pct_at_6.95s |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | context_action | 45 | 370 | 85 | 0.260 | 0 | 10 | 26.0 |
| Qwen/Qwen2.5-3B-Instruct | token_confidence | 30 | 389 | 81 | 0.222 | 0 | 6 | 20.9 |
| Qwen/Qwen2.5-3B-Instruct | attention | 40 | 392 | 68 | 0.216 | 1 | 4 | 20.3 |
| Qwen/Qwen2.5-3B-Instruct | attention+token_confidence | 40 | 371 | 89 | 0.258 | 0 | 6 | 24.5 |
| Qwen/Qwen2.5-3B-Instruct | all_internal | 41 | 347 | 112 | 0.306 | 0 | 14 | 29.1 |
| Qwen/Qwen2.5-3B-Instruct | context_action+all_internal | 65 | 304 | 131 | 0.392 | 1 | 17 | 37.7 |
| meta-llama/Llama-3.2-3B-Instruct | context_action | 39 | 268 | 193 | 0.464 | 0 | 14 | 46.4 |
| meta-llama/Llama-3.2-3B-Instruct | token_confidence | 0 | 386 | 114 | 0.228 | 0 | 11 | 22.0 |
| meta-llama/Llama-3.2-3B-Instruct | attention | 3 | 357 | 140 | 0.286 | 0 | 11 | 27.8 |
| meta-llama/Llama-3.2-3B-Instruct | attention+token_confidence | 25 | 309 | 166 | 0.382 | 0 | 17 | 37.4 |
| meta-llama/Llama-3.2-3B-Instruct | all_internal | 26 | 274 | 200 | 0.452 | 1 | 22 | 44.2 |
| meta-llama/Llama-3.2-3B-Instruct | context_action+all_internal | 47 | 235 | 218 | 0.530 | 2 | 22 | 52.0 |
| Qwen/Qwen2.5-1.5B-Instruct | context_action | 32 | 202 | 264 | 0.594 | 0 | 26 | 59.4 |
| Qwen/Qwen2.5-1.5B-Instruct | token_confidence | 3 | 359 | 136 | 0.279 | 0 | 11 | 27.0 |
| Qwen/Qwen2.5-1.5B-Instruct | attention | 35 | 230 | 233 | 0.538 | 3 | 18 | 52.9 |
| Qwen/Qwen2.5-1.5B-Instruct | attention+token_confidence | 31 | 250 | 217 | 0.498 | 2 | 14 | 48.9 |
| Qwen/Qwen2.5-1.5B-Instruct | all_internal | 28 | 287 | 183 | 0.424 | 2 | 14 | 41.4 |
| Qwen/Qwen2.5-1.5B-Instruct | context_action+all_internal | 28 | 260 | 210 | 0.478 | 0 | 17 | 46.8 |

## 3. Certification (independent cert set)

| model | policy | n_eligible | prevalence | allow | verify | disallow | calls_saved | failures_let_through | allow_rate_ub | cert_0.05 | cert_0.02 | lost_valid | whatif_saved_pct_at_6.95s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | P1_all_internal | 2994 | 0.523 | 231 | 2156 | 607 | 0.280 | 4 | 0.048 | True | False | 68 | 26.8 |
| Qwen/Qwen2.5-3B-Instruct | P2_attention_token | 2994 | 0.523 | 253 | 2210 | 531 | 0.262 | 2 | 0.032 | True | False | 49 | 25.1 |
| Qwen/Qwen2.5-3B-Instruct | P3_agree_context_all_internal | 2994 | 0.523 | 399 | 1988 | 607 | 0.336 | 23 | 0.089 | False | False | 68 | 32.4 |
| Qwen/Qwen2.5-3B-Instruct | P4_context_action_baseline | 2994 | 0.523 | 274 | 2224 | 496 | 0.257 | 5 | 0.046 | True | False | 46 | 25.7 |
| meta-llama/Llama-3.2-3B-Instruct | P1_all_internal | 2997 | 0.628 | 129 | 1624 | 1244 | 0.458 | 3 | 0.074 | False | False | 147 | 44.7 |
| meta-llama/Llama-3.2-3B-Instruct | P2_attention_token | 2997 | 0.628 | 118 | 1787 | 1092 | 0.404 | 2 | 0.067 | False | False | 124 | 39.5 |
| meta-llama/Llama-3.2-3B-Instruct | P3_agree_context_all_internal | 2997 | 0.628 | 198 | 1555 | 1244 | 0.481 | 10 | 0.097 | False | False | 147 | 47.0 |
| meta-llama/Llama-3.2-3B-Instruct | P4_context_action_baseline | 2997 | 0.628 | 198 | 1567 | 1232 | 0.477 | 3 | 0.048 | True | False | 132 | 47.7 |
| Qwen/Qwen2.5-1.5B-Instruct | P1_all_internal | 2959 | 0.671 | 147 | 1672 | 1140 | 0.435 | 1 | 0.043 | True | False | 84 | 42.5 |
| Qwen/Qwen2.5-1.5B-Instruct | P2_attention_token | 2959 | 0.671 | 141 | 1559 | 1259 | 0.473 | 2 | 0.056 | False | False | 97 | 46.5 |
| Qwen/Qwen2.5-1.5B-Instruct | P3_agree_context_all_internal | 2959 | 0.671 | 161 | 1658 | 1140 | 0.440 | 2 | 0.049 | True | False | 84 | 43.0 |
| Qwen/Qwen2.5-1.5B-Instruct | P4_context_action_baseline | 2959 | 0.671 | 194 | 1230 | 1535 | 0.584 | 0 | 0.022 | True | False | 153 | 58.4 |
