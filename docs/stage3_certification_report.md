# Stage 3 — certification across four models (FSM)

**Date:** 2026-09-24. The overnight run completed for all four models; every step returned 0.

**Sources:**
- Cross-model tables: `outputs/cross_model/20260924_043231_summary/` (`cross_model_summary.md`, `predictive.csv`, `routing_zero_failure.csv`, `certification.csv`).
- Per-model pipeline logs: `outputs/model_pipelines/*/pipeline.json`.
- Certification runs: `outputs/certification/*_certify/`.
- Method: `docs/risk_certification.md`.

## Setup (identical for every model)

| Component | Setting |
|---|---|
| Prompt and decoding | Prompt `fsm_v2.0` (chat template, date pinned); greedy decoding, `repetition_penalty=1.0`, stop at the first JSON object |
| Pilot data | Seed 20260923: train 1500 / dev_cal 500 / dev_thr 500 / test 500 |
| Certification data | Seed 20260924: 3000 graphs, disjoint from the pilot and from all legacy data. The same graphs are used for every model |
| Probes | Fit on train; Platt calibration on dev_cal |
| Thresholds | Selected on dev_thr: ALLOW with α = 0 (point rule), DISALLOW with β = 0.10 |
| Freeze | Per model, before that model's certification inference. Hashes are verified by `scripts/17` |
| Certification | ALLOW failure rate. One-sided Clopper–Pearson bound at δ_i = 0.05 / 4 = 0.0125, Bonferroni over the 4 policies within each model. Certified at α ⇔ bound ≤ α |

**Wall time per model:**

| Model | Pilot | Certification | Other steps |
|---|---|---|---|
| Llama-3.2-3B | 45 min | 47 min | < 4 min |
| Qwen2.5-1.5B | 46 min | 47 min | < 4 min |
| SmolLM2-1.7B | 33 min | 34 min | < 4 min |

## Certification results (independent set, n ≈ 3000 per model)

**Qwen2.5-3B** (failure prevalence 0.52):

| Policy | ALLOW (share) | Failures let through | Observed rate | 95%* bound | ≤ 5%? | Calls saved | Valid lost (% of valid) |
|---|---|---|---|---|---|---|---|
| P1 all_internal | 231 (7.7%) | 4 | 1.7% | 4.8% | **yes** | 28.0% | 68 (4.8%) |
| P2 attention+token | 253 (8.5%) | 2 | 0.8% | 3.2% | **yes** | 26.2% | 49 (3.4%) |
| P3 agreement | 399 (13.3%) | 23 | 5.8% | 8.9% | no | 33.6% | 68 (4.8%) |
| P4 context_action (baseline) | 274 (9.2%) | 5 | 1.8% | 4.6% | **yes** | 25.7% | 46 (3.2%) |

**Llama-3.2-3B** (failure prevalence 0.63):

| Policy | ALLOW (share) | Failures let through | Observed rate | 95%* bound | ≤ 5%? | Calls saved | Valid lost (% of valid) |
|---|---|---|---|---|---|---|---|
| P1 all_internal | 129 (4.3%) | 3 | 2.3% | 7.4% | no | 45.8% | 147 (13.2%) |
| P2 attention+token | 118 (3.9%) | 2 | 1.7% | 6.7% | no | 40.4% | 124 (11.1%) |
| P3 agreement | 198 (6.6%) | 10 | 5.1% | 9.7% | no | 48.1% | 147 (13.2%) |
| P4 context_action (baseline) | 198 (6.6%) | 3 | 1.5% | 4.8% | **yes** | 47.7% | 132 (11.8%) |

**Qwen2.5-1.5B** (failure prevalence 0.67):

| Policy | ALLOW (share) | Failures let through | Observed rate | 95%* bound | ≤ 5%? | Calls saved | Valid lost (% of valid) |
|---|---|---|---|---|---|---|---|
| P1 all_internal | 147 (5.0%) | 1 | 0.7% | 4.3% | **yes** | 43.5% | 84 (8.6%) |
| P2 attention+token | 141 (4.8%) | 2 | 1.4% | 5.6% | no | 47.3% | 97 (10.0%) |
| P3 agreement | 161 (5.4%) | 2 | 1.2% | 4.9% | **yes** | 44.0% | 84 (8.6%) |
| P4 context_action (baseline) | 194 (6.6%) | **0** | 0.0% | 2.2% | **yes** | 58.4% | 153 (15.7%) |

**SmolLM2-1.7B** (failure prevalence 0.83):

| Policy | ALLOW (share) | Failures let through | Observed rate | 95%* bound | ≤ 5%? | Calls saved | Valid lost (% of valid) |
|---|---|---|---|---|---|---|---|
| P1 all_internal | 22 (0.7%) | 3 | 13.6% | 37.9% | no | 87.3% | 244 (49%) |
| P2 attention+token | 8 (0.3%) | 1 | 12.5% | 57.5% | no | 86.7% | 233 (47%) |
| P3 agreement | 118 (4.0%) | **0** | 0.0% | 3.6% | **yes** | 90.4% | 239 (48%) |
| P4 context_action (baseline) | 125 (4.3%) | **0** | 0.0% | 3.4% | **yes** | 90.3% | 223 (45%) |

\* The bound is at confidence 1 − 0.0125 per policy, so each model's family of four holds jointly at 95%. No policy certifies at α = 0.02 for any model.

## Findings, stated plainly

1. **Certification works as designed, and the pilot's "zero failures" did not hold at scale.**
   - Qwen-3B P1 let through 0 failures of 41 ALLOWs on the pilot test, but 4 of 231 on the certification set.
   - Observed ALLOW failure rates for the certified policies are 0–2.3%, with certified upper bounds of 2.2–4.8%.

2. **The agreement rule (P3) overfits when the model has moderate failure prevalence.**
   - For Qwen-3B it let through 23 failures (5.8%) and failed certification. Its pilot result had been 1 of 68.
   - Searching two thresholds jointly on 500 dev samples is too flexible.
   - It certified for Qwen-1.5B and SmolLM2, but not better than the baseline.

3. **Internal signals add value over the observable baseline only for Qwen2.5-3B.**
   - On the pilot test, attention+token − context_action AUROC differences (95% CI) are:

     | Model | Δ AUROC |
     |---|---|
     | Qwen-3B | +0.032 [+0.010, +0.056] |
     | Llama-3B | −0.009 [−0.032, +0.016] |
     | Qwen-1.5B | +0.002 [−0.019, +0.025] |
     | SmolLM2 | −0.032 [−0.052, −0.014] |

   - In certification, the context_action baseline (P4) certifies for **every** model and gives the most or nearly the most calls saved.
   - Internal-signal policies certify for Qwen-3B (P1, P2) and Qwen-1.5B (P1) but do not beat P4.
   - **On FSM, the claim "internals beat observable features" is not supported across models.**

4. **Why the baseline is strong on FSM.**
   - `context_action` includes cheap structural facts about the proposed answer: endpoints match, repeated nodes, answer length. It also includes task difficulty: graph size and shortest-path length.
   - For weaker models most failures are structurally obvious, e.g. copied or truncated paths. That is visible without model internals.
   - `answer_endpoints_match` is effectively a partial verifier.

5. **For weaker models, "calls saved" comes mostly from DISALLOW, and DISALLOW is not free.**
   - SmolLM2 saves about 90% of calls, but DISALLOW rejects 45–49% of all *valid* answers.
   - In a closed loop every DISALLOW triggers a re-proposal, whose cost the per-candidate identity excludes.
   - ALLOW shares are small: Qwen-3B 8–9%, Llama/Qwen-1.5B 4–7%, SmolLM2 0.3–4%. The "what-if" CSTR savings for weak models are therefore overstated unless re-proposal cost is counted.

6. **The failure rate limits what can be ALLOWed.** Failure prevalence rises from 0.52 (Qwen-3B) to 0.83 (SmolLM2) and the ALLOW share shrinks accordingly. No policy reaches α = 0.02 with n = 3000.

## Implications for the paper

**Supported claims:**
- Three-way routing with an independently certified ALLOW failure rate works across four models from three families, using a pre-registered freeze and an independent set.
- It saves 26–58% of verifier calls at certified ALLOW risk ≤ 5% for models with prevalence ≤ 0.67. Much of this comes from DISALLOW.
- Certification catches an overfit policy (P3 on Qwen-3B) that looked safe on the pilot.

**Not supported on FSM:**
- That internal signals are needed for this. Observable context/action features match or beat them for 3 of 4 models.

**Open, and decisive:**
- Whether CSTR behaves differently (Stage 4). There, the "obvious structural error" signal is weaker: a setpoint triple has no endpoints to mismatch, and the observable context is plant state plus setpoint deltas.
- That is where internals could add value, and the design already includes the `context_action` comparison.

## Caveats

- **The pilot test was used to choose the α = 0 rule and the policy set.** The certification set was not; policies were frozen before it existed.
- **Certification is per model.** A single cross-model statement needs Bonferroni over all 16 policies (δ_i = 0.05/16 = 0.003125). At that level only four policies stay certified at α = 0.05 (bounds recomputed):

  | Model | Policy | Bound at δ/16 |
  |---|---|---|
  | Qwen-3B | P2 attention+token | 3.8% |
  | Qwen-1.5B | P4 context_action | 2.9% |
  | SmolLM2 | P3 agreement | 4.8% |
  | SmolLM2 | P4 context_action | 4.5% |

  These policies lose certification at δ/16:

  | Model | Policy | Bound at δ/16 |
  |---|---|---|
  | Qwen-3B | P1 | 5.6% |
  | Qwen-3B | P4 | 5.3% |
  | Llama-3B | P4 | 5.7% |
  | Qwen-1.5B | P1 | 5.3% |

  The paper should state claims per model, which is how they were declared.
- **The "what-if" CSTR savings are arithmetic** from FSM routing shares and exclude re-proposal cost.
