# Risk certification of routing policies (Stage 3)

This document covers the method and sample sizes. Results are produced by
`scripts/17_certify.py` and written under `outputs/certification/`.

## What is certified

A routing policy sends each schema-valid candidate to one of three routes:

- **ALLOW**: execute without the simulator.
- **VERIFY**: call the simulator.
- **DISALLOW**: reject without the simulator.

The certified quantity is the **ALLOW failure rate**, P(fail | ALLOW): the
fraction of actions executed without verification that are in fact invalid.

Two further quantities are **co-reported with bounds but not certified**:

- the marginal escaped-error rate, P(ALLOW ∧ fail), over all eligible candidates;
- the DISALLOW valid rate, i.e. valid actions thrown away.

## Procedure

1. **Fit.** Probes are fit on `train` and calibrated with Platt scaling on `dev_cal`.
2. **Select thresholds on `dev_thr` only.** The ALLOW threshold uses the point rule with α = 0: the largest threshold with zero ALLOW failures. The DISALLOW threshold uses the point rule with β = 0.10.
3. **Freeze.** `scripts/16_freeze_policies.py` writes the declared policy set, thresholds and fitted probes, with sha256 hashes. This happens **before** the certification partition is generated.
   - Refitting reproduces the stored Stage 1 predictions to 1e-16.
4. **Generate an independent certification partition.**
   - Fresh root seed.
   - Graph-disjoint from every pilot partition and every existing dataset.
   - One candidate per graph, so certification candidates are independent draws from the declared population.
5. **Certify.** `scripts/17_certify.py` verifies the hashes and checks that the certification inference postdates the freeze. It then routes each candidate and computes, for k failures among n ALLOWed candidates:
   - U = exact one-sided Clopper–Pearson upper bound at confidence 1 − δ/m;
   - certified at α ⇔ n > 0 and U ≤ α.

   Nothing is refit or re-tuned. A failed certification is reported as a failure. It is not repaired on the same data.

## Declared policy set (m = 4)

| Id | ALLOW rule | DISALLOW rule |
|---|---|---|
| P1_all_internal | all_internal probe < τ | all_internal > τ_high |
| P2_attention_token | attention + token-confidence probe < τ | same probe > τ_high |
| P3_agree_context_all_internal | context_action < τ_a **and** all_internal < τ_b | all_internal > τ_high |
| P4_context_action_baseline | context_action < τ (no model internals) | context_action > τ_high |

Certification levels are α ∈ {0.05, 0.02}, with family-wise δ = 0.05.

- Bonferroni over the 4 policies gives δ_i = 0.0125.
- One bound per policy covers both α levels, so multiplicity is over policies, not levels.

## Guarantee

With probability at least 1 − δ over the certification sample, no declared
policy is certified at a level below its true ALLOW failure rate.

This holds for the declared IID population:
- Qwen2.5-3B-Instruct (bf16), prompt `fsm_v2.0`, greedy decoding;
- graphs from the `fsm_dataset_v2.0` generator with 5/10/15/20 nodes.

The guarantee does **not** cover:
- other models, prompts, or instance distributions (OOD);
- individual actions (it is a rate statement, not a per-action certificate);
- closed-loop episodes with repeated decisions.

This is the standard fixed-policy binomial argument; see Learn then Test
(https://arxiv.org/abs/2110.01052). No new theorem is claimed.

## Sample sizes

The tables give the minimum number of ALLOWed candidates n needed for U ≤ α, given k failures (`src/evaluation/certification.py`, tested).

**δ = 0.05** (single policy):

| α | k=0 | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| 0.05 | 59 | 93 | 124 | 208 |
| 0.02 | 149 | 236 | 313 | 523 |
| 0.01 | 299 | 473 | 628 | 1049 |

**δ = 0.0125** (Bonferroni, 4 policies):

| α | k=0 | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| 0.05 | 86 | 125 | 160 | 252 |
| 0.02 | 217 | 317 | 404 | 635 |

**Certification partition size.** The protocol rule is n_required(k = 2) / expected ALLOW coverage × 1.5. The pilot ALLOW coverage at the frozen thresholds was about 8% (P1, P2, P4) to 14% (P3):

| Target | Candidates required |
|---|---|
| α = 0.05 | 3000 at 8%; 1715 at 14% |
| α = 0.02 | 7575 at 8% |

A partition of **3000** was generated (root seed 20260924). α = 0.05 is therefore the planned primary level. α = 0.02 is reachable only if ALLOW failures on the certification set are close to zero.

## Relation to "zero failures let through"

Zero observed ALLOW failures is not a zero rate. With k = 0 and n ALLOWs, the
bound is U = 1 − δ^(1/n). For example, 0 failures in 40 ALLOWs is compatible with
a true rate of up to 7% at 95% confidence. Certification replaces
"none observed" with a rate bound on independent data.
