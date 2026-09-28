# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 6; routed 2994; failure prevalence 0.523.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
**Disclosure:** Certification partition reused: its labels were previously used to certify the 'main' policy set (P1-P4). This policy set was selected on pilot data only and frozen before grounding features were computed on the certification data.
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1_context_grounding | 290 | 2364 | 340 | 21.0% | 8 | 0.0276 | 0.0578 | no | no | 0.0027 [0.0057] | 17 | -220.2 | +1.39 s (+20%) |
| G2_context_allinternal_grounding | 405 | 1805 | 784 | 39.7% | 15 | 0.0370 | 0.0640 | no | no | 0.0050 [0.0088] | 70 | -248.8 | +2.68 s (+39%) |
| G3_grounding_only | 245 | 2507 | 242 | 16.3% | 12 | 0.0490 | 0.0894 | no | no | 0.0040 [0.0075] | 36 | -220.5 | +1.06 s (+15%) |
| G4_context_action_baseline | 274 | 2224 | 496 | 25.7% | 5 | 0.0182 | 0.0459 | yes | no | 0.0017 [0.0043] | 46 | -0.0 | +1.79 s (+26%) |