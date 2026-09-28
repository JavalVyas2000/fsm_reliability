# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 6; routed 2994; failure prevalence 0.523.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_all_internal | 231 | 2156 | 607 | 28.0% | 4 | 0.0173 | 0.0481 | yes | no | 0.0013 [0.0038] | 68 | -249.8 | +1.86 s (+27%) |
| P2_attention_token | 253 | 2210 | 531 | 26.2% | 2 | 0.0079 | 0.0317 | yes | no | 0.0007 [0.0027] | 49 | -219.8 | +1.75 s (+25%) |
| P3_agree_context_all_internal | 399 | 1988 | 607 | 33.6% | 23 | 0.0576 | 0.0894 | no | no | 0.0077 [0.0121] | 68 | -251.7 | +2.25 s (+32%) |
| P4_context_action_baseline | 274 | 2224 | 496 | 25.7% | 5 | 0.0182 | 0.0459 | yes | no | 0.0017 [0.0043] | 46 | -0.0 | +1.79 s (+26%) |