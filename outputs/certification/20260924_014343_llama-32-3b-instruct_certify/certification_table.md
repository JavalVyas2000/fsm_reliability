# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 3; routed 2997; failure prevalence 0.628.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_all_internal | 129 | 1624 | 1244 | 45.8% | 3 | 0.0233 | 0.0735 | no | no | 0.0010 [0.0032] | 147 | -225.4 | +3.11 s (+45%) |
| P2_attention_token | 118 | 1787 | 1092 | 40.4% | 2 | 0.0169 | 0.0671 | no | no | 0.0007 [0.0027] | 124 | -184.3 | +2.74 s (+39%) |
| P3_agree_context_all_internal | 198 | 1555 | 1244 | 48.1% | 10 | 0.0505 | 0.0972 | no | no | 0.0033 [0.0066] | 147 | -227.4 | +3.27 s (+47%) |
| P4_context_action_baseline | 198 | 1567 | 1232 | 47.7% | 3 | 0.0152 | 0.0484 | yes | no | 0.0010 [0.0032] | 132 | -0.0 | +3.32 s (+48%) |