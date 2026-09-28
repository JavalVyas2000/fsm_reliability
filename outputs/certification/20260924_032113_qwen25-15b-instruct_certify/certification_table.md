# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 41; routed 2959; failure prevalence 0.671.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_all_internal | 147 | 1672 | 1140 | 43.5% | 1 | 0.0068 | 0.0426 | yes | no | 0.0003 [0.0022] | 84 | -201.3 | +2.95 s (+43%) |
| P2_attention_token | 141 | 1559 | 1259 | 47.3% | 2 | 0.0142 | 0.0564 | no | no | 0.0007 [0.0027] | 97 | -174.5 | +3.23 s (+46%) |
| P3_agree_context_all_internal | 161 | 1658 | 1140 | 44.0% | 2 | 0.0124 | 0.0495 | yes | no | 0.0007 [0.0027] | 84 | -204.0 | +2.99 s (+43%) |
| P4_context_action_baseline | 194 | 1230 | 1535 | 58.4% | 0 | 0.0000 | 0.0223 | yes | no | 0.0000 [0.0015] | 153 | -0.0 | +4.06 s (+58%) |