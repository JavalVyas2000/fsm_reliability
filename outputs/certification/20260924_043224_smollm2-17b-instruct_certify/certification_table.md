# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 65; routed 2935; failure prevalence 0.831.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_all_internal | 22 | 374 | 2539 | 87.3% | 3 | 0.1364 | 0.3795 | no | no | 0.0010 [0.0033] | 244 | -182.5 | +6.00 s (+86%) |
| P2_attention_token | 8 | 390 | 2537 | 86.7% | 1 | 0.1250 | 0.5755 | no | no | 0.0003 [0.0022] | 233 | -146.2 | +5.98 s (+86%) |
| P3_agree_context_all_internal | 118 | 283 | 2534 | 90.4% | 0 | 0.0000 | 0.0365 | yes | no | 0.0000 [0.0015] | 239 | -185.3 | +6.22 s (+89%) |
| P4_context_action_baseline | 125 | 286 | 2524 | 90.3% | 0 | 0.0000 | 0.0344 | yes | no | 0.0000 [0.0015] | 223 | -0.0 | +6.27 s (+90%) |