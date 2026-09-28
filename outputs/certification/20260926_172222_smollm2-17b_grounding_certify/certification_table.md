# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 65; routed 2935; failure prevalence 0.831.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
**Disclosure:** Certification partition reused: its labels were previously used to certify the 'main' policy set (P1-P4). This policy set was selected on pilot data only and frozen before grounding features were computed on the certification data.
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1_context_grounding | 56 | 216 | 2663 | 92.6% | 0 | 0.0000 | 0.0753 | no | no | 0.0000 [0.0015] | 256 | -146.4 | +6.39 s (+92%) |
| G2_context_allinternal_grounding | 138 | 137 | 2660 | 95.3% | 3 | 0.0217 | 0.0689 | no | no | 0.0010 [0.0033] | 249 | -191.3 | +6.56 s (+94%) |
| G3_grounding_only | 32 | 297 | 2606 | 89.9% | 1 | 0.0312 | 0.1834 | no | no | 0.0003 [0.0022] | 249 | -146.4 | +6.20 s (+89%) |
| G4_context_action_baseline | 125 | 286 | 2524 | 90.3% | 0 | 0.0000 | 0.0344 | yes | no | 0.0000 [0.0015] | 223 | -0.0 | +6.27 s (+90%) |