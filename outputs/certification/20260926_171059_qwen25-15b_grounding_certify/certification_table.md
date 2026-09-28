# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 41; routed 2959; failure prevalence 0.671.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
**Disclosure:** Certification partition reused: its labels were previously used to certify the 'main' policy set (P1-P4). This policy set was selected on pilot data only and frozen before grounding features were computed on the certification data.
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1_context_grounding | 231 | 1009 | 1719 | 65.9% | 1 | 0.0043 | 0.0273 | yes | no | 0.0003 [0.0022] | 141 | -174.6 | +4.52 s (+65%) |
| G2_context_allinternal_grounding | 170 | 1292 | 1497 | 56.3% | 1 | 0.0059 | 0.0369 | yes | no | 0.0003 [0.0022] | 119 | -204.5 | +3.85 s (+55%) |
| G3_grounding_only | 200 | 1458 | 1301 | 50.7% | 4 | 0.0200 | 0.0554 | no | no | 0.0014 [0.0038] | 101 | -175.1 | +3.47 s (+50%) |
| G4_context_action_baseline | 194 | 1230 | 1535 | 58.4% | 0 | 0.0000 | 0.0223 | yes | no | 0.0000 [0.0015] | 153 | -0.0 | +4.06 s (+58%) |