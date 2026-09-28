# Certification of frozen routing policies (FSM, partition `cert`)

Generated 3000; rejected by cheap format checks 3; routed 2997; failure prevalence 0.628.
Exact one-sided bounds at δ = 0.05 / 4 = 0.0125 per policy (Bonferroni).
**Disclosure:** Certification partition reused: its labels were previously used to certify the 'main' policy set (P1-P4). This policy set was selected on pilot data only and frozen before grounding features were computed on the certification data.
Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.

| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1_context_grounding | 160 | 1109 | 1728 | 63.0% | 2 | 0.0125 | 0.0498 | yes | no | 0.0007 [0.0027] | 172 | -184.4 | +4.32 s (+62%) |
| G2_context_allinternal_grounding | 260 | 1034 | 1703 | 65.5% | 2 | 0.0077 | 0.0309 | yes | no | 0.0007 [0.0027] | 161 | -236.0 | +4.47 s (+64%) |
| G3_grounding_only | 103 | 1548 | 1346 | 48.3% | 3 | 0.0291 | 0.0915 | no | no | 0.0010 [0.0032] | 116 | -184.1 | +3.30 s (+47%) |
| G4_context_action_baseline | 198 | 1567 | 1232 | 47.7% | 3 | 0.0152 | 0.0484 | yes | no | 0.0010 [0.0032] | 132 | +0.0 | +3.32 s (+48%) |