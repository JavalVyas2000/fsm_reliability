# Agreement routing vs single-probe routing (pilot, exploratory test_iid, n = 500)

ALLOW thresholds chosen on dev_thr for zero ALLOW failures (point rule, α = 0); DISALLOW β = 0.10 (point) from the single probe / the second probe of a pair. Thresholds frozen before test.
What-if = routing shares × CSTR legacy verifier cost (6.95 s/call) − measured FSM overhead; not a CSTR result.

| Policy | ALLOW (dev) | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through [95% UB rate] | Random | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|
| single: context_action |  | 39 | 268 | 193 | 46% | 0 [0.074] | 24.5 | 14 | -0.3 | +3.22 s (+46%) |
| single: attention+token_confidence |  | 25 | 309 | 166 | 38% | 0 [0.113] | 15.8 | 17 | -27.9 | +2.60 s (+37%) |
| single: all_internal |  | 26 | 274 | 200 | 45% | 1 [0.170] | 16.4 | 22 | -34.5 | +3.07 s (+44%) |
| single: context_action+all_internal |  | 47 | 235 | 218 | 53% | 2 [0.128] | 29.5 | 22 | -35.9 | +3.61 s (+52%) |
| agree: context_action & attention | 38 | 38 | 322 | 140 | 36% | 0 [0.076] | 23.8 | 11 | -28.1 | +2.42 s (+35%) |
| agree: context_action & attention+token_confidence | 38 | 38 | 296 | 166 | 41% | 0 [0.076] | 23.8 | 17 | -28.1 | +2.78 s (+40%) |
| agree: context_action & all_internal | 43 | 37 | 263 | 200 | 47% | 1 [0.122] | 23.2 | 22 | -34.8 | +3.22 s (+46%) |
| agree: token_confidence & attention | 31 | 33 | 327 | 140 | 35% | 1 [0.136] | 20.7 | 11 | -28.1 | +2.35 s (+34%) |