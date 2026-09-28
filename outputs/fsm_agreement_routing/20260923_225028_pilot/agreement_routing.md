# Agreement routing vs single-probe routing (pilot, exploratory test_iid, n = 500)

ALLOW thresholds chosen on dev_thr for zero ALLOW failures (point rule, α = 0); DISALLOW β = 0.10 (point) from the single probe / the second probe of a pair. Thresholds frozen before test.
What-if = routing shares × CSTR legacy verifier cost (6.95 s/call) − measured FSM overhead; not a CSTR result.

| Policy | ALLOW (dev) | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through [95% UB rate] | Random | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|
| single: context_action |  | 45 | 370 | 85 | 26% | 0 [0.064] | 21.9 | 10 | -0.3 | +1.81 s (+26%) |
| single: attention+token_confidence |  | 40 | 371 | 89 | 26% | 0 [0.072] | 19.5 | 6 | -46.1 | +1.70 s (+24%) |
| single: all_internal |  | 41 | 347 | 112 | 31% | 0 [0.070] | 20.0 | 14 | -51.0 | +2.02 s (+29%) |
| single: context_action+all_internal |  | 65 | 304 | 131 | 39% | 1 [0.071] | 31.7 | 17 | -50.6 | +2.62 s (+38%) |
| agree: context_action & attention | 55 | 46 | 386 | 68 | 23% | 1 [0.099] | 22.4 | 4 | -46.3 | +1.49 s (+21%) |
| agree: context_action & attention+token_confidence | 58 | 54 | 357 | 89 | 29% | 1 [0.085] | 26.3 | 6 | -46.3 | +1.90 s (+27%) |
| agree: context_action & all_internal | 65 | 68 | 320 | 112 | 36% | 1 [0.068] | 33.1 | 14 | -51.3 | +2.40 s (+35%) |
| agree: token_confidence & attention | 57 | 62 | 370 | 68 | 26% | 1 [0.074] | 30.2 | 4 | -46.3 | +1.71 s (+25%) |