# Agreement routing vs single-probe routing (pilot, exploratory test_iid, n = 500)

ALLOW thresholds chosen on dev_thr for zero ALLOW failures (point rule, α = 0); DISALLOW β = 0.10 (point) from the single probe / the second probe of a pair. Thresholds frozen before test.
What-if = routing shares × CSTR legacy verifier cost (6.95 s/call) − measured FSM overhead; not a CSTR result.

| Policy | ALLOW (dev) | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through [95% UB rate] | Random | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|
| single: context_action |  | 32 | 202 | 264 | 59% | 0 [0.089] | 22.1 | 26 | -0.3 | +4.13 s (+59%) |
| single: attention+token_confidence |  | 31 | 250 | 217 | 50% | 2 [0.189] | 21.4 | 14 | -30.4 | +3.40 s (+49%) |
| single: all_internal |  | 28 | 287 | 183 | 42% | 2 [0.208] | 19.3 | 14 | -35.1 | +2.87 s (+41%) |
| single: context_action+all_internal |  | 28 | 260 | 210 | 48% | 0 [0.101] | 19.3 | 17 | -33.4 | +3.25 s (+47%) |
| agree: context_action & attention | 28 | 32 | 233 | 233 | 53% | 0 [0.089] | 22.1 | 18 | -30.7 | +3.64 s (+52%) |
| agree: context_action & attention+token_confidence | 37 | 38 | 243 | 217 | 51% | 2 [0.157] | 26.2 | 14 | -30.7 | +3.50 s (+50%) |
| agree: context_action & all_internal | 32 | 32 | 283 | 183 | 43% | 0 [0.089] | 22.1 | 14 | -35.4 | +2.93 s (+42%) |
| agree: token_confidence & attention | 26 | 34 | 231 | 233 | 54% | 2 [0.174] | 23.4 | 18 | -30.6 | +3.66 s (+53%) |