# Agreement routing vs single-probe routing (pilot, exploratory test_iid, n = 500)

ALLOW thresholds chosen on dev_thr for zero ALLOW failures (point rule, α = 0); DISALLOW β = 0.10 (point) from the single probe / the second probe of a pair. Thresholds frozen before test.
What-if = routing shares × CSTR legacy verifier cost (6.95 s/call) − measured FSM overhead; not a CSTR result.

| Policy | ALLOW (dev) | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through [95% UB rate] | Random | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |
|---|---|---|---|---|---|---|---|---|---|---|
| single: context_action |  | 24 | 44 | 425 | 91% | 0 [0.117] | 19.9 | 31 | -0.3 | +6.33 s (+91%) |
| single: attention+token_confidence |  | 1 | 74 | 418 | 85% | 0 [0.950] | 0.8 | 29 | -24.4 | +5.86 s (+84%) |
| single: all_internal |  | 7 | 64 | 422 | 87% | 3 [0.775] | 5.8 | 32 | -30.6 | +5.99 s (+86%) |
| single: context_action+all_internal |  | 19 | 28 | 446 | 94% | 2 [0.296] | 15.7 | 44 | -28.3 | +6.50 s (+93%) |
| agree: context_action & attention | 23 | 24 | 54 | 415 | 89% | 0 [0.117] | 19.9 | 30 | -24.6 | +6.14 s (+88%) |
| agree: context_action & attention+token_confidence | 23 | 24 | 51 | 418 | 90% | 0 [0.117] | 19.9 | 29 | -24.7 | +6.18 s (+89%) |
| agree: context_action & all_internal | 23 | 24 | 47 | 422 | 90% | 0 [0.117] | 19.9 | 32 | -31.0 | +6.22 s (+90%) |
| agree: token_confidence & attention | 6 | 5 | 73 | 415 | 85% | 2 [0.811] | 4.1 | 30 | -24.6 | +5.87 s (+84%) |