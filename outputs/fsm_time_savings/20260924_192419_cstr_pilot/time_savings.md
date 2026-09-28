# CSTR routing table with computational time saved (pilot, exploratory test_iid, n = 50)

Measured per candidate (means): generation 9.73 s; feature pass 0.266 s; verifier 7.60 s (CSTR legacy verifier, sequential re-timing (outputs/cstr_labeled/20260924_184632_20260924_175326_qwen25-3b-instruct_pilot300_v21/verifier_timing_sequential_test_iid.json)). Probe scoring (median, single row): context_action 0.80 ms, token_confidence 0.74 ms, attention 0.88 ms, attention+token_confidence 0.86 ms, all_internal 13.95 ms, context_action+all_internal 13.67 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates), all measured. Re-proposal cost after DISALLOW is not included. The last column repeats the arithmetic at the Stage 0 reference cost (6.95 s/call) for comparison only.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| context_action | 7 | 32 | 11 | 36% | 0 | 2 | +135.26 | 0.00 | 0.002 | +2.50 (+36%) |
| token_confidence | 0 | 44 | 6 | 12% | 0 | 1 | +38.19 | 13.36 | 2.227 | +0.57 (+8%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 10 | 40 | 0 | 20% | 6 | 0 | +65.42 | 14.01 | 1.401 | +1.11 (+16%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| context_action | 7 | 32 | 11 | 36% | 0 | 2 | +135.26 | 0.00 | 0.002 | +2.50 (+36%) |
| token_confidence | 0 | 44 | 6 | 12% | 0 | 1 | +38.19 | 13.36 | 2.227 | +0.57 (+8%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 10 | 40 | 0 | 20% | 6 | 0 | +65.42 | 14.01 | 1.401 | +1.11 (+16%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| context_action | 15 | 24 | 11 | 52% | 5 | 2 | +198.98 | 0.00 | 0.002 | +3.61 (+52%) |
| token_confidence | 0 | 44 | 6 | 12% | 0 | 1 | +38.19 | 13.36 | 2.227 | +0.57 (+8%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 10 | 40 | 0 | 20% | 6 | 0 | +65.42 | 14.01 | 1.401 | +1.11 (+16%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| context_action | 24 | 9 | 17 | 82% | 9 | 4 | +308.31 | 0.00 | 0.001 | +5.70 (+82%) |
| token_confidence | 0 | 43 | 7 | 14% | 0 | 2 | +45.84 | 13.36 | 1.909 | +0.71 (+10%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 40 | 10 | 20% | 0 | 4 | +60.04 | 13.37 | 1.337 | +1.12 (+16%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 10 | 40 | 0 | 20% | 6 | 0 | +65.42 | 14.01 | 1.401 | +1.11 (+16%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 47 | 3 | 6% | 0 | 1 | +14.32 | 13.32 | 4.441 | +0.15 (+2%) |
| context_action | 0 | 50 | 0 | 0% | 0 | 0 | -0.00 | 0.00 | – | -0.00 (-0%) |
| token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.36 | 13.36 | – | -0.27 (-4%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.01 | 14.01 | – | -0.28 (-4%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 47 | 3 | 6% | 0 | 1 | +14.32 | 13.32 | 4.441 | +0.15 (+2%) |
| context_action | 0 | 50 | 0 | 0% | 0 | 0 | -0.00 | 0.00 | – | -0.00 (-0%) |
| token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.36 | 13.36 | – | -0.27 (-4%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.01 | 14.01 | – | -0.28 (-4%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 47 | 3 | 6% | 0 | 1 | +14.32 | 13.32 | 4.441 | +0.15 (+2%) |
| context_action | 0 | 50 | 0 | 0% | 0 | 0 | -0.00 | 0.00 | – | -0.00 (-0%) |
| token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.36 | 13.36 | – | -0.27 (-4%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.01 | 14.01 | – | -0.28 (-4%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | CSTR time saved (s, 50 cands, measured) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 50 | 0 | 0% | 0 | 0 | -13.32 | 13.32 | – | -0.27 (-4%) |
| token_entropy (untrained) | 0 | 47 | 3 | 6% | 0 | 1 | +14.32 | 13.32 | 4.441 | +0.15 (+2%) |
| context_action | 0 | 50 | 0 | 0% | 0 | 0 | -0.00 | 0.00 | – | -0.00 (-0%) |
| token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.36 | 13.36 | – | -0.27 (-4%) |
| attention | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| attention+token_confidence | 0 | 50 | 0 | 0% | 0 | 0 | -13.37 | 13.37 | – | -0.27 (-4%) |
| all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.02 | 14.02 | – | -0.28 (-4%) |
| context_action+all_internal | 0 | 50 | 0 | 0% | 0 | 0 | -14.01 | 14.01 | – | -0.28 (-4%) |
