# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 0.590 s; feature pass 0.049 s; FSM verifier (path check) 0.2 µs. Probe scoring (median, single row): context_action 0.62 ms, token_confidence 0.60 ms, attention 0.65 ms, attention+token_confidence 0.88 ms, all_internal 13.58 ms, context_action+all_internal 8.89 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result; CSTR routing fractions and CSTR feature cost (≈4.5k-token prompts) must be measured in Stage 4.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 | 115 | 376 | 77% | 1 | 32 | -23.96 | 23.96 | 0.063 | +5.28 (+76%) |
| token_entropy (untrained) | 2 | 118 | 373 | 76% | 1 | 36 | -23.96 | 23.96 | 0.064 | +5.24 (+75%) |
| context_action | 24 | 44 | 425 | 91% | 0 | 31 | -0.00 | 0.00 | 0.001 | +6.33 (+91%) |
| token_confidence | 3 | 108 | 382 | 78% | 1 | 28 | -24.25 | 24.25 | 0.063 | +5.38 (+77%) |
| attention | 2 | 76 | 415 | 85% | 0 | 30 | -24.27 | 24.27 | 0.058 | +5.83 (+84%) |
| attention+token_confidence | 1 | 74 | 418 | 85% | 0 | 29 | -24.39 | 24.39 | 0.058 | +5.86 (+84%) |
| all_internal | 7 | 64 | 422 | 87% | 3 | 32 | -30.65 | 30.65 | 0.071 | +5.99 (+86%) |
| context_action+all_internal | 19 | 28 | 446 | 94% | 2 | 44 | -28.34 | 28.34 | 0.061 | +6.50 (+93%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 | 269 | 222 | 45% | 1 | 12 | -23.96 | 23.96 | 0.107 | +3.11 (+45%) |
| token_entropy (untrained) | 2 | 246 | 245 | 50% | 1 | 11 | -23.96 | 23.96 | 0.097 | +3.43 (+49%) |
| context_action | 33 | 73 | 387 | 85% | 1 | 16 | -0.00 | 0.00 | 0.001 | +5.92 (+85%) |
| token_confidence | 3 | 246 | 244 | 50% | 1 | 6 | -24.25 | 24.25 | 0.098 | +3.43 (+49%) |
| attention | 2 | 132 | 359 | 73% | 0 | 16 | -24.27 | 24.27 | 0.067 | +5.04 (+73%) |
| attention+token_confidence | 1 | 126 | 366 | 74% | 0 | 18 | -24.39 | 24.39 | 0.066 | +5.12 (+74%) |
| all_internal | 7 | 109 | 377 | 78% | 3 | 15 | -30.65 | 30.65 | 0.080 | +5.35 (+77%) |
| context_action+all_internal | 19 | 88 | 386 | 82% | 2 | 17 | -28.34 | 28.34 | 0.070 | +5.65 (+81%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 | 115 | 376 | 77% | 1 | 32 | -23.96 | 23.96 | 0.063 | +5.28 (+76%) |
| token_entropy (untrained) | 2 | 118 | 373 | 76% | 1 | 36 | -23.96 | 23.96 | 0.064 | +5.24 (+75%) |
| context_action | 35 | 33 | 425 | 93% | 3 | 31 | -0.00 | 0.00 | 0.001 | +6.48 (+93%) |
| token_confidence | 3 | 108 | 382 | 78% | 1 | 28 | -24.25 | 24.25 | 0.063 | +5.38 (+77%) |
| attention | 2 | 76 | 415 | 85% | 0 | 30 | -24.27 | 24.27 | 0.058 | +5.83 (+84%) |
| attention+token_confidence | 1 | 74 | 418 | 85% | 0 | 29 | -24.39 | 24.39 | 0.058 | +5.86 (+84%) |
| all_internal | 7 | 64 | 422 | 87% | 3 | 32 | -30.65 | 30.65 | 0.071 | +5.99 (+86%) |
| context_action+all_internal | 35 | 12 | 446 | 98% | 4 | 44 | -28.34 | 28.34 | 0.059 | +6.72 (+97%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 8 | 0 | 485 | 100% | 3 | 77 | -23.96 | 23.96 | 0.049 | +6.90 (+99%) |
| token_entropy (untrained) | 2 | 0 | 491 | 100% | 1 | 81 | -23.96 | 23.96 | 0.049 | +6.90 (+99%) |
| context_action | 35 | 0 | 458 | 100% | 3 | 50 | -0.00 | 0.00 | 0.001 | +6.95 (+100%) |
| token_confidence | 11 | 0 | 482 | 100% | 2 | 73 | -24.25 | 24.25 | 0.049 | +6.90 (+99%) |
| attention | 23 | 0 | 470 | 100% | 4 | 63 | -24.27 | 24.27 | 0.049 | +6.90 (+99%) |
| attention+token_confidence | 27 | 0 | 466 | 100% | 5 | 60 | -24.39 | 24.39 | 0.049 | +6.90 (+99%) |
| all_internal | 7 | 0 | 486 | 100% | 3 | 78 | -30.65 | 30.65 | 0.062 | +6.89 (+99%) |
| context_action+all_internal | 47 | 0 | 446 | 100% | 9 | 44 | -28.34 | 28.34 | 0.057 | +6.89 (+99%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 232 | 261 | 53% | 0 | 16 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| token_entropy (untrained) | 0 | 232 | 261 | 53% | 0 | 13 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| context_action | 0 | 87 | 406 | 82% | 0 | 23 | -0.00 | 0.00 | 0.001 | +5.72 (+82%) |
| token_confidence | 0 | 181 | 312 | 63% | 0 | 14 | -24.25 | 24.25 | 0.078 | +4.35 (+63%) |
| attention | 0 | 103 | 390 | 79% | 0 | 22 | -24.27 | 24.27 | 0.062 | +5.45 (+78%) |
| attention+token_confidence | 0 | 111 | 382 | 77% | 0 | 20 | -24.39 | 24.39 | 0.064 | +5.34 (+77%) |
| all_internal | 0 | 91 | 402 | 82% | 0 | 25 | -30.65 | 30.65 | 0.076 | +5.60 (+81%) |
| context_action+all_internal | 0 | 80 | 413 | 84% | 0 | 29 | -28.34 | 28.34 | 0.069 | +5.76 (+83%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 337 | 156 | 32% | 0 | 8 | -23.96 | 23.96 | 0.154 | +2.15 (+31%) |
| token_entropy (untrained) | 0 | 297 | 196 | 40% | 0 | 8 | -23.96 | 23.96 | 0.122 | +2.71 (+39%) |
| context_action | 0 | 150 | 343 | 70% | 0 | 8 | -0.00 | 0.00 | 0.001 | +4.83 (+70%) |
| token_confidence | 0 | 330 | 163 | 33% | 0 | 1 | -24.25 | 24.25 | 0.149 | +2.25 (+32%) |
| attention | 0 | 170 | 323 | 66% | 0 | 10 | -24.27 | 24.27 | 0.075 | +4.50 (+65%) |
| attention+token_confidence | 0 | 181 | 312 | 63% | 0 | 9 | -24.39 | 24.39 | 0.078 | +4.35 (+63%) |
| all_internal | 0 | 180 | 313 | 63% | 0 | 7 | -30.65 | 30.65 | 0.098 | +4.35 (+63%) |
| context_action+all_internal | 0 | 127 | 366 | 74% | 0 | 8 | -28.34 | 28.34 | 0.077 | +5.10 (+73%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 232 | 261 | 53% | 0 | 16 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| token_entropy (untrained) | 0 | 232 | 261 | 53% | 0 | 13 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| context_action | 0 | 87 | 406 | 82% | 0 | 23 | -0.00 | 0.00 | 0.001 | +5.72 (+82%) |
| token_confidence | 0 | 181 | 312 | 63% | 0 | 14 | -24.25 | 24.25 | 0.078 | +4.35 (+63%) |
| attention | 0 | 103 | 390 | 79% | 0 | 22 | -24.27 | 24.27 | 0.062 | +5.45 (+78%) |
| attention+token_confidence | 0 | 111 | 382 | 77% | 0 | 20 | -24.39 | 24.39 | 0.064 | +5.34 (+77%) |
| all_internal | 0 | 91 | 402 | 82% | 0 | 25 | -30.65 | 30.65 | 0.076 | +5.60 (+81%) |
| context_action+all_internal | 0 | 80 | 413 | 84% | 0 | 29 | -28.34 | 28.34 | 0.069 | +5.76 (+83%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 2 | 491 | 100% | 0 | 81 | -23.96 | 23.96 | 0.049 | +6.87 (+99%) |
| token_entropy (untrained) | 0 | 2 | 491 | 100% | 0 | 81 | -23.96 | 23.96 | 0.049 | +6.87 (+99%) |
| context_action | 33 | 0 | 460 | 100% | 1 | 50 | -0.00 | 0.00 | 0.001 | +6.95 (+100%) |
| token_confidence | 0 | 1 | 492 | 100% | 0 | 82 | -24.25 | 24.25 | 0.049 | +6.89 (+99%) |
| attention | 0 | 1 | 492 | 100% | 0 | 81 | -24.27 | 24.27 | 0.049 | +6.89 (+99%) |
| attention+token_confidence | 0 | 1 | 492 | 100% | 0 | 81 | -24.39 | 24.39 | 0.050 | +6.89 (+99%) |
| all_internal | 0 | 0 | 493 | 100% | 0 | 82 | -30.65 | 30.65 | 0.062 | +6.89 (+99%) |
| context_action+all_internal | 19 | 0 | 474 | 100% | 2 | 65 | -28.34 | 28.34 | 0.057 | +6.89 (+99%) |
