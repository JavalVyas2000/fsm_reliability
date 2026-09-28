# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 493)

Measured per candidate (means): generation 0.590 s; feature pass 0.049 s; FSM verifier (path check) 0.3 µs. Probe scoring (median, single row): context_action 0.56 ms, token_confidence 0.55 ms, attention 0.70 ms, attention+token_confidence 0.66 ms, all_internal 10.17 ms, context_action+all_internal 9.94 ms, grounding 0.55 ms, context_action+grounding 0.56 ms, context_action+all_internal+grounding 9.54 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 | 115 | 376 | 77% | 1 | 32 | -23.96 | 23.96 | 0.063 | +5.28 (+76%) |
| token_entropy (untrained) | 2 | 118 | 373 | 76% | 1 | 36 | -23.96 | 23.96 | 0.064 | +5.24 (+75%) |
| context_action | 24 | 44 | 425 | 91% | 0 | 31 | -0.00 | 0.00 | 0.001 | +6.33 (+91%) |
| token_confidence | 3 | 108 | 382 | 78% | 1 | 28 | -24.23 | 24.23 | 0.063 | +5.38 (+77%) |
| attention | 2 | 76 | 415 | 85% | 0 | 30 | -24.30 | 24.30 | 0.058 | +5.83 (+84%) |
| attention+token_confidence | 1 | 74 | 418 | 85% | 0 | 29 | -24.28 | 24.28 | 0.058 | +5.86 (+84%) |
| all_internal | 7 | 64 | 422 | 87% | 3 | 32 | -28.97 | 28.97 | 0.068 | +5.99 (+86%) |
| context_action+all_internal | 19 | 28 | 446 | 94% | 2 | 44 | -28.86 | 28.86 | 0.062 | +6.50 (+93%) |
| grounding | 5 | 50 | 438 | 90% | 0 | 36 | -24.23 | 24.23 | 0.055 | +6.20 (+89%) |
| context_action+grounding | 11 | 38 | 444 | 92% | 0 | 35 | -24.23 | 24.23 | 0.053 | +6.37 (+92%) |
| context_action+all_internal+grounding | 28 | 22 | 443 | 96% | 1 | 38 | -28.66 | 28.66 | 0.061 | +6.58 (+95%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 | 269 | 222 | 45% | 1 | 12 | -23.96 | 23.96 | 0.107 | +3.11 (+45%) |
| token_entropy (untrained) | 2 | 246 | 245 | 50% | 1 | 11 | -23.96 | 23.96 | 0.097 | +3.43 (+49%) |
| context_action | 33 | 73 | 387 | 85% | 1 | 16 | -0.00 | 0.00 | 0.001 | +5.92 (+85%) |
| token_confidence | 3 | 246 | 244 | 50% | 1 | 6 | -24.23 | 24.23 | 0.098 | +3.43 (+49%) |
| attention | 2 | 132 | 359 | 73% | 0 | 16 | -24.30 | 24.30 | 0.067 | +5.04 (+73%) |
| attention+token_confidence | 1 | 126 | 366 | 74% | 0 | 18 | -24.28 | 24.28 | 0.066 | +5.12 (+74%) |
| all_internal | 7 | 109 | 377 | 78% | 3 | 15 | -28.97 | 28.97 | 0.075 | +5.35 (+77%) |
| context_action+all_internal | 19 | 88 | 386 | 82% | 2 | 17 | -28.86 | 28.86 | 0.071 | +5.65 (+81%) |
| grounding | 5 | 96 | 392 | 81% | 0 | 18 | -24.23 | 24.23 | 0.061 | +5.55 (+80%) |
| context_action+grounding | 11 | 69 | 413 | 86% | 0 | 19 | -24.23 | 24.23 | 0.057 | +5.93 (+85%) |
| context_action+all_internal+grounding | 44 | 34 | 415 | 93% | 4 | 22 | -28.66 | 28.66 | 0.062 | +6.41 (+92%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 | 115 | 376 | 77% | 1 | 32 | -23.96 | 23.96 | 0.063 | +5.28 (+76%) |
| token_entropy (untrained) | 2 | 118 | 373 | 76% | 1 | 36 | -23.96 | 23.96 | 0.064 | +5.24 (+75%) |
| context_action | 35 | 33 | 425 | 93% | 3 | 31 | -0.00 | 0.00 | 0.001 | +6.48 (+93%) |
| token_confidence | 3 | 108 | 382 | 78% | 1 | 28 | -24.23 | 24.23 | 0.063 | +5.38 (+77%) |
| attention | 2 | 76 | 415 | 85% | 0 | 30 | -24.30 | 24.30 | 0.058 | +5.83 (+84%) |
| attention+token_confidence | 1 | 74 | 418 | 85% | 0 | 29 | -24.28 | 24.28 | 0.058 | +5.86 (+84%) |
| all_internal | 7 | 64 | 422 | 87% | 3 | 32 | -28.97 | 28.97 | 0.068 | +5.99 (+86%) |
| context_action+all_internal | 35 | 12 | 446 | 98% | 4 | 44 | -28.86 | 28.86 | 0.060 | +6.72 (+97%) |
| grounding | 5 | 50 | 438 | 90% | 0 | 36 | -24.23 | 24.23 | 0.055 | +6.20 (+89%) |
| context_action+grounding | 29 | 20 | 444 | 96% | 0 | 35 | -24.23 | 24.23 | 0.051 | +6.62 (+95%) |
| context_action+all_internal+grounding | 62 | 0 | 431 | 100% | 9 | 29 | -28.66 | 28.66 | 0.058 | +6.89 (+99%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 8 | 0 | 485 | 100% | 3 | 77 | -23.96 | 23.96 | 0.049 | +6.90 (+99%) |
| token_entropy (untrained) | 2 | 0 | 491 | 100% | 1 | 81 | -23.96 | 23.96 | 0.049 | +6.90 (+99%) |
| context_action | 35 | 0 | 458 | 100% | 3 | 50 | -0.00 | 0.00 | 0.001 | +6.95 (+100%) |
| token_confidence | 11 | 0 | 482 | 100% | 2 | 73 | -24.23 | 24.23 | 0.049 | +6.90 (+99%) |
| attention | 23 | 0 | 470 | 100% | 4 | 63 | -24.30 | 24.30 | 0.049 | +6.90 (+99%) |
| attention+token_confidence | 27 | 0 | 466 | 100% | 5 | 60 | -24.28 | 24.28 | 0.049 | +6.90 (+99%) |
| all_internal | 7 | 0 | 486 | 100% | 3 | 78 | -28.97 | 28.97 | 0.059 | +6.89 (+99%) |
| context_action+all_internal | 47 | 0 | 446 | 100% | 9 | 44 | -28.86 | 28.86 | 0.059 | +6.89 (+99%) |
| grounding | 15 | 0 | 478 | 100% | 0 | 67 | -24.23 | 24.23 | 0.049 | +6.90 (+99%) |
| context_action+grounding | 61 | 0 | 432 | 100% | 6 | 27 | -24.23 | 24.23 | 0.049 | +6.90 (+99%) |
| context_action+all_internal+grounding | 73 | 0 | 420 | 100% | 14 | 23 | -28.66 | 28.66 | 0.058 | +6.89 (+99%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 232 | 261 | 53% | 0 | 16 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| token_entropy (untrained) | 0 | 232 | 261 | 53% | 0 | 13 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| context_action | 0 | 87 | 406 | 82% | 0 | 23 | -0.00 | 0.00 | 0.001 | +5.72 (+82%) |
| token_confidence | 0 | 181 | 312 | 63% | 0 | 14 | -24.23 | 24.23 | 0.078 | +4.35 (+63%) |
| attention | 0 | 103 | 390 | 79% | 0 | 22 | -24.30 | 24.30 | 0.062 | +5.45 (+78%) |
| attention+token_confidence | 0 | 111 | 382 | 77% | 0 | 20 | -24.28 | 24.28 | 0.064 | +5.34 (+77%) |
| all_internal | 0 | 91 | 402 | 82% | 0 | 25 | -28.97 | 28.97 | 0.072 | +5.61 (+81%) |
| context_action+all_internal | 0 | 80 | 413 | 84% | 0 | 29 | -28.86 | 28.86 | 0.070 | +5.76 (+83%) |
| grounding | 0 | 80 | 413 | 84% | 0 | 27 | -24.23 | 24.23 | 0.059 | +5.77 (+83%) |
| context_action+grounding | 0 | 67 | 426 | 86% | 0 | 26 | -24.23 | 24.23 | 0.057 | +5.96 (+86%) |
| context_action+all_internal+grounding | 0 | 62 | 431 | 87% | 0 | 29 | -28.66 | 28.66 | 0.066 | +6.02 (+87%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 337 | 156 | 32% | 0 | 8 | -23.96 | 23.96 | 0.154 | +2.15 (+31%) |
| token_entropy (untrained) | 0 | 297 | 196 | 40% | 0 | 8 | -23.96 | 23.96 | 0.122 | +2.71 (+39%) |
| context_action | 0 | 150 | 343 | 70% | 0 | 8 | -0.00 | 0.00 | 0.001 | +4.83 (+70%) |
| token_confidence | 0 | 330 | 163 | 33% | 0 | 1 | -24.23 | 24.23 | 0.149 | +2.25 (+32%) |
| attention | 0 | 170 | 323 | 66% | 0 | 10 | -24.30 | 24.30 | 0.075 | +4.50 (+65%) |
| attention+token_confidence | 0 | 181 | 312 | 63% | 0 | 9 | -24.28 | 24.28 | 0.078 | +4.35 (+63%) |
| all_internal | 0 | 180 | 313 | 63% | 0 | 7 | -28.97 | 28.97 | 0.093 | +4.35 (+63%) |
| context_action+all_internal | 0 | 127 | 366 | 74% | 0 | 8 | -28.86 | 28.86 | 0.079 | +5.10 (+73%) |
| grounding | 0 | 152 | 341 | 69% | 0 | 6 | -24.23 | 24.23 | 0.071 | +4.76 (+68%) |
| context_action+grounding | 0 | 98 | 395 | 80% | 0 | 12 | -24.23 | 24.23 | 0.061 | +5.52 (+79%) |
| context_action+all_internal+grounding | 0 | 123 | 370 | 75% | 0 | 6 | -28.66 | 28.66 | 0.077 | +5.16 (+74%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 232 | 261 | 53% | 0 | 16 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| token_entropy (untrained) | 0 | 232 | 261 | 53% | 0 | 13 | -23.96 | 23.96 | 0.092 | +3.63 (+52%) |
| context_action | 0 | 87 | 406 | 82% | 0 | 23 | -0.00 | 0.00 | 0.001 | +5.72 (+82%) |
| token_confidence | 0 | 181 | 312 | 63% | 0 | 14 | -24.23 | 24.23 | 0.078 | +4.35 (+63%) |
| attention | 0 | 103 | 390 | 79% | 0 | 22 | -24.30 | 24.30 | 0.062 | +5.45 (+78%) |
| attention+token_confidence | 0 | 111 | 382 | 77% | 0 | 20 | -24.28 | 24.28 | 0.064 | +5.34 (+77%) |
| all_internal | 0 | 91 | 402 | 82% | 0 | 25 | -28.97 | 28.97 | 0.072 | +5.61 (+81%) |
| context_action+all_internal | 0 | 80 | 413 | 84% | 0 | 29 | -28.86 | 28.86 | 0.070 | +5.76 (+83%) |
| grounding | 0 | 80 | 413 | 84% | 0 | 27 | -24.23 | 24.23 | 0.059 | +5.77 (+83%) |
| context_action+grounding | 0 | 67 | 426 | 86% | 0 | 26 | -24.23 | 24.23 | 0.057 | +5.96 (+86%) |
| context_action+all_internal+grounding | 0 | 62 | 431 | 87% | 0 | 29 | -28.66 | 28.66 | 0.066 | +6.02 (+87%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 493 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 2 | 491 | 100% | 0 | 81 | -23.96 | 23.96 | 0.049 | +6.87 (+99%) |
| token_entropy (untrained) | 0 | 2 | 491 | 100% | 0 | 81 | -23.96 | 23.96 | 0.049 | +6.87 (+99%) |
| context_action | 33 | 0 | 460 | 100% | 1 | 50 | -0.00 | 0.00 | 0.001 | +6.95 (+100%) |
| token_confidence | 0 | 1 | 492 | 100% | 0 | 82 | -24.23 | 24.23 | 0.049 | +6.89 (+99%) |
| attention | 0 | 1 | 492 | 100% | 0 | 81 | -24.30 | 24.30 | 0.049 | +6.89 (+99%) |
| attention+token_confidence | 0 | 1 | 492 | 100% | 0 | 81 | -24.28 | 24.28 | 0.049 | +6.89 (+99%) |
| all_internal | 0 | 0 | 493 | 100% | 0 | 82 | -28.97 | 28.97 | 0.059 | +6.89 (+99%) |
| context_action+all_internal | 19 | 0 | 474 | 100% | 2 | 65 | -28.86 | 28.86 | 0.059 | +6.89 (+99%) |
| grounding | 0 | 2 | 491 | 100% | 0 | 80 | -24.23 | 24.23 | 0.049 | +6.87 (+99%) |
| context_action+grounding | 0 | 2 | 491 | 100% | 0 | 80 | -24.23 | 24.23 | 0.049 | +6.87 (+99%) |
| context_action+all_internal+grounding | 63 | 0 | 430 | 100% | 10 | 29 | -28.66 | 28.66 | 0.058 | +6.89 (+99%) |
