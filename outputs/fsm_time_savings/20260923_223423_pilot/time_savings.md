# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 1.457 s; feature pass 0.092 s; FSM verifier (path check) 0.3 µs. Probe scoring (median, single row): context_action 0.59 ms, token_confidence 0.52 ms, attention 0.56 ms, attention+token_confidence 0.79 ms, all_internal 13.95 ms, context_action+all_internal 10.07 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result; CSTR routing fractions and CSTR feature cost (≈4.5k-token prompts) must be measured in Stage 4.

## rule = point, α = β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 37 | 458 | 5 | 8% | 2 | 0 | -45.76 | 45.76 | 1.090 | +0.49 (+7%) |
| token_entropy (untrained) | 37 | 456 | 7 | 9% | 2 | 0 | -45.76 | 45.76 | 1.040 | +0.52 (+7%) |
| context_action | 52 | 402 | 46 | 20% | 2 | 1 | -0.00 | 0.00 | 0.003 | +1.36 (+20%) |
| token_confidence | 58 | 395 | 47 | 21% | 2 | 5 | -46.02 | 46.02 | 0.438 | +1.37 (+20%) |
| attention | 64 | 374 | 62 | 25% | 2 | 4 | -46.04 | 46.04 | 0.365 | +1.66 (+24%) |
| attention+token_confidence | 75 | 368 | 57 | 26% | 1 | 1 | -46.16 | 46.16 | 0.350 | +1.74 (+25%) |
| all_internal | 79 | 381 | 40 | 24% | 3 | 3 | -52.74 | 52.74 | 0.443 | +1.55 (+22%) |
| context_action+all_internal | 89 | 330 | 81 | 34% | 3 | 6 | -50.80 | 50.80 | 0.299 | +2.26 (+33%) |

## rule = point, α = β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 53 | 402 | 45 | 20% | 5 | 6 | -45.76 | 45.76 | 0.467 | +1.27 (+18%) |
| token_entropy (untrained) | 53 | 380 | 67 | 24% | 5 | 7 | -45.76 | 45.76 | 0.381 | +1.58 (+23%) |
| context_action | 93 | 322 | 85 | 36% | 13 | 10 | -0.00 | 0.00 | 0.002 | +2.47 (+36%) |
| token_confidence | 75 | 344 | 81 | 31% | 5 | 6 | -46.02 | 46.02 | 0.295 | +2.08 (+30%) |
| attention | 86 | 346 | 68 | 31% | 5 | 4 | -46.04 | 46.04 | 0.299 | +2.05 (+29%) |
| attention+token_confidence | 84 | 327 | 89 | 35% | 2 | 6 | -46.16 | 46.16 | 0.267 | +2.31 (+33%) |
| all_internal | 121 | 267 | 112 | 47% | 9 | 14 | -52.74 | 52.74 | 0.226 | +3.13 (+45%) |
| context_action+all_internal | 120 | 249 | 131 | 50% | 7 | 17 | -50.80 | 50.80 | 0.202 | +3.39 (+49%) |

## rule = point, α = β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 103 | 292 | 105 | 42% | 12 | 23 | -45.76 | 45.76 | 0.220 | +2.80 (+40%) |
| token_entropy (untrained) | 103 | 275 | 122 | 45% | 12 | 27 | -45.76 | 45.76 | 0.203 | +3.04 (+44%) |
| context_action | 162 | 169 | 169 | 66% | 30 | 34 | -0.00 | 0.00 | 0.001 | +4.60 (+66%) |
| token_confidence | 116 | 248 | 136 | 50% | 13 | 19 | -46.02 | 46.02 | 0.183 | +3.41 (+49%) |
| attention | 159 | 173 | 168 | 65% | 26 | 32 | -46.04 | 46.04 | 0.141 | +4.45 (+64%) |
| attention+token_confidence | 152 | 188 | 160 | 62% | 22 | 25 | -46.16 | 46.16 | 0.148 | +4.24 (+61%) |
| all_internal | 193 | 103 | 204 | 79% | 35 | 46 | -52.74 | 52.74 | 0.133 | +5.41 (+78%) |
| context_action+all_internal | 195 | 110 | 195 | 78% | 34 | 41 | -50.80 | 50.80 | 0.130 | +5.32 (+77%) |

## rule = ucb, α = β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 500 | 0 | 0% | 0 | 0 | -0.00 | 0.00 | – | -0.00 (-0%) |
| token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.02 | 46.02 | – | -0.09 (-1%) |
| attention | 0 | 500 | 0 | 0% | 0 | 0 | -46.04 | 46.04 | – | -0.09 (-1%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.16 | 46.16 | – | -0.09 (-1%) |
| all_internal | 0 | 500 | 0 | 0% | 0 | 0 | -52.74 | 52.74 | – | -0.11 (-2%) |
| context_action+all_internal | 65 | 435 | 0 | 13% | 1 | 0 | -50.80 | 50.80 | 0.782 | +0.80 (+12%) |

## rule = ucb, α = β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 45 | 413 | 42 | 17% | 0 | 1 | -0.00 | 0.00 | 0.003 | +1.21 (+17%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.02 | 46.02 | 1.211 | +0.44 (+6%) |
| attention | 40 | 401 | 59 | 20% | 1 | 4 | -46.04 | 46.04 | 0.465 | +1.28 (+18%) |
| attention+token_confidence | 74 | 426 | 0 | 15% | 1 | 0 | -46.16 | 46.16 | 0.624 | +0.94 (+13%) |
| all_internal | 79 | 389 | 32 | 22% | 3 | 2 | -52.74 | 52.74 | 0.475 | +1.44 (+21%) |
| context_action+all_internal | 84 | 371 | 45 | 26% | 2 | 2 | -50.80 | 50.80 | 0.394 | +1.69 (+24%) |

## rule = ucb, α = β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 52 | 380 | 68 | 24% | 5 | 11 | -45.76 | 45.76 | 0.381 | +1.58 (+23%) |
| token_entropy (untrained) | 43 | 390 | 67 | 22% | 3 | 7 | -45.76 | 45.76 | 0.416 | +1.44 (+21%) |
| context_action | 101 | 288 | 111 | 42% | 15 | 13 | -0.00 | 0.00 | 0.001 | +2.95 (+42%) |
| token_confidence | 75 | 340 | 85 | 32% | 5 | 6 | -46.02 | 46.02 | 0.288 | +2.13 (+31%) |
| attention | 132 | 262 | 106 | 48% | 16 | 11 | -46.04 | 46.04 | 0.193 | +3.22 (+46%) |
| attention+token_confidence | 129 | 237 | 134 | 53% | 13 | 19 | -46.16 | 46.16 | 0.176 | +3.56 (+51%) |
| all_internal | 144 | 200 | 156 | 60% | 16 | 29 | -52.74 | 52.74 | 0.176 | +4.06 (+58%) |
| context_action+all_internal | 167 | 183 | 150 | 63% | 21 | 24 | -50.80 | 50.80 | 0.160 | +4.30 (+62%) |
