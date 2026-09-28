# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 0.789 s; feature pass 0.055 s; FSM verifier (path check) 0.3 µs. Probe scoring (median, single row): context_action 0.50 ms, token_confidence 0.51 ms, attention 0.53 ms, attention+token_confidence 0.53 ms, all_internal 13.89 ms, context_action+all_internal 16.71 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result; CSTR routing fractions and CSTR feature cost (≈4.5k-token prompts) must be measured in Stage 4.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 4 | 431 | 65 | 14% | 0 | 13 | -27.59 | 27.59 | 0.400 | +0.90 (+13%) |
| token_entropy (untrained) | 3 | 383 | 114 | 23% | 0 | 16 | -27.59 | 27.59 | 0.236 | +1.57 (+23%) |
| context_action | 39 | 268 | 193 | 46% | 0 | 14 | -0.00 | 0.00 | 0.001 | +3.22 (+46%) |
| token_confidence | 0 | 386 | 114 | 23% | 0 | 11 | -27.84 | 27.84 | 0.244 | +1.53 (+22%) |
| attention | 3 | 357 | 140 | 29% | 0 | 11 | -27.85 | 27.85 | 0.195 | +1.93 (+28%) |
| attention+token_confidence | 25 | 309 | 166 | 38% | 0 | 17 | -27.85 | 27.85 | 0.146 | +2.60 (+37%) |
| all_internal | 26 | 274 | 200 | 45% | 1 | 22 | -34.53 | 34.53 | 0.153 | +3.07 (+44%) |
| context_action+all_internal | 47 | 235 | 218 | 53% | 2 | 22 | -35.94 | 35.94 | 0.136 | +3.61 (+52%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 22 | 478 | 0 | 4% | 1 | 0 | -27.59 | 27.59 | 1.254 | +0.25 (+4%) |
| token_entropy (untrained) | 3 | 437 | 60 | 13% | 0 | 12 | -27.59 | 27.59 | 0.438 | +0.82 (+12%) |
| context_action | 69 | 292 | 139 | 42% | 4 | 8 | -0.00 | 0.00 | 0.001 | +2.89 (+42%) |
| token_confidence | 32 | 418 | 50 | 16% | 3 | 4 | -27.84 | 27.84 | 0.340 | +1.08 (+16%) |
| attention | 3 | 418 | 79 | 16% | 0 | 3 | -27.85 | 27.85 | 0.340 | +1.08 (+16%) |
| attention+token_confidence | 42 | 343 | 115 | 31% | 1 | 9 | -27.85 | 27.85 | 0.177 | +2.13 (+31%) |
| all_internal | 43 | 337 | 120 | 33% | 1 | 11 | -34.53 | 34.53 | 0.212 | +2.20 (+32%) |
| context_action+all_internal | 74 | 261 | 165 | 48% | 4 | 9 | -35.94 | 35.94 | 0.150 | +3.25 (+47%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 51 | 384 | 65 | 23% | 10 | 13 | -27.59 | 27.59 | 0.238 | +1.56 (+22%) |
| token_entropy (untrained) | 51 | 335 | 114 | 33% | 9 | 16 | -27.59 | 27.59 | 0.167 | +2.24 (+32%) |
| context_action | 71 | 236 | 193 | 53% | 6 | 14 | -0.00 | 0.00 | 0.001 | +3.67 (+53%) |
| token_confidence | 46 | 340 | 114 | 32% | 6 | 11 | -27.84 | 27.84 | 0.174 | +2.17 (+31%) |
| attention | 69 | 291 | 140 | 42% | 6 | 11 | -27.85 | 27.85 | 0.133 | +2.85 (+41%) |
| attention+token_confidence | 80 | 254 | 166 | 49% | 4 | 17 | -27.85 | 27.85 | 0.113 | +3.36 (+48%) |
| all_internal | 81 | 219 | 200 | 56% | 8 | 22 | -34.53 | 34.53 | 0.123 | +3.84 (+55%) |
| context_action+all_internal | 95 | 187 | 218 | 63% | 7 | 22 | -35.94 | 35.94 | 0.115 | +4.28 (+62%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 74 | 174 | 252 | 65% | 15 | 51 | -27.59 | 27.59 | 0.085 | +4.48 (+64%) |
| token_entropy (untrained) | 71 | 210 | 219 | 58% | 16 | 41 | -27.59 | 27.59 | 0.095 | +3.98 (+57%) |
| context_action | 117 | 52 | 331 | 90% | 21 | 58 | -0.00 | 0.00 | 0.001 | +6.23 (+90%) |
| token_confidence | 74 | 127 | 299 | 75% | 13 | 56 | -27.84 | 27.84 | 0.075 | +5.13 (+74%) |
| attention | 116 | 39 | 345 | 92% | 17 | 63 | -27.85 | 27.85 | 0.060 | +6.35 (+91%) |
| attention+token_confidence | 124 | 14 | 362 | 97% | 16 | 67 | -27.85 | 27.85 | 0.057 | +6.70 (+96%) |
| all_internal | 131 | 25 | 344 | 95% | 26 | 70 | -34.53 | 34.53 | 0.073 | +6.53 (+94%) |
| context_action+all_internal | 143 | 18 | 339 | 96% | 28 | 62 | -35.94 | 35.94 | 0.075 | +6.63 (+95%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| context_action | 0 | 339 | 161 | 32% | 0 | 11 | -0.00 | 0.00 | 0.002 | +2.24 (+32%) |
| token_confidence | 0 | 457 | 43 | 9% | 0 | 3 | -27.84 | 27.84 | 0.647 | +0.54 (+8%) |
| attention | 0 | 425 | 75 | 15% | 0 | 3 | -27.85 | 27.85 | 0.371 | +0.99 (+14%) |
| attention+token_confidence | 0 | 385 | 115 | 23% | 0 | 9 | -27.85 | 27.85 | 0.242 | +1.54 (+22%) |
| all_internal | 0 | 349 | 151 | 30% | 0 | 14 | -34.53 | 34.53 | 0.229 | +2.03 (+29%) |
| context_action+all_internal | 0 | 325 | 175 | 35% | 0 | 13 | -35.94 | 35.94 | 0.205 | +2.36 (+34%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| context_action | 0 | 380 | 120 | 24% | 0 | 7 | -0.00 | 0.00 | 0.002 | +1.67 (+24%) |
| token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -27.84 | 27.84 | – | -0.06 (-1%) |
| attention | 0 | 500 | 0 | 0% | 0 | 0 | -27.85 | 27.85 | – | -0.06 (-1%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -27.85 | 27.85 | – | -0.06 (-1%) |
| all_internal | 0 | 422 | 78 | 16% | 0 | 3 | -34.53 | 34.53 | 0.443 | +1.02 (+15%) |
| context_action+all_internal | 0 | 423 | 77 | 15% | 0 | 0 | -35.94 | 35.94 | 0.467 | +1.00 (+14%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| context_action | 39 | 300 | 161 | 40% | 0 | 11 | -0.00 | 0.00 | 0.001 | +2.78 (+40%) |
| token_confidence | 0 | 457 | 43 | 9% | 0 | 3 | -27.84 | 27.84 | 0.647 | +0.54 (+8%) |
| attention | 0 | 425 | 75 | 15% | 0 | 3 | -27.85 | 27.85 | 0.371 | +0.99 (+14%) |
| attention+token_confidence | 0 | 385 | 115 | 23% | 0 | 9 | -27.85 | 27.85 | 0.242 | +1.54 (+22%) |
| all_internal | 43 | 306 | 151 | 39% | 1 | 14 | -34.53 | 34.53 | 0.178 | +2.63 (+38%) |
| context_action+all_internal | 67 | 258 | 175 | 48% | 3 | 13 | -35.94 | 35.94 | 0.149 | +3.29 (+47%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 46 | 380 | 74 | 24% | 8 | 14 | -27.59 | 27.59 | 0.230 | +1.61 (+23%) |
| token_entropy (untrained) | 51 | 287 | 162 | 43% | 9 | 24 | -27.59 | 27.59 | 0.130 | +2.91 (+42%) |
| context_action | 71 | 139 | 290 | 72% | 6 | 39 | -0.00 | 0.00 | 0.001 | +5.02 (+72%) |
| token_confidence | 33 | 253 | 214 | 49% | 4 | 28 | -27.84 | 27.84 | 0.113 | +3.38 (+49%) |
| attention | 88 | 167 | 245 | 67% | 8 | 33 | -27.85 | 27.85 | 0.084 | +4.57 (+66%) |
| attention+token_confidence | 96 | 112 | 292 | 78% | 10 | 42 | -27.85 | 27.85 | 0.072 | +5.34 (+77%) |
| all_internal | 90 | 134 | 276 | 73% | 9 | 41 | -34.53 | 34.53 | 0.094 | +5.02 (+72%) |
| context_action+all_internal | 105 | 94 | 301 | 81% | 13 | 43 | -35.94 | 35.94 | 0.089 | +5.57 (+80%) |
