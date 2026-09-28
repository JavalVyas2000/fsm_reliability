# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 0.789 s; feature pass 0.055 s; FSM verifier (path check) 0.3 µs. Probe scoring (median, single row): context_action 0.62 ms, token_confidence 0.61 ms, attention 0.65 ms, attention+token_confidence 0.64 ms, all_internal 16.58 ms, context_action+all_internal 15.45 ms, grounding 0.62 ms, context_action+grounding 0.64 ms, context_action+all_internal+grounding 15.92 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 4 | 431 | 65 | 14% | 0 | 13 | -27.59 | 27.59 | 0.400 | +0.90 (+13%) |
| token_entropy (untrained) | 3 | 383 | 114 | 23% | 0 | 16 | -27.59 | 27.59 | 0.236 | +1.57 (+23%) |
| context_action | 39 | 268 | 193 | 46% | 0 | 14 | -0.00 | 0.00 | 0.001 | +3.22 (+46%) |
| token_confidence | 0 | 386 | 114 | 23% | 0 | 11 | -27.89 | 27.89 | 0.245 | +1.53 (+22%) |
| attention | 3 | 357 | 140 | 29% | 0 | 11 | -27.91 | 27.91 | 0.195 | +1.93 (+28%) |
| attention+token_confidence | 25 | 309 | 166 | 38% | 0 | 17 | -27.91 | 27.91 | 0.146 | +2.60 (+37%) |
| all_internal | 26 | 274 | 200 | 45% | 1 | 22 | -35.88 | 35.88 | 0.159 | +3.07 (+44%) |
| context_action+all_internal | 47 | 235 | 218 | 53% | 2 | 22 | -35.31 | 35.31 | 0.133 | +3.61 (+52%) |
| grounding | 20 | 261 | 219 | 48% | 1 | 17 | -27.90 | 27.90 | 0.117 | +3.27 (+47%) |
| context_action+grounding | 31 | 201 | 268 | 60% | 0 | 24 | -27.91 | 27.91 | 0.093 | +4.10 (+59%) |
| context_action+all_internal+grounding | 51 | 170 | 279 | 66% | 0 | 26 | -35.55 | 35.55 | 0.108 | +4.52 (+65%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 22 | 478 | 0 | 4% | 1 | 0 | -27.59 | 27.59 | 1.254 | +0.25 (+4%) |
| token_entropy (untrained) | 3 | 437 | 60 | 13% | 0 | 12 | -27.59 | 27.59 | 0.438 | +0.82 (+12%) |
| context_action | 69 | 292 | 139 | 42% | 4 | 8 | -0.00 | 0.00 | 0.001 | +2.89 (+42%) |
| token_confidence | 32 | 418 | 50 | 16% | 3 | 4 | -27.89 | 27.89 | 0.340 | +1.08 (+16%) |
| attention | 3 | 418 | 79 | 16% | 0 | 3 | -27.91 | 27.91 | 0.340 | +1.08 (+16%) |
| attention+token_confidence | 42 | 343 | 115 | 31% | 1 | 9 | -27.91 | 27.91 | 0.178 | +2.13 (+31%) |
| all_internal | 43 | 337 | 120 | 33% | 1 | 11 | -35.88 | 35.88 | 0.220 | +2.19 (+32%) |
| context_action+all_internal | 74 | 261 | 165 | 48% | 4 | 9 | -35.31 | 35.31 | 0.148 | +3.25 (+47%) |
| grounding | 21 | 329 | 150 | 34% | 1 | 8 | -27.90 | 27.90 | 0.163 | +2.32 (+33%) |
| context_action+grounding | 52 | 248 | 200 | 50% | 1 | 7 | -27.91 | 27.91 | 0.111 | +3.45 (+50%) |
| context_action+all_internal+grounding | 76 | 217 | 207 | 57% | 2 | 10 | -35.55 | 35.55 | 0.126 | +3.86 (+56%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 51 | 384 | 65 | 23% | 10 | 13 | -27.59 | 27.59 | 0.238 | +1.56 (+22%) |
| token_entropy (untrained) | 51 | 335 | 114 | 33% | 9 | 16 | -27.59 | 27.59 | 0.167 | +2.24 (+32%) |
| context_action | 71 | 236 | 193 | 53% | 6 | 14 | -0.00 | 0.00 | 0.001 | +3.67 (+53%) |
| token_confidence | 46 | 340 | 114 | 32% | 6 | 11 | -27.89 | 27.89 | 0.174 | +2.17 (+31%) |
| attention | 69 | 291 | 140 | 42% | 6 | 11 | -27.91 | 27.91 | 0.134 | +2.85 (+41%) |
| attention+token_confidence | 80 | 254 | 166 | 49% | 4 | 17 | -27.91 | 27.91 | 0.113 | +3.36 (+48%) |
| all_internal | 81 | 219 | 200 | 56% | 8 | 22 | -35.88 | 35.88 | 0.128 | +3.83 (+55%) |
| context_action+all_internal | 95 | 187 | 218 | 63% | 7 | 22 | -35.31 | 35.31 | 0.113 | +4.28 (+62%) |
| grounding | 43 | 238 | 219 | 52% | 3 | 17 | -27.90 | 27.90 | 0.106 | +3.59 (+52%) |
| context_action+grounding | 115 | 117 | 268 | 77% | 9 | 24 | -27.91 | 27.91 | 0.073 | +5.27 (+76%) |
| context_action+all_internal+grounding | 136 | 85 | 279 | 83% | 15 | 26 | -35.55 | 35.55 | 0.086 | +5.70 (+82%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 74 | 174 | 252 | 65% | 15 | 51 | -27.59 | 27.59 | 0.085 | +4.48 (+64%) |
| token_entropy (untrained) | 71 | 210 | 219 | 58% | 16 | 41 | -27.59 | 27.59 | 0.095 | +3.98 (+57%) |
| context_action | 117 | 52 | 331 | 90% | 21 | 58 | -0.00 | 0.00 | 0.001 | +6.23 (+90%) |
| token_confidence | 74 | 127 | 299 | 75% | 13 | 56 | -27.89 | 27.89 | 0.075 | +5.13 (+74%) |
| attention | 116 | 39 | 345 | 92% | 17 | 63 | -27.91 | 27.91 | 0.061 | +6.35 (+91%) |
| attention+token_confidence | 124 | 14 | 362 | 97% | 16 | 67 | -27.91 | 27.91 | 0.057 | +6.70 (+96%) |
| all_internal | 131 | 25 | 344 | 95% | 26 | 70 | -35.88 | 35.88 | 0.076 | +6.53 (+94%) |
| context_action+all_internal | 143 | 18 | 339 | 96% | 28 | 62 | -35.31 | 35.31 | 0.073 | +6.63 (+95%) |
| grounding | 156 | 0 | 344 | 100% | 33 | 62 | -27.90 | 27.90 | 0.056 | +6.89 (+99%) |
| context_action+grounding | 179 | 0 | 321 | 100% | 37 | 43 | -27.91 | 27.91 | 0.056 | +6.89 (+99%) |
| context_action+all_internal+grounding | 173 | 0 | 327 | 100% | 35 | 47 | -35.55 | 35.55 | 0.071 | +6.88 (+99%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| context_action | 0 | 339 | 161 | 32% | 0 | 11 | -0.00 | 0.00 | 0.002 | +2.24 (+32%) |
| token_confidence | 0 | 457 | 43 | 9% | 0 | 3 | -27.89 | 27.89 | 0.649 | +0.54 (+8%) |
| attention | 0 | 425 | 75 | 15% | 0 | 3 | -27.91 | 27.91 | 0.372 | +0.99 (+14%) |
| attention+token_confidence | 0 | 385 | 115 | 23% | 0 | 9 | -27.91 | 27.91 | 0.243 | +1.54 (+22%) |
| all_internal | 0 | 349 | 151 | 30% | 0 | 14 | -35.88 | 35.88 | 0.238 | +2.03 (+29%) |
| context_action+all_internal | 0 | 325 | 175 | 35% | 0 | 13 | -35.31 | 35.31 | 0.202 | +2.36 (+34%) |
| grounding | 0 | 346 | 154 | 31% | 0 | 9 | -27.90 | 27.90 | 0.181 | +2.08 (+30%) |
| context_action+grounding | 0 | 272 | 228 | 46% | 0 | 15 | -27.91 | 27.91 | 0.122 | +3.11 (+45%) |
| context_action+all_internal+grounding | 0 | 286 | 214 | 43% | 0 | 10 | -35.55 | 35.55 | 0.166 | +2.90 (+42%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| context_action | 0 | 380 | 120 | 24% | 0 | 7 | -0.00 | 0.00 | 0.003 | +1.67 (+24%) |
| token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -27.89 | 27.89 | – | -0.06 (-1%) |
| attention | 0 | 500 | 0 | 0% | 0 | 0 | -27.91 | 27.91 | – | -0.06 (-1%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -27.91 | 27.91 | – | -0.06 (-1%) |
| all_internal | 0 | 422 | 78 | 16% | 0 | 3 | -35.88 | 35.88 | 0.460 | +1.01 (+15%) |
| context_action+all_internal | 0 | 423 | 77 | 15% | 0 | 0 | -35.31 | 35.31 | 0.459 | +1.00 (+14%) |
| grounding | 0 | 500 | 0 | 0% | 0 | 0 | -27.90 | 27.90 | – | -0.06 (-1%) |
| context_action+grounding | 0 | 372 | 128 | 26% | 0 | 2 | -27.91 | 27.91 | 0.218 | +1.72 (+25%) |
| context_action+all_internal+grounding | 0 | 352 | 148 | 30% | 0 | 4 | -35.55 | 35.55 | 0.240 | +1.99 (+29%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -27.59 | 27.59 | – | -0.06 (-1%) |
| context_action | 39 | 300 | 161 | 40% | 0 | 11 | -0.00 | 0.00 | 0.002 | +2.78 (+40%) |
| token_confidence | 0 | 457 | 43 | 9% | 0 | 3 | -27.89 | 27.89 | 0.649 | +0.54 (+8%) |
| attention | 0 | 425 | 75 | 15% | 0 | 3 | -27.91 | 27.91 | 0.372 | +0.99 (+14%) |
| attention+token_confidence | 0 | 385 | 115 | 23% | 0 | 9 | -27.91 | 27.91 | 0.243 | +1.54 (+22%) |
| all_internal | 43 | 306 | 151 | 39% | 1 | 14 | -35.88 | 35.88 | 0.185 | +2.62 (+38%) |
| context_action+all_internal | 67 | 258 | 175 | 48% | 3 | 13 | -35.31 | 35.31 | 0.146 | +3.29 (+47%) |
| grounding | 0 | 346 | 154 | 31% | 0 | 9 | -27.90 | 27.90 | 0.181 | +2.08 (+30%) |
| context_action+grounding | 31 | 241 | 228 | 52% | 0 | 15 | -27.91 | 27.91 | 0.108 | +3.54 (+51%) |
| context_action+all_internal+grounding | 74 | 212 | 214 | 58% | 2 | 10 | -35.55 | 35.55 | 0.123 | +3.93 (+57%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 46 | 380 | 74 | 24% | 8 | 14 | -27.59 | 27.59 | 0.230 | +1.61 (+23%) |
| token_entropy (untrained) | 51 | 287 | 162 | 43% | 9 | 24 | -27.59 | 27.59 | 0.130 | +2.91 (+42%) |
| context_action | 71 | 139 | 290 | 72% | 6 | 39 | -0.00 | 0.00 | 0.001 | +5.02 (+72%) |
| token_confidence | 33 | 253 | 214 | 49% | 4 | 28 | -27.89 | 27.89 | 0.113 | +3.38 (+49%) |
| attention | 88 | 167 | 245 | 67% | 8 | 33 | -27.91 | 27.91 | 0.084 | +4.57 (+66%) |
| attention+token_confidence | 96 | 112 | 292 | 78% | 10 | 42 | -27.91 | 27.91 | 0.072 | +5.34 (+77%) |
| all_internal | 90 | 134 | 276 | 73% | 9 | 41 | -35.88 | 35.88 | 0.098 | +5.02 (+72%) |
| context_action+all_internal | 105 | 94 | 301 | 81% | 13 | 43 | -35.31 | 35.31 | 0.087 | +5.57 (+80%) |
| grounding | 40 | 153 | 307 | 69% | 1 | 44 | -27.90 | 27.90 | 0.080 | +4.77 (+69%) |
| context_action+grounding | 152 | 0 | 348 | 100% | 23 | 56 | -27.91 | 27.91 | 0.056 | +6.89 (+99%) |
| context_action+all_internal+grounding | 155 | 8 | 337 | 98% | 26 | 52 | -35.55 | 35.55 | 0.072 | +6.77 (+97%) |
