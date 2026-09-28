# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 498)

Measured per candidate (means): generation 0.827 s; feature pass 0.060 s; FSM verifier (path check) 0.4 µs. Probe scoring (median, single row): context_action 0.67 ms, token_confidence 0.64 ms, attention 0.67 ms, attention+token_confidence 0.66 ms, all_internal 7.60 ms, context_action+all_internal 9.55 ms, grounding 0.62 ms, context_action+grounding 0.60 ms, context_action+all_internal+grounding 8.16 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 487 | 11 | 2% | 0 | 6 | -29.92 | 29.92 | 2.720 | +0.09 (+1%) |
| token_entropy (untrained) | 0 | 342 | 156 | 31% | 0 | 14 | -29.92 | 29.92 | 0.192 | +2.12 (+30%) |
| context_action | 32 | 202 | 264 | 59% | 0 | 26 | -0.00 | 0.00 | 0.001 | +4.13 (+59%) |
| token_confidence | 3 | 359 | 136 | 28% | 0 | 11 | -30.23 | 30.23 | 0.217 | +1.88 (+27%) |
| attention | 35 | 230 | 233 | 54% | 3 | 18 | -30.25 | 30.25 | 0.113 | +3.68 (+53%) |
| attention+token_confidence | 31 | 250 | 217 | 50% | 2 | 14 | -30.25 | 30.25 | 0.122 | +3.40 (+49%) |
| all_internal | 28 | 287 | 183 | 42% | 2 | 14 | -33.70 | 33.70 | 0.160 | +2.88 (+41%) |
| context_action+all_internal | 28 | 260 | 210 | 48% | 0 | 17 | -34.67 | 34.67 | 0.146 | +3.25 (+47%) |
| grounding | 32 | 241 | 225 | 52% | 2 | 13 | -30.22 | 30.22 | 0.118 | +3.53 (+51%) |
| context_action+grounding | 35 | 177 | 286 | 64% | 0 | 23 | -30.21 | 30.21 | 0.094 | +4.42 (+64%) |
| context_action+all_internal+grounding | 24 | 230 | 244 | 54% | 0 | 18 | -33.98 | 33.98 | 0.127 | +3.67 (+53%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 496 | 2 | 0% | 0 | 2 | -29.92 | 29.92 | 14.958 | -0.03 (-0%) |
| token_entropy (untrained) | 0 | 496 | 2 | 0% | 0 | 0 | -29.92 | 29.92 | 14.958 | -0.03 (-0%) |
| context_action | 47 | 317 | 134 | 36% | 2 | 2 | -0.00 | 0.00 | 0.002 | +2.53 (+36%) |
| token_confidence | 3 | 473 | 22 | 5% | 0 | 0 | -30.23 | 30.23 | 1.209 | +0.29 (+4%) |
| attention | 36 | 321 | 141 | 36% | 3 | 4 | -30.25 | 30.25 | 0.171 | +2.41 (+35%) |
| attention+token_confidence | 32 | 355 | 111 | 29% | 2 | 2 | -30.25 | 30.25 | 0.212 | +1.93 (+28%) |
| all_internal | 35 | 371 | 92 | 26% | 2 | 2 | -33.70 | 33.70 | 0.265 | +1.70 (+25%) |
| context_action+all_internal | 35 | 327 | 136 | 34% | 1 | 4 | -34.67 | 34.67 | 0.203 | +2.32 (+33%) |
| grounding | 43 | 333 | 122 | 33% | 4 | 6 | -30.22 | 30.22 | 0.183 | +2.24 (+32%) |
| context_action+grounding | 48 | 273 | 177 | 45% | 2 | 3 | -30.21 | 30.21 | 0.134 | +3.08 (+44%) |
| context_action+all_internal+grounding | 40 | 305 | 153 | 39% | 2 | 5 | -33.98 | 33.98 | 0.176 | +2.63 (+38%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 487 | 11 | 2% | 0 | 6 | -29.92 | 29.92 | 2.720 | +0.09 (+1%) |
| token_entropy (untrained) | 0 | 342 | 156 | 31% | 0 | 14 | -29.92 | 29.92 | 0.192 | +2.12 (+30%) |
| context_action | 52 | 182 | 264 | 63% | 2 | 26 | -0.00 | 0.00 | 0.001 | +4.41 (+63%) |
| token_confidence | 13 | 349 | 136 | 30% | 2 | 11 | -30.23 | 30.23 | 0.203 | +2.02 (+29%) |
| attention | 51 | 214 | 233 | 57% | 5 | 18 | -30.25 | 30.25 | 0.107 | +3.90 (+56%) |
| attention+token_confidence | 57 | 224 | 217 | 55% | 8 | 14 | -30.25 | 30.25 | 0.110 | +3.76 (+54%) |
| all_internal | 42 | 273 | 183 | 45% | 3 | 14 | -33.70 | 33.70 | 0.150 | +3.07 (+44%) |
| context_action+all_internal | 49 | 239 | 210 | 52% | 4 | 17 | -34.67 | 34.67 | 0.134 | +3.54 (+51%) |
| grounding | 56 | 217 | 225 | 56% | 5 | 13 | -30.22 | 30.22 | 0.108 | +3.86 (+56%) |
| context_action+grounding | 73 | 139 | 286 | 72% | 8 | 23 | -30.21 | 30.21 | 0.084 | +4.95 (+71%) |
| context_action+all_internal+grounding | 70 | 184 | 244 | 63% | 8 | 18 | -33.98 | 33.98 | 0.108 | +4.31 (+62%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 18 | 127 | 353 | 74% | 3 | 74 | -29.92 | 29.92 | 0.081 | +5.12 (+74%) |
| token_entropy (untrained) | 12 | 163 | 323 | 67% | 1 | 57 | -29.92 | 29.92 | 0.089 | +4.62 (+66%) |
| context_action | 56 | 46 | 396 | 91% | 6 | 75 | -0.00 | 0.00 | 0.001 | +6.31 (+91%) |
| token_confidence | 24 | 120 | 354 | 76% | 3 | 61 | -30.23 | 30.23 | 0.080 | +5.21 (+75%) |
| attention | 71 | 22 | 405 | 96% | 12 | 81 | -30.25 | 30.25 | 0.064 | +6.58 (+95%) |
| attention+token_confidence | 77 | 21 | 400 | 96% | 14 | 77 | -30.25 | 30.25 | 0.063 | +6.60 (+95%) |
| all_internal | 66 | 30 | 402 | 94% | 9 | 81 | -33.70 | 33.70 | 0.072 | +6.46 (+93%) |
| context_action+all_internal | 78 | 9 | 411 | 98% | 13 | 87 | -34.67 | 34.67 | 0.071 | +6.75 (+97%) |
| grounding | 94 | 5 | 399 | 99% | 14 | 72 | -30.22 | 30.22 | 0.061 | +6.82 (+98%) |
| context_action+grounding | 122 | 0 | 376 | 100% | 22 | 55 | -30.21 | 30.21 | 0.061 | +6.89 (+99%) |
| context_action+all_internal+grounding | 121 | 0 | 377 | 100% | 24 | 58 | -33.98 | 33.98 | 0.068 | +6.88 (+99%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 493 | 5 | 1% | 0 | 4 | -29.92 | 29.92 | 5.983 | +0.01 (+0%) |
| token_entropy (untrained) | 0 | 498 | 0 | 0% | 0 | 0 | -29.92 | 29.92 | – | -0.06 (-1%) |
| context_action | 0 | 334 | 164 | 33% | 0 | 5 | -0.00 | 0.00 | 0.002 | +2.29 (+33%) |
| token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.23 | 30.23 | – | -0.06 (-1%) |
| attention | 0 | 357 | 141 | 28% | 0 | 4 | -30.25 | 30.25 | 0.215 | +1.91 (+27%) |
| attention+token_confidence | 0 | 381 | 117 | 23% | 0 | 3 | -30.25 | 30.25 | 0.259 | +1.57 (+23%) |
| all_internal | 0 | 410 | 88 | 18% | 0 | 2 | -33.70 | 33.70 | 0.383 | +1.16 (+17%) |
| context_action+all_internal | 0 | 361 | 137 | 28% | 0 | 4 | -34.67 | 34.67 | 0.253 | +1.84 (+27%) |
| grounding | 0 | 369 | 129 | 26% | 0 | 6 | -30.22 | 30.22 | 0.234 | +1.74 (+25%) |
| context_action+grounding | 0 | 277 | 221 | 44% | 0 | 8 | -30.21 | 30.21 | 0.137 | +3.02 (+44%) |
| context_action+all_internal+grounding | 0 | 332 | 166 | 33% | 0 | 7 | -33.98 | 33.98 | 0.205 | +2.25 (+32%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 493 | 5 | 1% | 0 | 4 | -29.92 | 29.92 | 5.983 | +0.01 (+0%) |
| token_entropy (untrained) | 0 | 498 | 0 | 0% | 0 | 0 | -29.92 | 29.92 | – | -0.06 (-1%) |
| context_action | 0 | 443 | 55 | 11% | 0 | 0 | -0.00 | 0.00 | 0.006 | +0.77 (+11%) |
| token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.23 | 30.23 | – | -0.06 (-1%) |
| attention | 0 | 498 | 0 | 0% | 0 | 0 | -30.25 | 30.25 | – | -0.06 (-1%) |
| attention+token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.25 | 30.25 | – | -0.06 (-1%) |
| all_internal | 0 | 498 | 0 | 0% | 0 | 0 | -33.70 | 33.70 | – | -0.07 (-1%) |
| context_action+all_internal | 0 | 416 | 82 | 16% | 0 | 1 | -34.67 | 34.67 | 0.423 | +1.07 (+15%) |
| grounding | 0 | 498 | 0 | 0% | 0 | 0 | -30.22 | 30.22 | – | -0.06 (-1%) |
| context_action+grounding | 0 | 396 | 102 | 20% | 0 | 0 | -30.21 | 30.21 | 0.296 | +1.36 (+20%) |
| context_action+all_internal+grounding | 0 | 386 | 112 | 22% | 0 | 2 | -33.98 | 33.98 | 0.303 | +1.49 (+22%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 493 | 5 | 1% | 0 | 4 | -29.92 | 29.92 | 5.983 | +0.01 (+0%) |
| token_entropy (untrained) | 0 | 498 | 0 | 0% | 0 | 0 | -29.92 | 29.92 | – | -0.06 (-1%) |
| context_action | 0 | 334 | 164 | 33% | 0 | 5 | -0.00 | 0.00 | 0.002 | +2.29 (+33%) |
| token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.23 | 30.23 | – | -0.06 (-1%) |
| attention | 0 | 357 | 141 | 28% | 0 | 4 | -30.25 | 30.25 | 0.215 | +1.91 (+27%) |
| attention+token_confidence | 0 | 381 | 117 | 23% | 0 | 3 | -30.25 | 30.25 | 0.259 | +1.57 (+23%) |
| all_internal | 0 | 410 | 88 | 18% | 0 | 2 | -33.70 | 33.70 | 0.383 | +1.16 (+17%) |
| context_action+all_internal | 28 | 333 | 137 | 33% | 0 | 4 | -34.67 | 34.67 | 0.210 | +2.23 (+32%) |
| grounding | 32 | 337 | 129 | 32% | 2 | 6 | -30.22 | 30.22 | 0.188 | +2.19 (+31%) |
| context_action+grounding | 35 | 242 | 221 | 51% | 0 | 8 | -30.21 | 30.21 | 0.118 | +3.51 (+51%) |
| context_action+all_internal+grounding | 24 | 308 | 166 | 38% | 0 | 7 | -33.98 | 33.98 | 0.179 | +2.58 (+37%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 498 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 350 | 148 | 30% | 0 | 22 | -29.92 | 29.92 | 0.202 | +2.01 (+29%) |
| token_entropy (untrained) | 0 | 237 | 261 | 52% | 0 | 34 | -29.92 | 29.92 | 0.115 | +3.58 (+52%) |
| context_action | 52 | 100 | 346 | 80% | 2 | 52 | -0.00 | 0.00 | 0.001 | +5.55 (+80%) |
| token_confidence | 0 | 220 | 278 | 56% | 0 | 31 | -30.23 | 30.23 | 0.109 | +3.82 (+55%) |
| attention | 36 | 102 | 360 | 80% | 3 | 53 | -30.25 | 30.25 | 0.076 | +5.47 (+79%) |
| attention+token_confidence | 57 | 81 | 360 | 84% | 8 | 57 | -30.25 | 30.25 | 0.073 | +5.76 (+83%) |
| all_internal | 36 | 127 | 335 | 74% | 2 | 53 | -33.70 | 33.70 | 0.091 | +5.11 (+74%) |
| context_action+all_internal | 49 | 116 | 333 | 77% | 4 | 54 | -34.67 | 34.67 | 0.091 | +5.26 (+76%) |
| grounding | 56 | 76 | 366 | 85% | 5 | 52 | -30.22 | 30.22 | 0.072 | +5.83 (+84%) |
| context_action+grounding | 78 | 56 | 364 | 89% | 10 | 47 | -30.21 | 30.21 | 0.068 | +6.11 (+88%) |
| context_action+all_internal+grounding | 84 | 37 | 377 | 93% | 11 | 58 | -33.98 | 33.98 | 0.074 | +6.37 (+92%) |
