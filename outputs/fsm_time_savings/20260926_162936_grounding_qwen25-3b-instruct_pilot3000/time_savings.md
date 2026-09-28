# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 1.457 s; feature pass 0.092 s; FSM verifier (path check) 0.3 µs. Probe scoring (median, single row): context_action 0.72 ms, token_confidence 0.82 ms, attention 0.72 ms, attention+token_confidence 0.78 ms, all_internal 9.35 ms, context_action+all_internal 11.37 ms, grounding 0.73 ms, context_action+grounding 0.70 ms, context_action+all_internal+grounding 11.39 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 | 423 | 45 | 15% | 2 | 6 | -45.76 | 45.76 | 0.594 | +0.98 (+14%) |
| token_entropy (untrained) | 32 | 401 | 67 | 20% | 2 | 7 | -45.76 | 45.76 | 0.462 | +1.28 (+18%) |
| context_action | 45 | 370 | 85 | 26% | 0 | 10 | -0.00 | 0.00 | 0.003 | +1.81 (+26%) |
| token_confidence | 30 | 389 | 81 | 22% | 0 | 6 | -46.17 | 46.17 | 0.416 | +1.45 (+21%) |
| attention | 40 | 392 | 68 | 22% | 1 | 4 | -46.12 | 46.12 | 0.427 | +1.41 (+20%) |
| attention+token_confidence | 40 | 371 | 89 | 26% | 0 | 6 | -46.15 | 46.15 | 0.358 | +1.70 (+24%) |
| all_internal | 41 | 347 | 112 | 31% | 0 | 14 | -50.44 | 50.44 | 0.330 | +2.03 (+29%) |
| context_action+all_internal | 65 | 304 | 131 | 39% | 1 | 17 | -51.45 | 51.45 | 0.262 | +2.62 (+38%) |
| grounding | 41 | 426 | 33 | 15% | 2 | 3 | -46.13 | 46.13 | 0.623 | +0.94 (+13%) |
| context_action+grounding | 47 | 390 | 63 | 22% | 1 | 4 | -46.11 | 46.11 | 0.419 | +1.44 (+21%) |
| context_action+all_internal+grounding | 72 | 306 | 122 | 39% | 1 | 12 | -51.46 | 51.46 | 0.265 | +2.59 (+37%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 37 | 458 | 5 | 8% | 2 | 0 | -45.76 | 45.76 | 1.090 | +0.49 (+7%) |
| token_entropy (untrained) | 37 | 456 | 7 | 9% | 2 | 0 | -45.76 | 45.76 | 1.040 | +0.52 (+7%) |
| context_action | 52 | 402 | 46 | 20% | 2 | 1 | -0.00 | 0.00 | 0.004 | +1.36 (+20%) |
| token_confidence | 58 | 395 | 47 | 21% | 2 | 5 | -46.17 | 46.17 | 0.440 | +1.37 (+20%) |
| attention | 64 | 374 | 62 | 25% | 2 | 4 | -46.12 | 46.12 | 0.366 | +1.66 (+24%) |
| attention+token_confidence | 75 | 368 | 57 | 26% | 1 | 1 | -46.15 | 46.15 | 0.350 | +1.74 (+25%) |
| all_internal | 79 | 381 | 40 | 24% | 3 | 3 | -50.44 | 50.44 | 0.424 | +1.55 (+22%) |
| context_action+all_internal | 89 | 330 | 81 | 34% | 3 | 6 | -51.45 | 51.45 | 0.303 | +2.26 (+33%) |
| grounding | 78 | 419 | 3 | 16% | 5 | 0 | -46.13 | 46.13 | 0.569 | +1.03 (+15%) |
| context_action+grounding | 103 | 349 | 48 | 30% | 7 | 2 | -46.11 | 46.11 | 0.305 | +2.01 (+29%) |
| context_action+all_internal+grounding | 111 | 302 | 87 | 40% | 3 | 5 | -51.46 | 51.46 | 0.260 | +2.65 (+38%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 53 | 402 | 45 | 20% | 5 | 6 | -45.76 | 45.76 | 0.467 | +1.27 (+18%) |
| token_entropy (untrained) | 53 | 380 | 67 | 24% | 5 | 7 | -45.76 | 45.76 | 0.381 | +1.58 (+23%) |
| context_action | 93 | 322 | 85 | 36% | 13 | 10 | -0.00 | 0.00 | 0.002 | +2.47 (+36%) |
| token_confidence | 75 | 344 | 81 | 31% | 5 | 6 | -46.17 | 46.17 | 0.296 | +2.08 (+30%) |
| attention | 86 | 346 | 68 | 31% | 5 | 4 | -46.12 | 46.12 | 0.300 | +2.05 (+29%) |
| attention+token_confidence | 84 | 327 | 89 | 35% | 2 | 6 | -46.15 | 46.15 | 0.267 | +2.31 (+33%) |
| all_internal | 121 | 267 | 112 | 47% | 9 | 14 | -50.44 | 50.44 | 0.216 | +3.14 (+45%) |
| context_action+all_internal | 120 | 249 | 131 | 50% | 7 | 17 | -51.45 | 51.45 | 0.205 | +3.39 (+49%) |
| grounding | 93 | 374 | 33 | 25% | 7 | 3 | -46.13 | 46.13 | 0.366 | +1.66 (+24%) |
| context_action+grounding | 127 | 310 | 63 | 38% | 12 | 4 | -46.11 | 46.11 | 0.243 | +2.55 (+37%) |
| context_action+all_internal+grounding | 138 | 240 | 122 | 52% | 9 | 12 | -51.46 | 51.46 | 0.198 | +3.51 (+51%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 103 | 292 | 105 | 42% | 12 | 23 | -45.76 | 45.76 | 0.220 | +2.80 (+40%) |
| token_entropy (untrained) | 103 | 275 | 122 | 45% | 12 | 27 | -45.76 | 45.76 | 0.203 | +3.04 (+44%) |
| context_action | 162 | 169 | 169 | 66% | 30 | 34 | -0.00 | 0.00 | 0.001 | +4.60 (+66%) |
| token_confidence | 116 | 248 | 136 | 50% | 13 | 19 | -46.17 | 46.17 | 0.183 | +3.41 (+49%) |
| attention | 159 | 173 | 168 | 65% | 26 | 32 | -46.12 | 46.12 | 0.141 | +4.45 (+64%) |
| attention+token_confidence | 152 | 188 | 160 | 62% | 22 | 25 | -46.15 | 46.15 | 0.148 | +4.24 (+61%) |
| all_internal | 193 | 103 | 204 | 79% | 35 | 46 | -50.44 | 50.44 | 0.127 | +5.42 (+78%) |
| context_action+all_internal | 195 | 110 | 195 | 78% | 34 | 41 | -51.45 | 51.45 | 0.132 | +5.32 (+77%) |
| grounding | 156 | 190 | 154 | 62% | 22 | 34 | -46.13 | 46.13 | 0.149 | +4.22 (+61%) |
| context_action+grounding | 178 | 94 | 228 | 81% | 25 | 54 | -46.11 | 46.11 | 0.114 | +5.55 (+80%) |
| context_action+all_internal+grounding | 213 | 62 | 225 | 88% | 33 | 47 | -51.46 | 51.46 | 0.117 | +5.99 (+86%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 458 | 42 | 8% | 0 | 1 | -0.00 | 0.00 | 0.009 | +0.58 (+8%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.17 | 46.17 | 1.215 | +0.44 (+6%) |
| attention | 0 | 441 | 59 | 12% | 0 | 4 | -46.12 | 46.12 | 0.782 | +0.73 (+10%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.15 | 46.15 | – | -0.09 (-1%) |
| all_internal | 0 | 468 | 32 | 6% | 0 | 2 | -50.44 | 50.44 | 1.576 | +0.34 (+5%) |
| context_action+all_internal | 0 | 455 | 45 | 9% | 0 | 2 | -51.45 | 51.45 | 1.143 | +0.52 (+8%) |
| grounding | 0 | 500 | 0 | 0% | 0 | 0 | -46.13 | 46.13 | – | -0.09 (-1%) |
| context_action+grounding | 0 | 453 | 47 | 9% | 0 | 2 | -46.11 | 46.11 | 0.981 | +0.56 (+8%) |
| context_action+all_internal+grounding | 0 | 418 | 82 | 16% | 0 | 5 | -51.46 | 51.46 | 0.628 | +1.04 (+15%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 500 | 0 | 0% | 0 | 0 | -0.00 | 0.00 | – | -0.00 (-0%) |
| token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.17 | 46.17 | – | -0.09 (-1%) |
| attention | 0 | 500 | 0 | 0% | 0 | 0 | -46.12 | 46.12 | – | -0.09 (-1%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.15 | 46.15 | – | -0.09 (-1%) |
| all_internal | 0 | 500 | 0 | 0% | 0 | 0 | -50.44 | 50.44 | – | -0.10 (-1%) |
| context_action+all_internal | 65 | 435 | 0 | 13% | 1 | 0 | -51.45 | 51.45 | 0.792 | +0.80 (+12%) |
| grounding | 0 | 500 | 0 | 0% | 0 | 0 | -46.13 | 46.13 | – | -0.09 (-1%) |
| context_action+grounding | 0 | 500 | 0 | 0% | 0 | 0 | -46.11 | 46.11 | – | -0.09 (-1%) |
| context_action+all_internal+grounding | 72 | 428 | 0 | 14% | 1 | 0 | -51.46 | 51.46 | 0.715 | +0.90 (+13%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 45 | 413 | 42 | 17% | 0 | 1 | -0.00 | 0.00 | 0.004 | +1.21 (+17%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.17 | 46.17 | 1.215 | +0.44 (+6%) |
| attention | 40 | 401 | 59 | 20% | 1 | 4 | -46.12 | 46.12 | 0.466 | +1.28 (+18%) |
| attention+token_confidence | 74 | 426 | 0 | 15% | 1 | 0 | -46.15 | 46.15 | 0.624 | +0.94 (+13%) |
| all_internal | 79 | 389 | 32 | 22% | 3 | 2 | -50.44 | 50.44 | 0.454 | +1.44 (+21%) |
| context_action+all_internal | 84 | 371 | 45 | 26% | 2 | 2 | -51.45 | 51.45 | 0.399 | +1.69 (+24%) |
| grounding | 63 | 437 | 0 | 13% | 3 | 0 | -46.13 | 46.13 | 0.732 | +0.78 (+11%) |
| context_action+grounding | 103 | 350 | 47 | 30% | 7 | 2 | -46.11 | 46.11 | 0.307 | +1.99 (+29%) |
| context_action+all_internal+grounding | 105 | 313 | 82 | 37% | 3 | 5 | -51.46 | 51.46 | 0.275 | +2.50 (+36%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 52 | 380 | 68 | 24% | 5 | 11 | -45.76 | 45.76 | 0.381 | +1.58 (+23%) |
| token_entropy (untrained) | 43 | 390 | 67 | 22% | 3 | 7 | -45.76 | 45.76 | 0.416 | +1.44 (+21%) |
| context_action | 101 | 288 | 111 | 42% | 15 | 13 | -0.00 | 0.00 | 0.002 | +2.95 (+42%) |
| token_confidence | 75 | 340 | 85 | 32% | 5 | 6 | -46.17 | 46.17 | 0.289 | +2.13 (+31%) |
| attention | 132 | 262 | 106 | 48% | 16 | 11 | -46.12 | 46.12 | 0.194 | +3.22 (+46%) |
| attention+token_confidence | 129 | 237 | 134 | 53% | 13 | 19 | -46.15 | 46.15 | 0.175 | +3.56 (+51%) |
| all_internal | 144 | 200 | 156 | 60% | 16 | 29 | -50.44 | 50.44 | 0.168 | +4.07 (+59%) |
| context_action+all_internal | 167 | 183 | 150 | 63% | 21 | 24 | -51.45 | 51.45 | 0.162 | +4.30 (+62%) |
| grounding | 128 | 372 | 0 | 26% | 15 | 0 | -46.13 | 46.13 | 0.360 | +1.69 (+24%) |
| context_action+grounding | 144 | 197 | 159 | 61% | 13 | 25 | -46.11 | 46.11 | 0.152 | +4.12 (+59%) |
| context_action+all_internal+grounding | 172 | 142 | 186 | 72% | 17 | 30 | -51.46 | 51.46 | 0.144 | +4.87 (+70%) |
