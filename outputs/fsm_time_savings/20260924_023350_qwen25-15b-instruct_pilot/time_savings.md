# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 0.827 s; feature pass 0.060 s; FSM verifier (path check) 0.3 µs. Probe scoring (median, single row): context_action 0.64 ms, token_confidence 0.58 ms, attention 0.86 ms, attention+token_confidence 0.90 ms, all_internal 10.37 ms, context_action+all_internal 6.93 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result; CSTR routing fractions and CSTR feature cost (≈4.5k-token prompts) must be measured in Stage 4.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 487 | 11 | 2% | 0 | 6 | -29.92 | 29.92 | 2.720 | +0.09 (+1%) |
| token_entropy (untrained) | 0 | 342 | 156 | 31% | 0 | 14 | -29.92 | 29.92 | 0.192 | +2.12 (+30%) |
| context_action | 32 | 202 | 264 | 59% | 0 | 26 | -0.00 | 0.00 | 0.001 | +4.13 (+59%) |
| token_confidence | 3 | 359 | 136 | 28% | 0 | 11 | -30.20 | 30.20 | 0.217 | +1.88 (+27%) |
| attention | 35 | 230 | 233 | 54% | 3 | 18 | -30.35 | 30.35 | 0.113 | +3.68 (+53%) |
| attention+token_confidence | 31 | 250 | 217 | 50% | 2 | 14 | -30.37 | 30.37 | 0.122 | +3.40 (+49%) |
| all_internal | 28 | 287 | 183 | 42% | 2 | 14 | -35.08 | 35.08 | 0.166 | +2.87 (+41%) |
| context_action+all_internal | 28 | 260 | 210 | 48% | 0 | 17 | -33.36 | 33.36 | 0.140 | +3.25 (+47%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 496 | 2 | 0% | 0 | 2 | -29.92 | 29.92 | 14.958 | -0.03 (-0%) |
| token_entropy (untrained) | 0 | 496 | 2 | 0% | 0 | 0 | -29.92 | 29.92 | 14.958 | -0.03 (-0%) |
| context_action | 47 | 317 | 134 | 36% | 2 | 2 | -0.00 | 0.00 | 0.002 | +2.53 (+36%) |
| token_confidence | 3 | 473 | 22 | 5% | 0 | 0 | -30.20 | 30.20 | 1.208 | +0.29 (+4%) |
| attention | 36 | 321 | 141 | 36% | 3 | 4 | -30.35 | 30.35 | 0.171 | +2.41 (+35%) |
| attention+token_confidence | 32 | 355 | 111 | 29% | 2 | 2 | -30.37 | 30.37 | 0.212 | +1.93 (+28%) |
| all_internal | 35 | 371 | 92 | 26% | 2 | 2 | -35.08 | 35.08 | 0.276 | +1.70 (+24%) |
| context_action+all_internal | 35 | 327 | 136 | 34% | 1 | 4 | -33.36 | 33.36 | 0.195 | +2.32 (+33%) |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 487 | 11 | 2% | 0 | 6 | -29.92 | 29.92 | 2.720 | +0.09 (+1%) |
| token_entropy (untrained) | 0 | 342 | 156 | 31% | 0 | 14 | -29.92 | 29.92 | 0.192 | +2.12 (+30%) |
| context_action | 52 | 182 | 264 | 63% | 2 | 26 | -0.00 | 0.00 | 0.001 | +4.41 (+63%) |
| token_confidence | 13 | 349 | 136 | 30% | 2 | 11 | -30.20 | 30.20 | 0.203 | +2.02 (+29%) |
| attention | 51 | 214 | 233 | 57% | 5 | 18 | -30.35 | 30.35 | 0.107 | +3.90 (+56%) |
| attention+token_confidence | 57 | 224 | 217 | 55% | 8 | 14 | -30.37 | 30.37 | 0.111 | +3.76 (+54%) |
| all_internal | 42 | 273 | 183 | 45% | 3 | 14 | -35.08 | 35.08 | 0.156 | +3.07 (+44%) |
| context_action+all_internal | 49 | 239 | 210 | 52% | 4 | 17 | -33.36 | 33.36 | 0.129 | +3.55 (+51%) |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 18 | 127 | 353 | 74% | 3 | 74 | -29.92 | 29.92 | 0.081 | +5.12 (+74%) |
| token_entropy (untrained) | 12 | 163 | 323 | 67% | 1 | 57 | -29.92 | 29.92 | 0.089 | +4.62 (+66%) |
| context_action | 56 | 46 | 396 | 91% | 6 | 75 | -0.00 | 0.00 | 0.001 | +6.31 (+91%) |
| token_confidence | 24 | 120 | 354 | 76% | 3 | 61 | -30.20 | 30.20 | 0.080 | +5.21 (+75%) |
| attention | 71 | 22 | 405 | 96% | 12 | 81 | -30.35 | 30.35 | 0.064 | +6.58 (+95%) |
| attention+token_confidence | 77 | 21 | 400 | 96% | 14 | 77 | -30.37 | 30.37 | 0.064 | +6.60 (+95%) |
| all_internal | 66 | 30 | 402 | 94% | 9 | 81 | -35.08 | 35.08 | 0.075 | +6.46 (+93%) |
| context_action+all_internal | 78 | 9 | 411 | 98% | 13 | 87 | -33.36 | 33.36 | 0.068 | +6.76 (+97%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 493 | 5 | 1% | 0 | 4 | -29.92 | 29.92 | 5.983 | +0.01 (+0%) |
| token_entropy (untrained) | 0 | 498 | 0 | 0% | 0 | 0 | -29.92 | 29.92 | – | -0.06 (-1%) |
| context_action | 0 | 334 | 164 | 33% | 0 | 5 | -0.00 | 0.00 | 0.002 | +2.29 (+33%) |
| token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.20 | 30.20 | – | -0.06 (-1%) |
| attention | 0 | 357 | 141 | 28% | 0 | 4 | -30.35 | 30.35 | 0.215 | +1.91 (+27%) |
| attention+token_confidence | 0 | 381 | 117 | 23% | 0 | 3 | -30.37 | 30.37 | 0.260 | +1.57 (+23%) |
| all_internal | 0 | 410 | 88 | 18% | 0 | 2 | -35.08 | 35.08 | 0.399 | +1.16 (+17%) |
| context_action+all_internal | 0 | 361 | 137 | 28% | 0 | 4 | -33.36 | 33.36 | 0.244 | +1.84 (+27%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 493 | 5 | 1% | 0 | 4 | -29.92 | 29.92 | 5.983 | +0.01 (+0%) |
| token_entropy (untrained) | 0 | 498 | 0 | 0% | 0 | 0 | -29.92 | 29.92 | – | -0.06 (-1%) |
| context_action | 0 | 443 | 55 | 11% | 0 | 0 | -0.00 | 0.00 | 0.006 | +0.77 (+11%) |
| token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.20 | 30.20 | – | -0.06 (-1%) |
| attention | 0 | 498 | 0 | 0% | 0 | 0 | -30.35 | 30.35 | – | -0.06 (-1%) |
| attention+token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.37 | 30.37 | – | -0.06 (-1%) |
| all_internal | 0 | 498 | 0 | 0% | 0 | 0 | -35.08 | 35.08 | – | -0.07 (-1%) |
| context_action+all_internal | 0 | 416 | 82 | 16% | 0 | 1 | -33.36 | 33.36 | 0.407 | +1.08 (+16%) |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 493 | 5 | 1% | 0 | 4 | -29.92 | 29.92 | 5.983 | +0.01 (+0%) |
| token_entropy (untrained) | 0 | 498 | 0 | 0% | 0 | 0 | -29.92 | 29.92 | – | -0.06 (-1%) |
| context_action | 0 | 334 | 164 | 33% | 0 | 5 | -0.00 | 0.00 | 0.002 | +2.29 (+33%) |
| token_confidence | 0 | 498 | 0 | 0% | 0 | 0 | -30.20 | 30.20 | – | -0.06 (-1%) |
| attention | 0 | 357 | 141 | 28% | 0 | 4 | -30.35 | 30.35 | 0.215 | +1.91 (+27%) |
| attention+token_confidence | 0 | 381 | 117 | 23% | 0 | 3 | -30.37 | 30.37 | 0.260 | +1.57 (+23%) |
| all_internal | 0 | 410 | 88 | 18% | 0 | 2 | -35.08 | 35.08 | 0.399 | +1.16 (+17%) |
| context_action+all_internal | 28 | 333 | 137 | 33% | 0 | 4 | -33.36 | 33.36 | 0.202 | +2.24 (+32%) |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 350 | 148 | 30% | 0 | 22 | -29.92 | 29.92 | 0.202 | +2.01 (+29%) |
| token_entropy (untrained) | 0 | 237 | 261 | 52% | 0 | 34 | -29.92 | 29.92 | 0.115 | +3.58 (+52%) |
| context_action | 52 | 100 | 346 | 80% | 2 | 52 | -0.00 | 0.00 | 0.001 | +5.55 (+80%) |
| token_confidence | 0 | 220 | 278 | 56% | 0 | 31 | -30.20 | 30.20 | 0.109 | +3.82 (+55%) |
| attention | 36 | 102 | 360 | 80% | 3 | 53 | -30.35 | 30.35 | 0.077 | +5.47 (+79%) |
| attention+token_confidence | 57 | 81 | 360 | 84% | 8 | 57 | -30.37 | 30.37 | 0.073 | +5.76 (+83%) |
| all_internal | 36 | 127 | 335 | 74% | 2 | 53 | -35.08 | 35.08 | 0.095 | +5.11 (+73%) |
| context_action+all_internal | 49 | 116 | 333 | 77% | 4 | 54 | -33.36 | 33.36 | 0.087 | +5.26 (+76%) |
