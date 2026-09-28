# FSM routing table with computational time saved (pilot, exploratory test_iid, n = 500)

Measured per candidate (means): generation 1.457 s; feature pass 0.092 s; FSM verifier (path check) 0.4 µs. Probe scoring (median, single row): context_action 0.55 ms, token_confidence 0.55 ms, attention 0.61 ms, attention+token_confidence 0.58 ms, all_internal 10.54 ms, context_action+all_internal 9.77 ms.
Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). Re-proposal cost after DISALLOW is not included.

**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — time saved is negative for every policy. FSM demonstrates routing quality, not time savings.

**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost (6.95 s/call, Stage 0 audit). This is arithmetic, not a CSTR result; CSTR routing fractions and CSTR feature cost (≈4.5k-token prompts) must be measured in Stage 4.

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 | 423 | 45 | 15% | 2 | 6 | -45.76 | 45.76 | 0.594 | +0.98 (+14%) |
| token_entropy (untrained) | 32 | 401 | 67 | 20% | 2 | 7 | -45.76 | 45.76 | 0.462 | +1.28 (+18%) |
| context_action | 45 | 370 | 85 | 26% | 0 | 10 | -0.00 | 0.00 | 0.002 | +1.81 (+26%) |
| token_confidence | 30 | 389 | 81 | 22% | 0 | 6 | -46.04 | 46.04 | 0.415 | +1.45 (+21%) |
| attention | 40 | 392 | 68 | 22% | 1 | 4 | -46.07 | 46.07 | 0.427 | +1.41 (+20%) |
| attention+token_confidence | 40 | 371 | 89 | 26% | 0 | 6 | -46.05 | 46.05 | 0.357 | +1.70 (+24%) |
| all_internal | 41 | 347 | 112 | 31% | 0 | 14 | -51.03 | 51.03 | 0.334 | +2.02 (+29%) |
| context_action+all_internal | 65 | 304 | 131 | 39% | 1 | 17 | -50.65 | 50.65 | 0.258 | +2.62 (+38%) |

## rule = point, ALLOW failure target α = 0.01, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 | 423 | 45 | 15% | 2 | 6 | -45.76 | 45.76 | 0.594 | +0.98 (+14%) |
| token_entropy (untrained) | 32 | 401 | 67 | 20% | 2 | 7 | -45.76 | 45.76 | 0.462 | +1.28 (+18%) |
| context_action | 45 | 370 | 85 | 26% | 0 | 10 | -0.00 | 0.00 | 0.002 | +1.81 (+26%) |
| token_confidence | 30 | 389 | 81 | 22% | 0 | 6 | -46.04 | 46.04 | 0.415 | +1.45 (+21%) |
| attention | 40 | 392 | 68 | 22% | 1 | 4 | -46.07 | 46.07 | 0.427 | +1.41 (+20%) |
| attention+token_confidence | 40 | 371 | 89 | 26% | 0 | 6 | -46.05 | 46.05 | 0.357 | +1.70 (+24%) |
| all_internal | 41 | 347 | 112 | 31% | 0 | 14 | -51.03 | 51.03 | 0.334 | +2.02 (+29%) |
| context_action+all_internal | 65 | 304 | 131 | 39% | 1 | 17 | -50.65 | 50.65 | 0.258 | +2.62 (+38%) |

## rule = point, ALLOW failure target α = 0.02, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 | 423 | 45 | 15% | 2 | 6 | -45.76 | 45.76 | 0.594 | +0.98 (+14%) |
| token_entropy (untrained) | 32 | 401 | 67 | 20% | 2 | 7 | -45.76 | 45.76 | 0.462 | +1.28 (+18%) |
| context_action | 45 | 370 | 85 | 26% | 0 | 10 | -0.00 | 0.00 | 0.002 | +1.81 (+26%) |
| token_confidence | 30 | 389 | 81 | 22% | 0 | 6 | -46.04 | 46.04 | 0.415 | +1.45 (+21%) |
| attention | 40 | 392 | 68 | 22% | 1 | 4 | -46.07 | 46.07 | 0.427 | +1.41 (+20%) |
| attention+token_confidence | 61 | 350 | 89 | 30% | 1 | 6 | -46.05 | 46.05 | 0.307 | +1.99 (+29%) |
| all_internal | 71 | 317 | 112 | 37% | 1 | 14 | -51.03 | 51.03 | 0.279 | +2.44 (+35%) |
| context_action+all_internal | 76 | 293 | 131 | 41% | 2 | 17 | -50.65 | 50.65 | 0.245 | +2.78 (+40%) |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 37 | 418 | 45 | 16% | 2 | 6 | -45.76 | 45.76 | 0.558 | +1.05 (+15%) |
| token_entropy (untrained) | 37 | 396 | 67 | 21% | 2 | 7 | -45.76 | 45.76 | 0.440 | +1.35 (+19%) |
| context_action | 52 | 363 | 85 | 27% | 2 | 10 | -0.00 | 0.00 | 0.002 | +1.90 (+27%) |
| token_confidence | 58 | 361 | 81 | 28% | 2 | 6 | -46.04 | 46.04 | 0.331 | +1.84 (+26%) |
| attention | 64 | 368 | 68 | 26% | 2 | 4 | -46.07 | 46.07 | 0.349 | +1.74 (+25%) |
| attention+token_confidence | 75 | 336 | 89 | 33% | 1 | 6 | -46.05 | 46.05 | 0.281 | +2.19 (+31%) |
| all_internal | 79 | 309 | 112 | 38% | 3 | 14 | -51.03 | 51.03 | 0.267 | +2.55 (+37%) |
| context_action+all_internal | 89 | 280 | 131 | 44% | 3 | 17 | -50.65 | 50.65 | 0.230 | +2.96 (+43%) |

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 458 | 42 | 8% | 0 | 1 | -0.00 | 0.00 | 0.007 | +0.58 (+8%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.04 | 46.04 | 1.212 | +0.44 (+6%) |
| attention | 0 | 441 | 59 | 12% | 0 | 4 | -46.07 | 46.07 | 0.781 | +0.73 (+10%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.05 | 46.05 | – | -0.09 (-1%) |
| all_internal | 0 | 468 | 32 | 6% | 0 | 2 | -51.03 | 51.03 | 1.595 | +0.34 (+5%) |
| context_action+all_internal | 0 | 455 | 45 | 9% | 0 | 2 | -50.65 | 50.65 | 1.126 | +0.52 (+8%) |

## rule = ucb, ALLOW failure target α = 0.01, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 458 | 42 | 8% | 0 | 1 | -0.00 | 0.00 | 0.007 | +0.58 (+8%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.04 | 46.04 | 1.212 | +0.44 (+6%) |
| attention | 0 | 441 | 59 | 12% | 0 | 4 | -46.07 | 46.07 | 0.781 | +0.73 (+10%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.05 | 46.05 | – | -0.09 (-1%) |
| all_internal | 0 | 468 | 32 | 6% | 0 | 2 | -51.03 | 51.03 | 1.595 | +0.34 (+5%) |
| context_action+all_internal | 0 | 455 | 45 | 9% | 0 | 2 | -50.65 | 50.65 | 1.126 | +0.52 (+8%) |

## rule = ucb, ALLOW failure target α = 0.02, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 458 | 42 | 8% | 0 | 1 | -0.00 | 0.00 | 0.007 | +0.58 (+8%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.04 | 46.04 | 1.212 | +0.44 (+6%) |
| attention | 0 | 441 | 59 | 12% | 0 | 4 | -46.07 | 46.07 | 0.781 | +0.73 (+10%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.05 | 46.05 | – | -0.09 (-1%) |
| all_internal | 0 | 468 | 32 | 6% | 0 | 2 | -51.03 | 51.03 | 1.595 | +0.34 (+5%) |
| context_action+all_internal | 0 | 455 | 45 | 9% | 0 | 2 | -50.65 | 50.65 | 1.126 | +0.52 (+8%) |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | FSM time saved (s, 500 cands) | Router overhead (s) | Break-even verifier cost (s) | What-if saved at 6.95 s/call (s per cand, % of verification) |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| token_entropy (untrained) | 0 | 500 | 0 | 0% | 0 | 0 | -45.76 | 45.76 | – | -0.09 (-1%) |
| context_action | 0 | 458 | 42 | 8% | 0 | 1 | -0.00 | 0.00 | 0.007 | +0.58 (+8%) |
| token_confidence | 0 | 462 | 38 | 8% | 0 | 5 | -46.04 | 46.04 | 1.212 | +0.44 (+6%) |
| attention | 0 | 441 | 59 | 12% | 0 | 4 | -46.07 | 46.07 | 0.781 | +0.73 (+10%) |
| attention+token_confidence | 0 | 500 | 0 | 0% | 0 | 0 | -46.05 | 46.05 | – | -0.09 (-1%) |
| all_internal | 0 | 468 | 32 | 6% | 0 | 2 | -51.03 | 51.03 | 1.595 | +0.34 (+5%) |
| context_action+all_internal | 65 | 390 | 45 | 22% | 1 | 2 | -50.65 | 50.65 | 0.460 | +1.43 (+21%) |
