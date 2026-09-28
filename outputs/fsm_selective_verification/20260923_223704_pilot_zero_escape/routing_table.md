# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)

Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; **DISALLOW** = reject without simulator. p = predicted failure probability.
Test candidates: 500 generated, 0 rejected by cheap format checks, 500 routed; failure prevalence 0.486.
Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; `point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over 2000 draws).

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 458 (92%) | 42 (8%) | 8% | 0 | – [–] | 0.0 | 1 | 0.024 [0.108] | 21.6 |
| token_confidence | 0 (0%) | 462 (92%) | 38 (8%) | 8% | 0 | – [–] | 0.0 | 5 | 0.132 [0.257] | 19.5 |
| attention | 0 (0%) | 441 (88%) | 59 (12%) | 12% | 0 | – [–] | 0.0 | 4 | 0.068 [0.148] | 30.3 |
| attention+token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 468 (94%) | 32 (6%) | 6% | 0 | – [–] | 0.0 | 2 | 0.062 [0.184] | 16.4 |
| context_action+all_internal | 0 (0%) | 455 (91%) | 45 (9%) | 9% | 0 | – [–] | 0.0 | 2 | 0.044 [0.133] | 23.1 |

## rule = ucb, ALLOW failure target α = 0.01, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 458 (92%) | 42 (8%) | 8% | 0 | – [–] | 0.0 | 1 | 0.024 [0.108] | 21.6 |
| token_confidence | 0 (0%) | 462 (92%) | 38 (8%) | 8% | 0 | – [–] | 0.0 | 5 | 0.132 [0.257] | 19.5 |
| attention | 0 (0%) | 441 (88%) | 59 (12%) | 12% | 0 | – [–] | 0.0 | 4 | 0.068 [0.148] | 30.3 |
| attention+token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 468 (94%) | 32 (6%) | 6% | 0 | – [–] | 0.0 | 2 | 0.062 [0.184] | 16.4 |
| context_action+all_internal | 0 (0%) | 455 (91%) | 45 (9%) | 9% | 0 | – [–] | 0.0 | 2 | 0.044 [0.133] | 23.1 |

## rule = ucb, ALLOW failure target α = 0.02, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 458 (92%) | 42 (8%) | 8% | 0 | – [–] | 0.0 | 1 | 0.024 [0.108] | 21.6 |
| token_confidence | 0 (0%) | 462 (92%) | 38 (8%) | 8% | 0 | – [–] | 0.0 | 5 | 0.132 [0.257] | 19.5 |
| attention | 0 (0%) | 441 (88%) | 59 (12%) | 12% | 0 | – [–] | 0.0 | 4 | 0.068 [0.148] | 30.3 |
| attention+token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 468 (94%) | 32 (6%) | 6% | 0 | – [–] | 0.0 | 2 | 0.062 [0.184] | 16.4 |
| context_action+all_internal | 0 (0%) | 455 (91%) | 45 (9%) | 9% | 0 | – [–] | 0.0 | 2 | 0.044 [0.133] | 23.1 |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 458 (92%) | 42 (8%) | 8% | 0 | – [–] | 0.0 | 1 | 0.024 [0.108] | 21.6 |
| token_confidence | 0 (0%) | 462 (92%) | 38 (8%) | 8% | 0 | – [–] | 0.0 | 5 | 0.132 [0.257] | 19.5 |
| attention | 0 (0%) | 441 (88%) | 59 (12%) | 12% | 0 | – [–] | 0.0 | 4 | 0.068 [0.148] | 30.3 |
| attention+token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 468 (94%) | 32 (6%) | 6% | 0 | – [–] | 0.0 | 2 | 0.062 [0.184] | 16.4 |
| context_action+all_internal | 65 (13%) | 390 (78%) | 45 (9%) | 22% | 1 | 0.015 [0.071] | 31.7 | 2 | 0.044 [0.133] | 23.3 |

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 (6%) | 423 (85%) | 45 (9%) | 15% | 2 | 0.062 [0.184] | 15.6 | 6 | 0.133 [0.246] | 23.1 |
| token_entropy (untrained) | 32 (6%) | 401 (80%) | 67 (13%) | 20% | 2 | 0.062 [0.184] | 15.6 | 7 | 0.104 [0.187] | 34.5 |
| context_action | 45 (9%) | 370 (74%) | 85 (17%) | 26% | 0 | 0.000 [0.064] | 21.9 | 10 | 0.118 [0.191] | 43.7 |
| token_confidence | 30 (6%) | 389 (78%) | 81 (16%) | 22% | 0 | 0.000 [0.095] | 14.6 | 6 | 0.074 [0.141] | 41.8 |
| attention | 40 (8%) | 392 (78%) | 68 (14%) | 22% | 1 | 0.025 [0.113] | 19.5 | 4 | 0.059 [0.130] | 35.1 |
| attention+token_confidence | 40 (8%) | 371 (74%) | 89 (18%) | 26% | 0 | 0.000 [0.072] | 19.5 | 6 | 0.067 [0.129] | 45.8 |
| all_internal | 41 (8%) | 347 (69%) | 112 (22%) | 31% | 0 | 0.000 [0.070] | 20.0 | 14 | 0.125 [0.188] | 57.7 |
| context_action+all_internal | 65 (13%) | 304 (61%) | 131 (26%) | 39% | 1 | 0.015 [0.071] | 31.7 | 17 | 0.130 [0.188] | 67.5 |

## rule = point, ALLOW failure target α = 0.01, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 (6%) | 423 (85%) | 45 (9%) | 15% | 2 | 0.062 [0.184] | 15.6 | 6 | 0.133 [0.246] | 23.1 |
| token_entropy (untrained) | 32 (6%) | 401 (80%) | 67 (13%) | 20% | 2 | 0.062 [0.184] | 15.6 | 7 | 0.104 [0.187] | 34.5 |
| context_action | 45 (9%) | 370 (74%) | 85 (17%) | 26% | 0 | 0.000 [0.064] | 21.9 | 10 | 0.118 [0.191] | 43.7 |
| token_confidence | 30 (6%) | 389 (78%) | 81 (16%) | 22% | 0 | 0.000 [0.095] | 14.6 | 6 | 0.074 [0.141] | 41.8 |
| attention | 40 (8%) | 392 (78%) | 68 (14%) | 22% | 1 | 0.025 [0.113] | 19.5 | 4 | 0.059 [0.130] | 35.1 |
| attention+token_confidence | 40 (8%) | 371 (74%) | 89 (18%) | 26% | 0 | 0.000 [0.072] | 19.5 | 6 | 0.067 [0.129] | 45.8 |
| all_internal | 41 (8%) | 347 (69%) | 112 (22%) | 31% | 0 | 0.000 [0.070] | 20.0 | 14 | 0.125 [0.188] | 57.7 |
| context_action+all_internal | 65 (13%) | 304 (61%) | 131 (26%) | 39% | 1 | 0.015 [0.071] | 31.7 | 17 | 0.130 [0.188] | 67.5 |

## rule = point, ALLOW failure target α = 0.02, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 32 (6%) | 423 (85%) | 45 (9%) | 15% | 2 | 0.062 [0.184] | 15.6 | 6 | 0.133 [0.246] | 23.1 |
| token_entropy (untrained) | 32 (6%) | 401 (80%) | 67 (13%) | 20% | 2 | 0.062 [0.184] | 15.6 | 7 | 0.104 [0.187] | 34.5 |
| context_action | 45 (9%) | 370 (74%) | 85 (17%) | 26% | 0 | 0.000 [0.064] | 21.9 | 10 | 0.118 [0.191] | 43.7 |
| token_confidence | 30 (6%) | 389 (78%) | 81 (16%) | 22% | 0 | 0.000 [0.095] | 14.6 | 6 | 0.074 [0.141] | 41.8 |
| attention | 40 (8%) | 392 (78%) | 68 (14%) | 22% | 1 | 0.025 [0.113] | 19.5 | 4 | 0.059 [0.130] | 35.1 |
| attention+token_confidence | 61 (12%) | 350 (70%) | 89 (18%) | 30% | 1 | 0.016 [0.075] | 29.7 | 6 | 0.067 [0.129] | 45.9 |
| all_internal | 71 (14%) | 317 (63%) | 112 (22%) | 37% | 1 | 0.014 [0.065] | 34.6 | 14 | 0.125 [0.188] | 57.7 |
| context_action+all_internal | 76 (15%) | 293 (59%) | 131 (26%) | 41% | 2 | 0.026 [0.081] | 37.0 | 17 | 0.130 [0.188] | 67.5 |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 37 (7%) | 418 (84%) | 45 (9%) | 16% | 2 | 0.054 [0.161] | 18.0 | 6 | 0.133 [0.246] | 23.1 |
| token_entropy (untrained) | 37 (7%) | 396 (79%) | 67 (13%) | 21% | 2 | 0.054 [0.161] | 18.0 | 7 | 0.104 [0.187] | 34.5 |
| context_action | 52 (10%) | 363 (73%) | 85 (17%) | 27% | 2 | 0.038 [0.116] | 25.3 | 10 | 0.118 [0.191] | 43.7 |
| token_confidence | 58 (12%) | 361 (72%) | 81 (16%) | 28% | 2 | 0.034 [0.105] | 28.2 | 6 | 0.074 [0.141] | 41.7 |
| attention | 64 (13%) | 368 (74%) | 68 (14%) | 26% | 2 | 0.031 [0.095] | 31.2 | 4 | 0.059 [0.130] | 35.0 |
| attention+token_confidence | 75 (15%) | 336 (67%) | 89 (18%) | 33% | 1 | 0.013 [0.062] | 36.5 | 6 | 0.067 [0.129] | 45.8 |
| all_internal | 79 (16%) | 309 (62%) | 112 (22%) | 38% | 3 | 0.038 [0.095] | 38.4 | 14 | 0.125 [0.188] | 57.7 |
| context_action+all_internal | 89 (18%) | 280 (56%) | 131 (26%) | 44% | 3 | 0.034 [0.085] | 43.3 | 17 | 0.130 [0.188] | 67.4 |
