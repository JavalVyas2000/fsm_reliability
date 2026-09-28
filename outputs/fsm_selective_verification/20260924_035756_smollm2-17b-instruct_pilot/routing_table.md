# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)

Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; **DISALLOW** = reject without simulator. p = predicted failure probability.
Test candidates: 500 generated, 7 rejected by cheap format checks, 493 routed; failure prevalence 0.834.
Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; `point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over 2000 draws).

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 232 (47%) | 261 (53%) | 53% | 0 | – [–] | 0.0 | 16 | 0.061 [0.092] | 43.5 |
| token_entropy (untrained) | 0 (0%) | 232 (47%) | 261 (53%) | 53% | 0 | – [–] | 0.0 | 13 | 0.050 [0.078] | 43.5 |
| context_action | 0 (0%) | 87 (18%) | 406 (82%) | 82% | 0 | – [–] | 0.0 | 23 | 0.057 [0.079] | 67.5 |
| token_confidence | 0 (0%) | 181 (37%) | 312 (63%) | 63% | 0 | – [–] | 0.0 | 14 | 0.045 [0.069] | 52.0 |
| attention | 0 (0%) | 103 (21%) | 390 (79%) | 79% | 0 | – [–] | 0.0 | 22 | 0.056 [0.080] | 65.0 |
| attention+token_confidence | 0 (0%) | 111 (23%) | 382 (77%) | 77% | 0 | – [–] | 0.0 | 20 | 0.052 [0.075] | 63.7 |
| all_internal | 0 (0%) | 91 (18%) | 402 (82%) | 82% | 0 | – [–] | 0.0 | 25 | 0.062 [0.086] | 66.9 |
| context_action+all_internal | 0 (0%) | 80 (16%) | 413 (84%) | 84% | 0 | – [–] | 0.0 | 29 | 0.070 [0.095] | 68.7 |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 337 (68%) | 156 (32%) | 32% | 0 | – [–] | 0.0 | 8 | 0.051 [0.091] | 26.1 |
| token_entropy (untrained) | 0 (0%) | 297 (60%) | 196 (40%) | 40% | 0 | – [–] | 0.0 | 8 | 0.041 [0.072] | 32.7 |
| context_action | 0 (0%) | 150 (30%) | 343 (70%) | 70% | 0 | – [–] | 0.0 | 8 | 0.023 [0.042] | 57.2 |
| token_confidence | 0 (0%) | 330 (67%) | 163 (33%) | 33% | 0 | – [–] | 0.0 | 1 | 0.006 [0.029] | 27.3 |
| attention | 0 (0%) | 170 (34%) | 323 (66%) | 66% | 0 | – [–] | 0.0 | 10 | 0.031 [0.052] | 53.8 |
| attention+token_confidence | 0 (0%) | 181 (37%) | 312 (63%) | 63% | 0 | – [–] | 0.0 | 9 | 0.029 [0.050] | 52.0 |
| all_internal | 0 (0%) | 180 (37%) | 313 (63%) | 63% | 0 | – [–] | 0.0 | 7 | 0.022 [0.042] | 52.1 |
| context_action+all_internal | 0 (0%) | 127 (26%) | 366 (74%) | 74% | 0 | – [–] | 0.0 | 8 | 0.022 [0.039] | 61.0 |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 232 (47%) | 261 (53%) | 53% | 0 | – [–] | 0.0 | 16 | 0.061 [0.092] | 43.5 |
| token_entropy (untrained) | 0 (0%) | 232 (47%) | 261 (53%) | 53% | 0 | – [–] | 0.0 | 13 | 0.050 [0.078] | 43.5 |
| context_action | 0 (0%) | 87 (18%) | 406 (82%) | 82% | 0 | – [–] | 0.0 | 23 | 0.057 [0.079] | 67.5 |
| token_confidence | 0 (0%) | 181 (37%) | 312 (63%) | 63% | 0 | – [–] | 0.0 | 14 | 0.045 [0.069] | 52.0 |
| attention | 0 (0%) | 103 (21%) | 390 (79%) | 79% | 0 | – [–] | 0.0 | 22 | 0.056 [0.080] | 65.0 |
| attention+token_confidence | 0 (0%) | 111 (23%) | 382 (77%) | 77% | 0 | – [–] | 0.0 | 20 | 0.052 [0.075] | 63.7 |
| all_internal | 0 (0%) | 91 (18%) | 402 (82%) | 82% | 0 | – [–] | 0.0 | 25 | 0.062 [0.086] | 66.9 |
| context_action+all_internal | 0 (0%) | 80 (16%) | 413 (84%) | 84% | 0 | – [–] | 0.0 | 29 | 0.070 [0.095] | 68.7 |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 2 (0%) | 491 (100%) | 100% | 0 | – [–] | 0.0 | 81 | 0.165 [0.195] | 81.7 |
| token_entropy (untrained) | 0 (0%) | 2 (0%) | 491 (100%) | 100% | 0 | – [–] | 0.0 | 81 | 0.165 [0.195] | 81.7 |
| context_action | 33 (7%) | 0 (0%) | 460 (93%) | 100% | 1 | 0.030 [0.136] | 27.4 | 50 | 0.109 [0.136] | 76.4 |
| token_confidence | 0 (0%) | 1 (0%) | 492 (100%) | 100% | 0 | – [–] | 0.0 | 82 | 0.167 [0.197] | 81.8 |
| attention | 0 (0%) | 1 (0%) | 492 (100%) | 100% | 0 | – [–] | 0.0 | 81 | 0.165 [0.195] | 81.8 |
| attention+token_confidence | 0 (0%) | 1 (0%) | 492 (100%) | 100% | 0 | – [–] | 0.0 | 81 | 0.165 [0.195] | 81.8 |
| all_internal | 0 (0%) | 0 (0%) | 493 (100%) | 100% | 0 | – [–] | 0.0 | 82 | 0.166 [0.196] | 82.0 |
| context_action+all_internal | 19 (4%) | 0 (0%) | 474 (96%) | 100% | 2 | 0.105 [0.296] | 15.7 | 65 | 0.137 [0.166] | 78.7 |

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 (0%) | 115 (23%) | 376 (76%) | 77% | 1 | 0.500 [0.975] | 1.7 | 32 | 0.085 [0.113] | 62.7 |
| token_entropy (untrained) | 2 (0%) | 118 (24%) | 373 (76%) | 76% | 1 | 0.500 [0.975] | 1.7 | 36 | 0.097 [0.125] | 62.2 |
| context_action | 24 (5%) | 44 (9%) | 425 (86%) | 91% | 0 | 0.000 [0.117] | 19.9 | 31 | 0.073 [0.097] | 70.6 |
| token_confidence | 3 (1%) | 108 (22%) | 382 (77%) | 78% | 1 | 0.333 [0.865] | 2.5 | 28 | 0.073 [0.099] | 63.6 |
| attention | 2 (0%) | 76 (15%) | 415 (84%) | 85% | 0 | 0.000 [0.776] | 1.7 | 30 | 0.072 [0.097] | 69.0 |
| attention+token_confidence | 1 (0%) | 74 (15%) | 418 (85%) | 85% | 0 | 0.000 [0.950] | 0.8 | 29 | 0.069 [0.093] | 69.5 |
| all_internal | 7 (1%) | 64 (13%) | 422 (86%) | 87% | 3 | 0.429 [0.775] | 5.8 | 32 | 0.076 [0.101] | 70.1 |
| context_action+all_internal | 19 (4%) | 28 (6%) | 446 (90%) | 94% | 2 | 0.105 [0.296] | 15.7 | 44 | 0.099 [0.125] | 74.2 |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 (0%) | 269 (55%) | 222 (45%) | 45% | 1 | 0.500 [0.975] | 1.7 | 12 | 0.054 [0.086] | 36.9 |
| token_entropy (untrained) | 2 (0%) | 246 (50%) | 245 (50%) | 50% | 1 | 0.500 [0.975] | 1.7 | 11 | 0.045 [0.073] | 40.8 |
| context_action | 33 (7%) | 73 (15%) | 387 (78%) | 85% | 1 | 0.030 [0.136] | 27.4 | 16 | 0.041 [0.062] | 64.2 |
| token_confidence | 3 (1%) | 246 (50%) | 244 (49%) | 50% | 1 | 0.333 [0.865] | 2.5 | 6 | 0.025 [0.048] | 40.6 |
| attention | 2 (0%) | 132 (27%) | 359 (73%) | 73% | 0 | 0.000 [0.776] | 1.7 | 16 | 0.045 [0.067] | 59.9 |
| attention+token_confidence | 1 (0%) | 126 (26%) | 366 (74%) | 74% | 0 | 0.000 [0.950] | 0.8 | 18 | 0.049 [0.072] | 61.0 |
| all_internal | 7 (1%) | 109 (22%) | 377 (76%) | 78% | 3 | 0.429 [0.775] | 5.8 | 15 | 0.040 [0.061] | 62.8 |
| context_action+all_internal | 19 (4%) | 88 (18%) | 386 (78%) | 82% | 2 | 0.105 [0.296] | 15.7 | 17 | 0.044 [0.065] | 64.1 |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 2 (0%) | 115 (23%) | 376 (76%) | 77% | 1 | 0.500 [0.975] | 1.7 | 32 | 0.085 [0.113] | 62.7 |
| token_entropy (untrained) | 2 (0%) | 118 (24%) | 373 (76%) | 76% | 1 | 0.500 [0.975] | 1.7 | 36 | 0.097 [0.125] | 62.2 |
| context_action | 35 (7%) | 33 (7%) | 425 (86%) | 93% | 3 | 0.086 [0.207] | 29.0 | 31 | 0.073 [0.097] | 70.6 |
| token_confidence | 3 (1%) | 108 (22%) | 382 (77%) | 78% | 1 | 0.333 [0.865] | 2.5 | 28 | 0.073 [0.099] | 63.6 |
| attention | 2 (0%) | 76 (15%) | 415 (84%) | 85% | 0 | 0.000 [0.776] | 1.7 | 30 | 0.072 [0.097] | 69.0 |
| attention+token_confidence | 1 (0%) | 74 (15%) | 418 (85%) | 85% | 0 | 0.000 [0.950] | 0.8 | 29 | 0.069 [0.093] | 69.5 |
| all_internal | 7 (1%) | 64 (13%) | 422 (86%) | 87% | 3 | 0.429 [0.775] | 5.8 | 32 | 0.076 [0.101] | 70.1 |
| context_action+all_internal | 35 (7%) | 12 (2%) | 446 (90%) | 98% | 4 | 0.114 [0.243] | 29.0 | 44 | 0.099 [0.125] | 74.0 |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 8 (2%) | 0 (0%) | 485 (98%) | 100% | 3 | 0.375 [0.711] | 6.6 | 77 | 0.159 [0.189] | 80.6 |
| token_entropy (untrained) | 2 (0%) | 0 (0%) | 491 (100%) | 100% | 1 | 0.500 [0.975] | 1.7 | 81 | 0.165 [0.195] | 81.7 |
| context_action | 35 (7%) | 0 (0%) | 458 (93%) | 100% | 3 | 0.086 [0.207] | 29.0 | 50 | 0.109 [0.136] | 76.0 |
| token_confidence | 11 (2%) | 0 (0%) | 482 (98%) | 100% | 2 | 0.182 [0.470] | 9.1 | 73 | 0.151 [0.181] | 80.1 |
| attention | 23 (5%) | 0 (0%) | 470 (95%) | 100% | 4 | 0.174 [0.355] | 19.1 | 63 | 0.134 [0.163] | 78.1 |
| attention+token_confidence | 27 (5%) | 0 (0%) | 466 (95%) | 100% | 5 | 0.185 [0.351] | 22.4 | 60 | 0.129 [0.157] | 77.4 |
| all_internal | 7 (1%) | 0 (0%) | 486 (99%) | 100% | 3 | 0.429 [0.775] | 5.8 | 78 | 0.160 [0.190] | 80.8 |
| context_action+all_internal | 47 (10%) | 0 (0%) | 446 (90%) | 100% | 9 | 0.191 [0.310] | 39.0 | 44 | 0.099 [0.125] | 74.0 |
