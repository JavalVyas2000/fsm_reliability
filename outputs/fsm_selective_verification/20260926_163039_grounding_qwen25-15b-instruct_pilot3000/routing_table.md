# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)

Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; **DISALLOW** = reject without simulator. p = predicted failure probability.
Test candidates: 500 generated, 2 rejected by cheap format checks, 498 routed; failure prevalence 0.689.
Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; `point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over 2000 draws).

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 493 (99%) | 5 (1%) | 1% | 0 | – [–] | 0.0 | 4 | 0.800 [0.990] | 1.6 |
| token_entropy (untrained) | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 334 (67%) | 164 (33%) | 33% | 0 | – [–] | 0.0 | 5 | 0.030 [0.063] | 50.9 |
| token_confidence | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 357 (72%) | 141 (28%) | 28% | 0 | – [–] | 0.0 | 4 | 0.028 [0.064] | 43.8 |
| attention+token_confidence | 0 (0%) | 381 (77%) | 117 (23%) | 23% | 0 | – [–] | 0.0 | 3 | 0.026 [0.065] | 36.4 |
| all_internal | 0 (0%) | 410 (82%) | 88 (18%) | 18% | 0 | – [–] | 0.0 | 2 | 0.023 [0.070] | 27.3 |
| context_action+all_internal | 0 (0%) | 361 (72%) | 137 (28%) | 28% | 0 | – [–] | 0.0 | 4 | 0.029 [0.066] | 42.5 |
| grounding | 0 (0%) | 369 (74%) | 129 (26%) | 26% | 0 | – [–] | 0.0 | 6 | 0.047 [0.090] | 40.0 |
| context_action+grounding | 0 (0%) | 277 (56%) | 221 (44%) | 44% | 0 | – [–] | 0.0 | 8 | 0.036 [0.064] | 68.7 |
| context_action+all_internal+grounding | 0 (0%) | 332 (67%) | 166 (33%) | 33% | 0 | – [–] | 0.0 | 7 | 0.042 [0.078] | 51.6 |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 493 (99%) | 5 (1%) | 1% | 0 | – [–] | 0.0 | 4 | 0.800 [0.990] | 1.6 |
| token_entropy (untrained) | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 443 (89%) | 55 (11%) | 11% | 0 | – [–] | 0.0 | 0 | 0.000 [0.053] | 17.1 |
| token_confidence | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 0 (0%) | 416 (84%) | 82 (16%) | 16% | 0 | – [–] | 0.0 | 1 | 0.012 [0.057] | 25.5 |
| grounding | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+grounding | 0 (0%) | 396 (80%) | 102 (20%) | 20% | 0 | – [–] | 0.0 | 0 | 0.000 [0.029] | 31.7 |
| context_action+all_internal+grounding | 0 (0%) | 386 (78%) | 112 (22%) | 22% | 0 | – [–] | 0.0 | 2 | 0.018 [0.055] | 34.9 |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 493 (99%) | 5 (1%) | 1% | 0 | – [–] | 0.0 | 4 | 0.800 [0.990] | 1.6 |
| token_entropy (untrained) | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 334 (67%) | 164 (33%) | 33% | 0 | – [–] | 0.0 | 5 | 0.030 [0.063] | 50.9 |
| token_confidence | 0 (0%) | 498 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 357 (72%) | 141 (28%) | 28% | 0 | – [–] | 0.0 | 4 | 0.028 [0.064] | 43.8 |
| attention+token_confidence | 0 (0%) | 381 (77%) | 117 (23%) | 23% | 0 | – [–] | 0.0 | 3 | 0.026 [0.065] | 36.4 |
| all_internal | 0 (0%) | 410 (82%) | 88 (18%) | 18% | 0 | – [–] | 0.0 | 2 | 0.023 [0.070] | 27.3 |
| context_action+all_internal | 28 (6%) | 333 (67%) | 137 (28%) | 33% | 0 | 0.000 [0.101] | 19.3 | 4 | 0.029 [0.066] | 42.6 |
| grounding | 32 (6%) | 337 (68%) | 129 (26%) | 32% | 2 | 0.062 [0.184] | 22.1 | 6 | 0.047 [0.090] | 40.0 |
| context_action+grounding | 35 (7%) | 242 (49%) | 221 (44%) | 51% | 0 | 0.000 [0.082] | 24.1 | 8 | 0.036 [0.064] | 68.6 |
| context_action+all_internal+grounding | 24 (5%) | 308 (62%) | 166 (33%) | 38% | 0 | 0.000 [0.117] | 16.5 | 7 | 0.042 [0.078] | 51.6 |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 350 (70%) | 148 (30%) | 30% | 0 | – [–] | 0.0 | 22 | 0.149 [0.205] | 45.9 |
| token_entropy (untrained) | 0 (0%) | 237 (48%) | 261 (52%) | 52% | 0 | – [–] | 0.0 | 34 | 0.130 [0.170] | 81.1 |
| context_action | 52 (10%) | 100 (20%) | 346 (69%) | 80% | 2 | 0.038 [0.116] | 35.9 | 52 | 0.150 [0.186] | 107.6 |
| token_confidence | 0 (0%) | 220 (44%) | 278 (56%) | 56% | 0 | – [–] | 0.0 | 31 | 0.112 [0.148] | 86.3 |
| attention | 36 (7%) | 102 (20%) | 360 (72%) | 80% | 3 | 0.083 [0.202] | 24.8 | 53 | 0.147 [0.181] | 112.0 |
| attention+token_confidence | 57 (11%) | 81 (16%) | 360 (72%) | 84% | 8 | 0.140 [0.239] | 39.3 | 57 | 0.158 [0.193] | 111.9 |
| all_internal | 36 (7%) | 127 (26%) | 335 (67%) | 74% | 2 | 0.056 [0.165] | 24.8 | 53 | 0.158 [0.195] | 104.1 |
| context_action+all_internal | 49 (10%) | 116 (23%) | 333 (67%) | 77% | 4 | 0.082 [0.177] | 33.8 | 54 | 0.162 [0.199] | 103.6 |
| grounding | 56 (11%) | 76 (15%) | 366 (73%) | 85% | 5 | 0.089 [0.179] | 38.6 | 52 | 0.142 [0.176] | 113.8 |
| context_action+grounding | 78 (16%) | 56 (11%) | 364 (73%) | 89% | 10 | 0.128 [0.208] | 53.7 | 47 | 0.129 [0.162] | 113.2 |
| context_action+all_internal+grounding | 84 (17%) | 37 (7%) | 377 (76%) | 93% | 11 | 0.131 [0.207] | 57.9 | 58 | 0.154 [0.188] | 117.3 |

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 487 (98%) | 11 (2%) | 2% | 0 | – [–] | 0.0 | 6 | 0.545 [0.800] | 3.4 |
| token_entropy (untrained) | 0 (0%) | 342 (69%) | 156 (31%) | 31% | 0 | – [–] | 0.0 | 14 | 0.090 [0.137] | 48.4 |
| context_action | 32 (6%) | 202 (41%) | 264 (53%) | 59% | 0 | 0.000 [0.089] | 22.1 | 26 | 0.098 [0.134] | 82.0 |
| token_confidence | 3 (1%) | 359 (72%) | 136 (27%) | 28% | 0 | 0.000 [0.632] | 2.1 | 11 | 0.081 [0.130] | 42.2 |
| attention | 35 (7%) | 230 (46%) | 233 (47%) | 54% | 3 | 0.086 [0.207] | 24.1 | 18 | 0.077 [0.112] | 72.4 |
| attention+token_confidence | 31 (6%) | 250 (50%) | 217 (44%) | 50% | 2 | 0.065 [0.189] | 21.4 | 14 | 0.065 [0.099] | 67.4 |
| all_internal | 28 (6%) | 287 (58%) | 183 (37%) | 42% | 2 | 0.071 [0.208] | 19.3 | 14 | 0.077 [0.117] | 56.9 |
| context_action+all_internal | 28 (6%) | 260 (52%) | 210 (42%) | 48% | 0 | 0.000 [0.101] | 19.3 | 17 | 0.081 [0.119] | 65.3 |
| grounding | 32 (6%) | 241 (48%) | 225 (45%) | 52% | 2 | 0.062 [0.184] | 22.1 | 13 | 0.058 [0.090] | 69.9 |
| context_action+grounding | 35 (7%) | 177 (36%) | 286 (57%) | 64% | 0 | 0.000 [0.082] | 24.1 | 23 | 0.080 [0.112] | 88.8 |
| context_action+all_internal+grounding | 24 (5%) | 230 (46%) | 244 (49%) | 54% | 0 | 0.000 [0.117] | 16.5 | 18 | 0.074 [0.107] | 75.8 |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 496 (100%) | 2 (0%) | 0% | 0 | – [–] | 0.0 | 2 | 1.000 [1.000] | 0.6 |
| token_entropy (untrained) | 0 (0%) | 496 (100%) | 2 (0%) | 0% | 0 | – [–] | 0.0 | 0 | 0.000 [0.776] | 0.6 |
| context_action | 47 (9%) | 317 (64%) | 134 (27%) | 36% | 2 | 0.043 [0.128] | 32.4 | 2 | 0.015 [0.046] | 41.7 |
| token_confidence | 3 (1%) | 473 (95%) | 22 (4%) | 5% | 0 | 0.000 [0.632] | 2.1 | 0 | 0.000 [0.127] | 6.8 |
| attention | 36 (7%) | 321 (64%) | 141 (28%) | 36% | 3 | 0.083 [0.202] | 24.8 | 4 | 0.028 [0.064] | 43.8 |
| attention+token_confidence | 32 (6%) | 355 (71%) | 111 (22%) | 29% | 2 | 0.062 [0.184] | 22.1 | 2 | 0.018 [0.056] | 34.4 |
| all_internal | 35 (7%) | 371 (74%) | 92 (18%) | 26% | 2 | 0.057 [0.169] | 24.1 | 2 | 0.022 [0.067] | 28.5 |
| context_action+all_internal | 35 (7%) | 327 (66%) | 136 (27%) | 34% | 1 | 0.029 [0.129] | 24.1 | 4 | 0.029 [0.066] | 42.2 |
| grounding | 43 (9%) | 333 (67%) | 122 (24%) | 33% | 4 | 0.093 [0.200] | 29.6 | 6 | 0.049 [0.095] | 37.9 |
| context_action+grounding | 48 (10%) | 273 (55%) | 177 (36%) | 45% | 2 | 0.042 [0.125] | 33.1 | 3 | 0.017 [0.043] | 55.0 |
| context_action+all_internal+grounding | 40 (8%) | 305 (61%) | 153 (31%) | 39% | 2 | 0.050 [0.149] | 27.6 | 5 | 0.033 [0.067] | 47.5 |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 487 (98%) | 11 (2%) | 2% | 0 | – [–] | 0.0 | 6 | 0.545 [0.800] | 3.4 |
| token_entropy (untrained) | 0 (0%) | 342 (69%) | 156 (31%) | 31% | 0 | – [–] | 0.0 | 14 | 0.090 [0.137] | 48.4 |
| context_action | 52 (10%) | 182 (37%) | 264 (53%) | 63% | 2 | 0.038 [0.116] | 35.9 | 26 | 0.098 [0.134] | 82.0 |
| token_confidence | 13 (3%) | 349 (70%) | 136 (27%) | 30% | 2 | 0.154 [0.410] | 9.0 | 11 | 0.081 [0.130] | 42.2 |
| attention | 51 (10%) | 214 (43%) | 233 (47%) | 57% | 5 | 0.098 [0.195] | 35.2 | 18 | 0.077 [0.112] | 72.4 |
| attention+token_confidence | 57 (11%) | 224 (45%) | 217 (44%) | 55% | 8 | 0.140 [0.239] | 39.3 | 14 | 0.065 [0.099] | 67.4 |
| all_internal | 42 (8%) | 273 (55%) | 183 (37%) | 45% | 3 | 0.071 [0.174] | 28.9 | 14 | 0.077 [0.117] | 56.9 |
| context_action+all_internal | 49 (10%) | 239 (48%) | 210 (42%) | 52% | 4 | 0.082 [0.177] | 33.8 | 17 | 0.081 [0.119] | 65.3 |
| grounding | 56 (11%) | 217 (44%) | 225 (45%) | 56% | 5 | 0.089 [0.179] | 38.6 | 13 | 0.058 [0.090] | 69.9 |
| context_action+grounding | 73 (15%) | 139 (28%) | 286 (57%) | 72% | 8 | 0.110 [0.189] | 50.3 | 23 | 0.080 [0.112] | 88.9 |
| context_action+all_internal+grounding | 70 (14%) | 184 (37%) | 244 (49%) | 63% | 8 | 0.114 [0.197] | 48.3 | 18 | 0.074 [0.107] | 75.8 |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 18 (4%) | 127 (26%) | 353 (71%) | 74% | 3 | 0.167 [0.377] | 12.4 | 74 | 0.210 [0.248] | 109.7 |
| token_entropy (untrained) | 12 (2%) | 163 (33%) | 323 (65%) | 67% | 1 | 0.083 [0.339] | 8.3 | 57 | 0.176 [0.215] | 100.4 |
| context_action | 56 (11%) | 46 (9%) | 396 (80%) | 91% | 6 | 0.107 [0.201] | 38.6 | 75 | 0.189 [0.225] | 123.2 |
| token_confidence | 24 (5%) | 120 (24%) | 354 (71%) | 76% | 3 | 0.125 [0.292] | 16.5 | 61 | 0.172 [0.209] | 110.1 |
| attention | 71 (14%) | 22 (4%) | 405 (81%) | 96% | 12 | 0.169 [0.259] | 48.9 | 81 | 0.200 [0.236] | 126.0 |
| attention+token_confidence | 77 (15%) | 21 (4%) | 400 (80%) | 96% | 14 | 0.182 [0.270] | 53.1 | 77 | 0.193 [0.228] | 124.4 |
| all_internal | 66 (13%) | 30 (6%) | 402 (81%) | 94% | 9 | 0.136 [0.226] | 45.5 | 81 | 0.201 [0.237] | 125.1 |
| context_action+all_internal | 78 (16%) | 9 (2%) | 411 (83%) | 98% | 13 | 0.167 [0.252] | 53.7 | 87 | 0.212 [0.248] | 127.9 |
| grounding | 94 (19%) | 5 (1%) | 399 (80%) | 99% | 14 | 0.149 [0.223] | 64.8 | 72 | 0.180 [0.215] | 124.2 |
| context_action+grounding | 122 (24%) | 0 (0%) | 376 (76%) | 100% | 22 | 0.180 [0.247] | 84.1 | 55 | 0.146 [0.180] | 117.1 |
| context_action+all_internal+grounding | 121 (24%) | 0 (0%) | 377 (76%) | 100% | 24 | 0.198 [0.267] | 83.4 | 58 | 0.154 [0.188] | 117.4 |
