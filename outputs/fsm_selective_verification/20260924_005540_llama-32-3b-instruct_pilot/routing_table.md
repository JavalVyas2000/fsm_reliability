# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)

Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; **DISALLOW** = reject without simulator. p = predicted failure probability.
Test candidates: 500 generated, 0 rejected by cheap format checks, 500 routed; failure prevalence 0.630.
Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; `point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over 2000 draws).

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 339 (68%) | 161 (32%) | 32% | 0 | – [–] | 0.0 | 11 | 0.068 [0.111] | 59.6 |
| token_confidence | 0 (0%) | 457 (91%) | 43 (9%) | 9% | 0 | – [–] | 0.0 | 3 | 0.070 [0.171] | 16.0 |
| attention | 0 (0%) | 425 (85%) | 75 (15%) | 15% | 0 | – [–] | 0.0 | 3 | 0.040 [0.100] | 27.9 |
| attention+token_confidence | 0 (0%) | 385 (77%) | 115 (23%) | 23% | 0 | – [–] | 0.0 | 9 | 0.078 [0.133] | 42.7 |
| all_internal | 0 (0%) | 349 (70%) | 151 (30%) | 30% | 0 | – [–] | 0.0 | 14 | 0.093 [0.141] | 55.9 |
| context_action+all_internal | 0 (0%) | 325 (65%) | 175 (35%) | 35% | 0 | – [–] | 0.0 | 13 | 0.074 [0.116] | 64.8 |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 380 (76%) | 120 (24%) | 24% | 0 | – [–] | 0.0 | 7 | 0.058 [0.107] | 44.6 |
| token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 422 (84%) | 78 (16%) | 16% | 0 | – [–] | 0.0 | 3 | 0.038 [0.096] | 29.0 |
| context_action+all_internal | 0 (0%) | 423 (85%) | 77 (15%) | 15% | 0 | – [–] | 0.0 | 0 | 0.000 [0.038] | 28.6 |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 39 (8%) | 300 (60%) | 161 (32%) | 40% | 0 | 0.000 [0.074] | 24.5 | 11 | 0.068 [0.111] | 59.6 |
| token_confidence | 0 (0%) | 457 (91%) | 43 (9%) | 9% | 0 | – [–] | 0.0 | 3 | 0.070 [0.171] | 16.0 |
| attention | 0 (0%) | 425 (85%) | 75 (15%) | 15% | 0 | – [–] | 0.0 | 3 | 0.040 [0.100] | 27.9 |
| attention+token_confidence | 0 (0%) | 385 (77%) | 115 (23%) | 23% | 0 | – [–] | 0.0 | 9 | 0.078 [0.133] | 42.7 |
| all_internal | 43 (9%) | 306 (61%) | 151 (30%) | 39% | 1 | 0.023 [0.106] | 27.0 | 14 | 0.093 [0.141] | 55.8 |
| context_action+all_internal | 67 (13%) | 258 (52%) | 175 (35%) | 48% | 3 | 0.045 [0.112] | 42.1 | 13 | 0.074 [0.116] | 64.6 |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 46 (9%) | 380 (76%) | 74 (15%) | 24% | 8 | 0.174 [0.292] | 28.9 | 14 | 0.189 [0.280] | 27.5 |
| token_entropy (untrained) | 51 (10%) | 287 (57%) | 162 (32%) | 43% | 9 | 0.176 [0.288] | 32.0 | 24 | 0.148 [0.202] | 59.9 |
| context_action | 71 (14%) | 139 (28%) | 290 (58%) | 72% | 6 | 0.085 [0.160] | 44.6 | 39 | 0.134 [0.172] | 107.1 |
| token_confidence | 33 (7%) | 253 (51%) | 214 (43%) | 49% | 4 | 0.121 [0.256] | 20.7 | 28 | 0.131 [0.175] | 79.1 |
| attention | 88 (18%) | 167 (33%) | 245 (49%) | 67% | 8 | 0.091 [0.158] | 55.4 | 33 | 0.135 [0.176] | 90.5 |
| attention+token_confidence | 96 (19%) | 112 (22%) | 292 (58%) | 78% | 10 | 0.104 [0.170] | 60.4 | 42 | 0.144 [0.182] | 108.0 |
| all_internal | 90 (18%) | 134 (27%) | 276 (55%) | 73% | 9 | 0.100 [0.168] | 56.6 | 41 | 0.149 [0.188] | 102.0 |
| context_action+all_internal | 105 (21%) | 94 (19%) | 301 (60%) | 81% | 13 | 0.124 [0.190] | 66.0 | 43 | 0.143 [0.180] | 111.3 |

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 4 (1%) | 431 (86%) | 65 (13%) | 14% | 0 | 0.000 [0.527] | 2.5 | 13 | 0.200 [0.299] | 24.2 |
| token_entropy (untrained) | 3 (1%) | 383 (77%) | 114 (23%) | 23% | 0 | 0.000 [0.632] | 1.9 | 16 | 0.140 [0.205] | 42.4 |
| context_action | 39 (8%) | 268 (54%) | 193 (39%) | 46% | 0 | 0.000 [0.074] | 24.5 | 14 | 0.073 [0.111] | 71.3 |
| token_confidence | 0 (0%) | 386 (77%) | 114 (23%) | 23% | 0 | – [–] | 0.0 | 11 | 0.096 [0.155] | 42.3 |
| attention | 3 (1%) | 357 (71%) | 140 (28%) | 29% | 0 | 0.000 [0.632] | 1.9 | 11 | 0.079 [0.127] | 51.9 |
| attention+token_confidence | 25 (5%) | 309 (62%) | 166 (33%) | 38% | 0 | 0.000 [0.113] | 15.8 | 17 | 0.102 [0.150] | 61.5 |
| all_internal | 26 (5%) | 274 (55%) | 200 (40%) | 45% | 1 | 0.038 [0.170] | 16.4 | 22 | 0.110 [0.153] | 74.0 |
| context_action+all_internal | 47 (9%) | 235 (47%) | 218 (44%) | 53% | 2 | 0.043 [0.128] | 29.5 | 22 | 0.101 [0.141] | 80.6 |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 22 (4%) | 478 (96%) | 0 (0%) | 4% | 1 | 0.045 [0.198] | 13.9 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 3 (1%) | 437 (87%) | 60 (12%) | 13% | 0 | 0.000 [0.632] | 1.9 | 12 | 0.200 [0.304] | 22.3 |
| context_action | 69 (14%) | 292 (58%) | 139 (28%) | 42% | 4 | 0.058 [0.128] | 43.3 | 8 | 0.058 [0.101] | 51.4 |
| token_confidence | 32 (6%) | 418 (84%) | 50 (10%) | 16% | 3 | 0.094 [0.225] | 20.1 | 4 | 0.080 [0.174] | 18.5 |
| attention | 3 (1%) | 418 (84%) | 79 (16%) | 16% | 0 | 0.000 [0.632] | 1.9 | 3 | 0.038 [0.095] | 29.3 |
| attention+token_confidence | 42 (8%) | 343 (69%) | 115 (23%) | 31% | 1 | 0.024 [0.108] | 26.4 | 9 | 0.078 [0.133] | 42.5 |
| all_internal | 43 (9%) | 337 (67%) | 120 (24%) | 33% | 1 | 0.023 [0.106] | 27.0 | 11 | 0.092 [0.147] | 44.4 |
| context_action+all_internal | 74 (15%) | 261 (52%) | 165 (33%) | 48% | 4 | 0.054 [0.119] | 46.5 | 9 | 0.055 [0.093] | 60.9 |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 51 (10%) | 384 (77%) | 65 (13%) | 23% | 10 | 0.196 [0.310] | 32.0 | 13 | 0.200 [0.299] | 24.1 |
| token_entropy (untrained) | 51 (10%) | 335 (67%) | 114 (23%) | 33% | 9 | 0.176 [0.288] | 32.0 | 16 | 0.140 [0.205] | 42.1 |
| context_action | 71 (14%) | 236 (47%) | 193 (39%) | 53% | 6 | 0.085 [0.160] | 44.6 | 14 | 0.073 [0.111] | 71.3 |
| token_confidence | 46 (9%) | 340 (68%) | 114 (23%) | 32% | 6 | 0.130 [0.241] | 28.9 | 11 | 0.096 [0.155] | 42.2 |
| attention | 69 (14%) | 291 (58%) | 140 (28%) | 42% | 6 | 0.087 [0.164] | 43.3 | 11 | 0.079 [0.127] | 51.7 |
| attention+token_confidence | 80 (16%) | 254 (51%) | 166 (33%) | 49% | 4 | 0.050 [0.111] | 50.3 | 17 | 0.102 [0.150] | 61.4 |
| all_internal | 81 (16%) | 219 (44%) | 200 (40%) | 56% | 8 | 0.099 [0.171] | 50.9 | 22 | 0.110 [0.153] | 73.9 |
| context_action+all_internal | 95 (19%) | 187 (37%) | 218 (44%) | 63% | 7 | 0.074 [0.134] | 59.7 | 22 | 0.101 [0.141] | 80.5 |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 74 (15%) | 174 (35%) | 252 (50%) | 65% | 15 | 0.203 [0.295] | 46.5 | 51 | 0.202 [0.249] | 93.0 |
| token_entropy (untrained) | 71 (14%) | 210 (42%) | 219 (44%) | 58% | 16 | 0.225 [0.322] | 44.6 | 41 | 0.187 [0.236] | 80.8 |
| context_action | 117 (23%) | 52 (10%) | 331 (66%) | 90% | 21 | 0.179 [0.248] | 73.5 | 58 | 0.175 [0.213] | 122.4 |
| token_confidence | 74 (15%) | 127 (25%) | 299 (60%) | 75% | 13 | 0.176 [0.265] | 46.5 | 56 | 0.187 [0.228] | 110.4 |
| attention | 116 (23%) | 39 (8%) | 345 (69%) | 92% | 17 | 0.147 [0.212] | 72.9 | 63 | 0.183 [0.220] | 127.6 |
| attention+token_confidence | 124 (25%) | 14 (3%) | 362 (72%) | 97% | 16 | 0.129 [0.189] | 78.0 | 67 | 0.185 [0.222] | 133.8 |
| all_internal | 131 (26%) | 25 (5%) | 344 (69%) | 95% | 26 | 0.198 [0.265] | 82.4 | 70 | 0.203 [0.242] | 127.2 |
| context_action+all_internal | 143 (29%) | 18 (4%) | 339 (68%) | 96% | 28 | 0.196 [0.258] | 90.0 | 62 | 0.183 [0.221] | 125.4 |
