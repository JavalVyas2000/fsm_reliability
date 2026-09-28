# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)

Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; **DISALLOW** = reject without simulator. p = predicted failure probability.
Test candidates: 500 generated, 0 rejected by cheap format checks, 500 routed; failure prevalence 0.486.
Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; `point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over 2000 draws).

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 65 (13%) | 435 (87%) | 0 (0%) | 13% | 1 | 0.015 [0.071] | 31.7 | 0 | – [–] | 0.0 |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 500 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 45 (9%) | 413 (83%) | 42 (8%) | 17% | 0 | 0.000 [0.064] | 21.9 | 1 | 0.024 [0.108] | 21.6 |
| token_confidence | 0 (0%) | 462 (92%) | 38 (8%) | 8% | 0 | – [–] | 0.0 | 5 | 0.132 [0.257] | 19.5 |
| attention | 40 (8%) | 401 (80%) | 59 (12%) | 20% | 1 | 0.025 [0.113] | 19.5 | 4 | 0.068 [0.148] | 30.4 |
| attention+token_confidence | 74 (15%) | 426 (85%) | 0 (0%) | 15% | 1 | 0.014 [0.063] | 36.0 | 0 | – [–] | 0.0 |
| all_internal | 79 (16%) | 389 (78%) | 32 (6%) | 22% | 3 | 0.038 [0.095] | 38.4 | 2 | 0.062 [0.184] | 16.6 |
| context_action+all_internal | 84 (17%) | 371 (74%) | 45 (9%) | 26% | 2 | 0.024 [0.073] | 40.8 | 2 | 0.044 [0.133] | 23.1 |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 52 (10%) | 380 (76%) | 68 (14%) | 24% | 5 | 0.096 [0.192] | 25.3 | 11 | 0.162 [0.254] | 35.1 |
| token_entropy (untrained) | 43 (9%) | 390 (78%) | 67 (13%) | 22% | 3 | 0.070 [0.171] | 20.9 | 7 | 0.104 [0.187] | 34.6 |
| context_action | 101 (20%) | 288 (58%) | 111 (22%) | 42% | 15 | 0.149 [0.219] | 49.0 | 13 | 0.117 [0.180] | 57.1 |
| token_confidence | 75 (15%) | 340 (68%) | 85 (17%) | 32% | 5 | 0.067 [0.135] | 36.5 | 6 | 0.071 [0.135] | 43.7 |
| attention | 132 (26%) | 262 (52%) | 106 (21%) | 48% | 16 | 0.121 [0.178] | 64.2 | 11 | 0.104 [0.166] | 54.6 |
| attention+token_confidence | 129 (26%) | 237 (47%) | 134 (27%) | 53% | 13 | 0.101 [0.155] | 62.7 | 19 | 0.142 [0.201] | 69.0 |
| all_internal | 144 (29%) | 200 (40%) | 156 (31%) | 60% | 16 | 0.111 [0.164] | 69.9 | 29 | 0.186 [0.245] | 80.2 |
| context_action+all_internal | 167 (33%) | 183 (37%) | 150 (30%) | 63% | 21 | 0.126 [0.176] | 81.2 | 24 | 0.160 [0.218] | 77.1 |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 37 (7%) | 458 (92%) | 5 (1%) | 8% | 2 | 0.054 [0.161] | 18.0 | 0 | 0.000 [0.451] | 2.6 |
| token_entropy (untrained) | 37 (7%) | 456 (91%) | 7 (1%) | 9% | 2 | 0.054 [0.161] | 18.0 | 0 | 0.000 [0.348] | 3.6 |
| context_action | 52 (10%) | 402 (80%) | 46 (9%) | 20% | 2 | 0.038 [0.116] | 25.3 | 1 | 0.022 [0.099] | 23.8 |
| token_confidence | 58 (12%) | 395 (79%) | 47 (9%) | 21% | 2 | 0.034 [0.105] | 28.2 | 5 | 0.106 [0.211] | 24.3 |
| attention | 64 (13%) | 374 (75%) | 62 (12%) | 25% | 2 | 0.031 [0.095] | 31.2 | 4 | 0.065 [0.142] | 32.0 |
| attention+token_confidence | 75 (15%) | 368 (74%) | 57 (11%) | 26% | 1 | 0.013 [0.062] | 36.5 | 1 | 0.018 [0.081] | 29.3 |
| all_internal | 79 (16%) | 381 (76%) | 40 (8%) | 24% | 3 | 0.038 [0.095] | 38.4 | 3 | 0.075 [0.183] | 20.7 |
| context_action+all_internal | 89 (18%) | 330 (66%) | 81 (16%) | 34% | 3 | 0.034 [0.085] | 43.3 | 6 | 0.074 [0.141] | 41.7 |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 53 (11%) | 402 (80%) | 45 (9%) | 20% | 5 | 0.094 [0.188] | 25.8 | 6 | 0.133 [0.246] | 23.3 |
| token_entropy (untrained) | 53 (11%) | 380 (76%) | 67 (13%) | 24% | 5 | 0.094 [0.188] | 25.8 | 7 | 0.104 [0.187] | 34.6 |
| context_action | 93 (19%) | 322 (64%) | 85 (17%) | 36% | 13 | 0.140 [0.213] | 45.2 | 10 | 0.118 [0.191] | 43.7 |
| token_confidence | 75 (15%) | 344 (69%) | 81 (16%) | 31% | 5 | 0.067 [0.135] | 36.5 | 6 | 0.074 [0.141] | 41.7 |
| attention | 86 (17%) | 346 (69%) | 68 (14%) | 31% | 5 | 0.058 [0.118] | 41.8 | 4 | 0.059 [0.130] | 35.0 |
| attention+token_confidence | 84 (17%) | 327 (65%) | 89 (18%) | 35% | 2 | 0.024 [0.073] | 40.8 | 6 | 0.067 [0.129] | 45.8 |
| all_internal | 121 (24%) | 267 (53%) | 112 (22%) | 47% | 9 | 0.074 [0.126] | 58.8 | 14 | 0.125 [0.188] | 57.6 |
| context_action+all_internal | 120 (24%) | 249 (50%) | 131 (26%) | 50% | 7 | 0.058 [0.107] | 58.3 | 17 | 0.130 [0.188] | 67.4 |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 103 (21%) | 292 (58%) | 105 (21%) | 42% | 12 | 0.117 [0.182] | 50.0 | 23 | 0.219 [0.296] | 54.1 |
| token_entropy (untrained) | 103 (21%) | 275 (55%) | 122 (24%) | 45% | 12 | 0.117 [0.182] | 50.0 | 27 | 0.221 [0.292] | 62.7 |
| context_action | 162 (32%) | 169 (34%) | 169 (34%) | 66% | 30 | 0.185 [0.243] | 78.7 | 34 | 0.201 [0.259] | 86.9 |
| token_confidence | 116 (23%) | 248 (50%) | 136 (27%) | 50% | 13 | 0.112 [0.172] | 56.3 | 19 | 0.140 [0.198] | 69.9 |
| attention | 159 (32%) | 173 (35%) | 168 (34%) | 65% | 26 | 0.164 [0.220] | 77.3 | 32 | 0.190 [0.247] | 86.3 |
| attention+token_confidence | 152 (30%) | 188 (38%) | 160 (32%) | 62% | 22 | 0.145 [0.200] | 73.8 | 25 | 0.156 [0.211] | 82.2 |
| all_internal | 193 (39%) | 103 (21%) | 204 (41%) | 79% | 35 | 0.181 [0.233] | 93.8 | 46 | 0.225 [0.279] | 104.9 |
| context_action+all_internal | 195 (39%) | 110 (22%) | 195 (39%) | 78% | 34 | 0.174 [0.225] | 94.7 | 41 | 0.210 [0.264] | 100.2 |
