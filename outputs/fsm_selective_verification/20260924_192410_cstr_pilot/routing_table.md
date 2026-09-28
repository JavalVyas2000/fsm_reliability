# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)

Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; **DISALLOW** = reject without simulator. p = predicted failure probability.
Test candidates: 50 generated, 0 rejected by cheap format checks, 50 routed; failure prevalence 0.560.
Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; `point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over 2000 draws).

## rule = ucb, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 47 (94%) | 3 (6%) | 6% | 0 | – [–] | 0.0 | 1 | 0.333 [0.865] | 1.3 |
| context_action | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |

## rule = ucb, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 47 (94%) | 3 (6%) | 6% | 0 | – [–] | 0.0 | 1 | 0.333 [0.865] | 1.3 |
| context_action | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |

## rule = ucb, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 47 (94%) | 3 (6%) | 6% | 0 | – [–] | 0.0 | 1 | 0.333 [0.865] | 1.3 |
| context_action | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |

## rule = ucb, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 47 (94%) | 3 (6%) | 6% | 0 | – [–] | 0.0 | 1 | 0.333 [0.865] | 1.3 |
| context_action | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |

## rule = point, ALLOW failure target α = 0.0, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 7 (14%) | 32 (64%) | 11 (22%) | 36% | 0 | 0.000 [0.348] | 4.0 | 2 | 0.182 [0.470] | 4.8 |
| token_confidence | 0 (0%) | 44 (88%) | 6 (12%) | 12% | 0 | – [–] | 0.0 | 1 | 0.167 [0.582] | 2.6 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 10 (20%) | 40 (80%) | 0 (0%) | 20% | 6 | 0.600 [0.850] | 5.6 | 0 | – [–] | 0.0 |

## rule = point, ALLOW failure target α = 0.05, DISALLOW valid target β = 0.05

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 7 (14%) | 32 (64%) | 11 (22%) | 36% | 0 | 0.000 [0.348] | 4.0 | 2 | 0.182 [0.470] | 4.8 |
| token_confidence | 0 (0%) | 44 (88%) | 6 (12%) | 12% | 0 | – [–] | 0.0 | 1 | 0.167 [0.582] | 2.6 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 10 (20%) | 40 (80%) | 0 (0%) | 20% | 6 | 0.600 [0.850] | 5.6 | 0 | – [–] | 0.0 |

## rule = point, ALLOW failure target α = 0.1, DISALLOW valid target β = 0.1

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 15 (30%) | 24 (48%) | 11 (22%) | 52% | 5 | 0.333 [0.577] | 8.5 | 2 | 0.182 [0.470] | 4.9 |
| token_confidence | 0 (0%) | 44 (88%) | 6 (12%) | 12% | 0 | – [–] | 0.0 | 1 | 0.167 [0.582] | 2.6 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 10 (20%) | 40 (80%) | 0 (0%) | 20% | 6 | 0.600 [0.850] | 5.6 | 0 | – [–] | 0.0 |

## rule = point, ALLOW failure target α = 0.2, DISALLOW valid target β = 0.2

| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |
|---|---|---|---|---|---|---|---|---|---|---|
| neg_mean_logprob (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| token_entropy (untrained) | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action | 24 (48%) | 9 (18%) | 17 (34%) | 82% | 9 | 0.375 [0.563] | 13.5 | 4 | 0.235 [0.461] | 7.5 |
| token_confidence | 0 (0%) | 43 (86%) | 7 (14%) | 14% | 0 | – [–] | 0.0 | 2 | 0.286 [0.659] | 3.0 |
| attention | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| attention+token_confidence | 0 (0%) | 40 (80%) | 10 (20%) | 20% | 0 | – [–] | 0.0 | 4 | 0.400 [0.696] | 4.4 |
| all_internal | 0 (0%) | 50 (100%) | 0 (0%) | 0% | 0 | – [–] | 0.0 | 0 | – [–] | 0.0 |
| context_action+all_internal | 10 (20%) | 40 (80%) | 0 (0%) | 20% | 6 | 0.600 [0.850] | 5.6 | 0 | – [–] | 0.0 |
