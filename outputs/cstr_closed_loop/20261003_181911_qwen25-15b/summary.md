# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261003_181911_qwen25-15b`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 34.5% | 0.0% | 65.5% | 1.05 | 4.23 | 0.00 (0.00) | 47.5 (37.0 + 0.0 + 0.0 + 10.5) |
| observables_probe | 33.0% | 3.0% | 64.0% | 0.35 | 4.12 | 2.42 (0.11) | 43.0 (38.9 + 0.0 + 0.0 + 4.1) |
| combined_probe | 33.2% | 2.2% | 64.5% | 0.38 | 4.13 | 3.23 (0.08) | 45.9 (39.9 + 0.5 + 1.2 + 4.3) |
| random_matched | 33.8% | 46.5% | 19.8% | 0.42 | 2.96 | 1.21 (0.21) | 32.9 (28.6 + 0.0 + 0.0 + 4.3) |
| never_validate | 34.5% | 64.8% | 0.8% | 0.00 | 1.00 | 0.00 (0.00) | 10.8 (10.8 + 0.0 + 0.0 + 0.0) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe | -1.50 [-3.00, -0.25] | +3.00 [+1.50, +4.75] | -0.70 [-0.75, -0.65] | -6.39 [-6.92, -5.82] | -4.47 [-5.90, -3.09] |
| combined_probe | -1.25 [-2.51, +0.00] | +2.25 [+1.00, +3.75] | -0.67 [-0.72, -0.61] | -6.15 [-6.72, -5.54] | -1.59 [-3.23, -0.05] |
| random_matched | -0.75 [-1.75, +0.00] | +46.50 [+41.50, +51.25] | -0.63 [-0.68, -0.57] | -6.22 [-6.80, -5.58] | -14.63 [-16.74, -12.60] |
| never_validate | +0.00 [+0.00, +0.00] | +64.75 [+60.00, +69.50] | -1.05 [-1.08, -1.03] | -10.47 [-10.89, -10.05] | -36.72 [-39.05, -34.53] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe | combined_probe | random_matched | never_validate |
|---|---|---|---|---|---|
| False | 0.0 | 0.004 | 0.004 | 0.0 | 0.0 |
| True | 0.993 | 0.942 | 0.95 | 0.971 | 0.993 |

Recovered by fault family:

| family | always_validate | observables_probe | combined_probe | random_matched | never_validate |
|---|---|---|---|---|---|
| cool_stuck_closed | 0.28 | 0.22 | 0.24 | 0.28 | 0.28 |
| fouling | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| outlet_block | 0.59 | 0.59 | 0.58 | 0.57 | 0.59 |
| pump_degrade | 0.5 | 0.5 | 0.5 | 0.49 | 0.5 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|
| always_validate | 0 | 0 | 138 | 258 | 4 |
| observables_probe | 8 | 4 | 132 | 249 | 7 |
| combined_probe | 5 | 4 | 133 | 249 | 9 |
| random_matched | 159 | 27 | 135 | 67 | 12 |
| never_validate | 218 | 41 | 138 | 0 | 3 |