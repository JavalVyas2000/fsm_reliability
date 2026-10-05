# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261004_225258_qwen25-7b`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 34.0% | 0.0% | 66.0% | 2.50 | 4.48 | 0.00 (0.00) | 71.1 (51.9 + 0.0 + 0.0 + 19.3) |
| observables_probe | 25.0% | 2.0% | 73.0% | 0.24 | 4.78 | 4.17 (0.14) | 59.5 (57.3 + 0.0 + 0.0 + 2.2) |
| combined_probe | 26.5% | 41.2% | 32.2% | 0.62 | 3.37 | 1.89 (0.03) | 49.5 (40.5 + 1.1 + 3.4 + 4.5) |
| internals_probe | 30.0% | 0.0% | 70.0% | 1.41 | 4.67 | 2.48 (0.14) | 73.2 (55.4 + 1.5 + 4.6 + 11.8) |
| random_matched | 19.2% | 26.8% | 54.0% | 0.55 | 4.64 | 3.45 (0.58) | 59.1 (55.1 + 0.0 + 0.0 + 4.0) |
| never_validate | 19.2% | 79.5% | 1.2% | 0.00 | 1.00 | 0.00 (0.00) | 11.9 (11.9 + 0.0 + 0.0 + 0.0) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe | -9.00 [-12.75, -5.50] | +2.00 [+0.75, +3.50] | -2.25 [-2.41, -2.10] | -17.08 [-18.45, -15.67] | -11.61 [-14.03, -9.06] |
| combined_probe | -7.50 [-11.00, -4.49] | +41.25 [+36.50, +45.76] | -1.88 [-2.04, -1.72] | -14.72 [-16.17, -13.23] | -21.65 [-24.72, -18.52] |
| internals_probe | -4.00 [-6.25, -1.75] | +0.00 [+0.00, +0.00] | -1.09 [-1.25, -0.93] | -7.51 [-8.74, -6.36] | +2.07 [+0.55, +3.60] |
| random_matched | -14.75 [-18.50, -11.00] | +26.75 [+22.25, +31.00] | -1.95 [-2.10, -1.78] | -15.28 [-16.63, -13.85] | -12.07 [-15.37, -8.77] |
| never_validate | -14.75 [-18.25, -11.25] | +79.50 [+75.50, +83.25] | -2.50 [-2.65, -2.34] | -19.27 [-20.72, -17.82] | -59.25 [-62.48, -55.89] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe | combined_probe | internals_probe | random_matched | never_validate |
|---|---|---|---|---|---|---|
| False | 0.134 | 0.061 | 0.073 | 0.103 | 0.031 | 0.008 |
| True | 0.727 | 0.604 | 0.626 | 0.669 | 0.496 | 0.54 |

Recovered by fault family:

| family | always_validate | observables_probe | combined_probe | internals_probe | random_matched | never_validate |
|---|---|---|---|---|---|---|
| cool_stuck_closed | 0.14 | 0.04 | 0.12 | 0.06 | 0.1 | 0.1 |
| fouling | 0.06 | 0.01 | 0.02 | 0.01 | 0.02 | 0.0 |
| outlet_block | 0.6 | 0.47 | 0.47 | 0.59 | 0.34 | 0.33 |
| pump_degrade | 0.56 | 0.48 | 0.45 | 0.54 | 0.31 | 0.34 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | executed_failure:unsafe_fraction | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|---|
| always_validate | 0 | 0 | 0 | 136 | 259 | 5 |
| observables_probe | 1 | 5 | 2 | 100 | 286 | 6 |
| combined_probe | 119 | 40 | 6 | 106 | 124 | 5 |
| internals_probe | 0 | 0 | 0 | 120 | 275 | 5 |
| random_matched | 93 | 12 | 2 | 77 | 211 | 5 |
| never_validate | 278 | 37 | 3 | 77 | 0 | 5 |