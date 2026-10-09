# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261009_160941_smollm2-17b`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 36.5% | 0.0% | 63.5% | 1.01 | 4.11 | 0.00 (0.00) | 27.6 (16.7 + 0.0 + 0.0 + 10.9) |
| observables_probe | 35.2% | 3.2% | 61.5% | 0.39 | 3.87 | 1.86 (0.02) | 29.0 (24.2 + 0.0 + 0.0 + 4.8) |
| combined_probe | 35.0% | 59.2% | 5.8% | 0.34 | 1.65 | 0.45 (0.03) | 16.5 (10.6 + 0.7 + 0.9 + 4.3) |
| internals_probe | 35.5% | 60.8% | 3.8% | 0.56 | 1.65 | 0.33 (0.02) | 17.6 (9.1 + 0.6 + 0.9 + 7.0) |
| random_matched | 34.0% | 43.0% | 23.0% | 0.36 | 2.83 | 1.26 (0.27) | 23.8 (19.9 + 0.0 + 0.0 + 4.0) |
| never_validate | 36.0% | 63.5% | 0.5% | 0.00 | 1.00 | 0.00 (0.00) | 4.5 (4.5 + 0.0 + 0.0 + 0.0) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe | -1.25 [-2.50, -0.25] | +3.25 [+1.75, +5.25] | -0.63 [-0.68, -0.57] | -6.13 [-6.79, -5.52] | +1.39 [-0.45, +3.49] |
| combined_probe | -1.50 [-2.75, -0.50] | +59.25 [+54.25, +64.00] | -0.67 [-0.72, -0.62] | -6.66 [-7.29, -6.04] | -11.15 [-12.41, -9.91] |
| internals_probe | -1.00 [-2.00, -0.25] | +60.75 [+55.75, +65.25] | -0.45 [-0.51, -0.40] | -3.93 [-4.49, -3.39] | -10.00 [-11.13, -8.97] |
| random_matched | -2.50 [-4.00, -1.00] | +43.00 [+38.24, +47.75] | -0.66 [-0.70, -0.60] | -6.95 [-7.57, -6.30] | -3.77 [-5.39, -2.00] |
| never_validate | -0.50 [-1.25, +0.00] | +63.50 [+58.75, +68.00] | -1.01 [-1.03, -1.00] | -10.93 [-11.43, -10.46] | -23.09 [-24.38, -21.85] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe | combined_probe | internals_probe | random_matched | never_validate |
|---|---|---|---|---|---|---|
| False | 0.027 | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 |
| True | 1.0 | 0.978 | 0.971 | 0.986 | 0.942 | 1.0 |

Recovered by fault family:

| family | always_validate | observables_probe | combined_probe | internals_probe | random_matched | never_validate |
|---|---|---|---|---|---|---|
| cool_stuck_closed | 0.29 | 0.27 | 0.25 | 0.27 | 0.27 | 0.29 |
| fouling | 0.08 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 |
| outlet_block | 0.59 | 0.59 | 0.59 | 0.59 | 0.55 | 0.59 |
| pump_degrade | 0.5 | 0.5 | 0.5 | 0.5 | 0.48 | 0.5 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | executed_failure:unsafe_fraction | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|---|
| always_validate | 0 | 0 | 0 | 146 | 247 | 7 |
| observables_probe | 9 | 4 | 0 | 141 | 226 | 20 |
| combined_probe | 196 | 41 | 0 | 140 | 2 | 21 |
| internals_probe | 202 | 41 | 0 | 142 | 0 | 15 |
| random_matched | 144 | 26 | 2 | 136 | 55 | 37 |
| never_validate | 213 | 41 | 0 | 144 | 0 | 2 |