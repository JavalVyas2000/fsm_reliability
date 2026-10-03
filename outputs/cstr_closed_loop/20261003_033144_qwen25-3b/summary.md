# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261003_033144_qwen25-3b`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 43.8% | 0.0% | 56.2% | 2.45 | 3.95 | 0.00 (0.00) | 75.2 (55.1 + 0.0 + 0.0 + 20.1) |
| observables_probe | 41.0% | 0.0% | 59.0% | 1.30 | 4.08 | 2.27 (0.12) | 73.3 (63.0 + 0.0 + 0.0 + 10.4) |
| combined_probe | 37.0% | 0.2% | 62.7% | 0.69 | 4.22 | 3.36 (0.20) | 77.4 (67.2 + 2.7 + 2.4 + 5.1) |
| random_matched | 37.8% | 27.5% | 34.8% | 1.31 | 3.69 | 1.59 (0.26) | 66.6 (56.8 + 0.0 + 0.0 + 9.8) |
| never_validate | 30.0% | 68.5% | 1.5% | 0.00 | 1.00 | 0.00 (0.00) | 16.5 (16.5 + 0.0 + 0.0 + 0.0) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe | -2.75 [-5.75, +0.25] | +0.00 [+0.00, +0.00] | -1.15 [-1.32, -0.97] | -9.77 [-11.33, -8.21] | -1.86 [-3.97, +0.26] |
| combined_probe | -6.75 [-10.00, -3.50] | +0.25 [+0.00, +0.75] | -1.76 [-1.95, -1.59] | -15.07 [-16.69, -13.47] | +2.24 [-0.34, +4.73] |
| random_matched | -6.00 [-9.00, -3.00] | +27.50 [+23.25, +31.75] | -1.14 [-1.31, -0.97] | -10.37 [-11.90, -8.92] | -8.62 [-12.29, -5.00] |
| never_validate | -13.75 [-17.00, -10.50] | +68.50 [+63.75, +73.00] | -2.45 [-2.61, -2.29] | -20.13 [-21.79, -18.58] | -58.72 [-62.98, -54.60] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe | combined_probe | random_matched | never_validate |
|---|---|---|---|---|---|
| False | 0.226 | 0.215 | 0.165 | 0.176 | 0.138 |
| True | 0.835 | 0.777 | 0.755 | 0.755 | 0.604 |

Recovered by fault family:

| family | always_validate | observables_probe | combined_probe | random_matched | never_validate |
|---|---|---|---|---|---|
| cool_stuck_closed | 0.39 | 0.27 | 0.21 | 0.33 | 0.23 |
| fouling | 0.33 | 0.31 | 0.25 | 0.32 | 0.26 |
| outlet_block | 0.55 | 0.55 | 0.54 | 0.45 | 0.37 |
| pump_degrade | 0.48 | 0.51 | 0.48 | 0.41 | 0.34 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | executed_failure:unsafe_fraction | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|---|
| always_validate | 0 | 0 | 0 | 175 | 219 | 6 |
| observables_probe | 0 | 0 | 0 | 164 | 229 | 7 |
| combined_probe | 1 | 0 | 0 | 148 | 242 | 9 |
| random_matched | 95 | 8 | 7 | 151 | 133 | 6 |
| never_validate | 209 | 32 | 33 | 120 | 0 | 6 |