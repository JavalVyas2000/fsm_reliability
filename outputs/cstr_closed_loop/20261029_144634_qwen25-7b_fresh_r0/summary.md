# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261029_144634_qwen25-7b_fresh_r0`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 37.2% | 0.0% | 62.7% | 2.48 | 4.36 | 0.00 (0.00) | 78.7 (52.5 + 0.0 + 0.0 + 26.2) |
| observables_probe_r0 | 24.2% | 0.0% | 75.8% | 0.28 | 4.90 | 4.39 (0.13) | 63.2 (59.9 + 0.0 + 0.0 + 3.3) |
| combined_probe | 27.5% | 38.0% | 34.5% | 0.65 | 3.53 | 1.95 (0.04) | 55.4 (43.2 + 1.2 + 3.8 + 7.2) |
| combined_probe_r0 | 33.8% | 0.0% | 66.2% | 1.42 | 4.63 | 1.99 (0.04) | 79.5 (56.8 + 1.6 + 5.0 + 16.1) |
| internals_probe | 33.8% | 0.0% | 66.2% | 1.34 | 4.54 | 2.51 (0.09) | 77.2 (55.6 + 1.6 + 4.8 + 15.3) |
| internals_probe_r0 | 33.8% | 0.0% | 66.2% | 1.34 | 4.54 | 2.51 (0.09) | 77.2 (55.6 + 1.6 + 4.8 + 15.3) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe_r0 | -13.00 [-16.75, -9.25] | +0.00 [+0.00, +0.00] | -2.19 [-2.35, -2.03] | -22.83 [-24.76, -20.93] | -15.44 [-18.10, -12.77] |
| combined_probe | -9.75 [-13.00, -6.75] | +38.00 [+33.00, +42.76] | -1.82 [-1.98, -1.67] | -18.93 [-20.77, -17.17] | -23.23 [-26.58, -19.99] |
| combined_probe_r0 | -3.50 [-6.25, -1.00] | +0.00 [+0.00, +0.00] | -1.05 [-1.21, -0.90] | -10.02 [-11.62, -8.43] | +0.79 [-1.37, +2.91] |
| internals_probe | -3.50 [-5.75, -1.50] | +0.00 [+0.00, +0.00] | -1.13 [-1.28, -0.97] | -10.89 [-12.54, -9.25] | -1.51 [-3.42, +0.46] |
| internals_probe_r0 | -3.50 [-5.75, -1.50] | +0.00 [+0.00, +0.00] | -1.13 [-1.28, -0.97] | -10.89 [-12.54, -9.25] | -1.51 [-3.42, +0.46] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe_r0 | combined_probe | combined_probe_r0 | internals_probe | internals_probe_r0 |
|---|---|---|---|---|---|---|
| False | 0.186 | 0.058 | 0.084 | 0.142 | 0.15 | 0.15 |
| True | 0.778 | 0.643 | 0.69 | 0.762 | 0.746 | 0.746 |

Recovered by fault family:

| family | always_validate | observables_probe_r0 | combined_probe | combined_probe_r0 | internals_probe | internals_probe_r0 |
|---|---|---|---|---|---|---|
| cool_stuck_closed | 0.1 | 0.02 | 0.08 | 0.08 | 0.05 | 0.05 |
| fouling | 0.1 | 0.0 | 0.01 | 0.02 | 0.0 | 0.0 |
| outlet_block | 0.68 | 0.46 | 0.52 | 0.65 | 0.69 | 0.69 |
| pump_degrade | 0.61 | 0.49 | 0.49 | 0.6 | 0.61 | 0.61 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | executed_failure:unsafe_fraction | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|---|
| always_validate | 0 | 0 | 0 | 149 | 249 | 2 |
| observables_probe_r0 | 0 | 0 | 0 | 97 | 302 | 1 |
| combined_probe | 118 | 30 | 4 | 110 | 137 | 1 |
| combined_probe_r0 | 0 | 0 | 0 | 135 | 263 | 2 |
| internals_probe | 0 | 0 | 0 | 135 | 264 | 1 |
| internals_probe_r0 | 0 | 0 | 0 | 135 | 264 | 1 |