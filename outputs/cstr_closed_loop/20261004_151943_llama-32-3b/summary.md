# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261004_151943_llama-32-3b`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 53.8% | 0.0% | 46.2% | 1.80 | 2.91 | 0.00 (0.00) | 181.6 (166.8 + 0.0 + 0.0 + 14.8) |
| observables_probe | 35.2% | 0.0% | 64.8% | 0.68 | 3.76 | 2.54 (0.27) | 253.9 (248.5 + 0.0 + 0.0 + 5.4) |
| combined_probe | 44.5% | 0.0% | 55.5% | 0.99 | 3.37 | 1.63 (0.10) | 252.7 (230.8 + 6.8 + 7.5 + 7.6) |
| internals_probe | 49.8% | 0.0% | 50.2% | 1.27 | 3.27 | 1.11 (0.06) | 233.4 (210.1 + 6.0 + 7.1 + 10.2) |
| random_matched | 29.5% | 0.0% | 70.5% | 0.58 | 4.00 | 2.76 (0.78) | 305.3 (300.5 + 0.0 + 0.0 + 4.8) |
| never_validate | 15.2% | 78.0% | 6.8% | 0.00 | 1.00 | 0.00 (0.00) | 56.5 (56.5 + 0.0 + 0.0 + 0.0) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe | -18.50 [-24.00, -13.25] | +0.00 [+0.00, +0.00] | -1.12 [-1.24, -1.01] | -9.40 [-10.51, -8.30] | +72.37 [+51.62, +93.90] |
| combined_probe | -9.25 [-14.00, -4.25] | +0.00 [+0.00, +0.00] | -0.81 [-0.93, -0.70] | -7.17 [-8.25, -6.09] | +71.14 [+49.88, +92.25] |
| internals_probe | -4.00 [-8.25, +0.25] | +0.00 [+0.00, +0.00] | -0.53 [-0.63, -0.44] | -4.62 [-5.61, -3.78] | +51.81 [+31.11, +72.90] |
| random_matched | -24.25 [-29.50, -18.75] | +0.00 [+0.00, +0.00] | -1.22 [-1.33, -1.10] | -9.99 [-11.06, -8.87] | +123.72 [+100.24, +147.76] |
| never_validate | -38.50 [-43.25, -33.75] | +78.00 [+74.00, +82.00] | -1.80 [-1.90, -1.70] | -14.80 [-15.89, -13.71] | -125.03 [-140.27, -110.95] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe | combined_probe | internals_probe | random_matched | never_validate |
|---|---|---|---|---|---|---|
| False | 0.41 | 0.257 | 0.349 | 0.402 | 0.215 | 0.111 |
| True | 0.777 | 0.532 | 0.626 | 0.676 | 0.446 | 0.23 |

Recovered by fault family:

| family | always_validate | observables_probe | combined_probe | internals_probe | random_matched | never_validate |
|---|---|---|---|---|---|---|
| cool_stuck_closed | 0.45 | 0.43 | 0.42 | 0.44 | 0.37 | 0.3 |
| fouling | 0.38 | 0.19 | 0.33 | 0.39 | 0.16 | 0.11 |
| outlet_block | 0.65 | 0.38 | 0.5 | 0.59 | 0.3 | 0.05 |
| pump_degrade | 0.67 | 0.41 | 0.53 | 0.57 | 0.35 | 0.15 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | executed_failure:unsafe_fraction | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|---|
| always_validate | 0 | 0 | 0 | 215 | 94 | 91 |
| observables_probe | 0 | 0 | 0 | 141 | 169 | 90 |
| combined_probe | 0 | 0 | 0 | 178 | 117 | 105 |
| internals_probe | 0 | 0 | 0 | 199 | 110 | 91 |
| random_matched | 0 | 0 | 0 | 118 | 145 | 137 |
| never_validate | 287 | 3 | 22 | 61 | 0 | 27 |