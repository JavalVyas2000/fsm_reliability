# Closed-loop CSTR summary (400 episodes, run `outputs/cstr_closed_loop/20261029_121653_smollm2-17b_fresh_r0`)

| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |
|---|---|---|---|---|---|---|---|
| always_validate | 32.2% | 0.0% | 67.8% | 0.99 | 4.28 | 0.00 (0.00) | 27.9 (16.8 + 0.0 + 0.0 + 11.1) |
| observables_probe_r0 | 31.0% | 2.0% | 67.0% | 0.39 | 4.10 | 1.98 (0.02) | 28.4 (23.6 + 0.0 + 0.0 + 4.8) |
| combined_probe | 31.0% | 60.8% | 8.2% | 0.34 | 1.68 | 0.46 (0.02) | 16.6 (10.5 + 1.1 + 0.8 + 4.2) |
| combined_probe_r0 | 31.2% | 2.2% | 66.5% | 0.34 | 3.98 | 0.47 (0.02) | 26.5 (19.1 + 1.6 + 1.7 + 4.2) |
| internals_probe | 31.2% | 63.0% | 5.8% | 0.56 | 1.69 | 0.34 (0.03) | 18.0 (9.3 + 0.9 + 0.9 + 7.0) |
| internals_probe_r0 | 31.5% | 0.0% | 68.5% | 0.56 | 4.17 | 0.34 (0.03) | 28.8 (18.6 + 1.4 + 1.8 + 7.0) |

Paired difference to always_validate (bootstrap 95% CI over episodes):

| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |
|---|---|---|---|---|---|
| observables_probe_r0 | -1.25 [-2.50, -0.25] | +2.00 [+0.75, +3.50] | -0.60 [-0.65, -0.55] | -6.30 [-6.88, -5.69] | +0.49 [-0.88, +1.92] |
| combined_probe | -1.25 [-2.50, -0.25] | +60.75 [+55.99, +65.75] | -0.66 [-0.70, -0.61] | -6.91 [-7.51, -6.28] | -11.28 [-12.42, -10.16] |
| combined_probe_r0 | -1.00 [-2.00, -0.25] | +2.25 [+1.00, +4.00] | -0.66 [-0.70, -0.60] | -6.90 [-7.50, -6.27] | -1.39 [-2.20, -0.56] |
| internals_probe | -1.00 [-2.00, -0.25] | +63.00 [+58.25, +68.00] | -0.43 [-0.48, -0.38] | -4.15 [-4.68, -3.60] | -9.86 [-10.91, -8.88] |
| internals_probe_r0 | -0.75 [-1.75, +0.00] | +0.00 [+0.00, +0.00] | -0.43 [-0.48, -0.38] | -4.11 [-4.63, -3.57] | +0.87 [+0.26, +1.46] |

Recovered by no-change stratum:

| nochange_pass | always_validate | observables_probe_r0 | combined_probe | combined_probe_r0 | internals_probe | internals_probe_r0 |
|---|---|---|---|---|---|---|
| False | 0.011 | 0.007 | 0.007 | 0.011 | 0.007 | 0.011 |
| True | 1.0 | 0.968 | 0.968 | 0.968 | 0.976 | 0.976 |

Recovered by fault family:

| family | always_validate | observables_probe_r0 | combined_probe | combined_probe_r0 | internals_probe | internals_probe_r0 |
|---|---|---|---|---|---|---|
| cool_stuck_closed | 0.3 | 0.26 | 0.26 | 0.26 | 0.27 | 0.27 |
| fouling | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| outlet_block | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 |
| pump_degrade | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 |

Outcome counts:

| policy | executed_failure:no_recovery | executed_failure:slow_recovery | executed_failure:unsafe_fraction | recovered | unresolved_fallback | unresolved_format |
|---|---|---|---|---|---|---|
| always_validate | 0 | 0 | 0 | 129 | 261 | 10 |
| observables_probe_r0 | 6 | 2 | 0 | 124 | 244 | 24 |
| combined_probe | 190 | 52 | 1 | 124 | 1 | 32 |
| combined_probe_r0 | 7 | 2 | 0 | 125 | 230 | 36 |
| internals_probe | 198 | 53 | 1 | 125 | 0 | 23 |
| internals_probe_r0 | 0 | 0 | 0 | 126 | 246 | 28 |