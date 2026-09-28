# CSTR severity calibration (no LLM; legacy verifier, deterministic replay)

Seeds per severity: 3; onset 2000.0 s; 16 grid actions (incl. no-change). Actionable = no-change fails for > 50% of triggered seeds AND some grid action passes for > 50%.

| family | severity | triggered/seeds | no-change fail rate | any grid pass rate | mean pass share of grid | actionable |
|---|---|---|---|---|---|---|
| fouling | {"fouling_max": 0.3, "fouling_tau": 2000.0} | 0/3 | – | – | – | no |
| fouling | {"fouling_max": 0.5, "fouling_tau": 2000.0} | 0/3 | – | – | – | no |
| fouling | {"fouling_max": 0.4, "fouling_tau": 2000.0} | 0/3 | – | – | – | no |
| pump_degrade | {"pump_degrade_factor": 0.65} | 0/3 | – | – | – | no |
| pump_degrade | {"pump_degrade_factor": 0.8} | 0/3 | – | – | – | no |
| fouling | {"fouling_max": 0.6, "fouling_tau": 2000.0} | 3/3 | 1.00 | 1.00 | 0.73 | yes |
| fouling | {"fouling_max": 0.7, "fouling_tau": 2000.0} | 3/3 | 1.00 | 1.00 | 0.67 | yes |
| fouling | {"fouling_max": 0.8, "fouling_tau": 2000.0} | 3/3 | 1.00 | 1.00 | 0.40 | yes |
| fouling | {"fouling_max": 0.9, "fouling_tau": 2000.0} | 3/3 | 1.00 | 1.00 | 0.27 | yes |
| cool_stuck_closed | {"stuck_opening": 0.6} | 0/3 | – | – | – | no |
| cool_stuck_closed | {"stuck_opening": 0.5} | 0/3 | – | – | – | no |
| pump_degrade | {"pump_degrade_factor": 0.5} | 3/3 | 0.00 | 1.00 | 0.93 | no |
| pump_degrade | {"pump_degrade_factor": 0.35} | 3/3 | 1.00 | 1.00 | 0.53 | yes |
| pump_degrade | {"pump_degrade_factor": 0.2} | 3/3 | 1.00 | 1.00 | 0.13 | yes |
| pump_degrade | {"pump_degrade_factor": 0.1} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| outlet_block | {"outlet_block_factor": 0.8} | 0/3 | – | – | – | no |
| cool_stuck_closed | {"stuck_opening": 0.4} | 3/3 | 1.00 | 1.00 | 0.80 | yes |
| cool_stuck_closed | {"stuck_opening": 0.3} | 3/3 | 1.00 | 1.00 | 0.67 | yes |
| cool_stuck_closed | {"stuck_opening": 0.2} | 3/3 | 1.00 | 1.00 | 0.27 | yes |
| cool_stuck_closed | {"stuck_opening": 0.1} | 3/3 | 1.00 | 1.00 | 0.13 | yes |
| cool_stuck_closed | {"stuck_opening": 0.0} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| outlet_block | {"outlet_block_factor": 0.6} | 3/3 | 0.00 | 1.00 | 1.00 | no |
| outlet_block | {"outlet_block_factor": 0.45} | 3/3 | 1.00 | 1.00 | 0.80 | yes |
| outlet_block | {"outlet_block_factor": 0.3} | 3/3 | 1.00 | 1.00 | 0.40 | yes |
| outlet_block | {"outlet_block_factor": 0.2} | 3/3 | 1.00 | 1.00 | 0.13 | yes |
| outlet_block | {"outlet_block_factor": 0.1} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| leak | {"leak_k": 0.001} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| leak | {"leak_k": 0.002} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| leak | {"leak_k": 0.003} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| leak | {"leak_k": 0.005} | 3/3 | 1.00 | 0.00 | 0.00 | no |
| leak | {"leak_k": 0.008} | 3/3 | 1.00 | 0.00 | 0.00 | no |