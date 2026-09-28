# CSTR severity calibration against the model's empirical v3.1 moves (no LLM)

10 random plants per severity; 20 empirical moves per episode (from outputs/cstr_collect/20260928_120902_qwen25-3b-instruct_v31_pilot40); plant and onset ranges from configs/cstr_severity_ranges_v3.json.

| family | severity | usable/n | no-change passes | v3.1-like move passes | ... when no-change fails | feed x0.65 passes | feed x0.5 passes |
|---|---|---|---|---|---|---|---|
| cool_stuck_closed | {"stuck_opening": 0.35} | 9/10 | 0.00 | 0.02 | 0.02 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.3} | 10/10 | 0.00 | 0.01 | 0.01 | 0.70 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.45} | 8/10 | 0.38 | 0.28 | 0.03 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.4} | 9/10 | 0.22 | 0.14 | 0.01 | 0.89 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.55} | 2/10 | 0.00 | 0.08 | 0.08 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.5} | 5/10 | 0.20 | 0.16 | 0.04 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.6} | 2/10 | 0.00 | 0.03 | 0.03 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.4, "fouling_tau": 2000.0} | 0/10 | – | – | – | – | – |
| fouling | {"fouling_max": 0.45, "fouling_tau": 2000.0} | 3/10 | 0.00 | 0.08 | 0.08 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.5, "fouling_tau": 2000.0} | 3/10 | 0.00 | 0.10 | 0.10 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.55, "fouling_tau": 2000.0} | 2/10 | 0.00 | 0.10 | 0.10 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.6, "fouling_tau": 2000.0} | 8/10 | 0.00 | 0.04 | 0.04 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.65, "fouling_tau": 2000.0} | 8/10 | 0.00 | 0.03 | 0.03 | 0.75 | 0.75 |
| fouling | {"fouling_max": 0.7, "fouling_tau": 2000.0} | 7/10 | 0.00 | 0.04 | 0.04 | 0.86 | 0.86 |
| outlet_block | {"outlet_block_factor": 0.35} | 10/10 | 0.00 | 0.01 | 0.01 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.45} | 10/10 | 0.60 | 0.43 | 0.04 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.4} | 10/10 | 0.10 | 0.03 | 0.01 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.55} | 10/10 | 1.00 | 0.73 | – | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.5} | 10/10 | 0.80 | 0.60 | 0.15 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.65} | 2/10 | 1.00 | 0.75 | – | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.6} | 7/10 | 1.00 | 0.75 | – | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.35} | 10/10 | 0.00 | 0.01 | 0.01 | 0.90 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.45} | 10/10 | 0.60 | 0.44 | 0.06 | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.4} | 10/10 | 0.30 | 0.21 | 0.01 | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.55} | 10/10 | 1.00 | 0.72 | – | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.5} | 9/10 | 0.56 | 0.46 | 0.15 | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.65} | 1/10 | 1.00 | 0.75 | – | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.6} | 7/10 | 1.00 | 0.75 | – | 1.00 | 1.00 |