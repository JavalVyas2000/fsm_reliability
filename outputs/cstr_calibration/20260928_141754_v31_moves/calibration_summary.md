# CSTR severity calibration against the model's empirical v3.1 moves (no LLM)

1 random plants per severity; 2 empirical moves per episode (from outputs/cstr_collect/20260928_120902_qwen25-3b-instruct_v31_pilot40); plant and onset ranges from configs/cstr_severity_ranges_v3.json.

| family | severity | usable/n | no-change passes | v3.1-like move passes | ... when no-change fails | feed x0.65 passes | feed x0.5 passes |
|---|---|---|---|---|---|---|---|
| cool_stuck_closed | {"stuck_opening": 0.35} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.3} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.45} | 0/1 | – | – | – | – | – |
| cool_stuck_closed | {"stuck_opening": 0.4} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.55} | 0/1 | – | – | – | – | – |
| cool_stuck_closed | {"stuck_opening": 0.5} | 0/1 | – | – | – | – | – |
| cool_stuck_closed | {"stuck_opening": 0.6} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.4, "fouling_tau": 2000.0} | 0/1 | – | – | – | – | – |
| fouling | {"fouling_max": 0.45, "fouling_tau": 2000.0} | 0/1 | – | – | – | – | – |
| fouling | {"fouling_max": 0.5, "fouling_tau": 2000.0} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.55, "fouling_tau": 2000.0} | 1/1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fouling | {"fouling_max": 0.6, "fouling_tau": 2000.0} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.65, "fouling_tau": 2000.0} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| fouling | {"fouling_max": 0.7, "fouling_tau": 2000.0} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.35} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.45} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.4} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.55} | 1/1 | 1.00 | 1.00 | – | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.5} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.65} | 1/1 | 1.00 | 1.00 | – | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.6} | 0/1 | – | – | – | – | – |
| pump_degrade | {"pump_degrade_factor": 0.35} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.45} | 1/1 | 1.00 | 1.00 | – | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.4} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.55} | 1/1 | 1.00 | 1.00 | – | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.5} | 1/1 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| pump_degrade | {"pump_degrade_factor": 0.65} | 0/1 | – | – | – | – | – |
| pump_degrade | {"pump_degrade_factor": 0.6} | 1/1 | 1.00 | 1.00 | – | 1.00 | 1.00 |