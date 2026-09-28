# CSTR varied-plant severity calibration (no LLM)

6 random plants per severity (onset [1500.0, 5000.0] s; plant ranges from configs/cstr_severity_ranges_v3_draft.json). Actions are relative to each episode's own setpoints.

| family | severity | triggered/n | pre-fault alarms | no-change fail | x0.9 feed passes | any grid pass | mean pass share |
|---|---|---|---|---|---|---|---|
| fouling | {"fouling_max": 0.5, "fouling_tau": 2000.0} | 2/6 | 0 | 1.00 | 0.50 | 1.00 | 0.75 |
| fouling | {"fouling_max": 0.55, "fouling_tau": 2000.0} | 2/6 | 0 | 1.00 | 0.50 | 1.00 | 0.83 |
| fouling | {"fouling_max": 0.6, "fouling_tau": 2000.0} | 2/6 | 0 | 1.00 | 0.50 | 1.00 | 0.75 |
| fouling | {"fouling_max": 0.65, "fouling_tau": 2000.0} | 5/6 | 0 | 1.00 | 0.00 | 1.00 | 0.60 |
| fouling | {"fouling_max": 0.7, "fouling_tau": 2000.0} | 5/6 | 0 | 1.00 | 0.20 | 1.00 | 0.70 |
| fouling | {"fouling_max": 0.8, "fouling_tau": 2000.0} | 6/6 | 0 | 1.00 | 0.00 | 1.00 | 0.53 |
| pump_degrade | {"pump_degrade_factor": 0.5} | 6/6 | 0 | 0.33 | 1.00 | 1.00 | 0.97 |
| pump_degrade | {"pump_degrade_factor": 0.45} | 6/6 | 0 | 0.67 | 0.50 | 1.00 | 0.83 |
| pump_degrade | {"pump_degrade_factor": 0.4} | 6/6 | 0 | 0.83 | 0.33 | 1.00 | 0.72 |
| pump_degrade | {"pump_degrade_factor": 0.35} | 6/6 | 0 | 1.00 | 0.50 | 1.00 | 0.67 |
| pump_degrade | {"pump_degrade_factor": 0.3} | 6/6 | 0 | 1.00 | 0.00 | 1.00 | 0.44 |
| pump_degrade | {"pump_degrade_factor": 0.25} | 6/6 | 0 | 1.00 | 0.00 | 1.00 | 0.31 |
| cool_stuck_closed | {"stuck_opening": 0.5} | 2/6 | 0 | 0.50 | 1.00 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.45} | 3/6 | 0 | 1.00 | 0.67 | 1.00 | 0.89 |
| cool_stuck_closed | {"stuck_opening": 0.4} | 3/6 | 0 | 0.67 | 1.00 | 1.00 | 1.00 |
| cool_stuck_closed | {"stuck_opening": 0.35} | 5/6 | 0 | 0.80 | 0.20 | 1.00 | 0.63 |
| cool_stuck_closed | {"stuck_opening": 0.3} | 5/6 | 0 | 1.00 | 0.20 | 1.00 | 0.57 |
| cool_stuck_closed | {"stuck_opening": 0.25} | 6/6 | 0 | 1.00 | 0.17 | 1.00 | 0.58 |
| outlet_block | {"outlet_block_factor": 0.6} | 5/6 | 0 | 0.00 | 1.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.55} | 6/6 | 0 | 0.00 | 1.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.5} | 6/6 | 0 | 0.17 | 1.00 | 1.00 | 1.00 |
| outlet_block | {"outlet_block_factor": 0.45} | 6/6 | 0 | 0.33 | 0.83 | 1.00 | 0.92 |
| outlet_block | {"outlet_block_factor": 0.4} | 6/6 | 0 | 0.67 | 0.50 | 1.00 | 0.72 |
| outlet_block | {"outlet_block_factor": 0.35} | 6/6 | 0 | 1.00 | 0.00 | 1.00 | 0.47 |