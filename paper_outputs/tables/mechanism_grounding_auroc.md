# Does low attention on the deciding prompt region predict failure? (train + dev, answer level)

| domain | model | feature | n | AUROC |
|---|---|---|---|---|
| FSM | Qwen2.5-3B | grd_min_share | 2494 | 0.748 |
| FSM | Llama-3.2-3B | grd_min_share | 2499 | 0.836 |
| FSM | Qwen2.5-1.5B | grd_min_share | 2472 | 0.845 |
| FSM | SmolLM2-1.7B | grd_min_share | 2422 | 0.882 |
| FSM | Qwen2.5-7B (4-bit) | grd_min_share | 2491 | 0.723 |
| CSTR | Qwen2.5-1.5B | g31_min_share | 3968 | 0.716 |
| CSTR | Qwen2.5-3B* | g31_min_share | 3954 | 0.496 |
| CSTR | Qwen2.5-7B (4-bit) | g31_min_share | 3988 | 0.479 |
| CSTR | Llama-3.2-3B | g31_min_share | 3817 | 0.522 |