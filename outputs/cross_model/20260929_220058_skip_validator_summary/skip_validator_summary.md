# Validator calls skipped per signal

Rules frozen on train/dev per model (scripts/39); ACCEPT if dev failure rate among accepted <= X, REJECT if dev rejected set >= 95% failures. Cells: % of calls skipped on cert (failures executed unchecked / accepted; * = certified at X, Clopper-Pearson, Bonferroni over signals).

## Tolerance X = 10% (cert)

| signal | FSM Qwen2.5-3B | FSM Llama-3.2-3B | FSM Qwen2.5-1.5B | FSM SmolLM2-1.7B | CSTR Qwen2.5-3B |
|---|---|---|---|---|---|
| observables (context + action) | 23% (42/417) | 43% (39/353) | 42% (23/322) | 85% (28/204) | 56% (11/76) |
| all internals | 32% (91/754) | 38% (44/434) | 30% (21/309) | 75% (3/27) | 15% (1/2) |
| all internals + grounding | 31% (87/776) | 41% (40/507) | 41% (18/321)* | 80% (4/72) | 12% (0/0) |
| attention by prompt region | 25% (35/445) | 27% (35/304) | 38% (20/313) | 73% (0/0) | 15% (0/0) |
| context + path + all internals + grounding | 41% (93/814) | 66% (67/763) | 44% (19/381)* | 95% (28/273) |  |
| hidden states | 25% (57/574) | 34% (45/411) | 34% (32/350) | 74% (3/26) | 13% (0/0) |
| plant readings only |  |  |  |  | 4% (4/10) |
| readings + change + all internals + grounding |  |  |  |  | 54% (10/65) |
| region-based attention grounding | 19% (54/520) | 37% (28/277) | 36% (19/332)* | 81% (4/63) | 10% (0/0) |
| token confidence | 19% (37/372) | 19% (46/279) | 17% (6/79) | 57% (1/18) | 0% (0/0) |

Change in calls skipped vs observables, test_iid, X = 10% (points, bootstrap 95% CI):

| signal | FSM Qwen2.5-3B | FSM Llama-3.2-3B | FSM Qwen2.5-1.5B | FSM SmolLM2-1.7B | CSTR Qwen2.5-3B |
|---|---|---|---|---|---|
| all internals | +7.8 [+3.6, +12.0] | -6.8 [-11.2, -2.2] | -12.9 [-16.5, -9.2] | -8.7 [-12.0, -5.5] | -41.8 [-46.2, -37.1] |
| all internals + grounding | +7.6 [+3.2, +11.8] | +0.0 [-5.0, +5.0] | -1.6 [-6.0, +2.8] | -4.1 [-7.5, -0.6] | -44.0 [-48.9, -39.4] |
| attention by prompt region | +4.0 [+0.4, +7.6] | -15.6 [-19.8, -11.4] | -1.6 [-4.8, +1.8] | -12.2 [-15.8, -8.7] | -41.6 [-46.5, -36.7] |
| context + path + all internals + grounding | +15.4 [+11.4, +19.4] | +22.4 [+17.6, +27.2] | +0.8 [-3.4, +4.8] | +8.9 [+5.9, +12.0] |  |
| hidden states | +1.6 [-2.0, +5.0] | -10.4 [-14.4, -6.2] | -8.8 [-11.8, -5.8] | -9.9 [-13.6, -6.7] | -41.8 [-46.5, -36.9] |
| plant readings only |  |  |  |  | -53.5 [-58.0, -48.9] |
| readings + change + all internals + grounding |  |  |  |  | -0.6 [-3.9, +3.0] |
| region-based attention grounding | -4.6 [-8.8, -0.6] | -4.0 [-9.2, +1.2] | -5.8 [-11.0, -0.4] | -2.8 [-6.1, +0.6] | -44.8 [-49.7, -40.2] |
| token confidence | -0.6 [-4.6, +3.4] | -24.6 [-29.2, -19.8] | -25.9 [-29.9, -21.9] | -28.6 [-33.1, -24.1] | -56.2 [-60.4, -51.7] |

## Tolerance X = 5% (cert)

| signal | FSM Qwen2.5-3B | FSM Llama-3.2-3B | FSM Qwen2.5-1.5B | FSM SmolLM2-1.7B | CSTR Qwen2.5-3B |
|---|---|---|---|---|---|
| observables (context + action) | 18% (7/289) | 38% (6/205) | 42% (12/311) | 84% (27/193) | 45% (2/18) |
| all internals | 23% (42/489) | 32% (17/250) | 26% (4/184) | 75% (3/22) | 15% (1/2) |
| all internals + grounding | 21% (24/466) | 36% (15/364) | 38% (4/240) | 78% (0/16) | 12% (0/0) |
| attention by prompt region | 21% (16/327) | 19% (4/70) | 34% (4/192) | 73% (0/0) | 15% (0/0) |
| context + path + all internals + grounding | 31% (24/510) | 56% (20/480) | 40% (4/260)* | 92% (8/186) |  |
| hidden states | 21% (34/439) | 26% (15/194) | 31% (8/260) | 74% (3/26) | 13% (0/0) |
| plant readings only |  |  |  |  | 3% (1/4) |
| readings + change + all internals + grounding |  |  |  |  | 50% (5/43) |
| region-based attention grounding | 10% (12/247) | 33% (8/152) | 32% (4/203) | 80% (2/34) | 10% (0/0) |
| token confidence | 17% (17/307) | 16% (26/204) | 14% (1/15) | 57% (1/18) | 0% (0/0) |

Change in calls skipped vs observables, test_iid, X = 5% (points, bootstrap 95% CI):

| signal | FSM Qwen2.5-3B | FSM Llama-3.2-3B | FSM Qwen2.5-1.5B | FSM SmolLM2-1.7B | CSTR Qwen2.5-3B |
|---|---|---|---|---|---|
| all internals | +4.2 [+0.6, +7.8] | -8.2 [-12.8, -3.8] | -15.5 [-19.1, -11.8] | -8.7 [-12.0, -5.5] | -33.1 [-37.5, -28.6] |
| all internals + grounding | +4.2 [+0.6, +7.8] | +0.6 [-4.2, +5.6] | -4.4 [-8.6, +0.0] | -6.3 [-9.7, -2.6] | -35.3 [-39.8, -30.8] |
| attention by prompt region | +2.8 [+0.0, +5.6] | -18.2 [-22.4, -13.8] | -4.6 [-8.0, -1.0] | -12.2 [-15.8, -8.7] | -32.9 [-37.3, -28.2] |
| context + path + all internals + grounding | +12.6 [+9.2, +16.4] | +22.0 [+17.0, +26.8] | -3.4 [-7.2, +0.6] | +5.7 [+2.6, +8.7] |  |
| hidden states | +2.6 [-0.8, +6.0] | -13.4 [-17.4, -9.6] | -10.6 [-13.7, -7.4] | -9.9 [-13.6, -6.7] | -33.1 [-37.5, -28.4] |
| plant readings only |  |  |  |  | -45.4 [-49.7, -41.0] |
| readings + change + all internals + grounding |  |  |  |  | +4.5 [+1.0, +8.1] |
| region-based attention grounding | -9.4 [-13.2, -5.6] | -2.4 [-7.4, +2.6] | -9.6 [-14.7, -4.8] | -4.7 [-8.1, -1.2] | -36.1 [-40.6, -31.4] |
| token confidence | +1.0 [-2.6, +4.8] | -22.2 [-26.8, -17.8] | -27.1 [-31.1, -23.3] | -28.6 [-33.1, -24.1] | -47.5 [-51.5, -43.0] |
