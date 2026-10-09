# CSTR (test_iid): what happens to the skipped validator calls

Per proposal set of n: calls avoided = accepted + rejected without validation.
Accepted unchecked: correct actions let through / **failing actions let through (unsafe)**.
Rejected unchecked: failing actions correctly stopped / good actions wrongly rejected (of all good actions).

## X = 10%

### Qwen2.5-1.5B (n = 494, 329 failing, 165 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 315 (64%) | 99 | **9** | 202 | 5 / 165 |  |
| Plant readings only | 312 (63%) | 96 | **10** | 201 | 5 / 165 |  |
| Obs. + grounding | 319 (65%) | 92 | **7** | 213 | 7 / 165 |  |
| Obs. + all internals | 307 (62%) | 94 | **7** | 200 | 6 / 165 |  |
| Obs. + internals + grounding | 296 (60%) | 93 | **9** | 188 | 6 / 165 |  |
| Region grounding | 138 (28%) | 0 | **0** | 132 | 6 / 165 |  |
| Attention (regions) | 143 (29%) | 0 | **0** | 137 | 6 / 165 |  |
| Hidden states | 159 (32%) | 0 | **0** | 147 | 12 / 165 |  |
| All internals | 158 (32%) | 0 | **0** | 147 | 11 / 165 |  |
| All internals + grounding | 151 (31%) | 0 | **0** | 140 | 11 / 165 |  |
| Token confidence | 16 (3%) | 0 | **0** | 15 | 1 / 165 |  |

### Qwen2.5-3B* (n = 493, 350 failing, 143 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 215 (44%) | 0 | **0** | 210 | 5 / 143 |  |
| Plant readings only | 7 (1%) | 0 | **0** | 7 | 0 / 143 |  |
| Obs. + grounding | 198 (40%) | 0 | **0** | 194 | 4 / 143 |  |
| Obs. + all internals | 268 (54%) | 40 | **4** | 215 | 9 / 143 |  |
| Obs. + internals + grounding | 256 (52%) | 45 | **4** | 201 | 6 / 143 |  |
| Region grounding | 56 (11%) | 0 | **0** | 50 | 6 / 143 |  |
| Attention (regions) | 72 (15%) | 0 | **0** | 65 | 7 / 143 |  |
| Hidden states | 70 (14%) | 0 | **0** | 67 | 3 / 143 |  |
| All internals | 69 (14%) | 0 | **0** | 68 | 1 / 143 |  |
| All internals + grounding | 60 (12%) | 0 | **0** | 60 | 0 / 143 |  |
| Token confidence | 0 (0%) | 0 | **0** | 0 | 0 / 143 |  |

### Qwen2.5-7B (4-bit) (n = 499, 395 failing, 104 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 441 (88%) | 58 | **0** | 365 | 18 / 104 | yes |
| Plant readings only | 411 (82%) | 45 | **2** | 344 | 20 / 104 |  |
| Obs. + grounding | 443 (89%) | 63 | **0** | 365 | 15 / 104 | yes |
| Obs. + all internals | 430 (86%) | 50 | **0** | 361 | 19 / 104 |  |
| Obs. + internals + grounding | 417 (84%) | 37 | **0** | 360 | 20 / 104 |  |
| Region grounding | 237 (47%) | 0 | **0** | 226 | 11 / 104 |  |
| Attention (regions) | 245 (49%) | 0 | **0** | 233 | 12 / 104 |  |
| Hidden states | 250 (50%) | 0 | **0** | 238 | 12 / 104 |  |
| All internals | 244 (49%) | 0 | **0** | 232 | 12 / 104 |  |
| All internals + grounding | 247 (49%) | 0 | **0** | 236 | 11 / 104 |  |
| Token confidence | 96 (19%) | 0 | **0** | 92 | 4 / 104 |  |

### Llama-3.2-3B (n = 473, 398 failing, 75 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 359 (76%) | 0 | **0** | 337 | 22 / 75 |  |
| Plant readings only | 24 (5%) | 0 | **0** | 22 | 2 / 75 |  |
| Obs. + grounding | 368 (78%) | 0 | **0** | 350 | 18 / 75 |  |
| Obs. + all internals | 345 (73%) | 0 | **0** | 337 | 8 / 75 |  |
| Obs. + internals + grounding | 355 (75%) | 0 | **0** | 347 | 8 / 75 |  |
| Region grounding | 191 (40%) | 0 | **0** | 184 | 7 / 75 |  |
| Attention (regions) | 251 (53%) | 0 | **0** | 245 | 6 / 75 |  |
| Hidden states | 254 (54%) | 0 | **0** | 250 | 4 / 75 |  |
| All internals | 268 (57%) | 0 | **0** | 263 | 5 / 75 |  |
| All internals + grounding | 268 (57%) | 0 | **0** | 263 | 5 / 75 |  |
| Token confidence | 173 (37%) | 0 | **0** | 164 | 9 / 75 |  |

### SmolLM2-1.7B (n = 499, 333 failing, 166 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 307 (62%) | 102 | **9** | 192 | 4 / 166 |  |
| Plant readings only | 291 (58%) | 97 | **10** | 181 | 3 / 166 |  |
| Obs. + grounding | 335 (67%) | 105 | **13** | 210 | 7 / 166 |  |
| Obs. + all internals | 333 (67%) | 90 | **8** | 224 | 11 / 166 |  |
| Obs. + internals + grounding | 329 (66%) | 91 | **8** | 221 | 9 / 166 |  |
| Region grounding | 132 (26%) | 0 | **0** | 127 | 5 / 166 |  |
| Attention (regions) | 164 (33%) | 0 | **0** | 152 | 12 / 166 |  |
| Hidden states | 233 (47%) | 66 | **2** | 160 | 5 / 166 |  |
| All internals | 208 (42%) | 39 | **2** | 164 | 3 / 166 |  |
| All internals + grounding | 215 (43%) | 34 | **2** | 173 | 6 / 166 |  |
| Token confidence | 117 (23%) | 0 | **0** | 110 | 7 / 166 |  |

## X = 5%

### Qwen2.5-1.5B (n = 494, 329 failing, 165 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 288 (58%) | 80 | **1** | 202 | 5 / 165 |  |
| Plant readings only | 273 (55%) | 67 | **0** | 201 | 5 / 165 |  |
| Obs. + grounding | 291 (59%) | 70 | **1** | 213 | 7 / 165 |  |
| Obs. + all internals | 273 (55%) | 65 | **2** | 200 | 6 / 165 |  |
| Obs. + internals + grounding | 230 (47%) | 36 | **0** | 188 | 6 / 165 |  |
| Region grounding | 138 (28%) | 0 | **0** | 132 | 6 / 165 |  |
| Attention (regions) | 143 (29%) | 0 | **0** | 137 | 6 / 165 |  |
| Hidden states | 159 (32%) | 0 | **0** | 147 | 12 / 165 |  |
| All internals | 158 (32%) | 0 | **0** | 147 | 11 / 165 |  |
| All internals + grounding | 151 (31%) | 0 | **0** | 140 | 11 / 165 |  |
| Token confidence | 16 (3%) | 0 | **0** | 15 | 1 / 165 |  |

### Qwen2.5-3B* (n = 493, 350 failing, 143 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 215 (44%) | 0 | **0** | 210 | 5 / 143 |  |
| Plant readings only | 7 (1%) | 0 | **0** | 7 | 0 / 143 |  |
| Obs. + grounding | 198 (40%) | 0 | **0** | 194 | 4 / 143 |  |
| Obs. + all internals | 224 (45%) | 0 | **0** | 215 | 9 / 143 |  |
| Obs. + internals + grounding | 207 (42%) | 0 | **0** | 201 | 6 / 143 |  |
| Region grounding | 56 (11%) | 0 | **0** | 50 | 6 / 143 |  |
| Attention (regions) | 72 (15%) | 0 | **0** | 65 | 7 / 143 |  |
| Hidden states | 70 (14%) | 0 | **0** | 67 | 3 / 143 |  |
| All internals | 69 (14%) | 0 | **0** | 68 | 1 / 143 |  |
| All internals + grounding | 60 (12%) | 0 | **0** | 60 | 0 / 143 |  |
| Token confidence | 0 (0%) | 0 | **0** | 0 | 0 / 143 |  |

### Qwen2.5-7B (4-bit) (n = 499, 395 failing, 104 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 430 (86%) | 47 | **0** | 365 | 18 / 104 |  |
| Plant readings only | 364 (73%) | 0 | **0** | 344 | 20 / 104 |  |
| Obs. + grounding | 407 (82%) | 27 | **0** | 365 | 15 / 104 |  |
| Obs. + all internals | 421 (84%) | 41 | **0** | 361 | 19 / 104 |  |
| Obs. + internals + grounding | 413 (83%) | 33 | **0** | 360 | 20 / 104 |  |
| Region grounding | 237 (47%) | 0 | **0** | 226 | 11 / 104 |  |
| Attention (regions) | 245 (49%) | 0 | **0** | 233 | 12 / 104 |  |
| Hidden states | 250 (50%) | 0 | **0** | 238 | 12 / 104 |  |
| All internals | 244 (49%) | 0 | **0** | 232 | 12 / 104 |  |
| All internals + grounding | 247 (49%) | 0 | **0** | 236 | 11 / 104 |  |
| Token confidence | 96 (19%) | 0 | **0** | 92 | 4 / 104 |  |

### Llama-3.2-3B (n = 473, 398 failing, 75 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 359 (76%) | 0 | **0** | 337 | 22 / 75 |  |
| Plant readings only | 24 (5%) | 0 | **0** | 22 | 2 / 75 |  |
| Obs. + grounding | 368 (78%) | 0 | **0** | 350 | 18 / 75 |  |
| Obs. + all internals | 345 (73%) | 0 | **0** | 337 | 8 / 75 |  |
| Obs. + internals + grounding | 355 (75%) | 0 | **0** | 347 | 8 / 75 |  |
| Region grounding | 191 (40%) | 0 | **0** | 184 | 7 / 75 |  |
| Attention (regions) | 251 (53%) | 0 | **0** | 245 | 6 / 75 |  |
| Hidden states | 254 (54%) | 0 | **0** | 250 | 4 / 75 |  |
| All internals | 268 (57%) | 0 | **0** | 263 | 5 / 75 |  |
| All internals + grounding | 268 (57%) | 0 | **0** | 263 | 5 / 75 |  |
| Token confidence | 173 (37%) | 0 | **0** | 164 | 9 / 75 |  |

### SmolLM2-1.7B (n = 499, 333 failing, 166 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 272 (55%) | 76 | **0** | 192 | 4 / 166 |  |
| Plant readings only | 253 (51%) | 69 | **0** | 181 | 3 / 166 |  |
| Obs. + grounding | 295 (59%) | 77 | **1** | 210 | 7 / 166 |  |
| Obs. + all internals | 305 (61%) | 69 | **1** | 224 | 11 / 166 |  |
| Obs. + internals + grounding | 294 (59%) | 63 | **1** | 221 | 9 / 166 |  |
| Region grounding | 132 (26%) | 0 | **0** | 127 | 5 / 166 |  |
| Attention (regions) | 164 (33%) | 0 | **0** | 152 | 12 / 166 |  |
| Hidden states | 165 (33%) | 0 | **0** | 160 | 5 / 166 |  |
| All internals | 192 (38%) | 24 | **1** | 164 | 3 / 166 |  |
| All internals + grounding | 207 (41%) | 27 | **1** | 173 | 6 / 166 |  |
| Token confidence | 117 (23%) | 0 | **0** | 110 | 7 / 166 |  |
