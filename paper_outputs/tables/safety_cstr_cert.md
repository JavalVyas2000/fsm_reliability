# CSTR (cert): what happens to the skipped validator calls

Per proposal set of n: calls avoided = accepted + rejected without validation.
Accepted unchecked: correct actions let through / **failing actions let through (unsafe)**.
Rejected unchecked: failing actions correctly stopped / good actions wrongly rejected (of all good actions).

## X = 10%

### Qwen2.5-1.5B (n = 492, 326 failing, 166 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 309 (63%) | 107 | **14** | 182 | 6 / 166 |  |
| Plant readings only | 292 (59%) | 99 | **14** | 174 | 5 / 166 |  |
| Obs. + grounding | 295 (60%) | 95 | **7** | 188 | 5 / 166 |  |
| Obs. + all internals | 302 (61%) | 97 | **13** | 185 | 7 / 166 |  |
| Obs. + internals + grounding | 298 (61%) | 97 | **14** | 181 | 6 / 166 |  |
| Region grounding | 135 (27%) | 0 | **0** | 132 | 3 / 166 |  |
| Attention (regions) | 148 (30%) | 0 | **0** | 141 | 7 / 166 |  |
| Hidden states | 142 (29%) | 0 | **0** | 138 | 4 / 166 |  |
| All internals | 145 (29%) | 0 | **0** | 137 | 8 / 166 |  |
| All internals + grounding | 139 (28%) | 0 | **0** | 135 | 4 / 166 |  |
| Token confidence | 18 (4%) | 0 | **0** | 15 | 3 / 166 |  |

### Qwen2.5-3B* (n = 498, 334 failing, 164 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 205 (41%) | 0 | **0** | 189 | 16 / 164 |  |
| Plant readings only | 9 (2%) | 0 | **0** | 9 | 0 / 164 |  |
| Obs. + grounding | 193 (39%) | 0 | **0** | 178 | 15 / 164 |  |
| Obs. + all internals | 252 (51%) | 34 | **4** | 196 | 18 / 164 |  |
| Obs. + internals + grounding | 248 (50%) | 38 | **5** | 190 | 15 / 164 |  |
| Region grounding | 50 (10%) | 0 | **0** | 43 | 7 / 164 |  |
| Attention (regions) | 77 (15%) | 0 | **0** | 70 | 7 / 164 |  |
| Hidden states | 67 (13%) | 0 | **0** | 64 | 3 / 164 |  |
| All internals | 73 (15%) | 0 | **0** | 70 | 3 / 164 |  |
| All internals + grounding | 61 (12%) | 0 | **0** | 58 | 3 / 164 |  |
| Token confidence | 0 (0%) | 0 | **0** | 0 | 0 / 164 |  |

### Qwen2.5-7B (4-bit) (n = 498, 400 failing, 98 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 426 (86%) | 52 | **2** | 358 | 14 / 98 |  |
| Plant readings only | 396 (80%) | 41 | **3** | 337 | 15 / 98 |  |
| Obs. + grounding | 427 (86%) | 53 | **4** | 355 | 15 / 98 |  |
| Obs. + all internals | 412 (83%) | 43 | **3** | 346 | 20 / 98 |  |
| Obs. + internals + grounding | 398 (80%) | 33 | **2** | 344 | 19 / 98 |  |
| Region grounding | 233 (47%) | 0 | **0** | 223 | 10 / 98 |  |
| Attention (regions) | 246 (49%) | 0 | **0** | 235 | 11 / 98 |  |
| Hidden states | 248 (50%) | 0 | **0** | 237 | 11 / 98 |  |
| All internals | 234 (47%) | 0 | **0** | 225 | 9 / 98 |  |
| All internals + grounding | 238 (48%) | 0 | **0** | 229 | 9 / 98 |  |
| Token confidence | 83 (17%) | 0 | **0** | 81 | 2 / 98 |  |

### Llama-3.2-3B (n = 474, 395 failing, 79 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 335 (71%) | 0 | **0** | 319 | 16 / 79 |  |
| Plant readings only | 19 (4%) | 0 | **0** | 18 | 1 / 79 |  |
| Obs. + grounding | 355 (75%) | 0 | **0** | 339 | 16 / 79 |  |
| Obs. + all internals | 339 (72%) | 0 | **0** | 331 | 8 / 79 |  |
| Obs. + internals + grounding | 351 (74%) | 0 | **0** | 341 | 10 / 79 |  |
| Region grounding | 178 (38%) | 0 | **0** | 171 | 7 / 79 |  |
| Attention (regions) | 226 (48%) | 0 | **0** | 219 | 7 / 79 |  |
| Hidden states | 227 (48%) | 0 | **0** | 221 | 6 / 79 |  |
| All internals | 252 (53%) | 0 | **0** | 244 | 8 / 79 |  |
| All internals + grounding | 250 (53%) | 0 | **0** | 244 | 6 / 79 |  |
| Token confidence | 160 (34%) | 0 | **0** | 150 | 10 / 79 |  |

## X = 5%

### Qwen2.5-1.5B (n = 492, 326 failing, 166 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 267 (54%) | 76 | **3** | 182 | 6 / 166 |  |
| Plant readings only | 242 (49%) | 63 | **0** | 174 | 5 / 166 |  |
| Obs. + grounding | 261 (53%) | 68 | **0** | 188 | 5 / 166 |  |
| Obs. + all internals | 245 (50%) | 51 | **2** | 185 | 7 / 166 |  |
| Obs. + internals + grounding | 218 (44%) | 31 | **0** | 181 | 6 / 166 |  |
| Region grounding | 135 (27%) | 0 | **0** | 132 | 3 / 166 |  |
| Attention (regions) | 148 (30%) | 0 | **0** | 141 | 7 / 166 |  |
| Hidden states | 142 (29%) | 0 | **0** | 138 | 4 / 166 |  |
| All internals | 145 (29%) | 0 | **0** | 137 | 8 / 166 |  |
| All internals + grounding | 139 (28%) | 0 | **0** | 135 | 4 / 166 |  |
| Token confidence | 18 (4%) | 0 | **0** | 15 | 3 / 166 |  |

### Qwen2.5-3B* (n = 498, 334 failing, 164 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 205 (41%) | 0 | **0** | 189 | 16 / 164 |  |
| Plant readings only | 9 (2%) | 0 | **0** | 9 | 0 / 164 |  |
| Obs. + grounding | 193 (39%) | 0 | **0** | 178 | 15 / 164 |  |
| Obs. + all internals | 214 (43%) | 0 | **0** | 196 | 18 / 164 |  |
| Obs. + internals + grounding | 205 (41%) | 0 | **0** | 190 | 15 / 164 |  |
| Region grounding | 50 (10%) | 0 | **0** | 43 | 7 / 164 |  |
| Attention (regions) | 77 (15%) | 0 | **0** | 70 | 7 / 164 |  |
| Hidden states | 67 (13%) | 0 | **0** | 64 | 3 / 164 |  |
| All internals | 73 (15%) | 0 | **0** | 70 | 3 / 164 |  |
| All internals + grounding | 61 (12%) | 0 | **0** | 58 | 3 / 164 |  |
| Token confidence | 0 (0%) | 0 | **0** | 0 | 0 / 164 |  |

### Qwen2.5-7B (4-bit) (n = 498, 400 failing, 98 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 417 (84%) | 43 | **2** | 358 | 14 / 98 |  |
| Plant readings only | 352 (71%) | 0 | **0** | 337 | 15 / 98 |  |
| Obs. + grounding | 392 (79%) | 21 | **1** | 355 | 15 / 98 |  |
| Obs. + all internals | 405 (81%) | 36 | **3** | 346 | 20 / 98 |  |
| Obs. + internals + grounding | 393 (79%) | 28 | **2** | 344 | 19 / 98 |  |
| Region grounding | 233 (47%) | 0 | **0** | 223 | 10 / 98 |  |
| Attention (regions) | 246 (49%) | 0 | **0** | 235 | 11 / 98 |  |
| Hidden states | 248 (50%) | 0 | **0** | 237 | 11 / 98 |  |
| All internals | 234 (47%) | 0 | **0** | 225 | 9 / 98 |  |
| All internals + grounding | 238 (48%) | 0 | **0** | 229 | 9 / 98 |  |
| Token confidence | 83 (17%) | 0 | **0** | 81 | 2 / 98 |  |

### Llama-3.2-3B (n = 474, 395 failing, 79 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 335 (71%) | 0 | **0** | 319 | 16 / 79 |  |
| Plant readings only | 19 (4%) | 0 | **0** | 18 | 1 / 79 |  |
| Obs. + grounding | 355 (75%) | 0 | **0** | 339 | 16 / 79 |  |
| Obs. + all internals | 339 (72%) | 0 | **0** | 331 | 8 / 79 |  |
| Obs. + internals + grounding | 351 (74%) | 0 | **0** | 341 | 10 / 79 |  |
| Region grounding | 178 (38%) | 0 | **0** | 171 | 7 / 79 |  |
| Attention (regions) | 226 (48%) | 0 | **0** | 219 | 7 / 79 |  |
| Hidden states | 227 (48%) | 0 | **0** | 221 | 6 / 79 |  |
| All internals | 252 (53%) | 0 | **0** | 244 | 8 / 79 |  |
| All internals + grounding | 250 (53%) | 0 | **0** | 244 | 6 / 79 |  |
| Token confidence | 160 (34%) | 0 | **0** | 150 | 10 / 79 |  |
