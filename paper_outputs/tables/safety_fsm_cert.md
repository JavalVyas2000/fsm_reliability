# FSM (cert): what happens to the skipped validator calls

Per proposal set of n: calls avoided = accepted + rejected without validation.
Accepted unchecked: correct actions let through / **failing actions let through (unsafe)**.
Rejected unchecked: failing actions correctly stopped / good actions wrongly rejected (of all good actions).

## X = 10%

### Qwen2.5-3B (n = 2992, 1525 failing, 1467 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 614 (21%) | 320 | **15** | 267 | 12 / 1467 | yes |
| Obs. + grounding | 621 (21%) | 328 | **11** | 273 | 9 / 1467 | yes |
| Obs. + all internals | 948 (32%) | 564 | **35** | 332 | 17 / 1467 | yes |
| Obs. + internals + grounding | 1032 (34%) | 576 | **27** | 412 | 17 / 1467 | yes |
| Region grounding | 431 (14%) | 349 | **18** | 53 | 11 / 1467 | yes |
| Attention (regions) | 640 (21%) | 332 | **19** | 267 | 22 / 1467 | yes |
| Hidden states | 714 (24%) | 480 | **26** | 201 | 7 / 1467 | yes |
| All internals | 760 (25%) | 495 | **31** | 223 | 11 / 1467 | yes |
| All internals + grounding | 675 (23%) | 462 | **21** | 188 | 4 / 1467 | yes |
| Token confidence | 536 (18%) | 301 | **28** | 188 | 19 / 1467 |  |

### Llama-3.2-3B (n = 2995, 1875 failing, 1120 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 1289 (43%) | 307 | **26** | 894 | 62 / 1120 |  |
| Obs. + grounding | 1754 (59%) | 424 | **20** | 1233 | 77 / 1120 | yes |
| Obs. + all internals | 1401 (47%) | 301 | **7** | 1015 | 78 / 1120 | yes |
| Obs. + internals + grounding | 1739 (58%) | 503 | **29** | 1139 | 68 / 1120 | yes |
| Region grounding | 939 (31%) | 100 | **2** | 786 | 51 / 1120 | yes |
| Attention (regions) | 495 (17%) | 0 | **0** | 465 | 30 / 1120 |  |
| Hidden states | 794 (27%) | 191 | **13** | 550 | 40 / 1120 |  |
| All internals | 920 (31%) | 232 | **8** | 634 | 46 / 1120 | yes |
| All internals + grounding | 1073 (36%) | 358 | **13** | 671 | 31 / 1120 | yes |
| Token confidence | 426 (14%) | 128 | **12** | 266 | 20 / 1120 |  |

### Qwen2.5-1.5B (n = 2945, 1989 failing, 956 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 1189 (40%) | 277 | **7** | 860 | 45 / 956 | yes |
| Obs. + grounding | 1340 (46%) | 227 | **1** | 1056 | 56 / 956 | yes |
| Obs. + all internals | 1081 (37%) | 256 | **6** | 782 | 37 / 956 | yes |
| Obs. + internals + grounding | 1176 (40%) | 254 | **5** | 877 | 40 / 956 | yes |
| Region grounding | 903 (31%) | 173 | **5** | 696 | 29 / 956 | yes |
| Attention (regions) | 1018 (35%) | 189 | **6** | 781 | 42 / 956 | yes |
| Hidden states | 654 (22%) | 0 | **0** | 625 | 29 / 956 |  |
| All internals | 750 (25%) | 190 | **6** | 537 | 17 / 956 | yes |
| All internals + grounding | 1118 (38%) | 232 | **8** | 833 | 45 / 956 | yes |
| Token confidence | 395 (13%) | 0 | **0** | 376 | 19 / 956 |  |

### SmolLM2-1.7B (n = 2946, 2456 failing, 490 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 2491 (85%) | 151 | **15** | 2191 | 134 / 490 |  |
| Obs. + grounding | 2509 (85%) | 0 | **0** | 2363 | 146 / 490 |  |
| Obs. + all internals | 2327 (79%) | 0 | **0** | 2191 | 136 / 490 |  |
| Obs. + internals + grounding | 2673 (91%) | 139 | **7** | 2362 | 165 / 490 |  |
| Region grounding | 2344 (80%) | 0 | **0** | 2224 | 120 / 490 |  |
| Attention (regions) | 2224 (75%) | 0 | **0** | 2099 | 125 / 490 |  |
| Hidden states | 2191 (74%) | 0 | **0** | 2083 | 108 / 490 |  |
| All internals | 2241 (76%) | 0 | **0** | 2116 | 125 / 490 |  |
| All internals + grounding | 2321 (79%) | 0 | **0** | 2209 | 112 / 490 |  |
| Token confidence | 1644 (56%) | 0 | **0** | 1567 | 77 / 490 |  |

### Qwen2.5-7B (4-bit) (n = 2973, 1450 failing, 1523 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 467 (16%) | 245 | **13** | 192 | 17 / 1523 | yes |
| Obs. + grounding | 660 (22%) | 472 | **41** | 143 | 4 / 1523 |  |
| Obs. + all internals | 458 (15%) | 343 | **25** | 86 | 4 / 1523 |  |
| Obs. + internals + grounding | 452 (15%) | 358 | **17** | 75 | 2 / 1523 | yes |
| Region grounding | 94 (3%) | 88 | **0** | 5 | 1 / 1523 | yes |
| Attention (regions) | 394 (13%) | 295 | **24** | 73 | 2 / 1523 |  |
| Hidden states | 286 (10%) | 219 | **15** | 50 | 2 / 1523 |  |
| All internals | 266 (9%) | 240 | **12** | 13 | 1 / 1523 | yes |
| All internals + grounding | 321 (11%) | 291 | **12** | 18 | 0 / 1523 | yes |
| Token confidence | 0 (0%) | 0 | **0** | 0 | 0 / 1523 |  |

## X = 5%

### Qwen2.5-3B (n = 2992, 1525 failing, 1467 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 578 (19%) | 296 | **3** | 267 | 12 / 1467 | yes |
| Obs. + grounding | 282 (9%) | 0 | **0** | 273 | 9 / 1467 |  |
| Obs. + all internals | 623 (21%) | 270 | **4** | 332 | 17 / 1467 | yes |
| Obs. + internals + grounding | 636 (21%) | 206 | **1** | 412 | 17 / 1467 | yes |
| Region grounding | 64 (2%) | 0 | **0** | 53 | 11 / 1467 |  |
| Attention (regions) | 462 (15%) | 170 | **3** | 267 | 22 / 1467 |  |
| Hidden states | 536 (18%) | 318 | **10** | 201 | 7 / 1467 |  |
| All internals | 423 (14%) | 186 | **3** | 223 | 11 / 1467 |  |
| All internals + grounding | 192 (6%) | 0 | **0** | 188 | 4 / 1467 |  |
| Token confidence | 207 (7%) | 0 | **0** | 188 | 19 / 1467 |  |

### Llama-3.2-3B (n = 2995, 1875 failing, 1120 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 1143 (38%) | 187 | **0** | 894 | 62 / 1120 | yes |
| Obs. + grounding | 1533 (51%) | 222 | **1** | 1233 | 77 / 1120 | yes |
| Obs. + all internals | 1339 (45%) | 243 | **3** | 1015 | 78 / 1120 | yes |
| Obs. + internals + grounding | 1536 (51%) | 326 | **3** | 1139 | 68 / 1120 | yes |
| Region grounding | 837 (28%) | 0 | **0** | 786 | 51 / 1120 |  |
| Attention (regions) | 495 (17%) | 0 | **0** | 465 | 30 / 1120 |  |
| Hidden states | 590 (20%) | 0 | **0** | 550 | 40 / 1120 |  |
| All internals | 680 (23%) | 0 | **0** | 634 | 46 / 1120 |  |
| All internals + grounding | 919 (31%) | 215 | **2** | 671 | 31 / 1120 | yes |
| Token confidence | 286 (10%) | 0 | **0** | 266 | 20 / 1120 |  |

### Qwen2.5-1.5B (n = 2945, 1989 failing, 956 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 1100 (37%) | 195 | **0** | 860 | 45 / 956 | yes |
| Obs. + grounding | 1316 (45%) | 204 | **0** | 1056 | 56 / 956 | yes |
| Obs. + all internals | 1020 (35%) | 199 | **2** | 782 | 37 / 956 | yes |
| Obs. + internals + grounding | 917 (31%) | 0 | **0** | 877 | 40 / 956 |  |
| Region grounding | 725 (25%) | 0 | **0** | 696 | 29 / 956 |  |
| Attention (regions) | 823 (28%) | 0 | **0** | 781 | 42 / 956 |  |
| Hidden states | 654 (22%) | 0 | **0** | 625 | 29 / 956 |  |
| All internals | 554 (19%) | 0 | **0** | 537 | 17 / 956 |  |
| All internals + grounding | 878 (30%) | 0 | **0** | 833 | 45 / 956 |  |
| Token confidence | 395 (13%) | 0 | **0** | 376 | 19 / 956 |  |

### SmolLM2-1.7B (n = 2946, 2456 failing, 490 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 2325 (79%) | 0 | **0** | 2191 | 134 / 490 |  |
| Obs. + grounding | 2509 (85%) | 0 | **0** | 2363 | 146 / 490 |  |
| Obs. + all internals | 2327 (79%) | 0 | **0** | 2191 | 136 / 490 |  |
| Obs. + internals + grounding | 2527 (86%) | 0 | **0** | 2362 | 165 / 490 |  |
| Region grounding | 2344 (80%) | 0 | **0** | 2224 | 120 / 490 |  |
| Attention (regions) | 2224 (75%) | 0 | **0** | 2099 | 125 / 490 |  |
| Hidden states | 2191 (74%) | 0 | **0** | 2083 | 108 / 490 |  |
| All internals | 2241 (76%) | 0 | **0** | 2116 | 125 / 490 |  |
| All internals + grounding | 2321 (79%) | 0 | **0** | 2209 | 112 / 490 |  |
| Token confidence | 1644 (56%) | 0 | **0** | 1567 | 77 / 490 |  |

### Qwen2.5-7B (4-bit) (n = 2973, 1450 failing, 1523 good)

| signal | calls avoided | correct let through | **failing let through** | failing stopped | good rejected | certified |
|---|---|---|---|---|---|---|
| Observables | 209 (7%) | 0 | **0** | 192 | 17 / 1523 |  |
| Obs. + grounding | 147 (5%) | 0 | **0** | 143 | 4 / 1523 |  |
| Obs. + all internals | 90 (3%) | 0 | **0** | 86 | 4 / 1523 |  |
| Obs. + internals + grounding | 380 (13%) | 293 | **10** | 75 | 2 / 1523 |  |
| Region grounding | 6 (0%) | 0 | **0** | 5 | 1 / 1523 |  |
| Attention (regions) | 75 (3%) | 0 | **0** | 73 | 2 / 1523 |  |
| Hidden states | 52 (2%) | 0 | **0** | 50 | 2 / 1523 |  |
| All internals | 14 (0%) | 0 | **0** | 13 | 1 / 1523 |  |
| All internals + grounding | 18 (1%) | 0 | **0** | 18 | 0 / 1523 |  |
| Token confidence | 0 (0%) | 0 | **0** | 0 | 0 / 1523 |  |
