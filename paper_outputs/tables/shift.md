# CSTR under plant / fault shift (`docs/cstr_shift_prereg.md`)

Probes and thresholds fitted on the source part only; evaluated once on the target. family (pooled) = the four held-out-family shifts, counts summed, AUROC averaged.

## X = 10%

### Qwen2.5-1.5B

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.927 | 1667 / 2478 (67%) | 424 | **14** | 1158 | 71 / 866 |
| control | Plant readings only | 0.920 | 1544 / 2478 (62%) | 419 | **21** | 1042 | 62 / 866 |
| control | Obs. + grounding | 0.925 | 1686 / 2478 (68%) | 454 | **21** | 1141 | 70 / 866 |
| control | Obs. + all internals | 0.919 | 1394 / 2478 (56%) | 310 | **4** | 1024 | 56 / 866 |
| control | Obs. + internals + grounding | 0.915 | 1495 / 2478 (60%) | 379 | **14** | 1043 | 59 / 866 |
| control | Region grounding | 0.726 | 654 / 2478 (26%) | 0 | **0** | 626 | 28 / 866 |
| control | Attention (regions) | 0.772 | 707 / 2478 (29%) | 0 | **0** | 665 | 42 / 866 |
| control | Hidden states | 0.795 | 751 / 2478 (30%) | 0 | **0** | 699 | 52 / 866 |
| control | All internals | 0.802 | 704 / 2478 (28%) | 0 | **0** | 663 | 41 / 866 |
| control | All internals + grounding | 0.804 | 752 / 2478 (30%) | 0 | **0** | 699 | 53 / 866 |
| control | Token confidence | 0.647 | 83 / 2478 (3%) | 0 | **0** | 77 | 6 / 866 |
| feed | Observables | 0.705 | 2010 / 2468 (81%) | 0 | **2** | 1845 | 163 / 267 |
| feed | Plant readings only | 0.689 | 1872 / 2468 (76%) | 0 | **0** | 1716 | 156 / 267 |
| feed | Obs. + grounding | 0.712 | 1961 / 2468 (79%) | 2 | **2** | 1797 | 160 / 267 |
| feed | Obs. + all internals | 0.724 | 1970 / 2468 (80%) | 12 | **22** | 1783 | 153 / 267 |
| feed | Obs. + internals + grounding | 0.728 | 1907 / 2468 (77%) | 13 | **22** | 1733 | 139 / 267 |
| feed | Region grounding | 0.604 | 807 / 2468 (33%) | 0 | **0** | 796 | 11 / 267 |
| feed | Attention (regions) | 0.608 | 850 / 2468 (34%) | 0 | **0** | 825 | 25 / 267 |
| feed | Hidden states | 0.636 | 893 / 2468 (36%) | 0 | **0** | 873 | 20 / 267 |
| feed | All internals | 0.644 | 929 / 2468 (38%) | 15 | **72** | 825 | 17 / 267 |
| feed | All internals + grounding | 0.643 | 925 / 2468 (37%) | 20 | **83** | 808 | 14 / 267 |
| feed | Token confidence | 0.544 | 12 / 2468 (0%) | 0 | **0** | 12 | 0 / 267 |
| cooling | Observables | 0.912 | 1767 / 2474 (71%) | 515 | **14** | 1113 | 125 / 939 |
| cooling | Plant readings only | 0.905 | 1673 / 2474 (68%) | 508 | **17** | 1036 | 112 / 939 |
| cooling | Obs. + grounding | 0.913 | 1772 / 2474 (72%) | 537 | **14** | 1101 | 120 / 939 |
| cooling | Obs. + all internals | 0.901 | 1399 / 2474 (57%) | 258 | **1** | 1033 | 107 / 939 |
| cooling | Obs. + internals + grounding | 0.901 | 1486 / 2474 (60%) | 325 | **6** | 1046 | 109 / 939 |
| cooling | Region grounding | 0.712 | 622 / 2474 (25%) | 0 | **0** | 582 | 40 / 939 |
| cooling | Attention (regions) | 0.759 | 703 / 2474 (28%) | 0 | **0** | 640 | 63 / 939 |
| cooling | Hidden states | 0.788 | 826 / 2474 (33%) | 0 | **0** | 735 | 91 / 939 |
| cooling | All internals | 0.787 | 781 / 2474 (32%) | 0 | **0** | 696 | 85 / 939 |
| cooling | All internals + grounding | 0.790 | 823 / 2474 (33%) | 0 | **0** | 724 | 99 / 939 |
| cooling | Token confidence | 0.629 | 41 / 2474 (2%) | 0 | **0** | 37 | 4 / 939 |
| family (pooled) | Observables | 0.722 | 3715 / 4954 (75%) | 846 | **28** | 2489 | 352 / 1691 |
| family (pooled) | Plant readings only | 0.824 | 3359 / 4954 (68%) | 831 | **38** | 2141 | 349 / 1691 |
| family (pooled) | Obs. + grounding | 0.716 | 3701 / 4954 (75%) | 848 | **27** | 2473 | 353 / 1691 |
| family (pooled) | Obs. + all internals | 0.740 | 3392 / 4954 (68%) | 935 | **349** | 1971 | 137 / 1691 |
| family (pooled) | Obs. + internals + grounding | 0.732 | 3272 / 4954 (66%) | 782 | **192** | 2106 | 192 / 1691 |
| family (pooled) | Region grounding | 0.544 | 251 / 4954 (5%) | 0 | **0** | 171 | 80 / 1691 |
| family (pooled) | Attention (regions) | 0.486 | 1208 / 4954 (24%) | 0 | **0** | 868 | 340 / 1691 |
| family (pooled) | Hidden states | 0.666 | 1223 / 4954 (25%) | 0 | **0** | 897 | 326 / 1691 |
| family (pooled) | All internals | 0.659 | 525 / 4954 (11%) | 0 | **45** | 374 | 106 / 1691 |
| family (pooled) | All internals + grounding | 0.650 | 1121 / 4954 (23%) | 1 | **341** | 583 | 196 / 1691 |
| family (pooled) | Token confidence | 0.504 | 356 / 4954 (7%) | 0 | **0** | 251 | 105 / 1691 |

### Qwen2.5-3B

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.870 | 1010 / 2473 (41%) | 0 | **0** | 955 | 55 / 749 |
| control | Plant readings only | 0.736 | 310 / 2473 (13%) | 0 | **0** | 282 | 28 / 749 |
| control | Obs. + grounding | 0.861 | 943 / 2473 (38%) | 0 | **0** | 889 | 54 / 749 |
| control | Obs. + all internals | 0.851 | 1171 / 2473 (47%) | 122 | **16** | 966 | 67 / 749 |
| control | Obs. + internals + grounding | 0.844 | 1100 / 2473 (44%) | 130 | **15** | 899 | 56 / 749 |
| control | Region grounding | 0.646 | 115 / 2473 (5%) | 0 | **0** | 109 | 6 / 749 |
| control | Attention (regions) | 0.688 | 0 / 2473 (0%) | 0 | **0** | 0 | 0 / 749 |
| control | Hidden states | 0.713 | 621 / 2473 (25%) | 0 | **0** | 547 | 74 / 749 |
| control | All internals | 0.714 | 523 / 2473 (21%) | 0 | **0** | 471 | 52 / 749 |
| control | All internals + grounding | 0.711 | 572 / 2473 (23%) | 0 | **0** | 507 | 65 / 749 |
| control | Token confidence | 0.568 | 0 / 2473 (0%) | 0 | **0** | 0 | 0 / 749 |
| feed | Observables | 0.725 | 1827 / 2473 (74%) | 6 | **0** | 1593 | 228 / 402 |
| feed | Plant readings only | 0.579 | 871 / 2473 (35%) | 0 | **0** | 789 | 82 / 402 |
| feed | Obs. + grounding | 0.727 | 1884 / 2473 (76%) | 7 | **0** | 1643 | 234 / 402 |
| feed | Obs. + all internals | 0.741 | 1667 / 2473 (67%) | 11 | **6** | 1483 | 167 / 402 |
| feed | Obs. + internals + grounding | 0.742 | 1630 / 2473 (66%) | 5 | **2** | 1461 | 162 / 402 |
| feed | Region grounding | 0.498 | 300 / 2473 (12%) | 0 | **0** | 267 | 33 / 402 |
| feed | Attention (regions) | 0.562 | 244 / 2473 (10%) | 0 | **0** | 221 | 23 / 402 |
| feed | Hidden states | 0.602 | 377 / 2473 (15%) | 0 | **0** | 353 | 24 / 402 |
| feed | All internals | 0.603 | 371 / 2473 (15%) | 0 | **0** | 360 | 11 / 402 |
| feed | All internals + grounding | 0.601 | 610 / 2473 (25%) | 17 | **38** | 523 | 32 / 402 |
| feed | Token confidence | 0.481 | 9 / 2473 (0%) | 0 | **0** | 8 | 1 / 402 |
| cooling | Observables | 0.860 | 1104 / 2466 (45%) | 0 | **0** | 1014 | 90 / 807 |
| cooling | Plant readings only | 0.725 | 394 / 2466 (16%) | 0 | **0** | 345 | 49 / 807 |
| cooling | Obs. + grounding | 0.859 | 1115 / 2466 (45%) | 0 | **0** | 1018 | 97 / 807 |
| cooling | Obs. + all internals | 0.851 | 968 / 2466 (39%) | 0 | **0** | 898 | 70 / 807 |
| cooling | Obs. + internals + grounding | 0.849 | 952 / 2466 (39%) | 0 | **0** | 883 | 69 / 807 |
| cooling | Region grounding | 0.650 | 173 / 2466 (7%) | 0 | **0** | 159 | 14 / 807 |
| cooling | Attention (regions) | 0.685 | 182 / 2466 (7%) | 0 | **0** | 173 | 9 / 807 |
| cooling | Hidden states | 0.712 | 524 / 2466 (21%) | 0 | **0** | 474 | 50 / 807 |
| cooling | All internals | 0.717 | 342 / 2466 (14%) | 0 | **0** | 317 | 25 / 807 |
| cooling | All internals + grounding | 0.717 | 416 / 2466 (17%) | 0 | **0** | 381 | 35 / 807 |
| cooling | Token confidence | 0.545 | 0 / 2466 (0%) | 0 | **0** | 0 | 0 / 807 |
| family (pooled) | Observables | 0.795 | 3270 / 4945 (66%) | 1 | **0** | 2745 | 524 / 1482 |
| family (pooled) | Plant readings only | 0.682 | 2048 / 4945 (41%) | 0 | **0** | 1653 | 395 / 1482 |
| family (pooled) | Obs. + grounding | 0.779 | 3247 / 4945 (66%) | 1 | **0** | 2721 | 525 / 1482 |
| family (pooled) | Obs. + all internals | 0.738 | 3299 / 4945 (67%) | 1 | **0** | 2750 | 548 / 1482 |
| family (pooled) | Obs. + internals + grounding | 0.732 | 3304 / 4945 (67%) | 1 | **0** | 2752 | 551 / 1482 |
| family (pooled) | Region grounding | 0.564 | 253 / 4945 (5%) | 0 | **0** | 228 | 25 / 1482 |
| family (pooled) | Attention (regions) | 0.602 | 125 / 4945 (3%) | 0 | **0** | 119 | 6 / 1482 |
| family (pooled) | Hidden states | 0.599 | 424 / 4945 (9%) | 0 | **0** | 394 | 30 / 1482 |
| family (pooled) | All internals | 0.617 | 630 / 4945 (13%) | 0 | **0** | 557 | 73 / 1482 |
| family (pooled) | All internals + grounding | 0.607 | 446 / 4945 (9%) | 0 | **0** | 408 | 38 / 1482 |
| family (pooled) | Token confidence | 0.498 | 12 / 4945 (0%) | 0 | **0** | 11 | 1 / 1482 |

### Qwen2.5-7B (4-bit)

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.934 | 2104 / 2494 (84%) | 291 | **12** | 1734 | 67 / 518 |
| control | Plant readings only | 0.895 | 1926 / 2494 (77%) | 191 | **21** | 1636 | 78 / 518 |
| control | Obs. + grounding | 0.929 | 2115 / 2494 (85%) | 268 | **13** | 1745 | 89 / 518 |
| control | Obs. + all internals | 0.923 | 1645 / 2494 (66%) | 0 | **0** | 1593 | 52 / 518 |
| control | Obs. + internals + grounding | 0.920 | 1627 / 2494 (65%) | 0 | **0** | 1573 | 54 / 518 |
| control | Region grounding | 0.754 | 998 / 2494 (40%) | 0 | **0** | 953 | 45 / 518 |
| control | Attention (regions) | 0.775 | 1144 / 2494 (46%) | 0 | **0** | 1089 | 55 / 518 |
| control | Hidden states | 0.786 | 1028 / 2494 (41%) | 0 | **0** | 985 | 43 / 518 |
| control | All internals | 0.803 | 1060 / 2494 (43%) | 0 | **0** | 1011 | 49 / 518 |
| control | All internals + grounding | 0.799 | 1091 / 2494 (44%) | 0 | **0** | 1038 | 53 / 518 |
| control | Token confidence | 0.678 | 377 / 2494 (15%) | 0 | **0** | 356 | 21 / 518 |
| feed | Observables | 0.657 | 2416 / 2494 (97%) | 4 | **0** | 2339 | 73 / 110 |
| feed | Plant readings only | 0.566 | 2460 / 2494 (99%) | 0 | **0** | 2356 | 104 / 110 |
| feed | Obs. + grounding | 0.649 | 2419 / 2494 (97%) | 4 | **0** | 2341 | 74 / 110 |
| feed | Obs. + all internals | 0.645 | 2375 / 2494 (95%) | 2 | **0** | 2303 | 70 / 110 |
| feed | Obs. + internals + grounding | 0.648 | 2357 / 2494 (95%) | 2 | **1** | 2284 | 70 / 110 |
| feed | Region grounding | 0.563 | 893 / 2494 (36%) | 0 | **0** | 867 | 26 / 110 |
| feed | Attention (regions) | 0.555 | 1475 / 2494 (59%) | 0 | **0** | 1410 | 65 / 110 |
| feed | Hidden states | 0.528 | 1455 / 2494 (58%) | 0 | **0** | 1390 | 65 / 110 |
| feed | All internals | 0.522 | 1407 / 2494 (56%) | 0 | **0** | 1345 | 62 / 110 |
| feed | All internals + grounding | 0.510 | 1323 / 2494 (53%) | 0 | **0** | 1266 | 57 / 110 |
| feed | Token confidence | 0.469 | 128 / 2494 (5%) | 0 | **0** | 127 | 1 / 110 |
| cooling | Observables | 0.924 | 2142 / 2492 (86%) | 319 | **18** | 1705 | 100 / 565 |
| cooling | Plant readings only | 0.884 | 1630 / 2492 (65%) | 0 | **0** | 1548 | 82 / 565 |
| cooling | Obs. + grounding | 0.922 | 2147 / 2492 (86%) | 307 | **16** | 1718 | 106 / 565 |
| cooling | Obs. + all internals | 0.909 | 1921 / 2492 (77%) | 162 | **10** | 1648 | 101 / 565 |
| cooling | Obs. + internals + grounding | 0.904 | 1861 / 2492 (75%) | 145 | **6** | 1612 | 98 / 565 |
| cooling | Region grounding | 0.748 | 1035 / 2492 (42%) | 0 | **0** | 972 | 63 / 565 |
| cooling | Attention (regions) | 0.742 | 1146 / 2492 (46%) | 0 | **0** | 1069 | 77 / 565 |
| cooling | Hidden states | 0.765 | 1151 / 2492 (46%) | 0 | **0** | 1072 | 79 / 565 |
| cooling | All internals | 0.770 | 1194 / 2492 (48%) | 0 | **0** | 1112 | 82 / 565 |
| cooling | All internals + grounding | 0.764 | 1189 / 2492 (48%) | 0 | **0** | 1100 | 89 / 565 |
| cooling | Token confidence | 0.655 | 580 / 2492 (23%) | 0 | **0** | 530 | 50 / 565 |
| family (pooled) | Observables | 0.773 | 4564 / 4985 (92%) | 571 | **40** | 3704 | 249 / 1034 |
| family (pooled) | Plant readings only | 0.723 | 3833 / 4985 (77%) | 195 | **27** | 3418 | 193 / 1034 |
| family (pooled) | Obs. + grounding | 0.755 | 4509 / 4985 (90%) | 562 | **38** | 3669 | 240 / 1034 |
| family (pooled) | Obs. + all internals | 0.707 | 4026 / 4985 (81%) | 181 | **16** | 3591 | 238 / 1034 |
| family (pooled) | Obs. + internals + grounding | 0.710 | 4016 / 4985 (81%) | 211 | **19** | 3561 | 225 / 1034 |
| family (pooled) | Region grounding | 0.620 | 1566 / 4985 (31%) | 0 | **0** | 1428 | 138 / 1034 |
| family (pooled) | Attention (regions) | 0.529 | 1373 / 4985 (28%) | 0 | **0** | 1233 | 140 / 1034 |
| family (pooled) | Hidden states | 0.519 | 1007 / 4985 (20%) | 0 | **0** | 897 | 110 / 1034 |
| family (pooled) | All internals | 0.547 | 1627 / 4985 (33%) | 0 | **0** | 1464 | 163 / 1034 |
| family (pooled) | All internals + grounding | 0.568 | 1903 / 4985 (38%) | 0 | **0** | 1738 | 165 / 1034 |
| family (pooled) | Token confidence | 0.538 | 771 / 4985 (15%) | 0 | **0** | 702 | 69 / 1034 |

### Llama-3.2-3B

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.900 | 1713 / 2383 (72%) | 0 | **0** | 1625 | 88 / 399 |
| control | Plant readings only | 0.676 | 12 / 2383 (1%) | 0 | **0** | 9 | 3 / 399 |
| control | Obs. + grounding | 0.907 | 1819 / 2383 (76%) | 0 | **0** | 1716 | 103 / 399 |
| control | Obs. + all internals | 0.924 | 1919 / 2383 (81%) | 0 | **0** | 1798 | 121 / 399 |
| control | Obs. + internals + grounding | 0.924 | 1888 / 2383 (79%) | 0 | **0** | 1776 | 112 / 399 |
| control | Region grounding | 0.705 | 913 / 2383 (38%) | 0 | **0** | 870 | 43 / 399 |
| control | Attention (regions) | 0.797 | 1234 / 2383 (52%) | 0 | **0** | 1168 | 66 / 399 |
| control | Hidden states | 0.838 | 1584 / 2383 (66%) | 0 | **0** | 1485 | 99 / 399 |
| control | All internals | 0.845 | 1664 / 2383 (70%) | 0 | **0** | 1560 | 104 / 399 |
| control | All internals + grounding | 0.847 | 1688 / 2383 (71%) | 0 | **0** | 1578 | 110 / 399 |
| control | Token confidence | 0.629 | 632 / 2383 (27%) | 0 | **0** | 611 | 21 / 399 |
| feed | Observables | 0.861 | 1969 / 2404 (82%) | 0 | **0** | 1838 | 131 / 372 |
| feed | Plant readings only | 0.615 | 12 / 2404 (0%) | 0 | **0** | 11 | 1 / 372 |
| feed | Obs. + grounding | 0.889 | 2004 / 2404 (83%) | 0 | **0** | 1843 | 161 / 372 |
| feed | Obs. + all internals | 0.905 | 2008 / 2404 (84%) | 0 | **0** | 1867 | 141 / 372 |
| feed | Obs. + internals + grounding | 0.902 | 2038 / 2404 (85%) | 0 | **0** | 1878 | 160 / 372 |
| feed | Region grounding | 0.673 | 848 / 2404 (35%) | 0 | **0** | 801 | 47 / 372 |
| feed | Attention (regions) | 0.805 | 1245 / 2404 (52%) | 0 | **0** | 1194 | 51 / 372 |
| feed | Hidden states | 0.812 | 1285 / 2404 (53%) | 0 | **0** | 1225 | 60 / 372 |
| feed | All internals | 0.829 | 1412 / 2404 (59%) | 0 | **0** | 1350 | 62 / 372 |
| feed | All internals + grounding | 0.821 | 1375 / 2404 (57%) | 0 | **0** | 1312 | 63 / 372 |
| feed | Token confidence | 0.570 | 645 / 2404 (27%) | 0 | **0** | 610 | 35 / 372 |
| cooling | Observables | 0.911 | 1841 / 2368 (78%) | 0 | **0** | 1738 | 103 / 399 |
| cooling | Plant readings only | 0.680 | 491 / 2368 (21%) | 0 | **0** | 459 | 32 / 399 |
| cooling | Obs. + grounding | 0.925 | 1911 / 2368 (81%) | 0 | **0** | 1793 | 118 / 399 |
| cooling | Obs. + all internals | 0.933 | 1963 / 2368 (83%) | 0 | **0** | 1835 | 128 / 399 |
| cooling | Obs. + internals + grounding | 0.933 | 1924 / 2368 (81%) | 0 | **0** | 1814 | 110 / 399 |
| cooling | Region grounding | 0.739 | 1106 / 2368 (47%) | 0 | **0** | 1050 | 56 / 399 |
| cooling | Attention (regions) | 0.826 | 1408 / 2368 (59%) | 0 | **0** | 1333 | 75 / 399 |
| cooling | Hidden states | 0.856 | 1533 / 2368 (65%) | 0 | **0** | 1455 | 78 / 399 |
| cooling | All internals | 0.859 | 1684 / 2368 (71%) | 0 | **0** | 1580 | 104 / 399 |
| cooling | All internals + grounding | 0.858 | 1685 / 2368 (71%) | 0 | **0** | 1577 | 108 / 399 |
| cooling | Token confidence | 0.663 | 821 / 2368 (35%) | 0 | **0** | 798 | 23 / 399 |
| family (pooled) | Observables | 0.801 | 3939 / 4764 (83%) | 0 | **0** | 3589 | 350 / 805 |
| family (pooled) | Plant readings only | 0.546 | 553 / 4764 (12%) | 0 | **0** | 502 | 51 / 805 |
| family (pooled) | Obs. + grounding | 0.791 | 4123 / 4764 (87%) | 0 | **0** | 3652 | 471 / 805 |
| family (pooled) | Obs. + all internals | 0.820 | 3929 / 4764 (82%) | 0 | **0** | 3535 | 394 / 805 |
| family (pooled) | Obs. + internals + grounding | 0.820 | 3942 / 4764 (83%) | 0 | **0** | 3526 | 416 / 805 |
| family (pooled) | Region grounding | 0.635 | 2618 / 4764 (55%) | 0 | **0** | 2218 | 400 / 805 |
| family (pooled) | Attention (regions) | 0.704 | 2769 / 4764 (58%) | 0 | **0** | 2395 | 374 / 805 |
| family (pooled) | Hidden states | 0.757 | 2808 / 4764 (59%) | 0 | **0** | 2464 | 344 / 805 |
| family (pooled) | All internals | 0.707 | 3074 / 4764 (65%) | 0 | **0** | 2669 | 405 / 805 |
| family (pooled) | All internals + grounding | 0.701 | 3043 / 4764 (64%) | 0 | **0** | 2641 | 402 / 805 |
| family (pooled) | Token confidence | 0.569 | 2109 / 4764 (44%) | 0 | **0** | 1808 | 301 / 805 |

## X = 5%

### Qwen2.5-1.5B

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.927 | 1640 / 2478 (66%) | 400 | **11** | 1158 | 71 / 866 |
| control | Plant readings only | 0.920 | 1440 / 2478 (58%) | 330 | **6** | 1042 | 62 / 866 |
| control | Obs. + grounding | 0.925 | 1547 / 2478 (62%) | 331 | **5** | 1141 | 70 / 866 |
| control | Obs. + all internals | 0.919 | 1080 / 2478 (44%) | 0 | **0** | 1024 | 56 / 866 |
| control | Obs. + internals + grounding | 0.915 | 1102 / 2478 (44%) | 0 | **0** | 1043 | 59 / 866 |
| control | Region grounding | 0.726 | 654 / 2478 (26%) | 0 | **0** | 626 | 28 / 866 |
| control | Attention (regions) | 0.772 | 707 / 2478 (29%) | 0 | **0** | 665 | 42 / 866 |
| control | Hidden states | 0.795 | 751 / 2478 (30%) | 0 | **0** | 699 | 52 / 866 |
| control | All internals | 0.802 | 704 / 2478 (28%) | 0 | **0** | 663 | 41 / 866 |
| control | All internals + grounding | 0.804 | 752 / 2478 (30%) | 0 | **0** | 699 | 53 / 866 |
| control | Token confidence | 0.647 | 83 / 2478 (3%) | 0 | **0** | 77 | 6 / 866 |
| feed | Observables | 0.705 | 2010 / 2468 (81%) | 0 | **2** | 1845 | 163 / 267 |
| feed | Plant readings only | 0.689 | 1872 / 2468 (76%) | 0 | **0** | 1716 | 156 / 267 |
| feed | Obs. + grounding | 0.712 | 1959 / 2468 (79%) | 0 | **2** | 1797 | 160 / 267 |
| feed | Obs. + all internals | 0.724 | 1954 / 2468 (79%) | 4 | **14** | 1783 | 153 / 267 |
| feed | Obs. + internals + grounding | 0.728 | 1888 / 2468 (76%) | 3 | **13** | 1733 | 139 / 267 |
| feed | Region grounding | 0.604 | 807 / 2468 (33%) | 0 | **0** | 796 | 11 / 267 |
| feed | Attention (regions) | 0.608 | 850 / 2468 (34%) | 0 | **0** | 825 | 25 / 267 |
| feed | Hidden states | 0.636 | 893 / 2468 (36%) | 0 | **0** | 873 | 20 / 267 |
| feed | All internals | 0.644 | 842 / 2468 (34%) | 0 | **0** | 825 | 17 / 267 |
| feed | All internals + grounding | 0.643 | 822 / 2468 (33%) | 0 | **0** | 808 | 14 / 267 |
| feed | Token confidence | 0.544 | 12 / 2468 (0%) | 0 | **0** | 12 | 0 / 267 |
| cooling | Observables | 0.912 | 1238 / 2474 (50%) | 0 | **0** | 1113 | 125 / 939 |
| cooling | Plant readings only | 0.905 | 1579 / 2474 (64%) | 423 | **8** | 1036 | 112 / 939 |
| cooling | Obs. + grounding | 0.913 | 1221 / 2474 (49%) | 0 | **0** | 1101 | 120 / 939 |
| cooling | Obs. + all internals | 0.901 | 1140 / 2474 (46%) | 0 | **0** | 1033 | 107 / 939 |
| cooling | Obs. + internals + grounding | 0.901 | 1155 / 2474 (47%) | 0 | **0** | 1046 | 109 / 939 |
| cooling | Region grounding | 0.712 | 622 / 2474 (25%) | 0 | **0** | 582 | 40 / 939 |
| cooling | Attention (regions) | 0.759 | 703 / 2474 (28%) | 0 | **0** | 640 | 63 / 939 |
| cooling | Hidden states | 0.788 | 826 / 2474 (33%) | 0 | **0** | 735 | 91 / 939 |
| cooling | All internals | 0.787 | 781 / 2474 (32%) | 0 | **0** | 696 | 85 / 939 |
| cooling | All internals + grounding | 0.790 | 823 / 2474 (33%) | 0 | **0** | 724 | 99 / 939 |
| cooling | Token confidence | 0.629 | 41 / 2474 (2%) | 0 | **0** | 37 | 4 / 939 |
| family (pooled) | Observables | 0.722 | 3231 / 4954 (65%) | 382 | **8** | 2489 | 352 / 1691 |
| family (pooled) | Plant readings only | 0.824 | 3048 / 4954 (62%) | 553 | **5** | 2141 | 349 / 1691 |
| family (pooled) | Obs. + grounding | 0.716 | 3607 / 4954 (73%) | 761 | **20** | 2473 | 353 / 1691 |
| family (pooled) | Obs. + all internals | 0.740 | 2759 / 4954 (56%) | 444 | **207** | 1971 | 137 / 1691 |
| family (pooled) | Obs. + internals + grounding | 0.732 | 2802 / 4954 (57%) | 391 | **113** | 2106 | 192 / 1691 |
| family (pooled) | Region grounding | 0.544 | 251 / 4954 (5%) | 0 | **0** | 171 | 80 / 1691 |
| family (pooled) | Attention (regions) | 0.486 | 1208 / 4954 (24%) | 0 | **0** | 868 | 340 / 1691 |
| family (pooled) | Hidden states | 0.666 | 1223 / 4954 (25%) | 0 | **0** | 897 | 326 / 1691 |
| family (pooled) | All internals | 0.659 | 480 / 4954 (10%) | 0 | **0** | 374 | 106 / 1691 |
| family (pooled) | All internals + grounding | 0.650 | 779 / 4954 (16%) | 0 | **0** | 583 | 196 / 1691 |
| family (pooled) | Token confidence | 0.504 | 356 / 4954 (7%) | 0 | **0** | 251 | 105 / 1691 |

### Qwen2.5-3B

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.870 | 1010 / 2473 (41%) | 0 | **0** | 955 | 55 / 749 |
| control | Plant readings only | 0.736 | 310 / 2473 (13%) | 0 | **0** | 282 | 28 / 749 |
| control | Obs. + grounding | 0.861 | 943 / 2473 (38%) | 0 | **0** | 889 | 54 / 749 |
| control | Obs. + all internals | 0.851 | 1033 / 2473 (42%) | 0 | **0** | 966 | 67 / 749 |
| control | Obs. + internals + grounding | 0.844 | 955 / 2473 (39%) | 0 | **0** | 899 | 56 / 749 |
| control | Region grounding | 0.646 | 115 / 2473 (5%) | 0 | **0** | 109 | 6 / 749 |
| control | Attention (regions) | 0.688 | 0 / 2473 (0%) | 0 | **0** | 0 | 0 / 749 |
| control | Hidden states | 0.713 | 621 / 2473 (25%) | 0 | **0** | 547 | 74 / 749 |
| control | All internals | 0.714 | 523 / 2473 (21%) | 0 | **0** | 471 | 52 / 749 |
| control | All internals + grounding | 0.711 | 572 / 2473 (23%) | 0 | **0** | 507 | 65 / 749 |
| control | Token confidence | 0.568 | 0 / 2473 (0%) | 0 | **0** | 0 | 0 / 749 |
| feed | Observables | 0.725 | 1821 / 2473 (74%) | 0 | **0** | 1593 | 228 / 402 |
| feed | Plant readings only | 0.579 | 871 / 2473 (35%) | 0 | **0** | 789 | 82 / 402 |
| feed | Obs. + grounding | 0.727 | 1877 / 2473 (76%) | 0 | **0** | 1643 | 234 / 402 |
| feed | Obs. + all internals | 0.741 | 1650 / 2473 (67%) | 0 | **0** | 1483 | 167 / 402 |
| feed | Obs. + internals + grounding | 0.742 | 1623 / 2473 (66%) | 0 | **0** | 1461 | 162 / 402 |
| feed | Region grounding | 0.498 | 300 / 2473 (12%) | 0 | **0** | 267 | 33 / 402 |
| feed | Attention (regions) | 0.562 | 244 / 2473 (10%) | 0 | **0** | 221 | 23 / 402 |
| feed | Hidden states | 0.602 | 377 / 2473 (15%) | 0 | **0** | 353 | 24 / 402 |
| feed | All internals | 0.603 | 371 / 2473 (15%) | 0 | **0** | 360 | 11 / 402 |
| feed | All internals + grounding | 0.601 | 555 / 2473 (22%) | 0 | **0** | 523 | 32 / 402 |
| feed | Token confidence | 0.481 | 9 / 2473 (0%) | 0 | **0** | 8 | 1 / 402 |
| cooling | Observables | 0.860 | 1104 / 2466 (45%) | 0 | **0** | 1014 | 90 / 807 |
| cooling | Plant readings only | 0.725 | 394 / 2466 (16%) | 0 | **0** | 345 | 49 / 807 |
| cooling | Obs. + grounding | 0.859 | 1115 / 2466 (45%) | 0 | **0** | 1018 | 97 / 807 |
| cooling | Obs. + all internals | 0.851 | 968 / 2466 (39%) | 0 | **0** | 898 | 70 / 807 |
| cooling | Obs. + internals + grounding | 0.849 | 952 / 2466 (39%) | 0 | **0** | 883 | 69 / 807 |
| cooling | Region grounding | 0.650 | 173 / 2466 (7%) | 0 | **0** | 159 | 14 / 807 |
| cooling | Attention (regions) | 0.685 | 182 / 2466 (7%) | 0 | **0** | 173 | 9 / 807 |
| cooling | Hidden states | 0.712 | 524 / 2466 (21%) | 0 | **0** | 474 | 50 / 807 |
| cooling | All internals | 0.717 | 342 / 2466 (14%) | 0 | **0** | 317 | 25 / 807 |
| cooling | All internals + grounding | 0.717 | 416 / 2466 (17%) | 0 | **0** | 381 | 35 / 807 |
| cooling | Token confidence | 0.545 | 0 / 2466 (0%) | 0 | **0** | 0 | 0 / 807 |
| family (pooled) | Observables | 0.795 | 3269 / 4945 (66%) | 0 | **0** | 2745 | 524 / 1482 |
| family (pooled) | Plant readings only | 0.682 | 2048 / 4945 (41%) | 0 | **0** | 1653 | 395 / 1482 |
| family (pooled) | Obs. + grounding | 0.779 | 3247 / 4945 (66%) | 1 | **0** | 2721 | 525 / 1482 |
| family (pooled) | Obs. + all internals | 0.738 | 3299 / 4945 (67%) | 1 | **0** | 2750 | 548 / 1482 |
| family (pooled) | Obs. + internals + grounding | 0.732 | 3304 / 4945 (67%) | 1 | **0** | 2752 | 551 / 1482 |
| family (pooled) | Region grounding | 0.564 | 253 / 4945 (5%) | 0 | **0** | 228 | 25 / 1482 |
| family (pooled) | Attention (regions) | 0.602 | 125 / 4945 (3%) | 0 | **0** | 119 | 6 / 1482 |
| family (pooled) | Hidden states | 0.599 | 424 / 4945 (9%) | 0 | **0** | 394 | 30 / 1482 |
| family (pooled) | All internals | 0.617 | 630 / 4945 (13%) | 0 | **0** | 557 | 73 / 1482 |
| family (pooled) | All internals + grounding | 0.607 | 446 / 4945 (9%) | 0 | **0** | 408 | 38 / 1482 |
| family (pooled) | Token confidence | 0.498 | 12 / 4945 (0%) | 0 | **0** | 11 | 1 / 1482 |

### Qwen2.5-7B (4-bit)

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.934 | 1801 / 2494 (72%) | 0 | **0** | 1734 | 67 / 518 |
| control | Plant readings only | 0.895 | 1714 / 2494 (69%) | 0 | **0** | 1636 | 78 / 518 |
| control | Obs. + grounding | 0.929 | 1834 / 2494 (74%) | 0 | **0** | 1745 | 89 / 518 |
| control | Obs. + all internals | 0.923 | 1645 / 2494 (66%) | 0 | **0** | 1593 | 52 / 518 |
| control | Obs. + internals + grounding | 0.920 | 1627 / 2494 (65%) | 0 | **0** | 1573 | 54 / 518 |
| control | Region grounding | 0.754 | 998 / 2494 (40%) | 0 | **0** | 953 | 45 / 518 |
| control | Attention (regions) | 0.775 | 1144 / 2494 (46%) | 0 | **0** | 1089 | 55 / 518 |
| control | Hidden states | 0.786 | 1028 / 2494 (41%) | 0 | **0** | 985 | 43 / 518 |
| control | All internals | 0.803 | 1060 / 2494 (43%) | 0 | **0** | 1011 | 49 / 518 |
| control | All internals + grounding | 0.799 | 1091 / 2494 (44%) | 0 | **0** | 1038 | 53 / 518 |
| control | Token confidence | 0.678 | 377 / 2494 (15%) | 0 | **0** | 356 | 21 / 518 |
| feed | Observables | 0.657 | 2414 / 2494 (97%) | 2 | **0** | 2339 | 73 / 110 |
| feed | Plant readings only | 0.566 | 2460 / 2494 (99%) | 0 | **0** | 2356 | 104 / 110 |
| feed | Obs. + grounding | 0.649 | 2418 / 2494 (97%) | 3 | **0** | 2341 | 74 / 110 |
| feed | Obs. + all internals | 0.645 | 2373 / 2494 (95%) | 0 | **0** | 2303 | 70 / 110 |
| feed | Obs. + internals + grounding | 0.648 | 2354 / 2494 (94%) | 0 | **0** | 2284 | 70 / 110 |
| feed | Region grounding | 0.563 | 893 / 2494 (36%) | 0 | **0** | 867 | 26 / 110 |
| feed | Attention (regions) | 0.555 | 1475 / 2494 (59%) | 0 | **0** | 1410 | 65 / 110 |
| feed | Hidden states | 0.528 | 1455 / 2494 (58%) | 0 | **0** | 1390 | 65 / 110 |
| feed | All internals | 0.522 | 1407 / 2494 (56%) | 0 | **0** | 1345 | 62 / 110 |
| feed | All internals + grounding | 0.510 | 1323 / 2494 (53%) | 0 | **0** | 1266 | 57 / 110 |
| feed | Token confidence | 0.469 | 128 / 2494 (5%) | 0 | **0** | 127 | 1 / 110 |
| cooling | Observables | 0.924 | 2115 / 2492 (85%) | 299 | **11** | 1705 | 100 / 565 |
| cooling | Plant readings only | 0.884 | 1630 / 2492 (65%) | 0 | **0** | 1548 | 82 / 565 |
| cooling | Obs. + grounding | 0.922 | 2029 / 2492 (81%) | 201 | **4** | 1718 | 106 / 565 |
| cooling | Obs. + all internals | 0.909 | 1749 / 2492 (70%) | 0 | **0** | 1648 | 101 / 565 |
| cooling | Obs. + internals + grounding | 0.904 | 1710 / 2492 (69%) | 0 | **0** | 1612 | 98 / 565 |
| cooling | Region grounding | 0.748 | 1035 / 2492 (42%) | 0 | **0** | 972 | 63 / 565 |
| cooling | Attention (regions) | 0.742 | 1146 / 2492 (46%) | 0 | **0** | 1069 | 77 / 565 |
| cooling | Hidden states | 0.765 | 1151 / 2492 (46%) | 0 | **0** | 1072 | 79 / 565 |
| cooling | All internals | 0.770 | 1194 / 2492 (48%) | 0 | **0** | 1112 | 82 / 565 |
| cooling | All internals + grounding | 0.764 | 1189 / 2492 (48%) | 0 | **0** | 1100 | 89 / 565 |
| cooling | Token confidence | 0.655 | 580 / 2492 (23%) | 0 | **0** | 530 | 50 / 565 |
| family (pooled) | Observables | 0.773 | 4249 / 4985 (85%) | 274 | **22** | 3704 | 249 / 1034 |
| family (pooled) | Plant readings only | 0.723 | 3611 / 4985 (72%) | 0 | **0** | 3418 | 193 / 1034 |
| family (pooled) | Obs. + grounding | 0.755 | 4197 / 4985 (84%) | 263 | **25** | 3669 | 240 / 1034 |
| family (pooled) | Obs. + all internals | 0.707 | 3966 / 4985 (80%) | 129 | **8** | 3591 | 238 / 1034 |
| family (pooled) | Obs. + internals + grounding | 0.710 | 3786 / 4985 (76%) | 0 | **0** | 3561 | 225 / 1034 |
| family (pooled) | Region grounding | 0.620 | 1566 / 4985 (31%) | 0 | **0** | 1428 | 138 / 1034 |
| family (pooled) | Attention (regions) | 0.529 | 1373 / 4985 (28%) | 0 | **0** | 1233 | 140 / 1034 |
| family (pooled) | Hidden states | 0.519 | 1007 / 4985 (20%) | 0 | **0** | 897 | 110 / 1034 |
| family (pooled) | All internals | 0.547 | 1627 / 4985 (33%) | 0 | **0** | 1464 | 163 / 1034 |
| family (pooled) | All internals + grounding | 0.568 | 1903 / 4985 (38%) | 0 | **0** | 1738 | 165 / 1034 |
| family (pooled) | Token confidence | 0.538 | 771 / 4985 (15%) | 0 | **0** | 702 | 69 / 1034 |

### Llama-3.2-3B

| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | failing stopped | good rejected |
|---|---|---|---|---|---|---|---|
| control | Observables | 0.900 | 1713 / 2383 (72%) | 0 | **0** | 1625 | 88 / 399 |
| control | Plant readings only | 0.676 | 12 / 2383 (1%) | 0 | **0** | 9 | 3 / 399 |
| control | Obs. + grounding | 0.907 | 1819 / 2383 (76%) | 0 | **0** | 1716 | 103 / 399 |
| control | Obs. + all internals | 0.924 | 1919 / 2383 (81%) | 0 | **0** | 1798 | 121 / 399 |
| control | Obs. + internals + grounding | 0.924 | 1888 / 2383 (79%) | 0 | **0** | 1776 | 112 / 399 |
| control | Region grounding | 0.705 | 913 / 2383 (38%) | 0 | **0** | 870 | 43 / 399 |
| control | Attention (regions) | 0.797 | 1234 / 2383 (52%) | 0 | **0** | 1168 | 66 / 399 |
| control | Hidden states | 0.838 | 1584 / 2383 (66%) | 0 | **0** | 1485 | 99 / 399 |
| control | All internals | 0.845 | 1664 / 2383 (70%) | 0 | **0** | 1560 | 104 / 399 |
| control | All internals + grounding | 0.847 | 1688 / 2383 (71%) | 0 | **0** | 1578 | 110 / 399 |
| control | Token confidence | 0.629 | 632 / 2383 (27%) | 0 | **0** | 611 | 21 / 399 |
| feed | Observables | 0.861 | 1969 / 2404 (82%) | 0 | **0** | 1838 | 131 / 372 |
| feed | Plant readings only | 0.615 | 12 / 2404 (0%) | 0 | **0** | 11 | 1 / 372 |
| feed | Obs. + grounding | 0.889 | 2004 / 2404 (83%) | 0 | **0** | 1843 | 161 / 372 |
| feed | Obs. + all internals | 0.905 | 2008 / 2404 (84%) | 0 | **0** | 1867 | 141 / 372 |
| feed | Obs. + internals + grounding | 0.902 | 2038 / 2404 (85%) | 0 | **0** | 1878 | 160 / 372 |
| feed | Region grounding | 0.673 | 848 / 2404 (35%) | 0 | **0** | 801 | 47 / 372 |
| feed | Attention (regions) | 0.805 | 1245 / 2404 (52%) | 0 | **0** | 1194 | 51 / 372 |
| feed | Hidden states | 0.812 | 1285 / 2404 (53%) | 0 | **0** | 1225 | 60 / 372 |
| feed | All internals | 0.829 | 1412 / 2404 (59%) | 0 | **0** | 1350 | 62 / 372 |
| feed | All internals + grounding | 0.821 | 1375 / 2404 (57%) | 0 | **0** | 1312 | 63 / 372 |
| feed | Token confidence | 0.570 | 645 / 2404 (27%) | 0 | **0** | 610 | 35 / 372 |
| cooling | Observables | 0.911 | 1841 / 2368 (78%) | 0 | **0** | 1738 | 103 / 399 |
| cooling | Plant readings only | 0.680 | 491 / 2368 (21%) | 0 | **0** | 459 | 32 / 399 |
| cooling | Obs. + grounding | 0.925 | 1911 / 2368 (81%) | 0 | **0** | 1793 | 118 / 399 |
| cooling | Obs. + all internals | 0.933 | 1963 / 2368 (83%) | 0 | **0** | 1835 | 128 / 399 |
| cooling | Obs. + internals + grounding | 0.933 | 1924 / 2368 (81%) | 0 | **0** | 1814 | 110 / 399 |
| cooling | Region grounding | 0.739 | 1106 / 2368 (47%) | 0 | **0** | 1050 | 56 / 399 |
| cooling | Attention (regions) | 0.826 | 1408 / 2368 (59%) | 0 | **0** | 1333 | 75 / 399 |
| cooling | Hidden states | 0.856 | 1533 / 2368 (65%) | 0 | **0** | 1455 | 78 / 399 |
| cooling | All internals | 0.859 | 1684 / 2368 (71%) | 0 | **0** | 1580 | 104 / 399 |
| cooling | All internals + grounding | 0.858 | 1685 / 2368 (71%) | 0 | **0** | 1577 | 108 / 399 |
| cooling | Token confidence | 0.663 | 821 / 2368 (35%) | 0 | **0** | 798 | 23 / 399 |
| family (pooled) | Observables | 0.801 | 3939 / 4764 (83%) | 0 | **0** | 3589 | 350 / 805 |
| family (pooled) | Plant readings only | 0.546 | 553 / 4764 (12%) | 0 | **0** | 502 | 51 / 805 |
| family (pooled) | Obs. + grounding | 0.791 | 4123 / 4764 (87%) | 0 | **0** | 3652 | 471 / 805 |
| family (pooled) | Obs. + all internals | 0.820 | 3929 / 4764 (82%) | 0 | **0** | 3535 | 394 / 805 |
| family (pooled) | Obs. + internals + grounding | 0.820 | 3942 / 4764 (83%) | 0 | **0** | 3526 | 416 / 805 |
| family (pooled) | Region grounding | 0.635 | 2618 / 4764 (55%) | 0 | **0** | 2218 | 400 / 805 |
| family (pooled) | Attention (regions) | 0.704 | 2769 / 4764 (58%) | 0 | **0** | 2395 | 374 / 805 |
| family (pooled) | Hidden states | 0.757 | 2808 / 4764 (59%) | 0 | **0** | 2464 | 344 / 805 |
| family (pooled) | All internals | 0.707 | 3074 / 4764 (65%) | 0 | **0** | 2669 | 405 / 805 |
| family (pooled) | All internals + grounding | 0.701 | 3043 / 4764 (64%) | 0 | **0** | 2641 | 402 / 805 |
| family (pooled) | Token confidence | 0.569 | 2109 / 4764 (44%) | 0 | **0** | 1808 | 301 / 805 |
