# Interpretable routing decisions — G1_context_grounding (outputs\fsm_baseline\20260926_154347_grounding_llama-32-3b-instruct_pilot3000)

Verifier verdicts are shown as offline truth only; they are never used by the policy.

## Failures caught (DISALLOW, verifier: invalid) — 1556 cases

### v2_cert_ee4af04aff38a5aa — route **DISALLOW**, predicted failure risk 0.89 (verifier, offline: **invalid**)
Task: path from 12 to 1. Answer: `{"path": [12, 10, 0, 6, 2, 1]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 10 — 27% on `12:` line — legal
- step 2: 10 → 0 — 21% on `10:` line — legal
- step 3: 0 → 6 — 18% on `0:` line (looked most at line `10:`) — legal
- step 4: 6 → 2 — 24% on `6:` line — legal
- step 5: 2 → 1 — 12% on `2:` line (looked most at line `6:`) — **ILLEGAL**

Weakest-grounded step 5: 2 → 1. While writing `1` the model put **12%** of its graph attention on the line that decides this step, `2: [6, 12, 14]`; it looked most at `6: [2, 7, 8, 11, 14]` (17%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  6: ███················· 16.9%
  2: ██·················· 12.4%  <- deciding line
  0: ██·················· 11.0%
  1: ██·················· 10.0%
 11: ██··················  8.4%
 10: █···················  7.3%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -0.72 (value 1)
- layer 075% depth: min share on deciding line: +0.61 (value 0.0867)
- answer path length: +0.39 (value 6)
- weakest step: attention share on the deciding line: +0.33 (value 0.124)

### v2_cert_0a9a2199fc2bb74a — route **DISALLOW**, predicted failure risk 1.00 (verifier, offline: **invalid**)
Task: path from 1 to 4. Answer: `{"path": [1, 0, 4, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 0 — 27% on `1:` line (looked most at line `3:`) — legal
- step 2: 0 → 4 — 26% on `0:` line (looked most at line `3:`) — legal
- step 3: 4 → 2 — 38% on `4:` line — legal
- the answer does not start at 1 and end at 4 (visible without the verifier)

Weakest-grounded step 2: 0 → 4. While writing `4` the model put **26%** of its graph attention on the line that decides this step, `0: [4]`; it looked most at `3: [0, 1, 2, 4]` (45%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  3: █████████··········· 45.2%
  0: █████··············· 25.5%  <- deciding line
  4: ███················· 12.7%
  1: ██··················  9.4%
  2: █···················  7.2%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +6.47 (value 0)
- weakest step: attention share on the deciding line: -1.22 (value 0.255)
- average attention share on the deciding line: -1.01 (value 0.302)
- layer 075% depth: min share on deciding line: -0.53 (value 0.325)

### v2_cert_c900ceed43e4fcd3 — route **DISALLOW**, predicted failure risk 0.93 (verifier, offline: **invalid**)
Task: path from 12 to 8. Answer: `{"path": [12, 0, 9, 8]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 0 — 16% on `12:` line — legal
- step 2: 0 → 9 — 7% on `0:` line (looked most at line `5:`) — legal
- step 3: 9 → 8 — 11% on `9:` line (looked most at line `5:`) — **ILLEGAL**

Weakest-grounded step 2: 0 → 9. While writing `9` the model put **7%** of its graph attention on the line that decides this step, `0: [5, 9]`; it looked most at `5: [1, 2, 7, 9, 10, 12]` (12%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  5: ██·················· 12.2%
  3: ██·················· 10.5%
  4: ██··················  9.8%
 13: ██··················  8.0%
 12: █···················  7.5%
 14: █···················  7.2%
  0: █···················  7.1%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- average attention share on the deciding line: +1.29 (value 0.115)
- weakest step: attention share on the deciding line: +0.96 (value 0.0712)
- shortest path length: +0.79 (value 6)
- layer 075% depth: min share on deciding line: +0.77 (value 0.0542)

### v2_cert_f22d42f8e2499610 — route **DISALLOW**, predicted failure risk 0.96 (verifier, offline: **invalid**)
Task: path from 13 to 2. Answer: `{"path": [13, 6, 9, 18, 4, 8, 0, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 13 → 6 — 8% on `13:` line (looked most at line `10:`) — legal
- step 2: 6 → 9 — 10% on `6:` line (looked most at line `2:`) — legal
- step 3: 9 → 18 — 12% on `9:` line — legal
- step 4: 18 → 4 — 26% on `18:` line — legal
- step 5: 4 → 8 — 8% on `4:` line (looked most at line `15:`) — **ILLEGAL**
- step 6: 8 → 0 — 6% on `8:` line (looked most at line `7:`) — **ILLEGAL**
- step 7: 0 → 2 — 8% on `0:` line (looked most at line `2:`) — legal

Weakest-grounded step 6: 8 → 0. While writing `0` the model put **6%** of its graph attention on the line that decides this step, `8: [3, 5, 16]`; it looked most at `7: [0, 4, 5, 10, 13, 14, 15, 16]` (11%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  7: ██·················· 11.3%
 14: ██·················· 10.3%
 18: ██··················  9.8%
 16: ██··················  8.2%
 10: █···················  7.3%
 19: █···················  6.7%
  8: █···················  5.7%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- average attention share on the deciding line: +1.33 (value 0.112)
- first step: attention share on the deciding line: -1.14 (value 0.082)
- weakest step: attention share on the deciding line: +1.13 (value 0.057)
- answer path length: +0.98 (value 8)

## Correctly trusted (ALLOW, verifier: valid) — 158 cases

### v2_cert_8bd1e8f33f210196 — route **ALLOW**, predicted failure risk 0.01 (verifier, offline: **valid**)
Task: path from 4 to 2. Answer: `{"path": [4, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 4 → 2 — 30% on `4:` line — legal

Weakest-grounded step 1: 4 → 2. While writing `2` the model put **30%** of its graph attention on the line that decides this step, `4: [0, 2]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  4: ██████·············· 29.8%  <- deciding line
  0: █████··············· 27.2%
  1: ████················ 20.7%
  2: ███················· 16.0%
  3: █···················  6.3%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -1.72 (value 0.298)
- layer 075% depth: min share on deciding line: -1.25 (value 0.476)
- average attention share on the deciding line: -0.96 (value 0.298)
- answer path length: -0.81 (value 2)

### v2_cert_6760e16fbcfd9499 — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 1 to 0. Answer: `{"path": [1, 0]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 0 — 45% on `1:` line — legal

Weakest-grounded step 1: 1 → 0. While writing `0` the model put **45%** of its graph attention on the line that decides this step, `1: [0, 3]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: █████████··········· 44.9%  <- deciding line
  0: █████··············· 23.5%
  3: ███················· 12.8%
  4: ██·················· 10.0%
  2: ██··················  8.9%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -3.51 (value 0.449)
- average attention share on the deciding line: -2.82 (value 0.449)
- layer 075% depth: min share on deciding line: -1.80 (value 0.589)
- first step: attention share on the deciding line: +1.50 (value 0.449)

### v2_cert_2e9ec6583d009845 — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 2 to 3. Answer: `{"path": [2, 3]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 2 → 3 — 34% on `2:` line — legal

Weakest-grounded step 1: 2 → 3. While writing `3` the model put **34%** of its graph attention on the line that decides this step, `2: [3]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  2: ███████············· 33.5%  <- deciding line
  3: ██████·············· 29.9%
  0: █████··············· 23.0%
  1: ██··················  8.1%
  4: █···················  5.4%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -2.17 (value 0.335)
- layer 075% depth: min share on deciding line: -1.48 (value 0.522)
- average attention share on the deciding line: -1.42 (value 0.335)
- answer path length: -0.81 (value 2)

### v2_cert_efa3ea01af011ef0 — route **ALLOW**, predicted failure risk 0.01 (verifier, offline: **valid**)
Task: path from 12 to 9. Answer: `{"path": [12, 9]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 9 — 22% on `12:` line — legal

Weakest-grounded step 1: 12 → 9. While writing `9` the model put **22%** of its graph attention on the line that decides this step, `12: [2, 8, 9, 13, 17, 19]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
 12: ████················ 22.1%  <- deciding line
 15: ██··················  8.6%
 11: ██··················  8.4%
  1: ██··················  8.2%
 17: ██··················  8.1%
  5: █···················  4.7%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -1.58 (value 0.543)
- answer path length: -0.81 (value 2)
- weakest step: attention share on the deciding line: -0.81 (value 0.221)
- answer starts/ends at the right states: -0.72 (value 1)

## Failures let through (ALLOW, verifier: invalid) — 2 cases

### v2_cert_12e869ad160c74fc — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **invalid**)
Task: path from 3 to 1. Answer: `{"path": [3, 1]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 1 — 53% on `3:` line — **ILLEGAL**

Weakest-grounded step 1: 3 → 1. While writing `1` the model put **53%** of its graph attention on the line that decides this step, `3: [0, 2, 4]`. (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  3: ███████████········· 52.8%  <- deciding line
  2: █████··············· 22.6%
  0: ██·················· 10.2%
  1: ██··················  9.0%
  4: █···················  5.5%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -4.45 (value 0.528)
- average attention share on the deciding line: -3.80 (value 0.528)
- first step: attention share on the deciding line: +2.07 (value 0.528)
- layer 025% depth: min share on deciding line: +1.98 (value 0.473)

### v2_cert_4f8140299e3037f6 — route **ALLOW**, predicted failure risk 0.01 (verifier, offline: **invalid**)
Task: path from 15 to 16. Answer: `{"path": [15, 16]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 15 → 16 — 25% on `15:` line — **ILLEGAL**

Weakest-grounded step 1: 15 → 16. While writing `16` the model put **25%** of its graph attention on the line that decides this step, `15: [6, 11, 12, 13, 14, 18]`. (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
 15: █████··············· 25.2%  <- deciding line
 16: ██·················· 12.4%
  0: ██··················  8.5%
 14: ██··················  8.1%
 17: █···················  5.4%
 18: █···················  5.2%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -1.42 (value 0.511)
- weakest step: attention share on the deciding line: -1.18 (value 0.252)
- answer path length: -0.81 (value 2)
- answer starts/ends at the right states: -0.72 (value 1)

## Valid answers rejected (DISALLOW, verifier: valid) — 172 cases

### v2_cert_a2a078babb82f774 — route **DISALLOW**, predicted failure risk 0.78 (verifier, offline: **valid**)
Task: path from 8 to 16. Answer: `{"path": [8, 1, 3, 5, 2, 16]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 8 → 1 — 26% on `8:` line — legal
- step 2: 1 → 3 — 22% on `1:` line — legal
- step 3: 3 → 5 — 12% on `3:` line (looked most at line `2:`) — legal
- step 4: 5 → 2 — 15% on `5:` line — legal
- step 5: 2 → 16 — 13% on `2:` line — legal

Weakest-grounded step 3: 3 → 5. While writing `5` the model put **12%** of its graph attention on the line that decides this step, `3: [5, 7, 14]`; it looked most at `2: [1, 3, 16, 19]` (16%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  2: ███················· 15.9%
  3: ██·················· 12.4%  <- deciding line
  0: ██··················  8.2%
 19: █···················  7.1%
 14: █···················  6.5%
  5: █···················  6.1%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -0.72 (value 1)
- average attention share on the deciding line: +0.51 (value 0.178)
- answer path length: +0.39 (value 6)
- graph size (nodes): -0.38 (value 20)

### v2_cert_8fe7be76e86ff7a8 — route **DISALLOW**, predicted failure risk 0.87 (verifier, offline: **valid**)
Task: path from 6 to 9. Answer: `{"path": [6, 2, 8, 7, 9]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 6 → 2 — 31% on `6:` line — legal
- step 2: 2 → 8 — 15% on `2:` line (looked most at line `4:`) — legal
- step 3: 8 → 7 — 13% on `8:` line (looked most at line `2:`) — legal
- step 4: 7 → 9 — 23% on `7:` line — legal

Weakest-grounded step 3: 8 → 7. While writing `7` the model put **13%** of its graph attention on the line that decides this step, `8: [2, 7]`; it looked most at `2: [7, 8, 9]` (18%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  2: ████················ 18.5%
  7: ████················ 18.3%
  9: ███················· 13.5%
  8: ███················· 13.0%  <- deciding line
  5: ██·················· 11.0%
  4: ██··················  9.5%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -0.72 (value 1)
- first step: attention share on the deciding line: +0.49 (value 0.309)
- weakest step: attention share on the deciding line: +0.26 (value 0.13)
- average attention share on the deciding line: +0.19 (value 0.204)

### v2_cert_5cb02cd5778479f9 — route **DISALLOW**, predicted failure risk 0.89 (verifier, offline: **valid**)
Task: path from 2 to 3. Answer: `{"path": [2, 11, 3]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 2 → 11 — 15% on `2:` line — legal
- step 2: 11 → 3 — 6% on `11:` line (looked most at line `0:`) — legal

Weakest-grounded step 2: 11 → 3. While writing `3` the model put **6%** of its graph attention on the line that decides this step, `11: [2, 3, 5, 6, 15]`; it looked most at `0: [2, 3, 7, 11, 12, 17]` (18%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ████················ 17.7%
  1: ███················· 13.5%
 19: ██··················  9.0%
 13: █···················  7.3%
  2: █···················  6.2%
 11: █···················  6.1%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- average attention share on the deciding line: +1.42 (value 0.105)
- weakest step: attention share on the deciding line: +1.08 (value 0.0611)
- layer 075% depth: min share on deciding line: +0.85 (value 0.0361)
- answer starts/ends at the right states: -0.72 (value 1)

### v2_cert_3e59415b3bca0029 — route **DISALLOW**, predicted failure risk 0.70 (verifier, offline: **valid**)
Task: path from 3 to 1. Answer: `{"path": [3, 0, 1]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 0 — 32% on `3:` line — legal
- step 2: 0 → 1 — 19% on `0:` line (looked most at line `1:`) — legal

Weakest-grounded step 2: 0 → 1. While writing `1` the model put **19%** of its graph attention on the line that decides this step, `0: [1]`; it looked most at `1: [0, 3]` (35%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ███████············· 35.0%
  2: ████················ 21.0%
  3: ████················ 18.9%
  0: ████················ 18.5%  <- deciding line
  4: █···················  6.5%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -0.72 (value 1)
- first step: attention share on the deciding line: +0.57 (value 0.319)
- answer path length: -0.51 (value 3)
- layer 025% depth: min share on deciding line: +0.46 (value 0.163)
