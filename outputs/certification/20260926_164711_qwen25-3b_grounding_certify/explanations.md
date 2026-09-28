# Interpretable routing decisions — G1_context_grounding (outputs\fsm_baseline\20260926_152318_grounding_qwen25-3b-instruct_pilot3000)

Verifier verdicts are shown as offline truth only; they are never used by the policy.

## Failures caught (DISALLOW, verifier: invalid) — 323 cases

### v2_cert_ee4af04aff38a5aa — route **DISALLOW**, predicted failure risk 1.00 (verifier, offline: **invalid**)
Task: path from 12 to 1. Answer: `{"path": [12, 7, 8, 9, 13, 5, 0, 1, 6, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 7 — 5% on `12:` line (looked most at line `1:`) — legal
- step 2: 7 → 8 — 5% on `7:` line (looked most at line `1:`) — legal
- step 3: 8 → 9 — 3% on `8:` line (looked most at line `1:`) — legal
- step 4: 9 → 13 — 7% on `9:` line (looked most at line `1:`) — legal
- step 5: 13 → 5 — 7% on `13:` line (looked most at line `1:`) — legal
- step 6: 5 → 0 — 3% on `5:` line (looked most at line `1:`) — legal
- step 7: 0 → 1 — 16% on `0:` line (looked most at line `1:`) — legal
- step 8: 1 → 6 — 24% on `1:` line — legal
- step 9: 6 → 2 — 7% on `6:` line (looked most at line `1:`) — legal
- the answer does not start at 12 and end at 1 (visible without the verifier)

Weakest-grounded step 6: 5 → 0. While writing `0` the model put **3%** of its graph attention on the line that decides this step, `5: [0]`; it looked most at `1: [6, 8, 11]` (21%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ████················ 21.1%
  0: ███················· 12.8%
 10: ██·················· 10.0%
 14: ██··················  8.1%
  4: █···················  7.3%
  6: █···················  6.8%
  5: █···················  2.8%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +5.10 (value 0)
- answer path length: +2.32 (value 10)
- weakest step: best single head's share on the deciding line: -0.81 (value 0.0571)
- layer 075% depth: min share on deciding line: +0.76 (value 0.041)

### v2_cert_f22d42f8e2499610 — route **DISALLOW**, predicted failure risk 0.92 (verifier, offline: **invalid**)
Task: path from 13 to 2. Answer: `{"path": [13, 9, 10, 16, 7, 15, 13, 16, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 13 → 9 — 3% on `13:` line (looked most at line `1:`) — legal
- step 2: 9 → 10 — 4% on `9:` line (looked most at line `1:`) — **ILLEGAL**
- step 3: 10 → 16 — 7% on `10:` line (looked most at line `1:`) — legal
- step 4: 16 → 7 — 7% on `16:` line (looked most at line `1:`) — legal
- step 5: 7 → 15 — 7% on `7:` line (looked most at line `1:`) — legal
- step 6: 15 → 13 — 5% on `15:` line (looked most at line `1:`) — legal
- step 7: 13 → 16 — 3% on `13:` line (looked most at line `1:`) — **ILLEGAL**
- step 8: 16 → 2 — 7% on `16:` line (looked most at line `1:`) — **ILLEGAL**

Weakest-grounded step 1: 13 → 9. While writing `9` the model put **3%** of its graph attention on the line that decides this step, `13: [6, 9, 10]`; it looked most at `1: [9, 10, 17]` (16%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ███················· 15.7%
 19: ██··················  8.8%
 18: ██··················  8.1%
  2: █···················  6.8%
  0: █···················  6.0%
 14: █···················  5.8%
 13: █···················  2.5%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: +1.86 (value 9)
- layer 075% depth: min share on deciding line: +0.83 (value 0.0357)
- first step: attention share on the deciding line: -0.80 (value 0.0255)
- weakest step: best single head's share on the deciding line: -0.79 (value 0.0605)

### v2_cert_387080e5877a09bf — route **DISALLOW**, predicted failure risk 0.95 (verifier, offline: **invalid**)
Task: path from 3 to 0. Answer: `{"path": [3, 6, 12, 8, 11, 14, 0]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 6 — 20% on `3:` line — legal
- step 2: 6 → 12 — 2% on `6:` line (looked most at line `3:`) — legal
- step 3: 12 → 8 — 3% on `12:` line (looked most at line `3:`) — **ILLEGAL**
- step 4: 8 → 11 — 2% on `8:` line (looked most at line `3:`) — **ILLEGAL**
- step 5: 11 → 14 — 3% on `11:` line (looked most at line `3:`) — **ILLEGAL**
- step 6: 14 → 0 — 10% on `14:` line (looked most at line `3:`) — **ILLEGAL**

Weakest-grounded step 4: 8 → 11. While writing `11` the model put **2%** of its graph attention on the line that decides this step, `8: [4]`; it looked most at `3: [6, 7, 10]` (18%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  3: ████················ 18.0%
 19: ██·················· 10.9%
 14: ██·················· 10.4%
  1: ██··················  9.6%
 10: █···················  5.4%
  0: █···················  5.0%
  8: ····················  1.6%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: +1.00 (value 0.0238)
- answer path length: +0.95 (value 7)
- weakest step: best single head's share on the deciding line: -0.86 (value 0.051)
- first step: attention share on the deciding line: +0.62 (value 0.202)

### v2_cert_21df9b3cd430c3d2 — route **DISALLOW**, predicted failure risk 0.99 (verifier, offline: **invalid**)
Task: path from 1 to 7. Answer: `{"path": [1, 8, 9]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 8 — 9% on `1:` line (looked most at line `3:`) — legal
- step 2: 8 → 9 — 12% on `8:` line (looked most at line `3:`) — **ILLEGAL**
- the answer does not start at 1 and end at 7 (visible without the verifier)

Weakest-grounded step 1: 1 → 8. While writing `8` the model put **9%** of its graph attention on the line that decides this step, `1: [8]`; it looked most at `3: [1, 4, 5, 6, 8]` (17%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  3: ███················· 16.5%
  2: ██·················· 11.9%
  8: ██·················· 10.2%
  4: ██·················· 10.2%
  9: ██··················  9.9%
  5: ██··················  9.6%
  1: ██··················  8.8%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +5.10 (value 0)
- answer path length: -0.87 (value 3)
- shortest path length: +0.47 (value 5)
- weakest step: best single head's share on the deciding line: +0.43 (value 0.233)

## Correctly trusted (ALLOW, verifier: valid) — 282 cases

### v2_cert_89c166db2b33ef3f — route **ALLOW**, predicted failure risk 0.04 (verifier, offline: **valid**)
Task: path from 3 to 4. Answer: `{"path": [3, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 4 — 16% on `3:` line (looked most at line `1:`) — legal

Weakest-grounded step 1: 3 → 4. While writing `4` the model put **16%** of its graph attention on the line that decides this step, `3: [4]`; it looked most at `1: [2, 3]` (35%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ███████············· 35.4%
  4: ████················ 18.7%
  3: ███················· 16.3%  <- deciding line
  0: ███················· 14.8%
  2: ███················· 14.8%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -2.20 (value 0.254)
- weakest step: best single head's share on the deciding line: +1.55 (value 0.392)
- answer path length: -1.33 (value 2)
- answer is a single hop: -0.80 (value 1)

### v2_cert_17ee72dcae84eca9 — route **ALLOW**, predicted failure risk 0.04 (verifier, offline: **valid**)
Task: path from 1 to 4. Answer: `{"path": [1, 0, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 0 — 25% on `1:` line (looked most at line `0:`) — legal
- step 2: 0 → 4 — 38% on `0:` line — legal

Weakest-grounded step 1: 1 → 0. While writing `0` the model put **25%** of its graph attention on the line that decides this step, `1: [0, 4]`; it looked most at `0: [2, 3, 4]` (38%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ████████············ 37.6%
  1: █████··············· 25.1%  <- deciding line
  4: ███················· 16.6%
  2: ██·················· 10.8%
  3: ██·················· 10.0%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -1.90 (value 0.232)
- weakest step: best single head's share on the deciding line: +1.82 (value 0.431)
- layer 050% depth: min share on deciding line: -1.82 (value 0.255)
- average attention share on the deciding line: -1.45 (value 0.316)

### v2_cert_13ab07750c9aa8ab — route **ALLOW**, predicted failure risk 0.04 (verifier, offline: **valid**)
Task: path from 3 to 4. Answer: `{"path": [3, 0, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 0 — 24% on `3:` line (looked most at line `0:`) — legal
- step 2: 0 → 4 — 30% on `0:` line — legal

Weakest-grounded step 1: 3 → 0. While writing `0` the model put **24%** of its graph attention on the line that decides this step, `3: [0, 1]`; it looked most at `0: [1, 4]` (29%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ██████·············· 28.5%
  3: █████··············· 23.8%  <- deciding line
  2: ████················ 19.1%
  1: ███················· 15.3%
  4: ███················· 13.3%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -3.71 (value 0.362)
- weakest step: best single head's share on the deciding line: +2.12 (value 0.473)
- average attention share on the deciding line: -1.12 (value 0.272)
- first step: attention share on the deciding line: +0.92 (value 0.238)

### v2_cert_06a65f961322ff4a — route **ALLOW**, predicted failure risk 0.06 (verifier, offline: **valid**)
Task: path from 0 to 2. Answer: `{"path": [0, 1, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 0 → 1 — 38% on `0:` line — legal
- step 2: 1 → 2 — 23% on `1:` line (looked most at line `0:`) — legal

Weakest-grounded step 2: 1 → 2. While writing `2` the model put **23%** of its graph attention on the line that decides this step, `1: [0, 2]`; it looked most at `0: [1, 2, 3]` (39%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ████████············ 39.5%
  1: █████··············· 22.7%  <- deciding line
  3: ███················· 15.7%
  4: ██·················· 11.4%
  2: ██·················· 10.8%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- first step: attention share on the deciding line: +2.02 (value 0.375)
- layer 075% depth: min share on deciding line: -1.96 (value 0.237)
- layer 050% depth: min share on deciding line: -1.82 (value 0.255)
- weakest step: best single head's share on the deciding line: +1.77 (value 0.423)

## Failures let through (ALLOW, verifier: invalid) — 8 cases

### v2_cert_d10c44c010c5da26 — route **ALLOW**, predicted failure risk 0.04 (verifier, offline: **invalid**)
Task: path from 1 to 2. Answer: `{"path": [1, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 2 — 17% on `1:` line (looked most at line `3:`) — **ILLEGAL**

Weakest-grounded step 1: 1 → 2. While writing `2` the model put **17%** of its graph attention on the line that decides this step, `1: [4]`; it looked most at `3: [0, 1, 2]` (33%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  3: ███████············· 33.0%
  4: █████··············· 25.3%
  1: ███················· 17.4%  <- deciding line
  2: ███················· 14.9%
  0: ██··················  9.5%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: -1.33 (value 2)
- layer 075% depth: min share on deciding line: -1.23 (value 0.184)
- layer 050% depth: min share on deciding line: -1.08 (value 0.184)
- answer is a single hop: -0.80 (value 1)

### v2_cert_cf33f5ddb3f8c311 — route **ALLOW**, predicted failure risk 0.10 (verifier, offline: **invalid**)
Task: path from 4 to 1. Answer: `{"path": [4, 2, 0, 1]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 4 → 2 — 24% on `4:` line (looked most at line `0:`) — legal
- step 2: 2 → 0 — 14% on `2:` line (looked most at line `4:`) — legal
- step 3: 0 → 1 — 27% on `0:` line — **ILLEGAL**

Weakest-grounded step 2: 2 → 0. While writing `0` the model put **14%** of its graph attention on the line that decides this step, `2: [0, 4]`; it looked most at `4: [1, 2, 3]` (27%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  4: █████··············· 27.0%
  1: █████··············· 24.1%
  0: █████··············· 23.6%
  2: ███················· 14.2%  <- deciding line
  3: ██·················· 11.1%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- edge density: -1.35 (value 0.65)
- layer 075% depth: min share on deciding line: -0.96 (value 0.164)
- first step: attention share on the deciding line: +0.91 (value 0.237)
- average attention share on the deciding line: -0.70 (value 0.216)

### v2_cert_b1780c3559fed5a0 — route **ALLOW**, predicted failure risk 0.10 (verifier, offline: **invalid**)
Task: path from 1 to 4. Answer: `{"path": [1, 3, 0, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 3 — 18% on `1:` line (looked most at line `0:`) — legal
- step 2: 3 → 0 — 32% on `3:` line — legal
- step 3: 0 → 4 — 31% on `0:` line — **ILLEGAL**

Weakest-grounded step 1: 1 → 3. While writing `3` the model put **18%** of its graph attention on the line that decides this step, `1: [3]`; it looked most at `0: [1, 2]` (35%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ███████············· 34.8%
  3: █████··············· 27.5%
  1: ████················ 18.3%  <- deciding line
  2: ██·················· 11.2%
  4: ██··················  8.2%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -1.72 (value 0.219)
- average attention share on the deciding line: -1.12 (value 0.272)
- layer 050% depth: min share on deciding line: -0.88 (value 0.166)
- weakest step: best single head's share on the deciding line: +0.79 (value 0.285)

### v2_cert_9c2a6c2831673236 — route **ALLOW**, predicted failure risk 0.06 (verifier, offline: **invalid**)
Task: path from 2 to 4. Answer: `{"path": [2, 1, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 2 → 1 — 16% on `2:` line (looked most at line `1:`) — legal
- step 2: 1 → 4 — 38% on `1:` line — **ILLEGAL**

Weakest-grounded step 1: 2 → 1. While writing `1` the model put **16%** of its graph attention on the line that decides this step, `2: [1, 4]`; it looked most at `1: [0, 2, 3]` (38%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ████████············ 37.7%
  4: ████················ 21.2%
  2: ███················· 16.1%  <- deciding line
  0: ███················· 14.0%
  3: ██·················· 11.0%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 050% depth: min share on deciding line: -1.26 (value 0.202)
- average attention share on the deciding line: -1.09 (value 0.268)
- layer 075% depth: min share on deciding line: -1.05 (value 0.171)
- weakest step: best single head's share on the deciding line: +0.95 (value 0.308)

## Valid answers rejected (DISALLOW, verifier: valid) — 17 cases

### v2_cert_b542085a03ec72c5 — route **DISALLOW**, predicted failure risk 0.89 (verifier, offline: **valid**)
Task: path from 0 to 14. Answer: `{"path": [0, 2, 12, 11, 4, 14]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 0 → 2 — 17% on `0:` line — legal
- step 2: 2 → 12 — 7% on `2:` line (looked most at line `0:`) — legal
- step 3: 12 → 11 — 5% on `12:` line (looked most at line `0:`) — legal
- step 4: 11 → 4 — 2% on `11:` line (looked most at line `17:`) — legal
- step 5: 4 → 14 — 4% on `4:` line (looked most at line `17:`) — legal

Weakest-grounded step 4: 11 → 4. While writing `4` the model put **2%** of its graph attention on the line that decides this step, `11: [4, 12]`; it looked most at `17: [0, 1, 3, 4, 7, 10, 11, 19]` (12%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
 17: ██·················· 12.2%
  0: ██·················· 10.6%
  1: ██·················· 10.5%
  2: ██··················  8.4%
 19: █···················  7.0%
 18: █···················  6.9%
 11: ····················  1.8%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: +0.97 (value 0.0259)
- weakest step: best single head's share on the deciding line: -0.83 (value 0.0551)
- layer 050% depth: min share on deciding line: +0.58 (value 0.0267)
- answer path length: +0.49 (value 6)

### v2_cert_50535420ad02bae2 — route **DISALLOW**, predicted failure risk 0.91 (verifier, offline: **valid**)
Task: path from 1 to 14. Answer: `{"path": [1, 9, 3, 7, 6, 2, 14]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 9 — 11% on `1:` line (looked most at line `0:`) — legal
- step 2: 9 → 3 — 6% on `9:` line (looked most at line `0:`) — legal
- step 3: 3 → 7 — 3% on `3:` line (looked most at line `0:`) — legal
- step 4: 7 → 6 — 3% on `7:` line (looked most at line `0:`) — legal
- step 5: 6 → 2 — 6% on `6:` line (looked most at line `0:`) — legal
- step 6: 2 → 14 — 8% on `2:` line (looked most at line `0:`) — legal

Weakest-grounded step 3: 3 → 7. While writing `7` the model put **3%** of its graph attention on the line that decides this step, `3: [7]`; it looked most at `0: [2, 3, 4, 7, 9]` (17%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ███················· 17.3%
  4: ██·················· 11.7%
 10: ██·················· 10.7%
 13: ██··················  9.1%
  2: ██··················  8.1%
  1: ██··················  7.9%
  3: █···················  2.8%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: +0.95 (value 7)
- layer 075% depth: min share on deciding line: +0.82 (value 0.0365)
- weakest step: best single head's share on the deciding line: -0.73 (value 0.0694)
- layer 050% depth: min share on deciding line: +0.57 (value 0.0275)

### v2_cert_b0249ac50d76afad — route **DISALLOW**, predicted failure risk 0.92 (verifier, offline: **valid**)
Task: path from 6 to 1. Answer: `{"path": [6, 1, 4, 2, 3, 7, 8, 9, 0, 1]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 6 → 1 — 11% on `6:` line (looked most at line `1:`) — legal
- step 2: 1 → 4 — 30% on `1:` line — legal
- step 3: 4 → 2 — 13% on `4:` line (looked most at line `1:`) — legal
- step 4: 2 → 3 — 5% on `2:` line (looked most at line `1:`) — legal
- step 5: 3 → 7 — 4% on `3:` line (looked most at line `1:`) — legal
- step 6: 7 → 8 — 6% on `7:` line (looked most at line `1:`) — legal
- step 7: 8 → 9 — 7% on `8:` line (looked most at line `1:`) — legal
- step 8: 9 → 0 — 18% on `9:` line (looked most at line `1:`) — legal
- step 9: 0 → 1 — 18% on `0:` line (looked most at line `1:`) — legal

Weakest-grounded step 5: 3 → 7. While writing `7` the model put **4%** of its graph attention on the line that decides this step, `3: [4, 7]`; it looked most at `1: [0, 4, 5, 6]` (29%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ██████·············· 29.2%
  0: ███················· 15.7%
  9: ███················· 13.7%
  4: ██·················· 10.7%
  6: ██··················  9.4%
  8: █···················  4.9%
  3: █···················  4.4%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: +2.32 (value 10)
- layer 075% depth: min share on deciding line: +0.66 (value 0.0483)
- weakest step: best single head's share on the deciding line: -0.44 (value 0.11)
- layer 100% depth: min share on deciding line: +0.27 (value 0.0053)

### v2_cert_c8bf52e0c2f450a3 — route **DISALLOW**, predicted failure risk 0.91 (verifier, offline: **valid**)
Task: path from 6 to 12. Answer: `{"path": [6, 5, 16, 9, 14, 11, 15, 12]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 6 → 5 — 2% on `6:` line (looked most at line `1:`) — legal
- step 2: 5 → 16 — 2% on `5:` line (looked most at line `1:`) — legal
- step 3: 16 → 9 — 7% on `16:` line (looked most at line `1:`) — legal
- step 4: 9 → 14 — 6% on `9:` line (looked most at line `1:`) — legal
- step 5: 14 → 11 — 6% on `14:` line (looked most at line `1:`) — legal
- step 6: 11 → 15 — 6% on `11:` line (looked most at line `1:`) — legal
- step 7: 15 → 12 — 8% on `15:` line (looked most at line `1:`) — legal

Weakest-grounded step 2: 5 → 16. While writing `16` the model put **2%** of its graph attention on the line that decides this step, `5: [16]`; it looked most at `1: [13, 15, 16, 17, 19]` (22%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ████················ 21.7%
  0: ███················· 13.8%
  9: █···················  5.9%
 19: █···················  5.4%
 16: █···················  5.1%
 17: █···················  4.9%
  5: ····················  1.6%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: +1.41 (value 8)
- layer 075% depth: min share on deciding line: +1.00 (value 0.0237)
- weakest step: best single head's share on the deciding line: -0.91 (value 0.0438)
- first step: attention share on the deciding line: -0.86 (value 0.0179)
