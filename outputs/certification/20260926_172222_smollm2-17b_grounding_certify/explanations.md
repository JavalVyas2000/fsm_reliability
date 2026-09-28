# Interpretable routing decisions — G1_context_grounding (outputs\fsm_baseline\20260926_161230_grounding_smollm2-17b-instruct_pilot3000)

Verifier verdicts are shown as offline truth only; they are never used by the policy.

## Failures caught (DISALLOW, verifier: invalid) — 2407 cases

### v2_cert_ee4af04aff38a5aa — route **DISALLOW**, predicted failure risk 1.00 (verifier, offline: **invalid**)
Task: path from 12 to 1. Answer: `{"path": [12, 10, 6, 2, 7, 8, 11, 0, 3, 4, 13]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 10 — 8% on `12:` line (looked most at line `0:`) — legal
- step 2: 10 → 6 — 17% on `10:` line — legal
- step 3: 6 → 2 — 15% on `6:` line (looked most at line `0:`) — legal
- step 4: 2 → 7 — 9% on `2:` line (looked most at line `0:`) — **ILLEGAL**
- step 5: 7 → 8 — 7% on `7:` line (looked most at line `0:`) — legal
- step 6: 8 → 11 — 5% on `8:` line (looked most at line `0:`) — **ILLEGAL**
- step 7: 11 → 0 — 9% on `11:` line (looked most at line `0:`) — **ILLEGAL**
- step 8: 0 → 3 — 15% on `0:` line — **ILLEGAL**
- step 9: 3 → 4 — 4% on `3:` line (looked most at line `0:`) — **ILLEGAL**
- step 10: 4 → 13 — 4% on `4:` line (looked most at line `0:`) — **ILLEGAL**
- the answer does not start at 12 and end at 1 (visible without the verifier)

Weakest-grounded step 9: 3 → 4. While writing `4` the model put **4%** of its graph attention on the line that decides this step, `3: [1]`; it looked most at `0: [1, 4, 6, 9, 12, 13]` (12%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  0: ██·················· 12.4%
 10: ██·················· 10.1%
  1: ██··················  9.2%
 14: ██··················  9.1%
  2: ██··················  7.8%
  7: █···················  7.0%
  3: █···················  4.4%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +4.62 (value 0)
- answer path length: +2.56 (value 11)
- layer 075% depth: min share on deciding line: +0.70 (value 0.0527)
- first step: attention share on the deciding line: -0.66 (value 0.08)

### v2_cert_a2a078babb82f774 — route **DISALLOW**, predicted failure risk 0.99 (verifier, offline: **invalid**)
Task: path from 8 to 16. Answer: `{"path": [8, 1, 11, 14, 16]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 8 → 1 — 13% on `8:` line — legal
- step 2: 1 → 11 — 4% on `1:` line (looked most at line `19:`) — **ILLEGAL**
- step 3: 11 → 14 — 6% on `11:` line (looked most at line `19:`) — **ILLEGAL**
- step 4: 14 → 16 — 8% on `14:` line (looked most at line `19:`) — **ILLEGAL**

Weakest-grounded step 2: 1 → 11. While writing `11` the model put **4%** of its graph attention on the line that decides this step, `1: [3, 8, 10]`; it looked most at `19: [0, 2, 6, 10, 16]` (11%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
 19: ██·················· 10.7%
  8: ██··················  8.4%
  2: █···················  6.8%
 16: █···················  6.5%
 18: █···················  6.1%
 12: █···················  5.9%
  1: █···················  4.5%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -2.13 (value 1)
- layer 075% depth: min share on deciding line: +0.92 (value 0.0384)
- average attention share on the deciding line: +0.74 (value 0.078)
- layer 025% depth: min share on deciding line: -0.24 (value 0.0278)

### v2_cert_89c166db2b33ef3f — route **DISALLOW**, predicted failure risk 0.71 (verifier, offline: **invalid**)
Task: path from 3 to 4. Answer: `{"path": [3, 0, 2, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 0 — 16% on `3:` line (looked most at line `1:`) — **ILLEGAL**
- step 2: 0 → 2 — 21% on `0:` line (looked most at line `1:`) — **ILLEGAL**
- step 3: 2 → 4 — 15% on `2:` line (looked most at line `1:`) — **ILLEGAL**

Weakest-grounded step 3: 2 → 4. While writing `4` the model put **15%** of its graph attention on the line that decides this step, `2: [0]`; it looked most at `1: [2, 3]` (36%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  1: ███████············· 36.1%
  4: ████················ 18.5%
  3: ███················· 16.7%
  2: ███················· 14.9%  <- deciding line
  0: ███················· 13.9%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -2.13 (value 1)
- average attention share on the deciding line: -0.68 (value 0.171)
- shortest path length: -0.56 (value 2)
- answer path length: -0.52 (value 4)

### v2_cert_bcbec0c371e300e5 — route **DISALLOW**, predicted failure risk 1.00 (verifier, offline: **invalid**)
Task: path from 9 to 10. Answer: `{"path": [9, 1, 3, 7, 12, 13, 5, 9, 11, 14]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 9 → 1 — 8% on `9:` line (looked most at line `1:`) — legal
- step 2: 1 → 3 — 10% on `1:` line (looked most at line `10:`) — legal
- step 3: 3 → 7 — 3% on `3:` line (looked most at line `10:`) — **ILLEGAL**
- step 4: 7 → 12 — 4% on `7:` line (looked most at line `10:`) — **ILLEGAL**
- step 5: 12 → 13 — 8% on `12:` line (looked most at line `10:`) — legal
- step 6: 13 → 5 — 11% on `13:` line (looked most at line `10:`) — legal
- step 7: 5 → 9 — 3% on `5:` line (looked most at line `14:`) — **ILLEGAL**
- step 8: 9 → 11 — 6% on `9:` line (looked most at line `14:`) — **ILLEGAL**
- step 9: 11 → 14 — 8% on `11:` line (looked most at line `13:`) — **ILLEGAL**
- the answer does not start at 9 and end at 10 (visible without the verifier)

Weakest-grounded step 3: 3 → 7. While writing `7` the model put **3%** of its graph attention on the line that decides this step, `3: [0, 10]`; it looked most at `10: [0, 4, 5, 8, 12, 13]` (14%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
 10: ███················· 13.7%
  2: ██·················· 10.2%
  9: ██··················  8.8%
 11: ██··················  8.5%
  8: ██··················  7.7%
 13: ██··················  7.7%
  3: █···················  2.9%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +4.62 (value 0)
- answer path length: +2.12 (value 10)
- layer 075% depth: min share on deciding line: +1.31 (value 0.0116)
- average attention share on the deciding line: +0.88 (value 0.0686)

## Correctly trusted (ALLOW, verifier: valid) — 56 cases

### v2_cert_17ee72dcae84eca9 — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 1 to 4. Answer: `{"path": [1, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 4 — 31% on `1:` line (looked most at line `0:`) — legal

Weakest-grounded step 1: 1 → 4. While writing `4` the model put **31%** of its graph attention on the line that decides this step, `1: [0, 4]`; it looked most at `0: [2, 3, 4]` (39%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ████████············ 39.0%
  1: ██████·············· 30.9%  <- deciding line
  4: ███················· 12.8%
  2: ██·················· 10.1%
  3: █···················  7.2%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -6.98 (value 0.572)
- average attention share on the deciding line: -2.81 (value 0.309)
- answer starts/ends at the right states: -2.13 (value 1)
- first step: attention share on the deciding line: +1.69 (value 0.309)

### v2_cert_3fa555692763429d — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 0 to 2. Answer: `{"path": [0, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 0 → 2 — 31% on `0:` line — legal

Weakest-grounded step 1: 0 → 2. While writing `2` the model put **31%** of its graph attention on the line that decides this step, `0: [2]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ██████·············· 31.4%  <- deciding line
  3: █████··············· 26.9%
  1: ████················ 18.2%
  2: ███················· 14.9%
  4: ██··················  8.6%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -4.77 (value 0.422)
- average attention share on the deciding line: -2.88 (value 0.314)
- answer starts/ends at the right states: -2.13 (value 1)
- first step: attention share on the deciding line: +1.74 (value 0.314)

### v2_cert_6760e16fbcfd9499 — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 1 to 0. Answer: `{"path": [1, 0]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 0 — 29% on `1:` line (looked most at line `0:`) — legal

Weakest-grounded step 1: 1 → 0. While writing `0` the model put **29%** of its graph attention on the line that decides this step, `1: [0, 3]`; it looked most at `0: [2, 3]` (36%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ███████············· 36.3%
  1: ██████·············· 29.1%  <- deciding line
  2: ███················· 13.6%
  4: ███················· 12.9%
  3: ██··················  8.1%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -6.49 (value 0.539)
- average attention share on the deciding line: -2.52 (value 0.291)
- answer starts/ends at the right states: -2.13 (value 1)
- first step: attention share on the deciding line: +1.50 (value 0.291)

### v2_cert_fb40185f67e0e37b — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 0 to 1. Answer: `{"path": [0, 1]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 0 → 1 — 32% on `0:` line — legal

Weakest-grounded step 1: 0 → 1. While writing `1` the model put **32%** of its graph attention on the line that decides this step, `0: [1]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ██████·············· 32.3%  <- deciding line
  3: ██████·············· 31.1%
  1: ███················· 14.8%
  2: ██·················· 12.0%
  4: ██··················  9.7%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- layer 075% depth: min share on deciding line: -5.03 (value 0.44)
- average attention share on the deciding line: -3.02 (value 0.323)
- answer starts/ends at the right states: -2.13 (value 1)
- layer 025% depth: min share on deciding line: +1.90 (value 0.329)

## Failures let through (ALLOW, verifier: invalid) — 0 cases

## Valid answers rejected (DISALLOW, verifier: valid) — 256 cases

### v2_cert_e7c4a8c58917fa6e — route **DISALLOW**, predicted failure risk 0.69 (verifier, offline: **valid**)
Task: path from 14 to 12. Answer: `{"path": [14, 12]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 14 → 12 — 12% on `14:` line — legal

Weakest-grounded step 1: 14 → 12. While writing `12` the model put **12%** of its graph attention on the line that decides this step, `14: [1, 2, 3, 6, 7, 11, 12, 13, 15]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
 14: ██·················· 12.0%  <- deciding line
 16: ██··················  8.6%
  0: ██··················  8.3%
 12: ██··················  7.7%
 19: █···················  7.0%
  2: █···················  6.7%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -2.13 (value 1)
- answer path length: -1.39 (value 2)
- layer 075% depth: min share on deciding line: -1.11 (value 0.175)
- layer 100% depth: fraction of steps deciding line is top: +1.11 (value 1)

### v2_cert_8fe7be76e86ff7a8 — route **DISALLOW**, predicted failure risk 0.60 (verifier, offline: **valid**)
Task: path from 6 to 9. Answer: `{"path": [6, 2, 7, 9]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 6 → 2 — 10% on `6:` line (looked most at line `0:`) — legal
- step 2: 2 → 7 — 16% on `2:` line — legal
- step 3: 7 → 9 — 20% on `7:` line — legal

Weakest-grounded step 1: 6 → 2. While writing `2` the model put **10%** of its graph attention on the line that decides this step, `6: [2, 4, 8]`; it looked most at `0: [5, 6, 7]` (14%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ███················· 13.7%
  7: ███················· 13.0%
  5: ███················· 12.6%
  2: ██·················· 12.3%
  4: ██·················· 11.7%
  8: ██·················· 10.0%
  6: ██··················  9.9%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -2.13 (value 1)
- layer 075% depth: min share on deciding line: -0.79 (value 0.154)
- answer path length: -0.52 (value 4)
- first step: attention share on the deciding line: -0.46 (value 0.0994)

### v2_cert_f22d42f8e2499610 — route **DISALLOW**, predicted failure risk 0.99 (verifier, offline: **valid**)
Task: path from 13 to 2. Answer: `{"path": [13, 10, 5, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 13 → 10 — 4% on `13:` line (looked most at line `2:`) — legal
- step 2: 10 → 5 — 7% on `10:` line (looked most at line `2:`) — legal
- step 3: 5 → 2 — 5% on `5:` line (looked most at line `2:`) — legal

Weakest-grounded step 1: 13 → 10. While writing `10` the model put **4%** of its graph attention on the line that decides this step, `13: [6, 9, 10]`; it looked most at `2: [0, 6, 10, 13, 16]` (9%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  2: ██··················  8.6%
 18: ██··················  8.5%
 16: █···················  7.4%
 12: █···················  7.1%
 19: █···················  6.7%
 10: █···················  6.6%
 13: █···················  3.9%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -2.13 (value 1)
- average attention share on the deciding line: +1.10 (value 0.0541)
- first step: attention share on the deciding line: -1.07 (value 0.0393)
- layer 075% depth: min share on deciding line: +0.79 (value 0.0468)

### v2_cert_e9efd827d1c585e7 — route **DISALLOW**, predicted failure risk 0.55 (verifier, offline: **valid**)
Task: path from 8 to 9. Answer: `{"path": [8, 4, 7, 9]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 8 → 4 — 13% on `8:` line (looked most at line `9:`) — legal
- step 2: 4 → 7 — 12% on `4:` line (looked most at line `8:`) — legal
- step 3: 7 → 9 — 15% on `7:` line (looked most at line `9:`) — legal

Weakest-grounded step 2: 4 → 7. While writing `7` the model put **12%** of its graph attention on the line that decides this step, `4: [6, 7, 9]`; it looked most at `8: [1, 2, 4, 7, 9]` (16%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  8: ███················· 15.7%
  9: ███················· 13.4%
  0: ██·················· 12.3%
  4: ██·················· 11.9%  <- deciding line
  7: ██··················  9.9%
  1: ██··················  8.7%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: -2.13 (value 1)
- layer 075% depth: min share on deciding line: -0.68 (value 0.146)
- shortest path length: -0.56 (value 2)
- answer path length: -0.52 (value 4)
