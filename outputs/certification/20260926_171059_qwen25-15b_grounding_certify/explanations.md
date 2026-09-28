# Interpretable routing decisions — G1_context_grounding (outputs\fsm_baseline\20260926_155839_grounding_qwen25-15b-instruct_pilot3000)

Verifier verdicts are shown as offline truth only; they are never used by the policy.

## Failures caught (DISALLOW, verifier: invalid) — 1578 cases

### v2_cert_ee4af04aff38a5aa — route **DISALLOW**, predicted failure risk 1.00 (verifier, offline: **invalid**)
Task: path from 12 to 1. Answer: `{"path": [12, 7, 4, 0, 1, 3, 5, 6, 9, 13, 14]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 7 — 7% on `12:` line (looked most at line `6:`) — legal
- step 2: 7 → 4 — 21% on `7:` line — legal
- step 3: 4 → 0 — 23% on `4:` line — **ILLEGAL**
- step 4: 0 → 1 — 21% on `0:` line — legal
- step 5: 1 → 3 — 4% on `1:` line (looked most at line `0:`) — **ILLEGAL**
- step 6: 3 → 5 — 4% on `3:` line (looked most at line `4:`) — **ILLEGAL**
- step 7: 5 → 6 — 11% on `5:` line (looked most at line `4:`) — **ILLEGAL**
- step 8: 6 → 9 — 8% on `6:` line (looked most at line `4:`) — **ILLEGAL**
- step 9: 9 → 13 — 11% on `9:` line (looked most at line `0:`) — legal
- step 10: 13 → 14 — 16% on `13:` line (looked most at line `0:`) — **ILLEGAL**
- the answer does not start at 12 and end at 1 (visible without the verifier)

Weakest-grounded step 5: 1 → 3. While writing `3` the model put **4%** of its graph attention on the line that decides this step, `1: [6, 8, 11]`; it looked most at `0: [1, 4, 6, 9, 12, 13]` (17%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
  0: ███················· 17.2%
 10: ███················· 14.3%
 14: ██·················· 10.7%
  4: ██··················  8.8%
 13: █···················  7.3%
  5: █···················  6.2%
  1: █···················  3.8%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +5.35 (value 0)
- answer path length: +2.42 (value 11)
- weakest step: attention share on the deciding line: +0.71 (value 0.038)
- layer 050% depth: min share on deciding line: -0.51 (value 0.0113)

### v2_cert_e7c4a8c58917fa6e — route **DISALLOW**, predicted failure risk 0.94 (verifier, offline: **invalid**)
Task: path from 14 to 12. Answer: `{"path": [14, 11, 2, 5, 17, 15, 16, 12]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 14 → 11 — 20% on `14:` line — legal
- step 2: 11 → 2 — 8% on `11:` line (looked most at line `14:`) — legal
- step 3: 2 → 5 — 17% on `2:` line — legal
- step 4: 5 → 17 — 5% on `5:` line (looked most at line `14:`) — **ILLEGAL**
- step 5: 17 → 15 — 11% on `17:` line (looked most at line `14:`) — legal
- step 6: 15 → 16 — 15% on `15:` line — legal
- step 7: 16 → 12 — 12% on `16:` line (looked most at line `15:`) — **ILLEGAL**

Weakest-grounded step 4: 5 → 17. While writing `17` the model put **5%** of its graph attention on the line that decides this step, `5: [8, 18]`; it looked most at `14: [1, 2, 3, 6, 7, 11, 12, 13, 15]` (12%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
 14: ██·················· 12.2%
 15: ██·················· 11.5%
 17: ██··················  9.5%
  0: ██··················  7.6%
  2: █···················  6.7%
 19: █···················  6.1%
  5: █···················  5.4%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: +1.24 (value 8)
- weakest step: attention share on the deciding line: +0.57 (value 0.054)
- layer 100% depth: min share on deciding line: +0.47 (value 0.0268)
- layer 050% depth: min share on deciding line: -0.39 (value 0.0264)

### v2_cert_89c166db2b33ef3f — route **DISALLOW**, predicted failure risk 0.92 (verifier, offline: **invalid**)
Task: path from 3 to 4. Answer: `{"path": [3, 4, 0]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 3 → 4 — 26% on `3:` line (looked most at line `1:`) — legal
- step 2: 4 → 0 — 32% on `4:` line — legal
- the answer does not start at 3 and end at 4 (visible without the verifier)

Weakest-grounded step 1: 3 → 4. While writing `4` the model put **26%** of its graph attention on the line that decides this step, `3: [4]`; it looked most at `1: [2, 3]` (36%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ███████············· 35.5%
  3: █████··············· 25.8%  <- deciding line
  4: ███················· 17.4%
  2: ██·················· 11.8%
  0: ██··················  9.4%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer starts/ends at the right states: +5.35 (value 0)
- weakest step: attention share on the deciding line: -1.21 (value 0.258)
- layer 100% depth: min share on deciding line: -1.14 (value 0.313)
- layer 075% depth: min share on deciding line: -0.85 (value 0.473)

### v2_cert_c900ceed43e4fcd3 — route **DISALLOW**, predicted failure risk 0.98 (verifier, offline: **invalid**)
Task: path from 12 to 8. Answer: `{"path": [12, 14, 13, 1, 2, 7, 8]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 14 — 3% on `12:` line (looked most at line `10:`) — **ILLEGAL**
- step 2: 14 → 13 — 12% on `14:` line (looked most at line `13:`) — legal
- step 3: 13 → 1 — 12% on `13:` line (looked most at line `8:`) — legal
- step 4: 1 → 2 — 5% on `1:` line (looked most at line `13:`) — **ILLEGAL**
- step 5: 2 → 7 — 6% on `2:` line (looked most at line `13:`) — **ILLEGAL**
- step 6: 7 → 8 — 5% on `7:` line (looked most at line `8:`) — **ILLEGAL**

Weakest-grounded step 1: 12 → 14. While writing `14` the model put **3%** of its graph attention on the line that decides this step, `12: [0]`; it looked most at `10: [2, 4, 6]` (13%). (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
 10: ███················· 12.8%
  2: ██·················· 12.3%
 13: ██·················· 12.2%
  3: ██··················  9.5%
 14: ██··················  8.8%
  8: ██··················  8.0%
 12: █···················  3.2%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- shortest path length: +0.96 (value 6)
- answer path length: +0.85 (value 7)
- weakest step: attention share on the deciding line: +0.77 (value 0.0319)
- first step: attention share on the deciding line: -0.55 (value 0.0319)

## Correctly trusted (ALLOW, verifier: valid) — 230 cases

### v2_cert_edb4ebbae370c576 — route **ALLOW**, predicted failure risk 0.06 (verifier, offline: **valid**)
Task: path from 2 to 3. Answer: `{"path": [2, 4, 3]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 2 → 4 — 28% on `2:` line (looked most at line `1:`) — legal
- step 2: 4 → 3 — 35% on `4:` line — legal

Weakest-grounded step 1: 2 → 4. While writing `4` the model put **28%** of its graph attention on the line that decides this step, `2: [1, 4]`; it looked most at `1: [0, 2, 4]` (30%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  1: ██████·············· 29.9%
  2: ██████·············· 27.9%  <- deciding line
  4: █████··············· 23.1%
  3: ██·················· 12.1%
  0: █···················  7.0%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -1.38 (value 0.279)
- layer 100% depth: min share on deciding line: -1.20 (value 0.326)
- layer 075% depth: min share on deciding line: -0.86 (value 0.473)
- answer path length: -0.73 (value 3)

### v2_cert_e59c73baededbf0c — route **ALLOW**, predicted failure risk 0.01 (verifier, offline: **valid**)
Task: path from 2 to 3. Answer: `{"path": [2, 3]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 2 → 3 — 48% on `2:` line — legal

Weakest-grounded step 1: 2 → 3. While writing `3` the model put **48%** of its graph attention on the line that decides this step, `2: [1, 3, 4]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  2: ██████████·········· 48.1%  <- deciding line
  3: █████··············· 25.8%
  4: ███················· 13.1%
  1: █···················  6.9%
  0: █···················  6.1%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -3.15 (value 0.481)
- layer 050% depth: min share on deciding line: +2.19 (value 0.365)
- layer 100% depth: min share on deciding line: -1.84 (value 0.44)
- layer 075% depth: min share on deciding line: -1.43 (value 0.658)

### v2_cert_3fa555692763429d — route **ALLOW**, predicted failure risk 0.02 (verifier, offline: **valid**)
Task: path from 0 to 2. Answer: `{"path": [0, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 0 → 2 — 31% on `0:` line — legal

Weakest-grounded step 1: 0 → 2. While writing `2` the model put **31%** of its graph attention on the line that decides this step, `0: [2]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ██████·············· 31.2%  <- deciding line
  1: █████··············· 24.4%
  3: █████··············· 23.7%
  2: ███················· 13.0%
  4: ██··················  7.6%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -1.67 (value 0.312)
- layer 100% depth: min share on deciding line: -1.37 (value 0.356)
- answer path length: -1.12 (value 2)
- answer is a single hop: -1.07 (value 1)

### v2_cert_8bd1e8f33f210196 — route **ALLOW**, predicted failure risk 0.00 (verifier, offline: **valid**)
Task: path from 4 to 2. Answer: `{"path": [4, 2]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 4 → 2 — 37% on `4:` line — legal

Weakest-grounded step 1: 4 → 2. While writing `2` the model put **37%** of its graph attention on the line that decides this step, `4: [0, 2]`. (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  4: ███████············· 36.7%  <- deciding line
  0: █████··············· 26.2%
  2: ███················· 17.4%
  1: ██·················· 12.3%
  3: █···················  7.3%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: -2.16 (value 0.367)
- layer 075% depth: min share on deciding line: -1.74 (value 0.758)
- layer 100% depth: min share on deciding line: -1.62 (value 0.399)
- answer path length: -1.12 (value 2)

## Failures let through (ALLOW, verifier: invalid) — 1 cases

### v2_cert_1704bfeb8f5593aa — route **ALLOW**, predicted failure risk 0.04 (verifier, offline: **invalid**)
Task: path from 12 to 3. Answer: `{"path": [12, 3]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 12 → 3 — 17% on `12:` line — **ILLEGAL**

Weakest-grounded step 1: 12 → 3. While writing `3` the model put **17%** of its graph attention on the line that decides this step, `12: [1, 2, 8, 18]`. (Offline truth: this step is illegal.)

Attention over graph lines for this step:
```
 12: ███················· 17.0%  <- deciding line
 19: ██·················· 11.8%
  0: ██··················  8.2%
  2: █···················  6.2%
  1: █···················  6.0%
 16: █···················  5.9%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: -1.12 (value 2)
- answer is a single hop: -1.07 (value 1)
- layer 075% depth: min share on deciding line: -0.73 (value 0.431)
- layer 100% depth: fraction of steps deciding line is top: -0.47 (value 1)

## Valid answers rejected (DISALLOW, verifier: valid) — 141 cases

### v2_cert_a2a078babb82f774 — route **DISALLOW**, predicted failure risk 0.88 (verifier, offline: **valid**)
Task: path from 8 to 16. Answer: `{"path": [8, 12, 11, 18, 16]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 8 → 12 — 18% on `8:` line — legal
- step 2: 12 → 11 — 4% on `12:` line (looked most at line `19:`) — legal
- step 3: 11 → 18 — 17% on `11:` line — legal
- step 4: 18 → 16 — 4% on `18:` line (looked most at line `19:`) — legal

Weakest-grounded step 2: 12 → 11. While writing `11` the model put **4%** of its graph attention on the line that decides this step, `12: [1, 11, 14, 15]`; it looked most at `19: [0, 2, 6, 10, 16]` (11%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
 19: ██·················· 11.2%
 11: ██·················· 10.2%
  8: █···················  7.5%
  2: █···················  7.4%
  0: █···················  6.7%
 13: █···················  5.7%
 12: █···················  3.9%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: +0.70 (value 0.0392)
- layer 075% depth: min share on deciding line: +0.53 (value 0.0233)
- weakest step: best single head's share on the deciding line: -0.42 (value 0.0617)
- layer 100% depth: min share on deciding line: +0.37 (value 0.0434)

### v2_cert_18618149a36ae2a5 — route **DISALLOW**, predicted failure risk 0.85 (verifier, offline: **valid**)
Task: path from 1 to 0. Answer: `{"path": [1, 2, 6, 9, 5, 0]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 2 — 16% on `1:` line — legal
- step 2: 2 → 6 — 16% on `2:` line — legal
- step 3: 6 → 9 — 9% on `6:` line (looked most at line `18:`) — legal
- step 4: 9 → 5 — 6% on `9:` line (looked most at line `0:`) — legal
- step 5: 5 → 0 — 19% on `5:` line — legal

Weakest-grounded step 4: 9 → 5. While writing `5` the model put **6%** of its graph attention on the line that decides this step, `9: [5, 7, 11, 17]`; it looked most at `0: [5, 6, 9]` (14%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ███················· 13.8%
  5: ██·················· 10.7%
 18: ██··················  9.3%
  6: █···················  7.0%
  4: █···················  6.4%
  9: █···················  6.2%  <- deciding line
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- weakest step: attention share on the deciding line: +0.51 (value 0.0616)
- answer path length: +0.45 (value 6)
- layer 050% depth: min share on deciding line: -0.43 (value 0.0217)
- answer starts/ends at the right states: -0.36 (value 1)

### v2_cert_a165d42df1d34df7 — route **DISALLOW**, predicted failure risk 0.92 (verifier, offline: **valid**)
Task: path from 13 to 4. Answer: `{"path": [13, 17, 11, 9, 16, 19, 4]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 13 → 17 — 6% on `13:` line (looked most at line `0:`) — legal
- step 2: 17 → 11 — 8% on `17:` line (looked most at line `15:`) — legal
- step 3: 11 → 9 — 8% on `11:` line (looked most at line `0:`) — legal
- step 4: 9 → 16 — 8% on `9:` line (looked most at line `19:`) — legal
- step 5: 16 → 19 — 9% on `16:` line (looked most at line `0:`) — legal
- step 6: 19 → 4 — 20% on `19:` line — legal

Weakest-grounded step 1: 13 → 17. While writing `17` the model put **6%** of its graph attention on the line that decides this step, `13: [12, 17]`; it looked most at `0: [3, 6, 8, 10, 13, 18, 19]` (15%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  0: ███················· 15.3%
  3: ██·················· 11.2%
 19: █···················  6.8%
 11: █···················  6.7%
 13: █···················  5.6%  <- deciding line
 10: █···················  5.1%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- answer path length: +0.85 (value 7)
- weakest step: attention share on the deciding line: +0.56 (value 0.0557)
- first step: attention share on the deciding line: -0.48 (value 0.0557)
- layer 075% depth: min share on deciding line: +0.36 (value 0.0787)

### v2_cert_21df9b3cd430c3d2 — route **DISALLOW**, predicted failure risk 0.85 (verifier, offline: **valid**)
Task: path from 1 to 7. Answer: `{"path": [1, 8, 5, 4, 7]}`

**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):
- step 1: 1 → 8 — 11% on `1:` line (looked most at line `3:`) — legal
- step 2: 8 → 5 — 12% on `8:` line (looked most at line `3:`) — legal
- step 3: 5 → 4 — 18% on `5:` line — legal
- step 4: 4 → 7 — 22% on `4:` line — legal

Weakest-grounded step 1: 1 → 8. While writing `8` the model put **11%** of its graph attention on the line that decides this step, `1: [8]`; it looked most at `3: [1, 4, 5, 6, 8]` (19%). (Offline truth: this step is legal.)

Attention over graph lines for this step:
```
  3: ████················ 18.5%
  2: ███················· 16.0%
  8: ███················· 12.7%
  1: ██·················· 10.7%  <- deciding line
  5: ██··················  8.4%
  4: ██··················  8.1%
```
**Why the probe scored this risk** (contribution to the risk logit; + raises risk):
- shortest path length: +0.64 (value 5)
- answer starts/ends at the right states: -0.36 (value 1)
- first step: attention share on the deciding line: -0.33 (value 0.107)
- layer 100% depth: fraction of steps deciding line is top: +0.24 (value 0)
