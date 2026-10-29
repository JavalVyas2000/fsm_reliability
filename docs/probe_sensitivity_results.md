# Results: probe-class and validator-derived-observables sensitivity (paper item 3a)

**Pre-registration:** `docs/probe_sensitivity_prereg.md` (commit 2703c9f, before any fit).
**Status:** sensitivity analysis on certification sets that had been examined once before; not confirmatory.
**Runs:** `outputs/sensitivity/20261029_104001_probe_class` (path planning) and
`outputs/sensitivity/20261029_104322_probe_class` (CSTR), `scripts/51_probe_sensitivity.py`.
**Sanity check:** the logistic arm reproduces all 60 confirmatory certification values exactly (`sanity.csv`).

All numbers: calls skipped at X = 10%, certification set. Δ: paired bootstrap 95% CI. * = certified
(Bonferroni over the 12 path-planning / 6 CSTR screens of this analysis).

## Path planning

Calls skipped (%):

| Signal | Probe | Qwen-1.5B | Qwen-3B | Qwen-7B | Llama-3B | SmolLM-1.7B |
|---|---|---|---|---|---|---|
| Observables | logistic | 40.4* | 20.5* | 15.7* | 43.0 | 84.6 |
| | gbm | 44.8* | 27.1* | 15.8* | 49.5 | 94.0* |
| Validator-free observables | logistic | 33.4* | 18.2* | 11.7* | 22.0 | 71.1 |
| | gbm | 32.3 | 12.5* | 8.9* | 13.2 | 72.4 |
| Internals only | logistic | 38.0* | 22.6* | 10.8* | 35.8* | 78.8 |
| | gbm | 47.1* | 30.0 | 11.9* | 45.9* | 86.7 |
| Combined | logistic | 39.9* | 34.5* | 15.2* | 58.1* | 90.7 |
| | gbm | 47.8* | 32.9* | 13.3* | 65.5* | 96.3 |
| Validator-free combined | logistic | 32.1* | 25.1* | 12.8* | 33.6* | 80.9 |
| | gbm | 42.9* | 27.5 | 25.0* | 49.8* | 87.0 |

- **Q1 / E1 (met).** With a flexible probe for every signal set, internals still add significantly for 4 of 5
  models (C1, gbm): Llama +15.9 [14.1, 17.8], Qwen-3B +5.8 [4.3, 7.3], Qwen-1.5B +3.0 [1.3, 4.9],
  SmolLM +2.4 [1.4, 3.4]. Qwen-7B reverses: −2.5 [−3.7, −1.3].
  - The gains shrink for Qwen-3B (logistic +14.0) and SmolLM (+6.2) and appear for Qwen-1.5B (logistic −0.4, n.s.).
- **Q2 / E3, first part (met).** The validator-derived features carry much of the observable screen.
  - Removing `shortest_length` and `answer_endpoints_match` costs 2.3–21.1 points (logistic) and 6.9–36.3 points
    (gbm), with all CIs below 0.
- **E3, second part (partly met).** Against validator-free observables, internals add +10.6 to +36.6 points for
  all 5 models under gbm.
  - Under logistic they add +1.1 to +11.6 for 4 of 5 (Qwen-1.5B −1.3, n.s.).
  - The added value exceeds the full-observables C1 for all 5 under gbm, but for only 2 of 5 under logistic.
- **Internals alone vs validator-free observables (gbm):** internals skip more calls for every model:
  47.1 vs 32.3, 30.0 vs 12.5, 11.9 vs 8.9, 45.9 vs 13.2, 86.7 vs 72.4.

## CSTR

Calls skipped (%), with failures/accepts:

| Signal | Probe | Qwen-1.5B | Qwen-3B† | Qwen-7B | Llama-3B | SmolLM-1.7B |
|---|---|---|---|---|---|---|
| Observables | logistic | 62.8 (14/121) | 41.2 (0/0) | 85.5 (2/54) | 70.7 (0/0) | 61.1 (14/128) |
| | gbm | 71.7 (17/135) | 81.5 (12/122) | 87.3 (3/54) | 89.5 (1/31) | 69.4 (15/133) |
| Internals only | logistic | 28.3 (0/0) | 12.2 (0/0) | 47.8 (0/0) | 52.7 (0/0) | 40.5 (2/39) |
| | gbm | 27.6 (0/0) | 22.5 (0/0) | 49.4 (0/0) | 54.2 (0/0) | 43.5 (4/42) |
| Combined | logistic | 60.6 (14/111) | 49.8 (5/43) | 79.9 (2/35) | 74.1 (0/0) | 65.7 (11/109) |
| | gbm | 70.9 (15/125) | 73.7 (9/105) | 84.1 (2/40) | 83.3 (0/0) | 67.5 (12/119) |

† exploratory. No CSTR screen is certified (power limit, as in the confirmatory analysis).

- **Q1 / E2 (met).** A flexible probe makes the CSTR observable screen much stronger: +1.8 to +40.4 points.
  - Under gbm, adding internals never helps. It reduces the calls skipped for Llama (−6.1 [−9.1, −3.4]),
    Qwen-3B (−7.8) and Qwen-7B (−3.2 [−6.0, −0.2]); Qwen-1.5B and SmolLM are n.s.
  - The two confirmatory CSTR gains (SmolLM, Llama, logistic) therefore do **not** survive a flexible probe.
    They should be presented as specific to the linear probe.
- **Internals only:** unchanged in character under gbm. It skips 22.5–54.2% almost entirely by rejection
  (SmolLM: 4 failures among 42 accepts).
- **Caveat:** the flexible observable screens accepted more, and their observed failure rates among accepts reached
  12.6% (Qwen-1.5B) and 11.3% (SmolLM). The certificate would refuse them.

## Implications for the paper

1. **Path planning.** The complementarity claim survives a flexible probe (4 of 5 models). It becomes much stronger
   when the baseline is restricted to information available without running part of the validator. Then the
   agent's internals alone beat the observables for every model.
2. **CSTR.** The observable screen is a learned surrogate of the plant response, and a flexible surrogate absorbs
   everything the internals added. The internals' role in the CSTR is plant-model-free rejection, not
   complementarity. Claim C3 must be narrowed accordingly.
3. Both results fit the mechanism (RQ3): internals add where errors are retrieval errors, not where the
   outcome is a function of the plant state and the action.
