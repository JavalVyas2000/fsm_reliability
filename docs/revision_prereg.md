# Pre-registration: revision analyses for the EAAI manuscript (items 1, 2, 4, 7)

**Frozen:** 2026-10-29, before any episode of the new data sets below was drawn and before any analysis in
this document was computed. The author authorised running these analyses unattended.

All probes, thresholds and rules used here are the **existing frozen ones**
(`outputs/certification/*_cstr_v4_*_skip_ucb*`, `outputs/certification/*_fsm_*_skip_ucb_cert2`). Nothing is
refitted, except in item 4, whose purpose is to refit.

---

## Item 2: the first-proposal-only rule (R0) on fresh closed-loop episodes

**Why.** R0 (closed-loop prereg, Amendment 4) was designed after the retry failure was seen on
`data/cstr/closedloop_v4`, and its effect was measured on the same 400 episodes. This item tests it on new
episodes.

**Data.** 400 new episodes, 100 per fault family.
- Specs: `scripts/33 --ranges configs/cstr_severity_ranges_v4.json --sizes closedloop2=400 --seed 20261029`
  → `data/cstr/closedloop2_v4_specs.xlsx`.
- Build: `scripts/35 --splits closedloop2 --replace_seed 20261030` → `data/cstr/closedloop2_v4`.
- KG file unchanged.

**Models.** The two models whose internals-based probes accepted failing retries: SmolLM2-1.7B and Qwen2.5-7B
(4-bit). Each uses its frozen CSTR probes.

**Policies (`scripts/42`).** `always_validate`, `combined_probe`, `internals_probe`, `observables_probe_r0`,
`combined_probe_r0`, `internals_probe_r0`. All other settings are as in the original closed loop.

**Primary criteria, per model, for each of `combined_probe_r0` and `internals_probe_r0`:**
- P2a (safety): the one-sided 95% Clopper-Pearson upper bound on the share of episodes with a failing action
  executed unchecked is <= 5%.
- P2b (recovery, non-inferiority): the lower end of the paired bootstrap 95% CI (episodes, 2000 resamples) of
  recovered(R0 policy) − recovered(always_validate) is > −5 points.

**Secondary.**
- S2a: the non-R0 policy of the same probe executes more failing actions than its R0 variant (paired by
  episode), i.e. the retry failure replicates.
- S2b: calls avoided.

**Reporting.** All outcomes, whatever they are, in the paper's closed-loop section (replacing the claim that R0
was tested only on the episodes that motivated it).

---

## Item 2b: validating rejections instead of discarding them (registered 2026-10-30, before any run)

**Why.** Item 2 (results in `paper/claims.md` C23, C24) passed P2a for every R0 screen and P2b for SmolLM2, but
failed P2b for Qwen2.5-7B (combined −3.5 [−6.2, −1.0], internals only −3.5 [−5.8, −1.5]). Every lost recovery
contained an unchecked rejection, and in most of them always-validate had recovered only after a failed
validation. Hypothesis: an unchecked rejection withholds the validator's measured feedback, which an agent that
repairs from feedback needs. The paper's deployment guide now recommends validating rejected proposals for such
agents; this item tests that recommendation.

**Policy (`scripts/42`, new).** `<screen>_r0v`: as `<screen>_r0` (retries are never accepted unchecked), and a
REJECT decision is validated instead of discarded. Only first-proposal accepts skip the validator. Frozen probes
and thresholds unchanged.

**Data and models.** The same 400 `closedloop2_v4` episodes as item 2. Qwen2.5-7B (4-bit) (primary) and
SmolLM2-1.7B (secondary). Policies: `always_validate` (re-run as a determinism check against item 2),
`observables_probe_r0v`, `combined_probe_r0v`, `internals_probe_r0v`.

**Primary criteria (Qwen2.5-7B; for each of `combined_probe_r0v` and `internals_probe_r0v`):**
- P2c (recovery, non-inferiority): lower end of the paired bootstrap 95% CI (episodes, 2000 resamples) of
  recovered(r0v) − recovered(always_validate) > −5 points.
- P2d (safety): one-sided 95% Clopper–Pearson upper bound on the share of episodes with an unchecked failing
  action ≤ 5%.

**Secondary.**
- S2c: recovery of `r0v` minus recovery of `r0` (item 2), paired by episode, with CI; expected ≥ 0.
- S2d: calls avoided relative to always-validate, and the compute saving at 60 s per call (scripts/47 re-costing).
  Expected smaller than under `r0`, because rejections no longer avoid calls.
- S2e: SmolLM2, same quantities; expected little recovery change (it rarely repairs from feedback).
- Determinism: `always_validate` outcomes identical to item 2 per episode; any mismatch is reported.

**Reporting.** All outcomes, whatever they are, in the closed-loop section, the deployment-guide rule (iv) and
the ledger. If P2c fails, rule (iv) is reworded or withdrawn.

---

## Item 1: a larger, fresh CSTR certification set (cert2)

**Why.** With 500 certification episodes no CSTR screen could be certified at X = 10% (power limit,
Eq. 5 of the paper).

**Data.** 2000 new episodes, 500 per fault family.
- Specs: `scripts/33 --sizes cert2=2000 --seed 20261031` → `data/cstr/cert2_v4_specs.xlsx`.
- Build: `scripts/35 --splits cert2 --replace_seed 20261101` → `data/cstr/cert2_v4`. Ranges and KG unchanged.
- The set is disjoint by construction from every earlier set (new parameter and noise draws).

**Per model (all five CSTR models):**
- First proposals collected with exactly the model's original collection arguments (`--partitions cert2`).
- Grounding computed by `scripts/31 --partitions cert2`.
- The model's **frozen** CSTR rules evaluated once by `scripts/40 --partition cert2`.

**Endpoints (as in the confirmatory analysis).**
- Calls skipped at X = 10% (primary) and 5%.
- Certification: one-sided Clopper-Pearson, Bonferroni over the 11 signals.
- Paired bootstrap Δ vs observables (2000 resamples).

**Status.**
- For Qwen-1.5B, Qwen-7B, Llama-3B and SmolLM2 this is a replication on fresh data.
- For **Qwen2.5-3B** the frozen rules are those of the exploratory freeze. Evaluation on cert2 is its **first
  evaluation on unseen data** and makes its CSTR result confirmatory.

**Also on cert2 (sensitivity, item 3a extended).** `scripts/51` re-run with cert2 as the evaluation set for the
CSTR models. The probes are refitted deterministically on the same open partitions with the same fixed
hyperparameters, so the screens are identical to those of `docs/probe_sensitivity_results.md`.

**Expectations (not hypotheses).**
- The observable screens of at least two models certify at X = 10%.
- The internals-only screens remain rejection-dominated.

---

## Item 4: stability of the screens across development splits (CPU)

**What.** For each model and domain, the open data (train + dev_cal + dev_thr) are re-partitioned at random
**10 times** (seeds 1-10) into the original sizes, stratified by graph size (path planning) or fault family (CSTR).
- On each split: refit the linear probe, Platt-calibrate, and rethreshold with the confirmatory rules.
- Evaluate once on the model's certification set: path-planning cert2; CSTR cert, and cert2 once available.

**Signal sets:** observables, internals only, combined.

**Report:** mean, SD, min-max of calls skipped at X = 10%, and in how many of the 10 splits the screen certifies.
Descriptive only; no hypothesis.

---

## Item 7: teacher-forcing audit and grounding-region sensitivity (CPU)

**7a.** Collect the teacher-forced argmax agreement recorded by every grounding run used in the paper (`run_manifest.json`, `tf_argmax_agreement`) for all models and domains, and report it in the paper.

**7b. CSTR region sensitivity.** The saved grounding features contain, per setpoint, the shares of four region
definitions:
- the pre-registered deciding regions (`g31_<sp>_share`);
- the physics section (`_physics_share`);
- the "how each setpoint acts" section (`_acts_share`);
- the whole snapshot block (`_snap_share`).

For each definition, report the AUROC of the negative minimum over setpoints for a failing first proposal (train + dev). The paper's claim that CSTR grounding is weak survives if **no** definition reaches AUROC >= 0.70 for the three models whose primary AUROC is near 0.5 (Qwen-3B, Qwen-7B, Llama-3B).
