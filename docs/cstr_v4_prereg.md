# Pre-registration: do model internals tell us when to accept a CSTR setpoint proposal?

**Frozen:** 2026-09-28. This is before any first proposal of the v4 train/val/test episodes was
generated, and before any internal feature was computed on them. The decisions left open in the
draft were resolved as proposed there and approved by the author.

## Question

When Qwen2.5-3B proposes setpoints for a faulty CSTR, can signals from inside the model tell us
whether to accept the proposal without running the verifier, beyond what the observable plant
readings and the proposed action already tell us?

The model's task success is not the object of study. It only matters because training and
certifying an ACCEPT decision needs both outcomes in sufficient numbers.

## Design choices (and why)

**Prompt v3.1** (`src/cstr/prompt_v3.py`):
- It gives no advice on which setpoint to change, in which direction, or by how much.
- It states plant physics as symmetric facts restating the KG equations.
- Retry feedback is the verifier's measured outcome only.
- Rationale: failures reflect the model's own reasoning, not prompt wording.

**Severity ranges v4** (`configs/cstr_severity_ranges_v4.json`), calibrated without the LLM against the model's empirical v3.1 moves (`outputs/cstr_calibration/20260928_142020_v31_moves`):
- The model's moves pass almost only where doing nothing also passes.
- Severity therefore sets how often the model passes, not whether it can fix a fault.
- The ranges target roughly 25–40% passing first proposals, so that an ACCEPT set is certifiable.
- Every fault is fixable: a strong feed cut passes almost always.

**What the outcome depends on.** A proposal passes when two conditions hold:
- the fault is mild enough for a small change to suffice (largely visible in the plant readings);
- the proposed change does not itself break a plant that would otherwise recover (determined by the model's action).

The second case is where internals can add information beyond the plant readings (H2).

## Pilot (100 separate episodes, `data/cstr/pilot_v4`; not part of any partition below)

Run: `outputs/cstr_collect/20260928_153222_qwen25-3b-instruct_v4ranges_v31_pilot100`.

| Quantity | Value |
|---|---|
| First-proposal pass rate | **31.3%** (stopping rule [20%, 50%] met) |
| Format failures | 1 / 100 |
| No-change passes | 37% of episodes |
| Model passes where no-change fails (n = 62) | 11%. These are real fixes, including 5/25 fouling episodes. |
| Model passes where no-change passes (n = 37) | 65% |
| Fin_sp raised while u_cool ≥ 0.95 and T_meas > 310 K | 11 proposals, 0 passed |

Pass rate by family:

| Family | Pass rate |
|---|---|
| pump_degrade | 48% |
| outlet_block | 46% |
| fouling | 20% |
| cool_stuck_closed | 12% |

## Data

**Episodes:** `data/cstr/cstr_episode_specs_v4.xlsx` (scripts/33, seed 20261004), simulated by
scripts/35 into `data/cstr/episodes_v4`.
- Balanced over four fault families.
- Each episode has a varied nominal plant, onset 1500–5000 s, and continuous severity.
- A slot that never triggers, or triggers before the fault onset, is redrawn within the same family.

**Partitions** (scripts/36; disjoint by episode):

| Spreadsheet split | Partition | Episodes | Use |
|---|---|---|---|
| train | train | 3000 | fitting |
| val | dev_cal | 500 | probe calibration |
| val | dev_thr | 500 | routing thresholds and all dev checks |
| test | test_iid | 500 | sealed; final predictive metrics (H1–H3), examined once |
| test | cert | 500 | sealed; certification (H4), used once after the policies are frozen |

Within each spreadsheet split and family, episodes are taken in slot order and alternate between the two halves (the 1st to the first half, the 2nd to the second, ...). Each half therefore has exactly 125 episodes per family.

**Unit of independence:** the episode. All bootstrap CIs resample episodes.

**Model and decoding:** Qwen2.5-3B-Instruct, bf16, sdpa attention, greedy, max 512 new tokens, prompt v3.1. Model revision, chat template and KG hash are recorded in `run_manifest.json`.

**Candidates:**
- Primary: the first proposal (round 0), the actual "accept this change?" decision.
- Secondary: retries (up to 5; v3.1 neutral feedback; an exact repeat ends the episode), analysed separately and grouped by episode, if collected.

**Label:** `verifier_pass` from the frozen CAR rollout validator (`VERIFIER_KNOBS` in `src/cstr/episodes.py`).
- Format failures are counted as "must verify" in routing and can never be accepted.
- Format failures are excluded from AUROC. Their rate is reported.

**Offline-only quantity:** `no_change_pass` (the verifier on the unchanged setpoints; `offline_nochange_pass`). It is never a feature. It is used only to define the H2 strata.

**Never features:** family, severity, onset, and plant parameters that are not in the prompt.

## Feature families

All are computed from the stored answer of the same forward pass.

| Family | Content |
|---|---|
| `context_action` (baseline) | observable snapshot fields + proposed setpoint changes (`src/cstr/llm_io.py`) |
| `token_confidence` | answer-token log-probabilities, entropies, margins (`tok_*`) |
| `attention` | attention mass by prompt region: system instructions, KG, snapshot, user instructions (`att_*`) |
| `hidden` | hidden states at relative depths 0.25 / 0.5 / 0.75 / 1.0, PCA to 64 on train |
| `grounding_v31` | below (`g31_*`) |

`all_internal` = token_confidence + attention + hidden.

**Grounding for prompt v3.1** (`g31_*`). This extends `docs/grounding_cstr_prereg.md` to the v3.1 prompt. Teacher-forced re-run of the stored answer tokens (no regeneration).
- **Decision rows:** the answer tokens of each setpoint number.
- **Attention:** head-mean, at relative depths 0.25, 0.5, 0.75 and 1.0.
- **Denominator:** all system-prompt tokens (instructions + KG) and snapshot tokens. Template and generated tokens are excluded.

Regions (character spans of the rendered prompt; a token belongs to a region on any overlap):
- `field:<name>`: each `- <name>:` line of the CURRENT SNAPSHOT block;
- `sec:acts`: the v3.1 section "# How each setpoint acts (KG)", header to the next header;
- `sec:physics`: the v3.1 section "# Plant physics (KG, PO5 dT/dt)", header to the next header;
- `kg:PO5`, `kg:PO6`, `kg:PO7`, `kg:PO8`: lines of the KG Turtle block (inside the ```turtle fence only) that mention that operator, where `kg:PO8` also includes PO1 and `kg:PO5` also includes `dT/dt` and `dV/dt`.

Decisive regions per setpoint:

| Setpoint | Decisive regions |
|---|---|
| Fin_sp | field:u_cool, field:T_meas, field:current setpoints, kg:PO8, kg:PO5 |
| L_sp | field:L_meas, field:u_pump, field:current setpoints, kg:PO6, kg:PO5 |
| T_sp | field:T_meas, field:u_cool, field:current setpoints, kg:PO7, kg:PO5 |

Features (each averaged over the four layers):
- `g31_<sp>_share`: decisive-region share, for sp in Fin, L, T;
- `g31_min_share`: the minimum of the three;
- `g31_<sp>_physics_share`, `g31_<sp>_acts_share`, `g31_<sp>_snap_share`.

## Hypotheses (directions fixed)

- **H1 (primary).** Adding internals to the observable probe improves failure prediction on first proposals.
  - Test: paired ΔAUROC(context_action + all_internal + grounding_v31 − context_action) on test_iid.
  - Uncertainty: grouped bootstrap 95% CI.
  - Supported if the CI lower bound is above 0.
  - Computed on dev_thr first and written down; test_iid examined once.
  - Secondary comparisons, same test: context_action + token_confidence; + hidden; + grounding_v31; + all_internal.
- **H2 (mechanism).** The same ΔAUROC is reported within each stratum of `no_change_pass`. Prediction: the gain is larger where no-change passes, because the outcome there depends on whether the model's own change breaks the plant.
- **H3 (grounding).** Among failing first proposals, those that raise Fin_sp while u_cool ≥ 0.95 and T_meas > 310 K have a lower `g31_Fin_physics_share` than other proposals.
  - Test: AUROC of −g31_Fin_physics_share for this proposal type against all other proposals.
  - Checked on train + dev, then on test_iid once.
- **H4 (routing and certification).**
  - Policies, frozen on dev before cert is touched:
    - P1 context_action;
    - P2 context_action + all_internal + grounding_v31;
    - P3 context_action + grounding_v31;
    - P4 all_internal + grounding_v31.
  - Rule: ACCEPT when calibrated risk is at or below the largest threshold with 0 accepted failures on dev_thr (α = 0).
  - On cert, certify that the failure rate among ACCEPTs is at most 5% at 95% confidence (Clopper–Pearson), Bonferroni over the 4 policies (δ = 0.0125 each).
  - Report ACCEPT counts, failures accepted, and verifier calls saved.

**Probes.** Logistic regression on train (median imputation, scaling, PCA-64 for hidden states), Platt calibration on dev_cal, as in `scripts/12`. One bounded nonlinear baseline may be added on dev only; it is labelled as such.

## Amendments

**Amendment 1 (2026-09-28).** Made the same day, before any v4 first proposal was generated or any feature computed on v4 data.
- **Change:** a `kg:<PO>` region is the whole Turtle statement that mentions the operator, from its first line to the line ending in ` .`, instead of only the matching line.
- **Reason:** a smoke test on 5 pilot proposals showed that the line rule missed the `rdfs:label` line holding the operator's equation, which is the most decisive text.
- **Implementation:** `_statement_span` in `src/features/grounding_cstr.py`.

**Amendment 2 (2026-09-29): primary endpoint = validator calls skipped.**

- **Timing:** written after the development results (`docs/cstr_v4_results.md` Part A) and before test_iid or cert were examined. At that point:
  - no test_iid metric or prediction had been computed (the probe run used `--seal_test`);
  - no cert grounding feature had been computed.
- **Reason:** the question of the study is how many validator calls each signal lets us skip, and how safely. That was the research plan's primary routing endpoint. AUROC (H1–H3) is kept as secondary.

**Signals** (one probe each, fitted as in `scripts/12`: train fit, Platt calibration on dev_cal):
- plant readings only (context_only_no_action);
- plant readings + proposed change (context_action, the baseline);
- token_confidence;
- attention;
- grounding_v31;
- hidden;
- all_internal;
- all_internal + grounding_v31;
- context_action + all_internal + grounding_v31.

**Rules** (thresholds on the Platt risk, chosen on dev_cal + dev_thr pooled, frozen in `scripts/39`):
- **ACCEPT** (execute without validation): risk ≤ t_acc. t_acc is the largest threshold whose accepted dev set has a failure rate ≤ X, with **X = 10% (primary)** and X = 5% (secondary).
- **REJECT** (send back to the model without validation): risk ≥ t_rej. t_rej is the smallest threshold whose rejected dev set is ≥ 95% failures.
- **Otherwise:** validate.

**Outcomes on test_iid** (examined once; `scripts/40`), per signal:
- share of validator calls skipped (ACCEPT + REJECT);
- failures executed without validation (count and rate among ACCEPTs);
- good proposals rejected without validation (count, and share of all good proposals);
- paired Δ(calls skipped) against context_action, with a bootstrap 95% CI over episodes.

**Certification on cert** (examined once, after the freeze), per signal:
- one-sided Clopper–Pearson upper bound on the failure rate among ACCEPTs at δ = 0.05 / 9 (Bonferroni over the 9 signals);
- a signal is certified at X if the bound ≤ X;
- calls skipped on cert are reported alongside.

**Unchanged:** the population, labels, features and partitions. H1–H3 are reported on test_iid as secondary.

**Amendment 3 (2026-09-29): threshold rule and signals for the new data.**

- **Scope:** CSTR Llama-3.2-3B, Qwen2.5-1.5B and Qwen2.5-7B (4-bit) on the same v4 episodes, and the fresh FSM certification set `data/v2/fsm_cert2_seed20260930` for the four FSM models.
- **Timing:** written before any of that data was analysed; the collections were still running.

**Why the rule changes:**
- Under Amendment 2 the ACCEPT thresholds came from the dev point estimate. For CSTR Qwen2.5-3B they turned out too loose: accepted sets had 11–15% failures on cert.
- An exploratory re-analysis of Qwen2.5-3B tested an upper-bound rule. Its test_iid and cert had already been used, so it was exploratory only (`outputs/certification/20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY`).
  - With that rule, observables alone accept nothing and skip 41–44% of calls by rejection.
  - Observables + internals (± grounding) accept 38–49 proposals, 8–12% of them failures, and skip 50–54% of calls.

**Changes:**
1. **Primary ACCEPT rule: `--threshold_rule ucb`.** The threshold is the largest t whose dev accepted set has a one-sided Clopper–Pearson upper bound (δ_sel = 0.05) on its failure rate ≤ X. The Amendment 2 point rule is reported as secondary. REJECT is unchanged.
2. **Two added signals,** to isolate grounding from the other internals: observables + grounding, and observables + all internals. There are then 11 CSTR signals and 10 FSM signals. The Bonferroni δ = 0.05 / (number of signals).
3. **Evaluation:**
   - CSTR models: test_iid once, then cert once, per model.
   - FSM: the fresh cert2 set once per model, with probes and rules frozen on the FSM pilot train/dev as before. The earlier FSM test_iid and cert are not re-used for claims.
4. **Power limitation, stated in advance:**
   - Certifying at X = 10% needs at least about 52 accepted proposals with zero failures (δ = 0.05/11). This holds for either tolerance: the point is that a certified bound needs a large accept set.
   - CSTR cert has 500 episodes. If a probe accepts about 10% of proposals, certification at 10% is infeasible from sample size alone.
   - CSTR certification results are therefore reported with this limitation. Calls skipped and the observed failure rate among ACCEPTs, with the bound, are the main CSTR outcomes.
   - A larger CSTR cert set is the remedy, for the closed-loop stage or later work.

## Reporting rules

- All four hypotheses are reported whatever the outcome.
- A null H1 bounds where internals help. H2/H3 then say whether the failure mechanism explains it, as in the FSM–CSTR contrast.
- No test_iid or cert tuning.
- Any deviation from this document is listed with its reason in the results document.

## Relation to earlier results

- **FSM:** grounding adds +0.03 to +0.05 AUROC over observables, for all 4 models, when failures are retrieval errors.
- **CSTR v2.1:** internals and field grounding add nothing in distribution, where failures were magnitude errors caused partly by the prompt's anchoring. Under plant shift, observables collapse while token confidence and hidden states transfer better.
- **v4 / v3.1** tests whether removing the prompt-induced anchoring changes that picture.

**Amendment 4 (2026-10-03): same model set in both domains.** Written before any of the data below existed.

**Qwen2.5-7B (4-bit) on FSM:**
- Data: the FSM pilot (`data/v2/fsm_pilot_seed20260923`), then the fresh cert2 set, with the same arguments as the other FSM models plus `--quantization 4bit`.
- Rules: frozen on the pilot train/dev with the Amendment 3 rules (FSM signal set), then evaluated once on cert2.

**SmolLM2-1.7B on CSTR**, gated by a pilot of 100 separate episodes (`data/cstr/pilot_v4`, first proposals only):
- The full collection runs only if at least 5% of schema-valid first proposals pass and at most 50% are format failures. Otherwise SmolLM2 is reported as "too weak for the CSTR task" with the pilot numbers.
- If it runs: collection on `data/cstr/episodes_v4`, grounding, freeze with the Amendment 3 rules, test_iid once, then cert grounding and cert once.
