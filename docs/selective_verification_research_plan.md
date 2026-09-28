# Knowing When to Verify

## Research and implementation plan

Working title: **Knowing When to Verify: Model-Internal Signals for Selective Validation of LLM Agent Actions**

Status: proposed protocol, not experimental findings. This document specifies successive tasks; it does not authorize running every stage at once.

### Central question

Can information extracted from an open-weight LLM reduce the cost of externally validating its proposed actions, while keeping the failure rate of unverified actions below a specified level?

The principal scientific test is whether internal representations add useful information beyond token confidence, observable task context, and the proposed action itself. The practical test is whether this information saves total computation after feature extraction and routing overhead are included.

Use FSM traversal as a cheap, exactly labelled test environment and the existing CSTR as the application with expensive dynamic validation. Success on FSMs is not evidence of automatic transfer to CSTR dynamics.

## Common instructions to prepend to each implementation task

1. Inspect the repository and applicable instructions before editing. Reuse functioning modules and preserve existing entry points where practical.
2. Work only on the requested stage. Finish with findings, artifacts, validation, limitations, and the next decision; do not start subsequent stages without an explicit request.
3. Do not fabricate measurements, populate result tables with illustrative numbers, or describe a proposed guarantee as an established result.
4. Keep old results immutable. Write new results to versioned experiment directories and record data provenance.
5. Keep preprocessing, model selection, calibration, routing thresholds, and layer selection independent of final test data.
6. Use explicit input-feature allowlists. Do not rely only on exclusions to prevent label leakage.
7. Record seeds, software versions, hardware, model/tokenizer revisions, precision, decoding configuration, dataset fingerprints, and prompt versions.
8. Add targeted sanity checks and tests for consequential new behaviour. Do not build a large framework before the pilot demonstrates a need.
9. Separate inference, verifier labelling, feature extraction, probe fitting, policy evaluation, and plotting so expensive work can be reused.
10. External verifier results may supply offline labels, but must never enter a feature available before the verification decision.
11. All direct-execution experiments in this project run in simulation. The protocol does not authorize bypassing checks on physical equipment.
12. Report files created/modified, commands, observed outputs, checks performed, unresolved issues, and any deviation from the protocol.

Repositories:

- Reliability experiments: `C:/Users/jv624/Desktop/fsm_reliability`.
- Existing process-recovery implementation: `C:/Users/jv624/Desktop/ctrl-alt-recover`.

Read the second repository to reuse its simulator, prompts, action schema, and validation logic. Keep integration code in the reliability workspace initially. Do not copy and silently diverge from the simulator or recreate a CSTR merely because it is outside the current workspace. Respect filesystem permissions for any later edits.

## Scientific contract

### Primary routing policy

For a schema-valid candidate, predict failure risk `p_hat`. For a frozen threshold `tau`:

```text
p_hat < tau  -> DIRECT: bypass the expensive verifier in the simulated experiment
p_hat >= tau -> VERIFY: invoke the existing verifier
```

Every candidate first passes cheap deterministic schema and admissibility checks that are available equally to all policies. Malformed or obviously forbidden actions are rejected or reprompted under a fixed rule. A numerical scoring failure or unsupported feature configuration routes to verification; an unusable action routes to rejection. These events remain visible in metrics.

A verified candidate executes only if validation passes. Specify the reprompt, fallback, deadline, and termination rules before comparison. Freeze them across policies.

Optional secondary policy, evaluated separately: reject or revise likely failures before simulation, but validate every applied candidate. This policy measures lost viable candidates and saved failed rollouts. Its guarantees do not justify the primary bypass policy.

### Labels and decisions

Do not call an observed failure label `verification_required`.

| Field | Meaning |
|---|---|
| `parse_success` | Whether a response can be parsed into the declared action schema |
| `schema_valid` | Whether the parsed action satisfies the common structural checks |
| `candidate_invalid` | FSM path violates the declared validity specification |
| `rollout_unacceptable` | Completed CSTR replay violates the frozen outcome contract |
| `label_status` | `known`, `simulator_error`, or another explicit unresolved condition |
| `predicted_failure_risk` | Learned estimate of the selected binary outcome |
| `route` | `DIRECT`, `VERIFY`, or `REJECT` |
| `verifier_pass` | Verdict of the existing verifier, when evaluated |

Maintain compatibility with `valid_path` for the existing FSM pipeline. Distinguish answer-format failure, invalid path, and suboptimal but valid path. The main semantic probe should be evaluated on schema-valid responses, with all-response pipeline results reported separately.

A simulator exception is an unknown physical outcome, not an unsafe-trajectory label. Report unresolved cases and retry policy explicitly. They cannot be counted as successful direct executions. Audit whether excluding them changes the evaluation population.

### Claims and endpoints

Distinguish:

1. Structural validity of an FSM answer.
2. Agreement with the existing rollout validator.
3. Physical outcomes within the specified simulation model and horizon.
4. Full-episode productive recovery.

Do not equate validator acceptance, safe shutdown, and productive recovery. Do not extrapolate simulated results into unconditional physical safety claims.

## Stage 0 — Audit and freeze the protocol

Inspect both repositories read-only before making research-code changes.

Audit data generation, split logic, prompts, model loading, decoding, token scores, attention extraction, hidden-state availability, labels, calibration, plots, traces, dependencies, and seed handling. Inspect at least:

- `scripts/03_extract_pilot_features.py`
- `scripts/06_calibrate_probability_model.py`
- `src/models/run_inference.py`
- `src/data/generate_fsm_dataset.py`
- `data/raw_10node_large/`, other existing raw splits, and `data/processed/`
- CSTR `_make_llm`, `clone_plant_obj`, `rollout_validate_setpoints`, and execution/reprompt routing
- `ctrl-alt-recover/docs/recovery_gap_study.md`

Check what is actually collected and persisted; library capabilities do not imply existing data availability. Check whether the evaluated answer is truncated while extracted features cover additional tokens. Identify whether logits are raw forward-pass logits or generation scores modified by decoding processors.

Audit split content, not just IDs: graphs, graph/query pairs, repeated prompts, episodes, model variants of the same example, and overlapping datasets from different seeds. Define grouping appropriate to each claimed generalization setting. Unique split-prefixed IDs are insufficient.

Freeze a protocol configuration covering target, eligible population, split/group policy, candidate models, feature groups, metrics, desired risk levels, uncertainty level, and compute budget. Define which results are exploratory and which will be confirmatory.

Deliver:

- `docs/selective_verification_repo_audit.md`
- `docs/selective_verification_protocol.md`
- A machine-readable protocol configuration using repository conventions

Stop after documenting reusable components, minimum additions, data-quality problems, unresolved assumptions, and the proposed pilot. Do not launch new inference.

## Stage 1 — Establish a clean, inexpensive FSM baseline

Reuse existing compatible features where provenance and splits are adequate. Start with token confidence and existing attention summaries; do not require hidden-state extraction yet.

Report raw sample counts, parse/schema failures, semantic invalidity, class balance, missing features, and unresolved labels per split and model. Audit repeated underlying examples across seeds before aggregating uncertainty estimates.

Use group-disjoint development data and retain an untouched final test set. If existing test results have already influenced design, call that set exploratory and reserve new final data. A 3000/500/500 split is a pilot example, not a certification sample-size prescription.

Fit logistic regression with training-only preprocessing. Fit any probability calibrator on a separate development partition. Report AUROC, failure-positive AUPRC with prevalence, Brier score, ECE with bin definition, and a reliability plot. Secondary classification metrics must use a development-selected threshold and state the positive class.

Use this stage to compare output confidence against attention, rather than presenting attention in isolation.

Deliver under a versioned `outputs/fsm_baseline/` directory:

- `metrics.json`, `dataset_summary.json`, `run_config.json`
- Per-example predictions with split/group identifiers
- Split-overlap and feature-availability audit results

Stop if leakage, ambiguous labels, or alignment problems prevent interpretable results. Correct these before expanding experiments.

## Stage 2 — Build minimal selective-routing evaluation

Implement reusable offline policy evaluation on frozen candidate records. Label all eligible candidates offline, even candidates the simulated routing policy would not verify. Account for offline dataset-construction cost separately from online policy cost.

Baselines:

1. Never invoke the expensive verifier, after common cheap checks.
2. Always invoke it for eligible candidates.
3. Random verification at matched budgets, repeated across routing seeds.
4. Mean selected-token log probability.
5. Token entropy.
6. Learned token-confidence probe.
7. Existing attention probe.
8. Combined token/attention probe.
9. Task-context/action-only probe where appropriate.

Report verifier-call rate, direct coverage, conditional direct failure risk, escaped-error rate over all eligible candidates, and counts. Also provide all-response operational denominators so malformed-output rates are not hidden.

Conditional risk is undefined when there are no direct executions. Always-verification has zero bypass errors by construction, but this does not imply zero simulator/plant failures or successful recovery.

Threshold sweeps are descriptive comparisons. Do not select a threshold from final test curves and then claim independently validated performance at that threshold. Clearly distinguish validation-budget-matched policies from descriptive test curves matched by realized call rate.

Deliver `risk_verification_curve.csv`, PDF plots, baseline comparisons, and per-example routing decisions under `outputs/fsm_selective_verification/`.

Stop and assess whether routing beats random, whether internal features improve over simple confidence, and whether the effect survives control for task/output structure. A weak FSM result should trigger a scoped decision about a small CSTR pilot, not an automatic large expansion or a claim that CSTR must also fail.

## Stage 3 — Add finite-sample risk certification

Separate predictive probability calibration from statistical certification of the routing policy.

The simplest primary protocol is:

1. Fit probes using training data.
2. Use development partitions for hyperparameters, probability calibration, feature selection, and threshold selection.
3. Freeze one primary policy before inspecting independent certification outcomes.
4. Certify that policy using an independent sample from the declared population.
5. Report final performance on untouched test episodes.

Development partitions can be implemented by explicit internal splits or documented cross-fitting. Do not demand nominal split sizes that leave rare-event assessment underpowered.

For a fixed policy with `k` failures among `n` independent directly executed candidates, compute a one-sided exact binomial upper confidence bound at confidence `1-delta`. Certification succeeds only if this upper bound is at most the prespecified risk target `alpha` and `n > 0`.

With zero failures, the upper bound is:

```text
U = 1 - delta ** (1 / n)
```

At 95% confidence, zero failures require at least 299 independent direct executions to support a 1% bound, and 598 to support a 0.5% bound. Total required candidates depend on direct coverage. Failures increase the requirement.

Do not assume independence for multiple decisions from one episode. For candidate-level binomial certification, use a prespecified independently sampled decision per independent episode, or justify another valid sampling scheme. Report episode-level events separately. Cluster bootstrap intervals can describe uncertainty but are not a substitute for this exact certification argument.

If thresholds or feature groups are selected using certification outcomes, use an appropriate simultaneous/multiple-testing risk-control procedure instead. A failed certification cannot be repaired by repeatedly tuning against the same set with unadjusted bounds. Do not apply a monotone-loss theorem without checking its assumptions: conditional selective risk need not be monotone in an arbitrary learned score.

For a fixed IID candidate population, the intended statement is:

```text
With probability at least 1-delta over certification data,
the procedure does not certify a policy whose direct-execution risk exceeds alpha.
```

State the population, independence assumptions, eligibility rules, and handling of unresolved labels. This is not an OOD guarantee or a per-action safety certificate.

Deliver `docs/risk_certification.md`, sample-size calculations, tested interval code, and certification artifacts. Reference existing methodology rather than claiming a new statistical theorem merely from applying it; see Learn then Test: https://arxiv.org/abs/2110.01052.

## Stage 4 — Integrate the existing CSTR and run a small snapshot pilot

Bring the CSTR feasibility check forward before investing in all feature families.

Expose a narrow adapter around the existing implementation:

```python
result = verifier.verify(snapshot=snapshot, action=action, config=config)
```

Use the existing supervisory action variables `T_sp`, `L_sp`, and `Fin_sp`. Do not substitute direct valve or pump commands. Use current prompts, retrieved context, and controller structure as the initial reference condition.

Snapshots must retain simulator time, phase, physical state, controller integrators, actuator state, fault configuration, and relevant random-generator state. Establish replay reproducibility and independence of hypothetical rollouts from their evaluation order. Measured temperature and level alone are not a complete snapshot.

Return label status, legacy verifier verdict, independent outcome-contract verdict, reason codes, relevant extrema, recovery timing, tracking metrics, and trajectory references. Preserve existing semantics as a reference; version any changed acceptance contract.

Freeze explicit physical limits, production criteria, allowed actions, evaluation horizon, transient allowances, recovery deadline, hold duration, shutdown handling, and relapse checks. Evaluate physical outcomes on simulated plant truth; retain sensor-based verdicts separately. Tracking a newly reduced setpoint must not automatically establish productive recovery.

Test known acceptable/unacceptable witnesses, schema/action bounds, constraint boundaries, simulator errors, snapshot replay, and rollout-order independence. A successful witness proves existence only under its declared conditions; failed search does not prove unrecoverability.

Generate a small, predeclared exploratory sample from supported faults and snapshots. Use one local model initially. Label all parseable eligible actions; retain failures and unresolved cases. Measure the wall-clock distribution of generation, feature extraction, and verification before launching a large campaign.

Compare token/attention probes with an observable plant-context/action-only probe and their combination. Privileged simulator state belongs to offline analysis, not the online predictor unless it is genuinely observable in deployment.

Stop with an explicit decision: continue, change the feature budget, narrow the claim, or report a negative result. Require plausible total-cost savings and enough outcome variation for meaningful evaluation.

## Stage 5 — Expand the common feature extractor only where justified

Preserve existing attention workflows and add optional feature groups behind configuration flags.

Define non-overlapping families:

- `token_confidence`: selected-token log probabilities, entropy, and top-two margins from a documented distribution.
- `attention`: masked layer/head and prompt-region summaries.
- `hidden`: selected-layer pooled representations.
- `context_action`: observable task/plant context and proposed-action features; these are not model internals.

Combinations include `token_plus_attention`, `token_plus_hidden`, `all_internal`, and `context_action_plus_internal`. Retain legacy group aliases where necessary. Avoid duplicate columns such as both mean log probability and its exact negative under a different name.

Specify token spans, causal/padding masks, prompt regions, EOS handling, truncation, layer numbering, and whether hidden states represent a token before or after it is consumed. Align the actual action evaluated with the tokens used for features. Document raw logits versus processed generation scores and chat-template differences.

Store compact summaries in tables and high-dimensional vectors in keyed binary artifacts. Fit scaling, PCA, imputation, and feature selection only on training data. Record missing-feature reasons; do not silently zero-fill unsupported attentions.

Avoid raw tensor dumps by default, and measure peak memory: not saving a tensor does not eliminate the memory required to materialize it. Begin with a few prespecified layer summaries before broad per-head searches.

Record checkpoint/tokenizer revision, precision/quantization, attention implementation, generation parameters, token IDs or reproducible references, latency, and feature schema version. Do not assume an Ollama response can be exactly reconstructed using a different Hugging Face checkpoint or quantization.

Deliver `docs/internal_feature_schema.md`, feature validation checks, and an extraction-cost report. Test alignment and missing-feature handling on a small sample before full inference.

## Stage 6 — Build the confirmatory datasets and fit risk estimators

Freeze choices justified by the pilots before final data generation. Create group-disjoint training, development, certification, IID test, and explicit OOD test partitions. Group repeated samples from the same graph/episode, alternate prompts, multiple model responses, and retries appropriately.

For CSTR, begin with the supported paper faults. Expand fault severity, initial conditions, or noise only through simulator-supported and explicitly recorded configurations. Define OOD axes in advance and do not recalibrate on OOD test data.

Reuse the common feature pipeline and frozen outcome contract. Preserve full response/action provenance. Retain counts and reason codes for every unsuccessful generation or labelling attempt.

Compare logistic regression with one justified nonlinear baseline using a bounded development search. Train separate model-specific probes unless cross-model transfer is an explicit experiment. Distinguish retraining the same method across domains from zero-shot transfer of a fitted probe.

Required comparisons: token confidence, attention, hidden representations if justified, context/action, and combinations. Fit Platt or isotonic calibration only with adequate development data and compare to uncalibrated scores. Probability calibration is secondary to the routing result; it cannot create ranking information absent from the base score.

Prespecify primary model/feature/policy claims. Correct for multiplicity if certifying several selected policies jointly. Report all planned comparisons and failures; do not identify a test-set winner and then describe it as prespecified.

Deliver datasets/manifests, fitted pipelines, predictions, comparison tables, and frozen thresholds. Do not overwrite exploratory outputs with confirmatory results.

## Stage 7 — Evaluate offline routing and total computational value

Evaluate frozen policies on identical held-out candidate/snapshot records. Report both legacy-verifier disagreement and independent outcome-contract failures where they differ.

Primary measurements:

- Direct coverage and verifier-call rate.
- Conditional failure risk among direct actions, with counts and uncertainty.
- Escaped failures over all eligible candidates and over all generated proposals.
- Total latency, feature overhead, verifier time, and peak memory.
- Verification calls and total time required at certified risk levels.

Include all baseline policies and show the incremental value of internals beyond `context_action`. Compare predictions and policy outcomes on paired samples; account for episode grouping in descriptive uncertainty estimates.

For constant verifier cost `C_v`, bypass fraction `a`, and added scoring/feature cost `C_f`, per-candidate saving relative to always verifying is:

```text
a * C_v - C_f
```

For variable costs, measure `E[1_DIRECT * C_v] - E[C_f]` using paired records. Report startup/warm-up, caching, hardware, repeat counts, and whether inference and verification compete for resources. Candidate-level identities exclude changes in retries and future state; those belong to the next stage.

Show training/label-generation cost separately, and estimate an amortization break-even point only if online savings are positive. FSM verifier-call savings alone are not a claim of FSM runtime acceleration.

## Stage 8 — Run full closed-loop and robustness experiments

Run Never DT, Always DT, Random DT, confidence routing, and the frozen learned policy through complete simulated episodes under the same cheap checks, fallback rules, attempt limits, and deadlines.

Freeze how decision time affects the plant: either the plant is paused during computation or it evolves while inference/verification runs. The first setting supports compute comparisons but does not establish performance under real decision delays. If claiming deadline benefits, model the latter explicitly and use state/snapshot freshness consistently across policies.

Use matched initial conditions and exogenous disturbance streams where valid. Keep hypothetical verifier randomness separate from executed-plant randomness so extra verifier calls cannot alter the plant disturbance sequence.

Report productive recovery, safe shutdown, unsafe termination, deadline failure, unresolved execution, production/tracking performance, maximum constraint excursion, retries, verifier calls, and total episode cost. Recheck initially successful recoveries through the declared remaining horizon.

Evaluate IID and OOD separately. A guarantee under the candidate-certification distribution does not automatically survive policy-induced state changes, repeated decisions, or OOD deployment. Report episode-level empirical failure bounds using independent episodes where applicable; do not derive a trajectory guarantee by multiplying marginal candidate success rates.

Aim for three open-weight model configurations if resources permit: two sizes from one family and one from another. Preserve prompt content except required chat formatting. Freeze layer and calibration choices before final comparisons.

Optional experiments: cross-domain probe transfer, model-size transfer, and reject-before-verification routing. Keep these separate from the main claim and report negative findings.

Stop and reassess if fewer rollouts do not reduce total time, or if saved verification causes unacceptable recovery losses. Such outcomes are research findings, not implementation failures to hide.

## Stage 9 — Assemble the paper and reproducible artifact

Generate all figures/tables from saved machine-readable outputs. Preserve protocol versions and list deviations.

Required figures:

1. Method and routing diagram with labels distinguished from online inputs.
2. FSM and CSTR conditional-risk versus verifier-call curves.
3. Escaped-error and total-cost comparisons at frozen policies.
4. Calibration and certified coverage at declared risk targets.
5. Incremental feature value, especially versus context/action baselines.
6. IID/OOD and closed-loop recovery comparisons.

Required tables: dataset counts/exclusions, predictive metrics, selective-routing outcomes with uncertainty, actual computational cost, and closed-loop/model robustness.

Create `paper_outputs/` with figures, tables, metrics, configurations, prediction references, and manifests. Create `REPRODUCE_PAPER.md` with separate commands for expensive inference/labelling, fitting, evaluation, and plot regeneration. Plots must not require new LLM calls.

Create `docs/paper_experiment_summary.md` containing observed results, failed hypotheses, limitations, and the relationship to the earlier JPC paper. Clearly identify the new contribution rather than presenting local inference alone as methodological novelty.

## Criteria for a strong paper

The strongest supported claim would be that a specified set of internal signals adds predictive value beyond strong observable baselines, enabling lower total validation cost at an independently assessed failure rate, with demonstrated recovery behaviour in complete simulated episodes.

Do not require every feature family or transfer experiment to succeed. If token confidence is sufficient, attention is too costly, hidden representations do not generalize, or selective bypass does not improve closed-loop outcomes, report that directly and adjust the contribution.

The elementary routing and cost identities support the formulation. The substantial evidence must come from valid risk assessment, incremental feature value, and system-level results. Any stronger theoretical claim requires additional assumptions and a separate proof.
