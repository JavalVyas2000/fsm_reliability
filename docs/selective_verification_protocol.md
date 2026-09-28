# Selective verification — frozen protocol (v1.0)

Working title: *Knowing When to Verify: Model-Internal Signals for Selective
Validation of LLM Agent Actions*.

**Status:** protocol only; no experimental findings yet. v1.0 was frozen on
2026-09-23, after the user resolved decisions D1–D7 (§13).

**Related files:**
- `docs/selective_verification_research_plan.md` — research plan
- `docs/selective_verification_repo_audit.md` — Stage 0 audit
- `configs/selective_verification_protocol_v1.json` — machine-readable parameters

If this document and the JSON disagree, the JSON is authoritative for
parameters and this document is authoritative for definitions. Any change after
v1.0 goes into the deviations log (§14).

---

## 1. Paper claim

> With a **fixed prompt template** and instances drawn from a declared
> distribution, a probe on the LLM's internal signals predicts whether the
> proposed action will pass external simulation-based validation. This lets a
> large fraction of simulator calls be skipped, at an independently certified
> failure rate among the skipped actions, with measured wall-clock savings.

- **Population.** "Same prompt, different instances" is the IID population.
  - The template, model, decoding settings and static retrieved context are fixed.
  - Only the instance varies: the graph and query for FSM; fault type, severity, onset time and noise seed for CSTR.
- **Out of scope.** No claim is made for other prompts, models or OOD instance distributions.

| ID | Claim | Domain | Status |
|---|---|---|---|
| C1 | Internal signals (attention, token confidence, hidden states) predict validator failure beyond `context_action` features | FSM + CSTR | confirmatory |
| C2 | A frozen routing policy is certified at risk ≤ α with non-trivial direct coverage | FSM + CSTR | confirmatory |
| C3 | On CSTR, the online saving `E[1_DIRECT·C_v] − E[C_f]` is positive, measured in wall-clock time over a large instance set | CSTR | **confirmatory, headline** |
| C4 | Closed-loop recovery is not degraded by selective bypass | CSTR | stretch |
| C5 | Cross-model / OOD / transfer | both | stretch |

**Why both domains:**
- **FSM** is a testbed with exact labels and near-zero verifier cost. It establishes that the predictive method works and supports C1/C2 with large, cheap samples. **No runtime-saving claim is made on FSM.**
- **CSTR** validation costs seconds per call. This is where the cost saving (C3) is demonstrated.

## 2. Labels

| Field | Definition |
|---|---|
| `parse_success` | FSM: the first JSON object parses into the path schema. CSTR: the structured output parses into `SetpointPlan` |
| `schema_valid` | FSM: a list of ints in `[0, num_nodes)`, of length 1 to `4·num_nodes`. CSTR: finite floats inside declared admissible bounds (§7.4) |
| `candidate_invalid` (FSM target) | `schema_valid ∧ ¬valid_path` |
| `suboptimal_valid` (FSM, secondary) | `valid_path ∧ ¬optimal_path`. This is **not** a routing failure: the verifier accepts it |
| `verifier_reject` (CSTR target) | The existing `rollout_validate_setpoints` returns FAIL under deterministic replay (§7.3) |
| `label_status` | One of `known`, `simulator_error`, `timeout`, `unresolved` |
| `predicted_failure_risk`, `route`, `verifier_pass` | As defined in the research plan |

- `valid_path` is kept for compatibility with earlier results.
- Format failures are excluded from the semantic-probe population. They are counted in the all-response operational denominators.
- The positive class is always **failure**.

**CSTR label semantics (D6).** The CSTR target is *acceptance by the existing
validator*. The paper's claim is replacing that simulator call, so agreement with
it is the correct target. No separate production or outcome contract is added.

Limitation, stated in the paper:
- Validator acceptance is not the same as productive recovery.
- The audit found that the validator accepts near-shutdown feeds.
- As a diagnostic only, we report the distribution of accepted `Fin_sp` relative to nominal. It is not used as a label.

## 3. Routing policy

- **Cheap checks**, the same for every policy:
  - parse and schema checks;
  - FSM node range check;
  - CSTR admissible bounds.
- **Failure handling:**
  - A format failure gets one fixed format-error reprompt, then REJECT.
  - A failure to score the candidate routes it to VERIFY.
  - All such events are logged with reason codes.
- **Primary policy:** `p_hat < τ → DIRECT`, otherwise VERIFY. A VERIFY candidate executes only if the verifier passes.
- **Offline datasets.** Each instance yields **one first-proposal candidate**.
  - Reprompt candidates depend on verifier feedback, which is unavailable under DIRECT, so they are excluded from the primary evaluation.
  - Reprompt, fallback and deadline rules apply only to the closed-loop stretch (C4).
- **Secondary policy (optional):** reject-before-verify, evaluated separately.

## 4. Models, prompts, decoding

**Models** (8 GB RTX 4060 Laptop GPU, bf16):

| Role | Model |
|---|---|
| Primary | `Qwen/Qwen2.5-3B-Instruct` |
| Second family | `meta-llama/Llama-3.2-3B-Instruct` |

Revisions are pinned at first download and recorded. Probes are trained separately
for each model.

**FSM prompt v2:**
- Uses the tokenizer's chat template.
- JSON schema output: `{"path": [...]}`.
- No empty-path option, since every query is reachable.
- Includes a worked example on a fixed graph that never occurs in any split, and whose answer cannot be copied into an instance. The example must not have the form `[start, x, goal]` with instance-specific values.
- The prompt text is versioned and hashed.

**CSTR prompt:**
- The existing `action_propose` system and user prompts from `ctrl-alt-recover`, rendered through the chat template.
- The static KG Turtle context is **frozen from a stored trace subgraph**. It was fetched once per run from GraphDB and is static, so there is no runtime GraphDB dependency.
- The frozen context is hashed. Identity across stored traces is checked before it is frozen.
- `symptoms` are populated as the prompt intends. The upstream bug that emptied them is not reproduced. This is recorded as a deviation from the historical runs.

**Decoding:**
- Greedy.
- Fixed `max_new_tokens` per domain, never derived from ground truth.
- FSM stops at the end of the first JSON object.
- Persisted per candidate:
  - generated ids and the answer span;
  - raw and processed selected-token log-probabilities;
  - pooled features (§6);
  - generation latency, feature latency and peak memory.

## 5. Instance distributions and splits

### 5.1 FSM

- **Generator:**
  - directed ER reachability graphs;
  - `num_nodes ∈ {5,10,15,20}`;
  - `edge_prob = {5:.35, 10:.30, 15:.25, 20:.20}`;
  - the start/goal pair must be reachable.
- **Seeding:** `numpy.random.SeedSequence(root).spawn` per partition.
- **Group key:** canonical adjacency hash (graph group). Partitions are disjoint on graph group (D4).
- **Overlap check:** zero overlap with the union of all existing `data/raw*` sets is a required check.
- **Prevalence adjustment:** the node mix may be adjusted **once**, on pilot data, towards overall failure prevalence of 10–40%. It is then frozen.

### 5.2 CSTR (instance = one fault episode, snapshot taken at the first action trigger)

**Fault families and severity** (D6: varying severity makes the setpoint change matter):

| Family | Severity parameter | Proposed range (subject to §5.3) | Onset |
|---|---|---|---|
| fouling | `fouling_max` (fraction of UA lost), `fouling_tau` | 0.3 – 0.9; tau 1000 – 3000 s | 1800 – 2200 s |
| pump_degrade | `pump_degrade_factor` (remaining flow) | 0.2 – 0.8 | 1800 – 2200 s |
| cool_stuck_closed | stuck valve opening | 0.0 – 0.6 | 1800 – 2200 s |

- The stuck opening is hard-coded to 0.3 in `FaultManager.actuator_cool_opening`. It is parameterised through a subclass in `fsm_reliability`, **not** by editing `ctrl-alt-recover`. The value 0.3 stays inside the range, so the original fault is included.
- Optional extra families: `outlet_block_factor`, `leak_k`. Added only if §5.3 shows useful outcome variation.
- Per-instance draws:
  - fault family (uniform);
  - severity and onset (uniform over the frozen ranges);
  - noise seed.
  - All draws come from `SeedSequence` streams.
- **Independence unit:** the episode. There is one candidate per episode. Episodes are disjoint across partitions by construction, since each has its own draws.

### 5.3 Severity-range calibration (no LLM; before any CSTR labelling)

1. For a grid of severities per family, run a fixed grid of setpoint actions around nominal, using the deterministic verifier.
2. Keep the severity region where **both** conditions hold:
   - (a) the no-change action fails for most instances;
   - (b) at least one grid action passes.
   This makes the region recoverable, but only with a meaningful setpoint change.
3. Freeze the ranges in the JSON before any LLM generation.
4. Report the witness grid as evidence of recoverability under the declared conditions.

### 5.4 Partition sizes

| Partition | FSM pilot | FSM confirmatory | CSTR pilot | CSTR confirmatory |
|---|---|---|---|---|
| train | 1500 | 3000 | 150 | 1500 |
| dev_cal | 500 | 1000 | 50 | 400 |
| dev_thr | 500 | 1000 | 50 | 400 |
| cert | — | sized by §8 | — | sized by §8 (≈800 planned) |
| test_iid | 500 (exploratory) | 1000 | 50 (exploratory) | 800 |

- Confirmatory sizes are provisional. They are finalised from pilot prevalence and coverage via §8, and recorded before generation.
- CSTR confirmatory budget estimate: about 3900 episodes.
  - Validation at ~7 s/call ≈ 7.6 CPU-hours, parallelised across cores.
  - Generation must be measured in the pilot.

## 6. Feature families

All feature groups are explicit allowlists.

| Family | Contents | Role |
|---|---|---|
| `attention` | At prespecified layers (relative depth 0.25/0.5/0.75, plus the last layer): attention mass from **answer-span queries** to prompt regions, head-mean and head-max. Regions — FSM: graph, start, goal, instruction. CSTR: KG context, plant snapshot, feedback/goal, instruction. Also row entropy over answer queries | **primary internal family** (D5) |
| `token_confidence` | Over the answer span, from raw logits: mean/min selected log-prob, mean/max entropy, min top-2 margin | primary internal family |
| `hidden` | Residual stream at the same layers, mean over the answer span plus the last prompt token; train-only PCA (k ≤ 64) | comparator; nearly free to collect |
| `context_action` | FSM: `num_nodes`, `num_edges`, density, `shortest_length`, answer length and pattern flags. CSTR: observable snapshot (measured T and L, actuator positions, anomaly ratio, control zone, current setpoints) plus the proposed setpoint deltas. **Fault family, severity and onset are privileged and excluded** | required baseline |

- **Combinations:** each internal family alone; `attention+token_confidence`; `all_internal`; `context_action`; `context_action+attention`; `context_action+all_internal`.
- **Attention extraction (cost-aware).**
  - Forward hooks at the selected layers compute softmax attention only for answer-span query rows against all keys.
  - Memory is O(heads · |answer| · T), not O(heads · T²), and full attention tensors are never materialised.
  - This is what makes attention feasible for ~4.5k-token CSTR prompts on 8 GB.
  - The hook output is validated against eager full attention on a small sample (max abs diff ≤ 1e-3).
- **Alignment test:** re-decoding the feature token span must reproduce the parsed substring.
- **Speed claim (D5).** For every family we report the wall time to produce a valid/invalid prediction, as feature extraction plus probe scoring, set against the verifier's wall time. This is the "how fast can we say valid or invalid without the simulator" result.

## 7. CSTR verifier adapter

1. **Interface:** `verify(snapshot, action, config) -> VerifierResult`, implemented in `fsm_reliability`. It imports `CSTRSimulation`, the checkers and `rollout_validate_setpoints` semantics from `ctrl-alt-recover`, without copying or diverging from them.
2. **Snapshot contents:**
   - plant deepcopy at the trigger;
   - realised fault config (family, severity, onset);
   - `t0`, mode and phase;
   - the rollout seed.
3. **Determinism** (a correctness fix, not a semantic change):
   - Each rollout runs with the global NumPy and Python RNG set from the snapshot's rollout seed.
   - The caller's RNG state is restored afterwards.
   - Required tests: replay determinism, independence from evaluation order, and no perturbation of the executed plant.
4. **Admissible action bounds** (cheap check):
   - declared physical ranges for `T_sp`, `L_sp`, `Fin_sp`, taken from the prompt's stated limits;
   - out-of-range actions go to REJECT;
   - the grid rule is recorded but not enforced.
5. **Result fields:**
   - `label_status`, `verifier_pass`, `fail_reason`;
   - `time_to_safe`, `unsafe_fraction`, peaks;
   - `n_steps`, `wall_s`;
   - a trajectory reference stored as compact parquet, only on request.
6. **Cost baselines:**
   - **Legacy verifier:** as-is. This is the headline C3 comparison.
   - **Early-exit verifier:** same verdict semantics, exiting as soon as the verdict is irrevocable. Reported as a sensitivity analysis, so the saving is not attributed to an obviously inefficient verifier. The early-exit variant must agree 100% with the legacy verdict on the pilot.

## 8. Metrics, risk targets, certification (D7)

**Predictive metrics:**
- AUROC;
- failure-positive AUPRC, with prevalence;
- Brier score;
- ECE (15 equal-width bins, with p = 1 in the top bin);
- reliability diagrams.

**Routing metrics:**
- verifier-call rate;
- direct coverage;
- conditional direct risk `k/n` with exact CI (undefined when n = 0);
- marginal escaped-error rate over eligible candidates and over all responses;
- counts.

**Cost metrics:**
- per-candidate generation, feature and verifier time;
- saving versus always-verify, for both verifier variants;
- peak memory.

**Risk targets:**
- Primary α ∈ {0.05, 0.02}; secondary α = 0.01; δ = 0.05, one-sided.
- Primary quantity: conditional risk P(fail | DIRECT).
- Co-reported: marginal risk P(DIRECT ∧ fail).

**Minimum number of direct executions for an exact (Clopper–Pearson) one-sided 95% upper bound ≤ α, given k observed failures:**

| α | k=0 | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| 0.05 | 59 | 93 | 124 | 208 |
| 0.02 | 149 | 236 | 313 | 523 |
| 0.01 | 299 | 473 | 628 | 1049 |

- **Cert size:** `n_required(k=2) / direct_coverage(dev_thr) × 1.5`, computed from pilot or dev data before cert is generated.
- **Procedure:**
  - One frozen policy per (model, domain, α), with the threshold chosen on `dev_thr`.
  - Certify on `cert`, using one candidate per independence unit.
  - Joint certification uses Bonferroni over the declared set.
  - A failed certification is reported, not re-tuned.
- **Guarantee statement:**
  - With probability ≥ 1−δ over cert data, the procedure does not certify a policy whose direct-execution risk exceeds α.
  - This holds for the declared IID instance population (fixed prompt, model and distribution).
  - It is not an OOD guarantee and not a per-action safety certificate.

## 9. Stage plan and gates

| Stage | Content | Gate to continue |
|---|---|---|
| 1 | FSM pilot: new generator, prompt v2, Qwen2.5-3B, all four families, cost measurement | Some internal family improves over `context_action` (grouped-bootstrap 95% CI on ΔAUROC excludes 0), **or** a documented decision to continue on CSTR regardless |
| 2 | Offline routing evaluation on FSM pilot | Routing beats random at matched budget |
| 3 | Certification code and tests | Tests pass |
| 4 | CSTR adapter, determinism tests, severity calibration (§5.3), 300-episode pilot | Outcome variation is present (failure prevalence 10–70%), the saving is plausibly positive, and the early-exit verifier agrees with the legacy verdict |
| 5 | Feature-extractor consolidation (only as needed) | — |
| 6 | Confirmatory datasets, both domains | — |
| 7 | Offline routing plus cost evaluation | — |
| 8 | Stretch: closed loop, second model, OOD | — |
| 9 | Paper artefacts | — |

**FSM pilot compute estimate:** about 3000 generations × ≤2 s, ≈ 2–3 GPU-hours. This is replaced by the measured timing on a 200-instance subset before the full run.

## 10. Exploratory vs confirmatory

- **Exploratory:**
  - all existing data and outputs in both repos;
  - all pilots;
  - severity calibration.
- **Confirmatory:**
  - Stage 6 data, generated with root seeds recorded in the JSON before generation;
  - test and cert outcomes are not inspected until every fitting choice is frozen.

## 11. Reproducibility

Every run writes `run_manifest.json`, recording:
- git commit and dirty flag, for both repos where they are used;
- package versions and GPU/driver;
- model and tokenizer revision, dtype, attention implementation;
- decoding config;
- prompt version and hash, and the KG context hash;
- dataset fingerprint and seed streams;
- wall time and peak memory.

Outputs are written to `outputs/<stage>/<run_id>/` and never overwritten.

## 12. Upstream issues in ctrl-alt-recover

These bugs are not fixed here. The adapter does not depend on the affected code paths:

- **F5:** `llm_model` is dropped from `GraphState`, so every run used gpt-4o-mini.
- **F6:** the fallback is never applied.
- **Dropped state keys:** `symptoms` and detector buffers.

Fixing them upstream is a separate decision for the user.

## 13. Resolved decisions (2026-09-23)

| ID | Resolution |
|---|---|
| D1 | Headline: internals reliably reduce simulation cost at scale under a fixed prompt with varying instances. CSTR C3 promoted to confirmatory headline; FSM is the exact-label testbed for C1/C2 |
| D2 | Prompt v2 as specified in §4 (delegated to the assistant) |
| D3 | Qwen2.5-3B-Instruct primary, Llama-3.2-3B-Instruct second family |
| D4 | Graph-disjoint partitions; all data generated fresh |
| D5 | Attention is a primary family. Report how fast validity can be predicted from internals versus the simulator's cost. Hidden states retained as a cheap comparator |
| D6 | No separate outcome contract; the target is acceptance by the existing validator. Fault severity is varied so that setpoint changes are meaningful (§5.2–5.3) |
| D7 | α ∈ {0.05, 0.02} primary, 0.01 secondary, δ = 0.05; conditional and marginal risk both reported (delegated to the assistant) |

## 14. Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-09-23 | v0.1 → v1.0: CSTR promoted to headline; outcome contract dropped in favour of the legacy validator target; attention promoted to primary (hook-based answer-query extraction); severity variation added; CSTR sizes added | User decisions D1–D7 |
| 2026-09-23 | Attention extraction is implemented as KV-cache crop + eager re-run of the decision positions, not forward hooks. It gives the same answer-query rows at O(H·k·T) memory | Simpler and exact; verified against a full eager pass (below) |
| 2026-09-23 | Attention validation criterion is applied in **fp32** (elementwise ≤ 1e-3, measured 3.5e-4). In bf16, even a full eager pass differs elementwise by ~0.2 from incremental computation; the region-mass difference is ~0.009. The bf16 figure is reported as a measurement-noise limit | bf16 precision, not an extraction error |
| 2026-09-23 | `token_confidence` also includes digit-token variants (`tok_num_*`: the same five statistics over answer tokens containing a digit). Fixed before any pilot data was inspected | The JSON scaffolding tokens are near-deterministic and dilute span means |
| 2026-09-23 | Attention regions (FSM) are system, instruction, example, graph, query, template and generated. `template` mass is excluded from the feature set because all masses sum to 1 | Removes exact collinearity |
| 2026-09-23 | Explicit `repetition_penalty=1.0`. Qwen's `generation_config` otherwise applies 1.1 under greedy decoding, making processed scores differ from raw logits by up to 0.48 | Greedy token = argmax of raw logits |
| 2026-09-23 | FSM `max_new_tokens = 96`; pilot root seed 20260923 | Fixed before pilot |
| 2026-09-23 | The format-failure reprompt is not run in the offline FSM pilot. Format failures are counted and excluded from the semantic population | Offline pilot has one first-proposal candidate per instance (section 3) |
| 2026-09-23 | Routing is three-way: ALLOW / VERIFY / DISALLOW (the protocol's primary bypass policy plus the optional reject-before-verify arm), with time saved as the reported cost metric | User request |
| 2026-09-23 | FSM task mix kept at {5, 10, 15, 20} nodes. The one allowed prevalence adjustment is not used | User decision |
| 2026-09-23 | Stage 3 ALLOW selection rule: point rule with α = 0 on dev_thr (largest threshold with zero dev ALLOW failures); DISALLOW β = 0.10. Declared set of m = 4 policies (P1–P4, `docs/risk_certification.md`), frozen with hashes in `outputs/certification/20260923_230127_frozen_policies/` before the cert partition existed | User goal: zero failures let through |
| 2026-09-23 | Certification partition: 3000 graphs, root seed 20260924, graph-disjoint from all pilot and legacy data. Levels α ∈ {0.05, 0.02}, Bonferroni δ_i = 0.0125. Sized by the protocol rule for α = 0.05 at ~8% ALLOW coverage | Compute budget (~80 min) |
| 2026-09-23 | The agreement rule (option 2, `scripts/15`) was evaluated on the pilot test (exploratory). It is included in the declared set as P3 | Exploratory result: +5 to +27 ALLOWs, 1 failure |
| 2026-09-23 | Additional models run through the identical pipeline (`scripts/18`): Llama-3.2-3B-Instruct (second family), Qwen2.5-1.5B-Instruct (same family, smaller) and SmolLM2-1.7B-Instruct (third family). Each has its own probes, thresholds, freeze and certification. Certification claims are per model, with Bonferroni over 4 policies within each model | User request for ≥ 2 more models; 8 GB VRAM excludes ≥ 3.8B models in bf16 |
| 2026-09-24 | CSTR adapter (`src/cstr/`) imports the unmodified ctrl-alt-recover code through `car_bridge.py`. Unused LangChain/LangGraph modules are replaced by stand-ins that raise if used; no packages are installed; CAR file sha256 is recorded (its working tree has uncommitted user changes). Episodes are driven node by node (initializing → plant → monitoring), not through LangGraph | Reuse without copying or divergence |
| 2026-09-24 | Verifier determinism: rollouts use numpy's global RNG state captured at the snapshot and restore the caller's RNG afterwards. Tests cover replay, order independence, no side effects, witnesses and the simulator-error path (`tests/test_cstr_episodes.py`) | F7 in the audit |
| 2026-09-24 | Episodes differ from the historical runs: deterministic onset and severity from the spec, a per-episode simulator noise seed, and `symptoms` plus cooling-limited detector state persisting between steps (LangGraph used to drop them) | User request: no two identical faults; audit F5 side effects |
| 2026-09-24 | Severity ranges frozen from the no-LLM calibration (`configs/cstr_severity_ranges_v1.json`): fouling_max 0.6–0.9, pump factor 0.2–0.35, stuck opening 0.1–0.4, outlet factor 0.2–0.45; onset 1800–2200 s. Leak excluded (no grid action passed at any tested severity). Cool-stuck-closed opening parameterised through a FaultManager subclass | Protocol §5.3 |
| 2026-09-24 | KG context fetched live from GraphDB, frozen per dataset (`kg_context.ttl`). It is identical to the historical trace subgraph after newline normalisation (both CRLF) | Fixed-prompt population |
| 2026-09-24 | Chunked prefill (1024-token chunks through the KV cache) for prompts longer than one chunk. A single SDPA prefill of the 4.8k-token CSTR prompt falls back to the math kernel (10 GB, 61 s vs 6.4 GB, 1.5 s). Tested equal to plain generation in fp32 | 8 GB GPU |
| 2026-09-24 | With the reference prompt (v1), Qwen2.5-3B and Llama-3.2-3B first proposals barely move the setpoints: 8 distinct proposals of 30 each, 0/30 pass; sampling 0/40. User chose to test CSTR prompt v2. v2.0 (anchoring wording removed, goal and ranges stated): 30/30 identical no-change proposals, because the numbers are emitted before the reasoning. v2.1 (reasoning first): 9 distinct proposals, 2/30 pass. All on train snapshots (exploratory) | User decision |
| 2026-09-24 | Qwen2.5-7B-Instruct in 4-bit NF4 (bitsandbytes installed with user approval) was tested with v2.1: 14 distinct proposals, 0/30 pass. Not used further | User decision; negative result |
| 2026-09-24 | Severity ranges v2 (`configs/cstr_severity_ranges_v2.json`) straddle the boundary where the typical LLM move (Fin_sp −10%) passes: fouling 0.6–0.7, pump 0.35–0.45, stuck 0.3–0.45, outlet 0.35–0.55. This is a declared, milder population; v1 is reported separately as a regime where local models fail. Pilot root seed 20260926 | User decision ("if this makes sense from the paper perspective") |
| 2026-09-24 | CSTR time accounting uses the verifier wall time re-measured sequentially on an idle machine (`scripts/24`; mean 7.6 s, 0 verdict mismatches). Parallel labelling times (about 13.8 s) are inflated by contention | Honest C_v |
| 2026-09-24 | CSTR main dataset: 3000 episodes, root seed 20260927, v2 ranges; train 1400 / dev_cal 300 / dev_thr 300 / cert 700 / test_iid 300 (4198 drawn, 4149 triggered, first 3000 kept in draw order). Collection with the CAR reprompt loop (`scripts/25`): every proposal is a labelled candidate, grouped by episode. The episode stops on an exact repeat (verdict copied; deterministic verifier). Certification uses round-0 candidates only. Shared-prefix KV cache for the fixed system prompt + KG context | User request: scale to 3000; reprompts as data. Justification in `docs/cstr_data_generation.md` |
| 2026-09-23 | Chat-template date pinned (`date_string = "23 Sep 2026"`). Llama-3.x templates otherwise insert the run date, making the prompt run-date dependent. Qwen and SmolLM2 renders are byte-identical with or without it. Each record now stores `prompt_sha256` | Fixed-prompt population |
