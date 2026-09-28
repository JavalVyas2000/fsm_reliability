# Selective verification — Stage 0 repository audit

Date: 2026-09-23. Scope: read-only audit of `fsm_reliability` (this repo) and
`ctrl-alt-recover` (CSTR/mixer case studies) against
`docs/selective_verification_research_plan.md`. No research code was modified,
no inference was run, no LLM/API was called. Findings marked **[verified]** were
re-checked directly after the initial audit pass; others are from code reading
and read-only analysis scripts (kept outside both repos, see §6).

Paths without a prefix are relative to `fsm_reliability`. Paths prefixed `CAR/`
are relative to `ctrl-alt-recover`.

---

## 0. Summary for decision-making

| # | Finding | Severity | Consequence |
|---|---|---|---|
| F1 | Multi-seed datasets overlap: seed `s` uses `s, s+1, s+2` for train/val/test, so `seed_42/test ≡ seed_43/val ⊂ seed_44/train` **[verified]** | Critical | Multi-seed variance is not independent replication; seed 43/44 calibrate/train on seed 42's test set |
| F2 | Token and attention features cover *all* generated tokens (incl. EOS and post-answer continuation up to the cap), but only the first line is parsed and labelled | Critical | Features are not aligned with the evaluated action. Qwen2.5-1.5B rows hit the 200-token cap in 100% of cases. Needs new inference to fix (ids/tensors not persisted) |
| F3 | Every existing FSM test split has been used in reported results across several paper iterations | Critical | All existing FSM test sets are **exploratory**. Confirmatory evaluation needs newly generated data |
| F4 | FSM provenance is broken: no model id recorded anywhere; all multiseed `run_metadata.json` point at shared, overwritten `data/processed_seed_4X` | Critical | Model identity of several processed files is inferred, not recorded |
| F5 | All CSTR "local model" runs in `CAR/traces/cstr/` were actually served by `gpt-4o-mini-2024-07-18` **[verified]** | Critical (for CAR; affects the JPC results) | `llm_model` is not declared in `GraphState` (`CAR/case_studies/cstr_case/cstr_case.py:435-514`), so LangGraph drops it and `action_propose` falls back to `gpt-4o-mini` (`:1268`). Directories named `llama3_8b`, `qwen3_8b`, `mistral_7b`, `phi3_3.8b`, `gpt-4.1-mini` all contain gpt-4o-mini responses |
| F6 | CSTR fallback is never applied: `route_after_evaluation` mutates state inside a router (`cstr_case.py:1879-1893`) **[verified: code]**. The audit found all 66 retry-exhausted runs applied the last *rejected* proposal | Critical (CAR) | Existing "success" metrics and the fallback path are unreliable |
| F7 | CSTR verifier verdict depends on hidden global RNG state; repeated evaluation of the same (snapshot, action) gives different verdicts and perturbs the real plant's future noise | Critical for labels | Verdict is not a function of (snapshot, action). Snapshots must include RNG state, or use isolated generators |
| F8 | The CSTR validator accepts near-shutdown actions (`Fin_sp = 0.001` passed for all three paper faults) and never requires production or T tracking | High | Validator acceptance ≠ productive recovery; an outcome contract is needed before labelling |
| F9 | Metadata shortcuts are strong: `shortest_length` gives AUROC 0.70–0.99; pooled attention entropy has ρ≈0.97 with `num_nodes`; `num_generated_tokens` (in `token_only`) proxies output length | High | No existing baseline tests whether internals add value beyond task/output structure |
| F10 | Existing CSTR verifier cost: ~7 s/call (4.1–10.4 s, n=21), ~1.3–1.8 ms per simulated second, no early exit, simulates through shutdown to `T_end` | Informative | `C_v` is moderate and could be reduced substantially by engineering; a cheap-surrogate baseline is required for an honest cost claim |

---

## 1. FSM repository (`fsm_reliability`)

### 1.1 Data generation and labels

- Graphs are directed Erdős–Rényi reachability graphs, not labelled FSMs.
  Each ordered pair is kept with probability `edge_prob` (`src/data/graph_utils.py:11-68`).
- Queries: `start ≠ goal`, and the goal is always reachable (`graph_utils.py:139-179`).
  - Consequence: the prompt's `{"path": []}` option (`src/prompts/fsm_prompts.py:26,31`) is **always wrong**.
  - `min_path_length=2` counts nodes, so ~35% of queries are a single edge.
- `build_default_dataset` hard-codes `edge_prob={5:.35,10:.30,15:.25,20:.20}` at `src/data/generate_fsm_dataset.py:190`, ignoring its argument **[verified]**.
  - The CLI values in `scripts/01` (0.24) and `scripts/07` (0.30) therefore do not match the data.
- Seeding **[verified]**: train/val/test use seeds `s, s+1, s+2` (`:191-211`), and each node-count block uses `seed+idx` (`:155`). This causes F1.
- `instance_id = f"{split}_n{N}_p{p}_{idx:05d}"` is not globally unique across seeds or families.
- Labels (`src/data/labels.py`, `src/evaluation/correctness.py`):
  - `valid_path`: non-empty, correct endpoints, every hop is an edge. Cycles are allowed.
  - `optimal_path`: valid and the same node count as BFS.
  - Suboptimal-but-valid is `valid ∧ ¬optimal`; there is no explicit column.
  - A parse failure is scored as `valid=0` (`correctness.py:20-28`), which conflates format failure with semantic invalidity.

### 1.2 Split content audit

| Relation | Shared graph+query rows |
|---|---|
| Within any one family, train/val/test | ≤1 graph or prompt (≤0.2%) |
| `raw_multi_node_large` vs `raw_seed_42` | byte-identical |
| `raw_seed_42/test` → `raw_seed_43/val` | 500/500 |
| `raw_seed_42/test` → `raw_seed_44/train` | 500/3000 |
| `raw_seed_43/test` → `raw_seed_44/val` | 500/500 |
| `raw_seed_42/val` → `raw_seed_43/train` | 500/3000 |
| `raw_10node_large/val` → `raw_multi_node_large/train` | 500/500 of val |
| `raw_10node_large/test` → `raw_seed_43/train` | 500/500 of test |
| `raw/test` → `raw_seed_43/train` | 200/200 of test |

- 12,000 multi-seed rows contain only 9,995 distinct graph+query pairs.
- Greedy decoding makes the processed rows for shared instances byte-identical, including features and labels.
- Model variants of the same instances exist across `data/processed/*_multi_*`, `data/processed_seed_*`, and `data/{qwen_25_15b,llama32_1b,smollm_1b}/`.
  - Pooling across models repeats prompts; group by `(graph, query)`.

**Grouping required for every future split:** group key = canonical
`(graph adjacency, start, goal)`. Groups must be disjoint across train, dev,
certification and test, and across any datasets that are combined. A stricter key
(canonical graph only) is required if a claim concerns generalization to unseen
graphs.

### 1.3 Existing processed data

Valid rate (test), with the model inferred from outputs and layer count; no file records a model id:

| Source | Model | Valid rate | Notes |
|---|---|---|---|
| `data/processed_seed_4X` | Llama-3-8B (inferred, 32 layers) | 0.30–0.32 | untracked in git; outputs `[start, 3, goal]` (copying the prompt example) in 51% of rows |
| `data/qwen_25_15b/*` | Qwen2.5-1.5B | 0.26–0.27 | never emits EOS; 100% of rows hit the 200-token cap |
| `data/smollm_1b/*` | SmolLM | 0.33–0.35 | outputs `[start, goal]` in 94.6% of rows; `shortest_length` AUROC 0.99 |
| `data/llama32_1b/*` | Llama-3.2-1B | 0.004–0.012 | ~98.5% `{"path": []}`; degenerate |
| `data/processed/*_multi_phi35_mini` | Phi-3.5-mini | 0.46–0.50 | |
| `data/processed/*_multi_llama32_3b` | Llama-3.2-3B | 0.34–0.36 | |
| `data/processed/*_10node` | unknown (probably Qwen2.5-1.5B) | 0.24 | 24-token cap; truncation-driven parse failures |

- Missing values: only `parsed_path`, and exactly on parse failures. No feature column has NaNs.
- `pilot_features.csv` and `data/runs/sanity_inference.jsonl` are inconsistent with the current parser. All other files re-score with 0 label mismatches.

**Implication for the protocol:** at the observed failure prevalence (50–75%),
certifying a 1% conditional direct-execution risk would leave very little
coverage. The pilot model and task distribution must be chosen with this in mind
(see protocol §4).

### 1.4 Inference and feature extraction

| Aspect | Current state | Cite |
|---|---|---|
| Model loading | `device_map="auto"`, `torch_dtype="auto"`, no `revision`, no `attn_implementation` | `src/models/load_model.py:20-37` |
| Prompt | raw completion string ending in `Output:`; the chat variant exists but is unused; the few-shot examples cause copying | `src/prompts/fsm_prompts.py:4-69` |
| Decoding | greedy; explicit `eos_token_id` overrides the model's generation-config EOS list; no stop strings | `src/models/run_inference.py:24-32` |
| Scores | `generation_output.scores`, i.e. **after logits processors**. Repetition penalty from `generation_config` still applies under greedy decoding. Raw logits (`output_logits`) are never collected | `run_inference.py:28-29,64` |
| Evaluated answer | decoded text, stripped, **first line only** | `run_inference.py:38-40` |
| Token features | over all generated steps, including EOS and post-answer text | `src/features/token_confidence_features.py:45-76`; `scripts/03_extract_pilot_features.py:132-135` |
| Attentions | second full forward pass, `use_cache=False`, all layers materialised at once, whether eager is used depends on the transformers version | `run_inference.py:44-55` |
| Pooled attention | row entropy/max averaged over all heads **and all query positions, including the prompt**, so it largely encodes sequence length | `src/features/attention_features.py:56-78` |
| Region attention | masks found by literal prompt substrings; `prompt_mass + output_mass = 1` (collinear) | `src/features/attention_region_features.py:108-270` |
| Hidden states | never collected | — |
| Persistence | one CSV row of scalars per example; no ids, tensors, per-token log-probs, or full raw text | `scripts/03:155-188` |
| Latency / memory | never recorded | — |
| Token budget | `choose_max_new_tokens` reads the ground-truth path length (oracle); `scripts/07` default of 24 can truncate | `scripts/03:66-79`; `scripts/07:34` |

### 1.5 Training, calibration and evaluation

- Target: `valid_path`, with **positive class = correct**. AUROC, AUPRC and ECE are reported for the correct class. The protocol needs the failure class as positive for AUPRC.
- Features come from named allowlist groups intersected with an exclusion list (`scripts/06_calibrate_probability_model.py:23-75,204-316`).
  - `feature_group="all"` relies only on exclusions.
  - The group logic is copy-pasted in three files.
- Preprocessing (median imputation, standard scaling, balanced logistic regression) is fit on train only (`06:346-355`). Good.
- The isotonic calibrator **and** the routing thresholds are both fit on val (`06:358-366, 647-661`).
  - Threshold fallbacks are silent (`06:411-436`); `low=0.0` occurs in most runs.
- ECE bug: `np.digitize` puts `p == 1.0` into an unsummed bin (`06:156-170`, and duplicated in two other files). Isotonic outputs hit 1.0 for up to 6% of rows.
- `scripts/05` is broken: `src/training/__init__.py` imports a non-existent `build_reliability_table`.
- `probability_model.evaluate_feature_groups` would rank groups by test AUROC; it is unreachable today.
- `visualize_internal_risk.py` reports metrics on training rows using all numeric columns, including `num_nodes` and `path_length`, so it leaks.

### 1.6 Reproducibility

- `requirements.txt` is unpinned and missing scikit-learn, numpy, matplotlib and tqdm. There is no lockfile, no tests, no README, and `configs/` is empty.
- Not recorded anywhere: software versions, hardware, model or tokenizer revision, dtype, decoding config, dataset hashes, prompt version, git commit.

### 1.7 Reusable FSM components

- Graph generation, BFS and serialisation (`graph_utils.py`), once the seeding and `edge_prob` are fixed.
- Label functions (`labels.py`, `correctness.py`). Parse status needs to be separated from validity.
- `output_parser.py` (JSON then list fallback, with `parse_mode`).
- `build_prompt_regions` (offset-mapping region masks); extend it to the parsed-answer span.
- `load_hf_model_and_tokenizer`, once `revision`, `attn_implementation` and `dtype` are added.
- From `06`: `fit_model`, `wilson_interval`, `routing_table`, and the plotting helpers, once the ECE is fixed and calibration and thresholds are separated.
- `build_selective_table_by_coverage` (`src/training/probability_model.py:285-330`) and `compute_risk_coverage` (`scripts/plot_risk_coverage.py:23-59`).

---

## 2. CSTR case study (`ctrl-alt-recover`)

### 2.1 LLM and prompting

- `_make_llm` (`CAR/case_studies/cstr_case/cstr_case.py:161-177`): `ChatOpenAI` if the name starts with `gpt`, otherwise `ChatOllama` on localhost. Temperature 0.
  - No logprobs are requested; stored responses have `"logprobs": null`.
  - Neither backend exposes hidden states or attention. **A Hugging Face backend is required for any internal features.**
- The prompt is built inline in `action_propose` (`:1180-1266`).
  - It contains a static knowledge-graph Turtle subgraph (fetched once from GraphDB at `localhost:7200`), a snapshot of measured values, and grid rules. Stored prompts are ~4.5k tokens.
- Action schema `SetpointPlan{T_sp, Fin_sp, L_sp, reasoning}` (`:180-184`). There are **no bounds and no grid enforcement**.
  - Parse failures abort the episode; there is no reprompt.
  - A repeated proposal after a failure silently gets `Fin_sp *= 0.9` (`:1405-1413`), so the validated action can differ from the emitted one.
- Undeclared `GraphState` keys are dropped. Besides F5, this makes `symptoms` always `{}` and the cooling-limited detector dead.

### 2.2 Graph and routing

- There is one intervention per episode (`should_act` requires `not sp_change_done`, `:1146`), and `reprompt_max=5`.
- The plant is frozen while the LLM and validator run, so decision time never affects the plant.
- Episode success is defined as validator acceptance (`:2099-2103`), not physical outcome.

### 2.3 Simulator

- `CSTRSimulation` (`CAR/case_studies/cstr_case/cstr_digital_twin.py:244-548`):
  - state: six ODE states, three actuators with dynamics, and three PID integrators;
  - `dt=1 s`, RK45 per step;
  - phases: STARTUP → NORMAL → SHUTDOWN at t=7000.
- Faults supported via the CLI: normal, fouling, pump_degrade, cool_stuck_closed. `cool_stuck_closed` forces the cooling valve to 0.3, not 0.
  - Onset for fouling, pump_degrade and cooling faults is `2000 ± U(200)` s from **unseeded** Python `random` (`cstr_case.py:719-728`).
- Noise comes from the global `np.random`, which the constructor reseeds with 42 (`cstr_digital_twin.py:280-281`).
- `clone_plant_obj` (deepcopy, `cstr_case.py:1434-1450`) is **not a complete snapshot**: it excludes the NumPy and Python RNG states.

### 2.4 Verifier (`rollout_validate_setpoints`, `cstr_case.py:1492-1728`)

- Holds the proposed triple constant and simulates to `T_end`, through shutdown. There is no early exit.
- `safe_now` requires measured `T ≤ 313` (noisy, σ=0.5 K), `2 ≤ L_true ≤ 11`, and no overflow.
- UNSAFE if an actuator is saturated at ≥0.95 (or trending towards it) while |T_meas−310| > 0.5.
- PASS requires 60 consecutive safe s before 600 s, and `unsafe_fraction ≤ 0.2`.
  - The fraction's denominator includes grace and shutdown steps, which dilutes it.
- Not required: production, T tracking, or recovery to the original setpoint. Relapse is tolerated up to the fraction budget.
- Returns `(ok, summary, fail_reason, metrics, trajectory)`. The trajectory holds ~5800 dicts per call.
  - Bug: `recovered_now` reads the wrong key (`:1624`), so it is always False.
- Measured on an Intel Core Ultra 9 185H, Python 3.13.1, by extracting the functions via `ast` because langgraph is not installed:

| Fault | Steps/call | Wall/call |
|---|---|---|
| cool_stuck_closed | 5802 | 7.7–10.4 s |
| fouling | 3244 | 4.1–4.4 s |
| pump_degrade | 5807 | 7.2–9.5 s |

  - Mean 6.95 s (n=21). A bare `simulate_step` costs ~1.2–1.6 ms; `deepcopy` ~70 µs.
- Verdict sensitivity observed:
  - near-shutdown `Fin_sp=0.001` → PASS for all three faults;
  - `T_sp=300` → FAIL for all three;
  - pump_degrade passes even with "no change".
  - Outcome variation therefore exists for cool_stuck_closed and fouling, but pump_degrade is nearly always acceptable.

### 2.5 Existing CSTR data

- 874 `validation_decision` events across 13 experiments, 236 of them accepted. Each has its prompt and raw response.
- Per fault:

| Fault | Accepted | Rejected |
|---|---|---|
| cool_stuck_closed | 40 | 431 no_recovery, 40 slow_recovery |
| fouling | 96 | 167 unsafe_fraction |
| pump_degrade | 100 | 0 |

- **Not usable as pilot labels for this project:**
  - one model (gpt-4o-mini) regardless of directory name;
  - only 26 distinct setpoint triples;
  - the same triple receives both verdicts depending on t0 and RNG;
  - no snapshot or RNG state, so decisions cannot be replayed;
  - no internals.

### 2.6 Existing recovery-gap protocol

`CAR/docs/recovery_gap_study.md` (17 Sep 2026) is a protocol draft, not a results document. It already calls for:
- an outcome contract with production, deadline, hold and relapse requirements;
- deterministic snapshots and RNG handling;
- the witness-versus-unresolved distinction.

It does **not** record F5, F6 or the near-shutdown pass. The CSTR adapter in this
project should adopt that document's outcome contract rather than define a
competing one.

### 2.7 Gaps for `verifier.verify(snapshot, action, config)`

1. **Snapshot:** plant deepcopy plus isolated RNG state (a per-snapshot `np.random.Generator`, or a pre-drawn disturbance tape), `t0`, the evaluation horizon, mode, fault config and severity, and the fault onset time.
2. **Isolation:** verification must not consume the executed plant's RNG stream. Restore the caller's state afterwards.
3. **Horizon:** a declared operating window that ends before shutdown, plus an early-exit option that preserves verdict semantics. Version the contract if the semantics change.
4. **Outcome contract:** production floor, T tracking, and verdicts on true state, kept separate from sensor-based verdicts.
5. **Action contract:** bounds and grid checks; record any post-hoc modification such as the 0.9 nudge.
6. **Compact output:** verdict, reason codes, extrema, timing, cost, and a trajectory reference.
7. **Upstream CAR bugs (F5, F6, dropped state keys):** these need fixing in `ctrl-alt-recover` itself, which requires the user's approval because that repo has uncommitted user changes. Until then the adapter must call the simulator and checker primitives directly, not the LangGraph application.

### 2.8 Environment

- langgraph and langchain are not installed in the Python 3.13 environment used for the audit.
- CAR has no pinned requirements, and it depends on GraphDB (and Ollama for local models).
- A `.env` file with API keys exists. Its values were not read.

---

## 3. Minimum additions before Stage 1

1. New dataset generator entry point with:
   - disjoint seed streams (e.g. `SeedSequence(root).spawn`);
   - honoured `edge_prob`;
   - globally unique, content-hashed ids and group keys;
   - a dataset fingerprint;
   - a cross-dataset overlap check against all existing `data/raw*`.
2. An inference path that persists generated ids, per-token raw log-probs (`output_logits=True`) *and* processed scores, answer-span token indices, latency, and peak memory. It should use:
   - the chat template for instruct models;
   - explicit `attn_implementation`, revision and dtype;
   - a stop criterion at the end of the first JSON object or line.
3. Features restricted to the parsed-answer token span, with an explicit allowlist per feature family.
4. A `context_action` baseline family (`num_nodes`, `num_edges`, `shortest_length`, parsed path length, and output-pattern indicators such as copy-of-example).
5. Fixes: failure as the positive class for AUPRC, the ECE edge bin, separate calibration and threshold partitions, and non-silent threshold fallbacks.
6. A run manifest recording versions, hardware, model or tokenizer revision, precision, decoding config, prompt version, dataset fingerprint, and git commit.

## 4. Unresolved assumptions

- GPU availability and VRAM for `fsm_reliability` inference. This determines feasible model sizes and whether attention extraction is affordable.
- Whether the user wants CAR bugs F5 and F6 fixed upstream now. They affect the JPC paper's reported per-model comparisons independently of this project.
- Which pilot model(s): existing small models have 50–75% failure prevalence, which makes low-alpha bypass nearly empty.
- Whether the prompt may be changed (chat template, removing the copyable few-shot example, dropping the always-wrong `[]` option). Any change makes a new prompt version incomparable with old results.

## 5. Proposed pilot

See `docs/selective_verification_protocol.md` §9.

## 6. Audit artefacts

- The read-only analysis scripts used for §1.2, §1.3 and §2.4 were run from a session scratchpad outside both repositories. They are not part of this repo and are not needed to reproduce later stages.
- The facts above that later stages depend on (F1, F2, F5, F6, F7) will be re-established by tests or checks in the relevant stage, not by reference to this audit.
