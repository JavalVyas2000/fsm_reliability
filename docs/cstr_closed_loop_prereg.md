# Pre-registration: closed-loop CSTR recovery with skip-the-validator routing

**Frozen:** 2026-09-29, before any closed-loop episode was run. The author authorised running the full pipeline.

**Question.** When a trained probe decides whether to execute a proposal unchecked, send it back unchecked, or validate it, how many faulty CSTR episodes still recover, and how many validator calls and how much compute does that save, compared with always validating?

## Setup

- **Episodes:** 400 fresh episodes, 100 per fault family, v4 ranges (`data/cstr/closedloop_v4_specs.xlsx`, seed 20261010; simulated by `scripts/35` into `data/cstr/closedloop_v4`). They are disjoint from all probe training, dev, test and cert data.
- **Model:** Qwen2.5-3B-Instruct, bf16, prompt v3.1, greedy decoding, max 512 new tokens. The probes were trained on this model.
- **Probes and thresholds:** frozen in `outputs/certification/20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY`:
  - upper-bound ACCEPT rule (Amendment 3), X = 10%;
  - REJECT when the dev rejected set is ≥ 95% failures.

  This freeze was evaluated on Qwen's already-used test_iid/cert, which does not affect the closed-loop episodes.
- **Plant timing:** the plant is paused during decision-making; the snapshot does not age while the LLM or the validator runs. This supports compute and recovery comparisons, not deadline claims.
- **Execution:** executing a proposal means applying its setpoints to the plant from the snapshot, with the plant's own future noise (the RNG state captured at the trigger).
  - The outcome is the CAR rollout validator's verdict on that trajectory: recovered = reached SAFE and held it as the outcome contract requires.
  - The validator therefore doubles as the plant simulator. A "validator call" is the decision-time check; the executed outcome is charged separately as the real plant, not as compute.

## Policies

Each policy handles up to **6 proposals per episode** (rounds 0–5).

| Policy | Decision for each proposal |
|---|---|
| **Always validate** (CAR default) | validate. Pass → execute; fail → re-prompt with the verifier's measured outcome (v3.1 neutral feedback). |
| **Never validate** | execute the first proposal. |
| **Observables probe** | risk from "plant readings + proposed change". ACCEPT → execute; REJECT → re-prompt without validation; else validate as above. |
| **Observables + internals + grounding probe** | the same, with the combined probe (internals and grounding computed online for each proposal). |
| **Random routing** | ACCEPT / REJECT / validate at random with the combined probe's dev rates (ACCEPT 92/989, REJECT 427/989), seeded per (episode, round). |

**Shared rules:**
- **Unchecked-reject feedback:** "The previous proposal was rejected by a risk screen before validation: you MUST propose a DIFFERENT set of setpoints." It contains no measured metrics.
- **Repeats:** a proposal that repeats an earlier one in the same episode keeps the earlier verdict (the verifier is deterministic) and costs no new validator call.
- **Fallback:** if no proposal is executed after 6 proposals, the current setpoints are held (no change) and that outcome is recorded as *unresolved*.
- **Probes on retries:** the probes were trained on first proposals only. Applying them to retries (prompts that contain feedback) is a declared distribution shift.

## Outcomes per policy (one run of the 400 episodes)

- **Recovered:** an executed proposal passes.
- **Executed failure:** an executed proposal fails; also counted by fail reason.
- **Unresolved:** fallback reached.
- Validator calls per episode.
- Proposals per episode.
- Compute per episode:
  - generation + internals pass (+ grounding pass for the combined probe);
  - validator wall time per call.

  Measured times are charged per policy even where generations are shared through the cache.
- **Paired differences against always-validate:** recovered rate, validator calls, compute. Bootstrap 95% CI over episodes, 2000 resamples.
- **Strata:** no-change passes / fails (offline, from the no-change action).

## Expectations (not hypotheses)

- Never-validate recovers only as often as first proposals pass (about 30%).
- The probe policies should need clearly fewer validator calls than always-validate.
  - Some recovery may be lost through unchecked executions of failing proposals, and through unchecked rejections of good proposals (extra retries).
  - How much is the open question.

## Amendment 1 (2026-10-03): the other three CSTR models

Added after the Qwen2.5-3B closed-loop result, at the author's request:
- **Models and frozen probes** (Amendment 3 rules; each evaluated once on its own test_iid/cert):
  - Qwen2.5-1.5B: `outputs/certification/20261002_094601_cstr_v4_qwen25-15b_skip_ucb`;
  - Llama-3.2-3B: `outputs/certification/20261001_112113_cstr_v4_llama-32-3b_skip_ucb`;
  - Qwen2.5-7B (4-bit): `outputs/certification/20261003_164452_cstr_v4_qwen25-7b_skip_ucb`.
- **Unchanged:** the same 400 episodes, policies, rules and outcomes.
- **Random routing** uses each model's own combined-probe dev rates.
- **Reporting:** each model is reported separately. No pooling across models.

## Amendment 2 (2026-10-03): internals-only policy

Written before any internals-only closed-loop result existed.

- **New policy `internals_probe`:** risk from model internals only (token confidence, attention, hidden states, region grounding). It sees no plant readings and no proposed-action features, and uses the same frozen rules as the other probes (signal "all internals + grounding" in each model's freeze).
- **Purpose:** it measures how many simulator calls the agent's own internal state can avoid without any learned model of the plant, which answers the concern that the observable probe is merely a surrogate of the CSTR dynamics.
- **Runs:**
  - Llama-3.2-3B and Qwen2.5-7B: included in their closed-loop runs.
  - Qwen2.5-3B (done) and Qwen2.5-1.5B (already running): supplementary runs of this policy only (`--policies internals_probe`) on the same 400 episodes, paired with the main runs by episode.
- **Determinism:** generation is greedy and the verifier is deterministic, so an episode's trajectory under a given policy does not depend on which other policies run alongside it.
- **Outcomes:** as for the other policies.
