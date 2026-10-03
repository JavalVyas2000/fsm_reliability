# Closed-loop CSTR results (pre-registered: `docs/cstr_closed_loop_prereg.md`)

**Date:** 2026-10-03.
**Run:** `outputs/cstr_closed_loop/20261003_033144_qwen25-3b`, 400 fresh episodes (100 per fault family), Qwen2.5-3B, prompt v3.1, up to 6 proposals per episode.
**Probes:** frozen in `outputs/certification/20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY` (upper-bound ACCEPT rule, X = 10%).
**Full tables:** `summary.md` in the run folder (`scripts/43`).

## Main result

| Policy | Recovered | Executed a failing action | Unresolved (6 proposals used) | Validator calls / ep | Proposals / ep | Compute / ep |
|---|---|---|---|---|---|---|
| Always validate (CAR default) | **43.8%** | 0 | 56.2% | 2.45 | 3.95 | 75.2 s |
| Observables probe | 41.0% | **0** | 59.0% | 1.30 | 4.08 | 73.3 s |
| Observables + internals + grounding probe | 37.0% | **1 (0.25%)** | 62.7% | 0.69 | 4.22 | 77.4 s |
| Random routing at the combined probe's rates | 37.8% | **110 (27.5%)** | 34.8% | 1.31 | 3.69 | 66.6 s |
| Never validate | 30.0% | 274 (68.5%) | 1.5% | 0 | 1.00 | 16.5 s |

Paired differences against always-validate (bootstrap 95% CI over episodes):

| Policy | Recovered (pts) | Validator calls / ep | Validator time / ep | Total compute / ep |
|---|---|---|---|---|
| Observables probe | −2.75 [−5.75, +0.25] | −1.15 [−1.32, −0.97] (**−47%**) | −9.8 s | −1.9 s [−4.0, +0.3] |
| Combined probe | −6.75 [−10.00, −3.50] | −1.76 [−1.95, −1.59] (**−72%**) | −15.1 s | +2.2 s [−0.3, +4.7] |
| Random routing | −6.00 [−9.00, −3.00] | −1.14 | −10.4 s | −8.6 s |
| Never validate | −13.75 [−17.00, −10.50] | −2.45 | −20.1 s | −58.7 s |

## What it shows

1. **The trained probes are safe when they skip the validator.**
   - The observables probe never executed a failing action unchecked; the combined probe did once in 400 episodes.
   - Random routing with the same skip rates executed a failing action in 27.5% of episodes.

   The probes know which proposals can be trusted. Unchecked execution, the step that matters for safety, held up in closed loop on fresh episodes.
2. **Validator calls drop sharply.**
   - Observables probe: −47%, for a recovery change of −2.8 points whose CI includes 0.
   - Combined probe: −72%, at a recovery cost of −6.8 points [−10.0, −3.5].
3. **Recovery is lost through rejection, not through unsafe execution.**
   - Each unchecked reject needs a new proposal, and some rejected proposals were good: 0.12 and 0.20 per episode.
   - More episodes therefore use up the 6-proposal budget (unresolved: 56% → 59% / 63%).
   - The loss is concentrated in cool_stuck_closed (39% → 27% / 21%). For outlet_block and pump_degrade, recovery is unchanged.
4. **Total compute does not fall in this setting.**
   - A rejected proposal is replaced by a new generation (about 16 s on this laptop), which costs more than the validator call it avoids (about 10 s, wall time with 4 parallel workers).
   - Skipping saves time only through ACCEPTs (one validator call saved, nothing regenerated). Rejecting saves time only when the validator costs more than generating a proposal.
   - With a higher-fidelity or longer-horizon validator, the 47–72% fewer calls translate into time saved. With CAR's 10 s rollout and a local 3B model, they do not.
5. **Internals and grounding in closed loop:** the combined probe saves 0.6 more validator calls per episode than the observables probe. It also loses 4 more points of recovery, and its extra feature passes (internals 2.7 s + grounding 2.4 s per episode) cancel its time advantage. In CSTR, the observables probe is the better operating point. This is consistent with the offline results, where grounding never helped in CSTR.

## Caveats

- **One model** (Qwen2.5-3B).
- **Plant paused** while the model decides.
- **The validator is also the plant simulator.**
- **Probes on retries:** they were trained on first proposals and applied to retries.
- **Exploratory freeze:** the probe rules come from a freeze whose test sets had been used for the exploratory UCB re-analysis. The closed-loop episodes themselves are fresh.
- **Compute times** are this laptop's (RTX 4060 8 GB, 4 verifier workers).
