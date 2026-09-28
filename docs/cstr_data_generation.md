# How CSTR faults and recovery actions are generated (Stage 4 scale-up)

This document justifies the data-generation procedure for the 3000-episode CSTR
run. It states what is taken unchanged from ctrl-alt-recover (CAR), what differs
and why, and what each design choice allows the paper to claim.

## 1. The plant

**Code.** The simulator is CAR's `CSTRSimulation` (`case_studies/cstr_case/cstr_digital_twin.py`),
used unmodified and imported rather than copied.

**Model.**
- Six ODE states: volume, four concentrations, temperature.
- An inlet valve and a cooling valve, each with first-order lags.
- An outlet pump.
- Three PID loops: flow → Fin, level → pump, temperature → cooling.
- Measurement and inlet-composition noise.

**Operating phases.**

| Phase | Time |
|---|---|
| STARTUP | until the level reaches its setpoint |
| NORMAL | until 7000 s |
| SHUTDOWN | after 7000 s |

**Nominal setpoints.** T_sp = 310 K, L_sp = 10, Fin_sp = 0.0333 L/s.

## 2. Fault generation — one independent draw per episode

Each episode draws the following independently. The draws come from
`numpy.random.SeedSequence(20260927)`, with a separate stream per partition.

1. **Fault family** (uniform over four). Every family is one of CAR's own `FaultManager` mechanisms:

   | Family | Physical meaning | Severity parameter (continuous, uniform) |
   |---|---|---|
   | fouling | heat-transfer coefficient UA decays towards (1 − fouling_max)·UA with time constant τ | fouling_max ∈ [0.6, 0.7], τ ∈ [1500, 2500] s |
   | pump_degrade | outlet pump delivers only a fraction of commanded flow | factor ∈ [0.35, 0.45] |
   | cool_stuck_closed | cooling valve stuck at a fixed opening | opening ∈ [0.30, 0.45] |
   | outlet_block | outlet partially blocked | factor ∈ [0.35, 0.55] |

2. **Onset time** ∈ [1800, 2200] s, always inside the NORMAL phase.
3. **Simulator noise seed.** CAR's constructor otherwise reseeds every run with 42, which is why historical runs of the same fault behaved identically.

As a result, **no two episodes share the same fault instance**: family, severity,
onset and noise all differ.

In the pilot:
- all 300 prompts were distinct;
- trigger times were distinct in 283 of 300 cases (counted within partitions). They are whole seconds, so different episodes can coincide.

**Why these families.**
- The three paper faults (fouling, pump degradation, stuck cooling) are included.
- Outlet blockage is a fourth mechanism already in CAR's simulator.
- Leak is excluded: in calibration no tested setpoint action recovered it at any severity. A failed search does not prove it is unrecoverable; the exclusion is stated as a scope limit.

**Why these severity ranges** (`configs/cstr_severity_ranges_v2.json`).
- A no-LLM calibration (5 families × 5–7 severities × 3 seeds × 16 setpoint actions) mapped where recovery is possible at all, and how large a setpoint change it needs.
- A boundary check then showed that the tested local models propose feed cuts of about 10%. These recover only at the mild edge of the originally calibrated ranges.
- The ranges were therefore placed to **straddle that boundary**:
  - at the mild end, no change or a small change recovers;
  - towards the severe end, cuts of more than 15% are needed.
- The outcome therefore depends on the specific instance **and** on the size of the proposed correction. That is the regime where predicting "will this pass verification?" is non-trivial and worth something.
- The more severe original ranges are reported separately as a regime where these models do not produce recovering first proposals. Claims are restricted to the declared population.

## 3. When the agent is asked to act

The episode runs from t = 0 through CAR's own monitoring node (`monitoring()`)
until its first action trigger. This is the same trigger logic as the CAR agent:
- a temperature-threshold detector with persistence and a settle gate;
- a cooling-limited detector;
- a level error above 0.7;
- entry into the UNSAFE control zone.

At that instant the **snapshot** is taken:
- a full copy of the plant, including controller integrators and actuator states;
- the random-generator state, i.e. the noise the plant will see next;
- the monitoring outputs shown to the agent.

As in CAR, the plant is frozen while the agent proposes and the verifier checks.
In the pilot, 415 of 420 drawn episodes triggered; the 5 that did not (mild
stuck-cooling cases) are recorded and excluded.

## 4. How recovery actions are generated

**Proposer.** Qwen2.5-3B-Instruct (bf16), greedy decoding, run locally so that
its internal signals can be read.

**Action space.** CAR's supervisory setpoints only: T_sp, L_sp, Fin_sp (its
`SetpointPlan` schema). There are no direct valve or pump commands.

**Prompt v2.1** (`src/cstr/prompt_v2.py`):

| Part | Content | Source |
|---|---|---|
| System prompt | CAR's agent role, its knowledge-graph usage rules and causal hints (PO5–PO8, e.g. "if cooling is saturated, reduce Fin_sp") | CAR, unchanged |
| KG context | the KG Turtle context from GraphDB | CAR's own query and formatting function; frozen per dataset, identical to the historical traces |
| Snapshot block | the measured values the agent sees | cut verbatim from CAR's own prompt for that snapshot |
| Goal and ranges | the validation goal stated plainly; the admissible setpoint ranges | added |
| Output format | reasoning first, then the three numbers | changed from CAR |

What was removed and why:
- CAR's "prefer SMALL changes" and 0.05-grid wording were removed.
- The output order was changed because, with CAR's numbers-first order, the models copied the current setpoints in 30 of 30 cases and only afterwards reasoned that the feed should be reduced.

No privileged information (fault family, severity, onset) is ever in the prompt.

**Cheap admissibility check** before any simulation: parseable JSON with numeric
values inside the declared ranges (T_sp 290–330 K, L_sp 2–11, Fin_sp 0–0.1 L/s).

**Observed behaviour** in the pilot (300 episodes):
- 35 distinct first proposals; 59% are the same move (feed cut to 0.03).
- About 36% pass overall, ranging from 17% to 51% by family.
- In 63 episodes the proposal passes where "no change" would fail. So the model's action matters, but its correction is often too small.

## 5. How actions are verified (the label)

**Validator.** CAR's `rollout_validate_setpoints`, unmodified and with CAR's
knobs. The proposed setpoints are held constant while a copy of the plant is
simulated from the snapshot to the end of the run.

**PASS requires:**
- reaching SAFE within 600 s;
- then staying SAFE for 60 consecutive seconds;
- spending no more than 20% of the time unsafe.

SAFE means:
- measured T ≤ 313 K;
- 2 ≤ level ≤ 11;
- no overflow;
- no saturated actuator while the temperature is off target.

**Determinism.** Each rollout uses the snapshot's own future noise and leaves
the caller's random state untouched. The verdict is therefore a function of
(snapshot, action) only, independent of evaluation order.
- Tested: replay, order independence, no side effects, known pass/fail witnesses, simulator error → "unknown" (never a pass).
- Re-timing 50 pilot verdicts sequentially reproduced every label.

**What a PASS means.** Acceptance by CAR's validator, which is what the paper's
cost claim replaces. It is **not** proof of productive recovery: the validator
also accepts near-shutdown feeds. The paper states this, and the accepted-feed
distribution is reported as a diagnostic.

**Cost.** 7.6 s mean per verification, measured sequentially on this machine.

## 6. Reprompt rounds as additional data

Following CAR's loop:
- After a failed verification, the model is re-asked with CAR's own `reprompting()` feedback text, e.g. `FAIL: did not reach SAFE…; hint=Reduce load (Fin_sp down)…`.
- The previous proposal is included, with the instruction to propose a different triple.
- Up to 5 reprompts (CAR's `reprompt_max`).

**Stop rules:**
- first passing proposal (as in CAR);
- last reprompt;
- unparseable answer (CAR aborts the episode);
- **exact repeat of an earlier proposal of the episode**. Its verdict is known because the verifier is deterministic; it is recorded with the copied label and the episode ends. CAR instead nudges the feed by 10%, which would make the verified action differ from the LLM's.

**Use in the analysis:**
- Every proposal is a labelled candidate: `<episode>_r<round>`.
- Reprompt candidates are used for **training and descriptive analysis**, always grouped by episode.
- The round number and previous fail reason are observable (they are in the prompt), so they are features.
- **Certification uses only round-0 candidates**, one per episode, so that certified samples are independent.
- Reprompt candidates are conditioned on verifier feedback. A deployed router that skipped verification would not have that feedback, so reprompt-round results describe the full CAR loop, not the bypass policy.

## 7. Partitions (3000 episodes; episodes are disjoint by construction)

| Partition | Episodes | Role |
|---|---|---|
| train | 1400 | fit probes (all rounds) |
| dev_cal | 300 | probability calibration |
| dev_thr | 300 | threshold selection |
| cert | 700 | certification of frozen policies (round 0 only; not analysed before the freeze) |
| test_iid | 300 | final reporting |

**Certification power.** At the pilot's ALLOW share (about 14%), 700 certification
episodes give about 100 round-0 ALLOWs:
- enough to certify α = 0.05 for one declared policy with at most 1 failure (59 needed with 0 failures, 93 with 1);
- borderline for 4 policies under Bonferroni (86 with 0 failures).

## 8. Runtime and how to run it

**Measured with the collector** (Qwen2.5-3B, shared-prefix cache, 4 verifier workers):
- about 10–12 s per proposal;
- about 2.2 proposals per episode.

**Estimates:**

| Scope | Time |
|---|---|
| Round 0 only (3000 first proposals) | ≈ 8.5–9.5 h |
| Full loop with reprompts (≈ 6600 candidates) | ≈ 18–22 h |

**Two nights, resumable.** Rounds are scheduled round-major: every first proposal
before any reprompt. **After night 1 all 3000 first proposals are complete**, and
reprompt rounds continue on night 2 from where the run stopped.

```
cd C:\Users\jv624\Desktop\fsm_reliability
set PYTHONIOENCODING=utf-8

:: night 1 (stops cleanly after 11 h)
python -m scripts.25_cstr_collect --dataset_dir data/v2/cstr_main_seed20260927 --tag main3000 --max_hours 11 --local_files_only

:: night 2 (resume the same run directory)
python -m scripts.25_cstr_collect --dataset_dir data/v2/cstr_main_seed20260927 --run_dir outputs/cstr_collect/<run from night 1> --max_hours 11 --local_files_only
```

**Keep the laptop plugged in and awake,** and do not run other GPU work. If the
run is interrupted, re-run the same command with `--run_dir`; completed
candidates are never redone.
