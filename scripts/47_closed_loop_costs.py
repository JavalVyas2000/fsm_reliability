"""
Closed-loop computational cost and savings across models (scripts/42 runs), plus the
"accept unchecked only on first proposals" rule (R0-accept) derived from the logged trajectories.

Per model and policy: recovered, failing actions executed unchecked, validator calls, compute per episode
(generation + internals pass + grounding pass + validator), and savings against always-validate at the
measured validator cost and when each validator call costs 30 s or 60 s (decisions do not depend on the
validator's latency, so the logged decisions are re-costed exactly). Break-even validator cost per policy.

R0-accept (probe policies): identical to the logged trajectory up to the first unchecked accept of a retry.
That retry is validated instead: a passing retry is executed as before (recovered, +1 validator call); a failing
one is rejected with the verifier's feedback and the episode would continue, which the log does not show. So the
unsafe count and the minimum extra validator calls are exact, and recovery is bounded: lower bound = the
logged recoveries; upper bound = lower bound + every such continued episode.

Example:
    python -m scripts.47_closed_loop_costs --runs "Qwen2.5-3B=outputs/cstr_closed_loop/<run>" ...
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.manifest import REPO_ROOT, build_manifest, write_json  # noqa: E402

OUT = REPO_ROOT / "paper_outputs"
ORDER = ["always_validate", "observables_probe", "observables_probe_r0", "combined_probe", "combined_probe_r0",
         "internals_probe", "internals_probe_r0", "random_matched", "never_validate"]
NAMES = {"always_validate": "Always validate", "observables_probe": "Observables probe",
         "combined_probe": "Obs. + internals + grounding probe", "internals_probe": "Internals-only probe",
         "random_matched": "Random routing", "never_validate": "Never validate",
         "observables_probe_r0": "Observables probe, first-proposal-only", "combined_probe_r0": "Obs. + internals + grounding, first-proposal-only",
         "internals_probe_r0": "Internals-only, first-proposal-only"}
COLORS = {"always_validate": "#2a78d6", "observables_probe": "#eb6834", "combined_probe": "#1baf7a",
          "internals_probe": "#e87ba4", "random_matched": "#eda100", "never_validate": "#898781", "observables_probe_r0": "#eb6834",
          "combined_probe_r0": "#1baf7a", "internals_probe_r0": "#e87ba4"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
PROBES = ("observables_probe", "combined_probe", "internals_probe")  # the rule bounds apply to the unrestricted probes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='"<model>=<run dir>[,<extra run dir>...]"')
    return p.parse_args()


def load(spec):
    dirs = spec.split(",")
    eps, steps = [], []
    for d in dirs:
        e = pd.DataFrame([json.loads(l) for l in open(REPO_ROOT / d / "episodes.jsonl", encoding="utf-8")])
        s = pd.DataFrame([json.loads(l) for l in open(REPO_ROOT / d / "steps.jsonl", encoding="utf-8")])
        have = set(pd.concat(eps).policy) if eps else set()
        eps.append(e[~e.policy.isin(have)])
        steps.append(s[~s.policy.isin(have)])
    e, s = pd.concat(eps, ignore_index=True), pd.concat(steps, ignore_index=True)
    common = set(e[e.policy == "always_validate"].episode_id)
    for p in e.policy.unique():
        common &= set(e[e.policy == p].episode_id)
    return e[e.episode_id.isin(common)].copy(), s[s.episode_id.isin(common)].copy()


def r0_accept(e, s, policy):
    """Exact unsafe executions / extra validator calls and recovery bounds for the R0-accept rule."""
    ep = e[e.policy == policy].set_index("episode_id")
    st = s[s.policy == policy]
    retry_acc = st[(st.decision == "accept") & (st["round"] > 0)]
    first_bad = st[(st.decision == "accept") & (st["round"] == 0) & (st.verifier_pass == False)]  # noqa: E712
    cont = retry_acc[retry_acc.verifier_pass == False].episode_id.nunique()  # noqa: E712
    n = len(ep)
    return {"unsafe_executions": int(len(first_bad)), "extra_validator_calls_min": int(len(retry_acc)),
            "recovered_lower": float(ep.recovered.mean()), "recovered_upper": float((ep.recovered.sum() + cont) / n),
            "continued_episodes": int(cont), "validator_calls_min": float((ep.validator_calls.sum() + len(retry_acc)) / n)}


def main():
    args = parse_args()
    runs = dict(a.split("=", 1) for a in args.runs)
    rows, r0rows, curves = [], [], {}
    for model, spec in runs.items():
        e, s = load(spec)
        e["executed_failure"] = e.outcome.str.startswith("executed_failure")
        e["fixed_s"] = e.t_generation_s + e.t_internals_s + e.t_grounding_s
        cv = e.t_validator_s.sum() / max(1, e.validator_calls.sum())
        g = e.groupby("policy").agg(n=("episode_id", "nunique"), recovered=("recovered", "mean"),
                                    unsafe=("executed_failure", "sum"), calls=("validator_calls", "mean"),
                                    gen=("t_generation_s", "mean"), internals=("t_internals_s", "mean"),
                                    grounding=("t_grounding_s", "mean"), validator=("t_validator_s", "mean"),
                                    fixed=("fixed_s", "mean"))
        g = g.reindex([p for p in ORDER if p in g.index])
        base = g.loc["always_validate"]
        curves[model] = (g, cv)
        for p, r in g.iterrows():
            row = {"model": model, "policy": NAMES[p], "episodes": int(r.n), "recovered_pct": 100 * r.recovered,
                   "unsafe_executions": int(r.unsafe), "validator_calls_per_ep": r.calls,
                   "calls_avoided_pct": 100 * (1 - r.calls / base.calls) if base.calls else np.nan,
                   "gen_s": r.gen, "internals_s": r.internals, "grounding_s": r.grounding, "validator_s": r.validator,
                   "measured_validator_s_per_call": cv}
            for c in (cv, 30.0, 60.0):
                t, tb = r.fixed + r.calls * c, base.fixed + base.calls * c
                key = "measured" if c == cv else f"{int(c)}s"
                row[f"compute_s_{key}"] = t
                row[f"saving_pct_{key}"] = 100 * (tb - t) / tb
            dc = base.calls - r.calls
            row["breakeven_validator_s"] = (r.fixed - base.fixed) / dc if dc > 0 and p != "never_validate" else np.nan
            rows.append(row)
        for p in PROBES:
            if p in g.index:
                r0rows.append({"model": model, "policy": NAMES[p], **r0_accept(e, s, p),
                               "logged_unsafe": int(g.loc[p].unsafe), "logged_calls": float(g.loc[p].calls),
                               "logged_recovered": float(g.loc[p].recovered)})
    T = pd.DataFrame(rows)
    R0 = pd.DataFrame(r0rows)
    T.to_csv(OUT / "tables" / "closed_loop_costs_all.csv", index=False)
    R0.to_csv(OUT / "tables" / "closed_loop_r0_accept.csv", index=False)

    md = ["# Closed-loop CSTR: outcomes, compute and savings per model", "",
          "Compute per episode = generation + internals pass + grounding pass + validator calls x validator cost. "
          "Savings are against always-validate on the same episodes; the 30 s / 60 s columns re-cost the same logged "
          "decisions for a slower validator.", ""]
    for model in runs:
        sub = T[T.model == model]
        cv = sub.measured_validator_s_per_call.iloc[0]
        md += [f"## {model} ({int(sub.episodes.iloc[0])} episodes; measured validator {cv:.1f} s/call)", "",
               "| policy | recovered | failing executed unchecked | validator calls / ep (avoided) | compute / ep: gen + int + grd + val | "
               f"saving @ {cv:.1f} s | @ 30 s | @ 60 s | break-even validator s |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            be = "" if pd.isna(r.breakeven_validator_s) else (
                "any (cheaper even at 0 s)" if r.breakeven_validator_s <= 0 else f"{r.breakeven_validator_s:.1f}")
            av = "" if r.policy == "Always validate" else f" (−{r.calls_avoided_pct:.0f}%)"
            md.append(f"| {r.policy} | {r.recovered_pct:.1f}% | {r.unsafe_executions} | {r.validator_calls_per_ep:.2f}{av} | "
                      f"{r.compute_s_measured:.1f} s = {r.gen_s:.1f} + {r.internals_s:.1f} + {r.grounding_s:.1f} + {r.validator_s:.1f} | "
                      f"{r.saving_pct_measured:+.0f}% | {r.saving_pct_30s:+.0f}% | {r.saving_pct_60s:+.0f}% | {be} |")
        md.append("")
    md += ["## Rule: accept unchecked only on first proposals (validate every retry)", "",
           "Derived from the logged trajectories (see script docstring): unsafe executions and the minimum extra validator "
           "calls are exact; recovery is bounded because a blocked failing retry would have continued.", "",
           "| model | probe | failing executed: logged → rule | validator calls / ep: logged → rule (min) | recovered: logged → rule (bounds) |",
           "|---|---|---|---|---|"]
    for _, r in R0.iterrows():
        md.append(f"| {r.model} | {r.policy} | {r.logged_unsafe} → **{r.unsafe_executions}** | {r.logged_calls:.2f} → {r.validator_calls_min:.2f} | "
                  f"{100 * r.logged_recovered:.1f}% → [{100 * r.recovered_lower:.1f}%, {100 * r.recovered_upper:.1f}%] |")
    (OUT / "tables" / "closed_loop_costs_all.md").write_text("\n".join(md), encoding="utf-8")

    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.2), sharey=False)
    for ax, (model, (g, cv)) in zip(np.atleast_1d(axes), curves.items()):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7, length=0)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        x = np.linspace(0, 60, 121)
        for p, r in g.iterrows():
            ax.plot(x, r.fixed + r.calls * x, color=COLORS[p], linewidth=2, label=NAMES[p])
        ax.axvline(cv, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(model, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("validator cost per call (s)", fontsize=8, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel("compute per episode (s)", fontsize=8, color=INK2)
    seen = {}
    for ax in np.atleast_1d(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    l, h = list(seen), list(seen.values())
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.12))
    fig.text(0.01, -0.04, "Dashed: measured validator cost. Same logged decisions re-costed for a slower validator.",
             fontsize=7, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"closed_loop_compute_all.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    write_json(OUT / "tables" / "closed_loop_costs_all.json", build_manifest(runs=runs))
    print("\n".join(md))


if __name__ == "__main__":
    main()
