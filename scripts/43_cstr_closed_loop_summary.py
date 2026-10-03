"""
Summarise a closed-loop run (scripts/42; docs/cstr_closed_loop_prereg.md).

Per policy: recovered / executed failure / unresolved rates, validator calls, proposals, unchecked
rejects (and how many were good), compute per episode; paired bootstrap 95% CIs (over episodes)
of the difference to always_validate; strata by no-change outcome and by fault family.

Example:
    python -m scripts.43_cstr_closed_loop_summary --run_dir outputs/cstr_closed_loop/<run>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.manifest import REPO_ROOT, write_json

ORDER = ["always_validate", "observables_probe", "combined_probe", "internals_probe", "random_matched", "never_validate"]
BASE = "always_validate"


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    rows = ["| " + " | ".join([str(i)] + [("" if pd.isna(v) else str(v)) for v in r]) + " |" for i, r in df.iterrows()]
    return "\n".join([head, "|---" * (len(df.columns) + 1) + "|"] + rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    d = REPO_ROOT / args.run_dir
    e = pd.DataFrame([json.loads(l) for l in open(d / "episodes.jsonl", encoding="utf-8")])
    e["executed_failure"] = e["outcome"].str.startswith("executed_failure")
    e["unresolved"] = e["outcome"].str.startswith("unresolved")
    n_ep = e["episode_id"].nunique()
    agg = e.groupby("policy").agg(
        episodes=("episode_id", "nunique"), recovered=("recovered", "mean"), executed_failure=("executed_failure", "mean"),
        unresolved=("unresolved", "mean"), validator_calls=("validator_calls", "mean"), proposals=("proposals", "mean"),
        unchecked_rejects=("unchecked_rejects", "mean"), good_rejected=("good_rejected", "mean"),
        t_generation_s=("t_generation_s", "mean"), t_internals_s=("t_internals_s", "mean"),
        t_grounding_s=("t_grounding_s", "mean"), t_validator_s=("t_validator_s", "mean"), t_total_s=("t_total_s", "mean"),
    ).reindex(ORDER)

    wide = {m: e.pivot(index="episode_id", columns="policy", values=m) for m in ("recovered", "validator_calls", "t_total_s",
                                                                                "t_validator_s", "executed_failure")}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_ep, size=(args.n_boot, n_ep))
    deltas = {}
    for p in ORDER:
        if p == BASE:
            continue
        deltas[p] = {}
        for m, w in wide.items():
            dv = (w[p].astype(float) - w[BASE].astype(float)).to_numpy()
            b = dv[idx].mean(axis=1)
            deltas[p][m] = {"delta": float(dv.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    strata = e.groupby(["nochange_pass", "policy"])["recovered"].mean().unstack().reindex(columns=ORDER)
    family = e.groupby(["family", "policy"])["recovered"].mean().unstack().reindex(columns=ORDER)
    outcomes = e.groupby(["policy", "outcome"]).size().unstack(fill_value=0).reindex(ORDER)

    f = lambda v: f"{v:+.3f}"  # noqa: E731
    lines = [f"# Closed-loop CSTR summary ({n_ep} episodes, run `{args.run_dir}`)", "",
             "| policy | recovered | executed failure | unresolved | validator calls / ep | proposals / ep | "
             "unchecked rejects / ep (good) | compute s / ep (gen + internals + grounding + validator) |",
             "|---|---|---|---|---|---|---|---|"]
    for p, r in agg.iterrows():
        lines.append(f"| {p} | {r.recovered:.1%} | {r.executed_failure:.1%} | {r.unresolved:.1%} | {r.validator_calls:.2f} | "
                     f"{r.proposals:.2f} | {r.unchecked_rejects:.2f} ({r.good_rejected:.2f}) | {r.t_total_s:.1f} "
                     f"({r.t_generation_s:.1f} + {r.t_internals_s:.1f} + {r.t_grounding_s:.1f} + {r.t_validator_s:.1f}) |")
    lines += ["", f"Paired difference to {BASE} (bootstrap 95% CI over episodes):", "",
              "| policy | recovered (pts) | executed failures (pts) | validator calls / ep | validator s / ep | compute s / ep |",
              "|---|---|---|---|---|---|"]
    for p, dm in deltas.items():
        c = lambda m, s=1: f"{s * dm[m]['delta']:+.2f} [{s * dm[m]['ci95'][0]:+.2f}, {s * dm[m]['ci95'][1]:+.2f}]"  # noqa: E731
        lines.append(f"| {p} | {c('recovered', 100)} | {c('executed_failure', 100)} | {c('validator_calls')} | "
                     f"{c('t_validator_s')} | {c('t_total_s')} |")
    lines += ["", "Recovered by no-change stratum:", "", md_table(strata.round(3)), "",
              "Recovered by fault family:", "", md_table(family.round(3)), "", "Outcome counts:", "",
              md_table(outcomes)]
    (d / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    agg.to_csv(d / "summary_by_policy.csv")
    write_json(d / "summary.json", {"n_episodes": n_ep, "by_policy": agg.reset_index().to_dict("records"), "paired_vs_always": deltas})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
