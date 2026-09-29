"""
One table across domains and models: validator calls skipped per signal (scripts/39 + 40 runs).

Example:
    python -m scripts.41_skip_validator_summary --runs "CSTR Qwen2.5-3B=outputs/certification/<frozen>" ...
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='"<name>=<frozen dir>"')
    return p.parse_args()


def main():
    args = parse_args()
    runs = dict(a.split("=", 1) for a in args.runs)
    rows = []
    for name, d in runs.items():
        d = REPO_ROOT / d
        base = json.loads((d / "rules.json").read_text()).get("baseline", "plant readings + proposed change")
        deltas = pd.read_csv(d / "eval_test_iid_delta_vs_baseline.csv")
        for part in ("test_iid", "cert"):
            ev = json.loads((d / f"eval_{part}.json").read_text())
            ub_key = next(k for k in ev["rows"][0] if k.startswith("cp_upper"))
            for r in ev["rows"]:
                sig = "observables (context + action)" if r["signal"] == base else r["signal"]
                dl = deltas[(deltas.tolerance == r["tolerance"]) & (deltas.signal == r["signal"])]
                rows.append({"run": name, "partition": part, "tolerance": r["tolerance"], "signal": sig,
                             "n": r["n"], "failure_rate": ev["failure_rate"], "skipped_pct": r["skipped_pct"],
                             "accepted": r["accepted"], "accepted_failures": r["accepted_failures"],
                             "cp_upper": r[ub_key], "certified": r["certified"], "rejected": r["rejected"],
                             "good_rejected_pct": r["good_rejected_pct_of_good"],
                             "test_delta_vs_obs": None if part != "test_iid" or dl.empty else
                             f"{dl.delta_skipped_pts.iloc[0]:+.1f} [{dl.ci95_lo.iloc[0]:+.1f}, {dl.ci95_hi.iloc[0]:+.1f}]"})
    T = pd.DataFrame(rows)
    out = make_run_dir(REPO_ROOT / "outputs/cross_model", "skip_validator_summary")
    T.to_csv(out / "skip_validator_all.csv", index=False)

    lines = ["# Validator calls skipped per signal", "",
             "Rules frozen on train/dev per model (scripts/39); ACCEPT if dev failure rate among accepted <= X, "
             "REJECT if dev rejected set >= 95% failures. Cells: % of calls skipped on cert "
             "(failures executed unchecked / accepted; * = certified at X, Clopper-Pearson, Bonferroni over signals).", ""]
    for x in (0.10, 0.05):
        c = T[(T.partition == "cert") & (T.tolerance == x)].copy()
        c["cell"] = c.apply(lambda r: f"{r.skipped_pct:.0f}% ({r.accepted_failures}/{r.accepted}){'*' if r.certified else ''}", axis=1)
        piv = c.pivot(index="signal", columns="run", values="cell").reindex(columns=list(runs))
        order = ["observables (context + action)"] + [s for s in piv.index if s != "observables (context + action)"]
        piv = piv.reindex(order)
        lines += [f"## Tolerance X = {x:.0%} (cert)", "", "| signal | " + " | ".join(piv.columns) + " |",
                  "|---" * (len(piv.columns) + 1) + "|"]
        lines += ["| " + s + " | " + " | ".join("" if pd.isna(v) else v for v in r) + " |" for s, r in piv.iterrows()]
        t = T[(T.partition == "test_iid") & (T.tolerance == x) & T.test_delta_vs_obs.notna()]
        piv2 = t.pivot(index="signal", columns="run", values="test_delta_vs_obs").reindex(columns=list(runs))
        lines += ["", f"Change in calls skipped vs observables, test_iid, X = {x:.0%} (points, bootstrap 95% CI):", "",
                  "| signal | " + " | ".join(piv2.columns) + " |", "|---" * (len(piv2.columns) + 1) + "|"]
        lines += ["| " + s + " | " + " | ".join("" if pd.isna(v) else v for v in r) + " |" for s, r in piv2.iterrows()]
        lines.append("")
    (out / "skip_validator_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "run_config.json", build_manifest(runs=runs))
    print("\n".join(lines))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
