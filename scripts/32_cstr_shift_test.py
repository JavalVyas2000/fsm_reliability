"""
Distribution-shift test for CSTR failure probes: train on the FIXED nominal plant (main
dataset), evaluate on VARIED nominal plants (varied dataset), and compare with probes
trained in-distribution on the varied plants.

Question: does the observable context/action probe degrade under plant shift more than
probes built on model internals?

All probes: first proposals (round 0), the same feature lists (the fixed-plant groups;
setpoint features that are constant on the fixed plant are not used), the same pipeline as
scripts/12. AUROC on the varied test_iid with bootstrap CIs; paired differences are over
the same varied test episodes.

Example:
    python -m scripts.32_cstr_shift_test --fixed outputs/cstr_collect/<main run> --varied outputs/cstr_collect/<varied run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics_v2 import bootstrap_auroc_ci, bootstrap_delta_auroc
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

baseline = importlib.import_module("scripts.12_fit_fsm_baseline")
FAMILIES = ["context_action", "context_only_no_action", "token_confidence", "attention", "hidden",
            "all_internal", "context_action+all_internal"]


def load(run: Path):
    df = baseline.load_records(run)
    df = df[(df["round"] == 0) & (df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()]
    df = df[df["partition"].isin(["train", "test_iid"])].copy()
    df["y"] = df["candidate_invalid"].astype(int)
    hidden, _ = baseline.load_hidden(run)
    return df, hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", required=True)
    p.add_argument("--varied", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    fixed, h_fixed = load(REPO_ROOT / args.fixed)
    varied, h_varied = load(REPO_ROOT / args.varied)
    groups = baseline.feature_groups(fixed, "cstr")  # feature lists defined on the fixed plant
    run_dir = make_run_dir(REPO_ROOT / "outputs/cstr_shift", "fixed_to_varied")

    ftr, fte = fixed[fixed.partition == "train"], fixed[fixed.partition == "test_iid"]
    vtr, vte = varied[varied.partition == "train"], varied[varied.partition == "test_iid"]
    y_vte = vte["y"].to_numpy()
    rows, preds = [], {}
    for fam in FAMILIES:
        spec = groups[fam]
        def fit(tr, h):
            X = baseline.build_matrix(tr, spec["scalar"], spec["hidden"], h)
            return baseline.make_pipeline(len(spec["scalar"]), X.shape[1]).fit(X, tr["y"])
        m_fixed, m_varied = fit(ftr, h_fixed), fit(vtr, h_varied)
        p_id = m_fixed.predict_proba(baseline.build_matrix(fte, spec["scalar"], spec["hidden"], h_fixed))[:, 1]
        X_vte = baseline.build_matrix(vte, spec["scalar"], spec["hidden"], h_varied)
        p_shift, p_in = m_fixed.predict_proba(X_vte)[:, 1], m_varied.predict_proba(X_vte)[:, 1]
        preds[fam] = (p_shift, p_in)
        a_id, a_shift, a_in = roc_auc_score(fte["y"], p_id), roc_auc_score(y_vte, p_shift), roc_auc_score(y_vte, p_in)
        d = bootstrap_delta_auroc(y_vte, p_shift, p_in, vte["graph_hash"].to_numpy(), args.n_boot)
        rows.append({"family": fam, "auroc_fixed_to_fixed": a_id, "auroc_fixed_to_varied": a_shift,
                     "auroc_varied_to_varied": a_in, "shift_drop_vs_fixed_test": a_id - a_shift,
                     "shift_vs_indist_delta": d["delta_auroc"], "shift_vs_indist_ci": [d["ci95_lo"], d["ci95_hi"]],
                     "fixed_to_varied_ci": list(bootstrap_auroc_ci(y_vte, p_shift, args.n_boot).values())})
    # paired: does adding internals help the observable probe UNDER SHIFT?
    add = bootstrap_delta_auroc(y_vte, preds["context_action+all_internal"][0], preds["context_action"][0],
                                vte["graph_hash"].to_numpy(), args.n_boot)
    T = pd.DataFrame(rows)
    T.to_csv(run_dir / "shift_results.csv", index=False)
    lines = ["# CSTR distribution shift: fixed nominal plant → varied nominal plants (first proposals)", "",
             f"Fixed-plant train n={len(ftr)}, fixed test n={len(fte)}; varied train n={len(vtr)}, varied test n={len(vte)} "
             f"(failure prevalence fixed test {fte.y.mean():.2f}, varied test {vte.y.mean():.2f}).", "",
             "| Family | AUROC fixed→fixed | AUROC fixed→varied [95% CI] | AUROC varied→varied | drop under shift | shift − in-distribution [95% CI] |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['auroc_fixed_to_fixed']:.3f} | {r['auroc_fixed_to_varied']:.3f} "
                     f"[{r['fixed_to_varied_ci'][0]:.3f}, {r['fixed_to_varied_ci'][1]:.3f}] | {r['auroc_varied_to_varied']:.3f} | "
                     f"{r['shift_drop_vs_fixed_test']:+.3f} | {r['shift_vs_indist_delta']:+.3f} "
                     f"[{r['shift_vs_indist_ci'][0]:+.3f}, {r['shift_vs_indist_ci'][1]:+.3f}] |")
    lines += ["", f"Under shift, context_action + all internals − context_action: {add['delta_auroc']:+.3f} "
                  f"[{add['ci95_lo']:+.3f}, {add['ci95_hi']:+.3f}]"]
    (run_dir / "shift_results.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(fixed=args.fixed, varied=args.varied, families=FAMILIES,
                                                          added_internals_under_shift=add))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
