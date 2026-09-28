"""
Collect per-model results (probes, zero-failure routing with time, agreement routing,
certification) into one cross-model table for the paper.

Model entries come from completed scripts/18 pipeline logs plus an optional JSON file of
manually listed runs (for models run step by step), each of the form
    {"model": ..., "probes": <baseline dir>, "time": <time dir>, "agreement": <dir>, "certify": <dir>}

Example:
    python -m scripts.19_cross_model_summary --manual outputs/cross_model/qwen25_3b_manual.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

FAMILIES = ["context_action", "token_confidence", "attention", "attention+token_confidence", "all_internal",
            "context_action+all_internal"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manual", type=str, nargs="*", default=[])
    p.add_argument("--out_root", type=str, default="outputs/cross_model")
    return p.parse_args()


def entries(args):
    out = []
    for m in args.manual:
        out.append(json.loads((REPO_ROOT / m).read_text()))
    for pj in sorted((REPO_ROOT / "outputs/model_pipelines").glob("*/pipeline.json")):
        log = json.loads(pj.read_text())
        steps = {s["step"]: s.get("output_dir") for s in log["steps"]}
        out.append({"model": log["model"], "status": log["status"], "probes": steps.get("probes"),
                    "time": steps.get("time_savings"), "agreement": steps.get("agreement"), "certify": steps.get("certify")})
    return out


def main():
    args = parse_args()
    run_dir = make_run_dir(REPO_ROOT / args.out_root, "summary")
    rows_pred, rows_route, rows_cert, notes = [], [], [], []

    for e in entries(args):
        model = e["model"]
        if e.get("probes"):
            m = json.loads((REPO_ROOT / e["probes"] / "metrics.json").read_text())
            s = json.loads((REPO_ROOT / e["probes"] / "dataset_summary.json").read_text())
            row = {"model": model, "test_prevalence_fail": s["partitions"]["test_iid"]["prevalence_fail"],
                   "gen_latency_mean_s": s["cost"]["gen_latency_s"]["mean"], "feat_latency_mean_s": s["cost"]["feat_latency_s"]["mean"]}
            for f in FAMILIES:
                row[f"auroc_{f}"] = m["groups"][f]["test_iid"]["raw"]["auroc"]
            gate = m["paired_delta_auroc"]["test_iid"]
            for k in ["attention+token_confidence - context_action", "context_action+all_internal - context_action"]:
                row[f"delta[{k}]"] = f"{gate[k]['delta_auroc']:+.3f} [{gate[k]['ci95_lo']:+.3f}, {gate[k]['ci95_hi']:+.3f}]"
            rows_pred.append(row)
        if e.get("time"):
            t = pd.read_csv(REPO_ROOT / e["time"] / "time_savings.csv")
            t = t[(t.rule == "point") & (t.alpha_allow == 0.0) & (t.beta_disallow == 0.10) & t.family.isin(FAMILIES)]
            for _, r in t.iterrows():
                rows_route.append({"model": model, "family": r.family, "allow": r.n_allow, "verify": r.n_verify,
                                   "disallow": r.n_disallow, "calls_saved": r.simulator_calls_saved_frac,
                                   "failures_let_through": r.escaped_failures, "lost_valid": r.lost_viable,
                                   "whatif_saved_pct_at_6.95s": r.whatif_saved_pct_of_verification_at_cstr_cv})
        if e.get("certify"):
            c = json.loads((REPO_ROOT / e["certify"] / "certification_results.json").read_text())
            for r in c["results"]:
                rows_cert.append({"model": model, "policy": r["policy"], "n_eligible": c["n_eligible"],
                                  "prevalence": c["failure_prevalence"], "allow": r["n_allow"], "verify": r["n_verify"],
                                  "disallow": r["n_disallow"], "calls_saved": r["simulator_calls_saved_frac"],
                                  "failures_let_through": r["escaped_failures"], "allow_rate_ub": r["allow_rate_upper_bound"],
                                  "cert_0.05": r["certified"]["0.05"], "cert_0.02": r["certified"]["0.02"],
                                  "lost_valid": r["lost_viable"], "whatif_saved_pct_at_6.95s": r["whatif_saved_pct_at_cstr_cv"]})
        if e.get("status") and e["status"] != "complete":
            notes.append(f"{model}: pipeline status = {e['status']}")

    P, R, C = pd.DataFrame(rows_pred), pd.DataFrame(rows_route), pd.DataFrame(rows_cert)
    P.to_csv(run_dir / "predictive.csv", index=False)
    R.to_csv(run_dir / "routing_zero_failure.csv", index=False)
    C.to_csv(run_dir / "certification.csv", index=False)

    def md(df, fmt=None):
        if df.empty:
            return "_no data_"
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    v = f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
                cells.append(str(v))
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    text = [
        "# Cross-model summary: selective verification on FSM (Qwen/Llama/SmolLM families)",
        "",
        "All models: prompt fsm_v2.0, greedy decoding, same pilot dataset (seed 20260923) and certification dataset "
        "(seed 20260924, 3000 graphs). Per-model policies frozen before that model's certification inference.",
        "Pilot test = exploratory; certification = independent, Bonferroni over 4 policies per model (δ_i = 0.0125).",
        "What-if column = routing share × CSTR legacy verifier cost (6.95 s/call) − measured FSM overhead; not a CSTR result.",
        "",
        "## 1. Failure prediction (pilot test_iid AUROC; failure = positive)",
        "", md(P), "",
        "## 2. Zero-failure routing on pilot test (α = 0 on dev_thr, β = 0.10)",
        "", md(R), "",
        "## 3. Certification (independent cert set)",
        "", md(C), "",
    ]
    if notes:
        text += ["## Notes", ""] + [f"- {n}" for n in notes]
    (run_dir / "cross_model_summary.md").write_text("\n".join(text), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(entries=entries(args)))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
