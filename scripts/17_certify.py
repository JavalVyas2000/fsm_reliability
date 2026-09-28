"""
Certify the frozen routing policies on the independent certification partition (Stage 3).

Inputs: a frozen-policy directory (scripts/16; hashes are verified) and inference records
for a certification partition generated AFTER the freeze. Nothing is refit or re-tuned.

Per declared policy:
  - route every schema-valid candidate (ALLOW / VERIFY / DISALLOW);
  - certify the ALLOW failure rate: exact one-sided upper bound at delta/m (Bonferroni);
  - co-report the marginal escaped-error rate P(ALLOW and fail) with its bound, the
    DISALLOW valid rate, and computational time saved (same identity as scripts/14).

Example:
    python -m scripts.17_certify --frozen_dir outputs/certification/<freeze> \
        --inference_dir outputs/fsm_inference/<cert run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.evaluation.certification import certify
from src.evaluation.routing_v2 import cp_upper, route, route_joint, routing_summary
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, sha256_file, write_json

baseline = importlib.import_module("scripts.12_fit_fsm_baseline")
timing = importlib.import_module("scripts.14_time_savings")

CSTR_LEGACY_VERIFIER_S = 6.95


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen_dir", type=str, required=True)
    p.add_argument("--inference_dir", type=str, required=True)
    p.add_argument("--partition", type=str, default="cert")
    p.add_argument("--out_root", type=str, default="outputs/certification")
    p.add_argument("--tag", type=str, default="certify")
    p.add_argument("--verifier_repeats", type=int, default=500)
    p.add_argument("--dry_run_pre_freeze_data", action="store_true",
                   help="code check only: skip the freeze-timestamp check; output is NOT a certification")
    return p.parse_args()


def main():
    args = parse_args()
    frozen_dir = (REPO_ROOT / args.frozen_dir).resolve()
    inf_dir = (REPO_ROOT / args.inference_dir).resolve()
    fm = json.loads((frozen_dir / "freeze_manifest.json").read_text())
    for fname, h in fm["sha256"].items():
        if sha256_file(frozen_dir / fname) != h:
            raise SystemExit(f"Frozen artefact modified after freeze: {fname}")
    policies = json.loads((frozen_dir / "policies.json").read_text())
    probes = joblib.load(frozen_dir / "probes.joblib")
    inf_manifest = json.loads((inf_dir / "run_manifest.json").read_text())
    disclosure = None
    if "grounding" in inf_manifest:
        # Augmented cert records (scripts/27): the grounding features must postdate the freeze.
        g_manifest = json.loads((inf_dir.parent / "run_manifest.json").read_text())
        if g_manifest["created_utc"] <= fm["created_utc"] and not args.dry_run_pre_freeze_data:
            raise SystemExit("Grounding features on the certification data predate the policy freeze")
        disclosure = ("Certification partition reused: its labels were previously used to certify the 'main' policy "
                      "set (P1-P4). This policy set was selected on pilot data only and frozen before grounding "
                      "features were computed on the certification data.")
    elif inf_manifest["created_utc"] <= fm["created_utc"] and not args.dry_run_pre_freeze_data:
        raise SystemExit("Certification inference predates the policy freeze")
    run_dir = make_run_dir(REPO_ROOT / args.out_root, args.tag)

    df = baseline.load_records(inf_dir)
    df = df[df["partition"] == args.partition].reset_index(drop=True)
    hidden, hidden_keys = baseline.load_hidden(inf_dir)
    if hidden_keys != policies["hidden_keys"]:
        raise SystemExit("Hidden-state layout differs from the frozen probes")
    n_generated = len(df)
    fmt_fail = int((df["schema_valid"] == 0).sum())
    elig = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok")].reset_index(drop=True)
    y = elig["candidate_invalid"].astype(int).to_numpy()
    if not elig["graph_hash"].is_unique:
        raise SystemExit("Certification requires one candidate per graph")

    p, probe_time = {}, {}
    for name, pr in probes.items():
        X = baseline.build_matrix(elig, pr["scalar"], pr["hidden"], hidden)
        raw = pr["pipeline"].predict_proba(X)[:, 1]
        p[name] = pr["platt"].predict_proba(logit(raw)[:, None])[:, 1]
        ts = []
        for i in range(min(200, len(X))):
            t0 = time.perf_counter()
            pr["platt"].predict_proba(logit(pr["pipeline"].predict_proba(X[i : i + 1])[:, 1])[:, None])
            ts.append(time.perf_counter() - t0)
        probe_time[name] = float(np.median(ts))

    c_gen, c_feat = elig["gen_latency_s"].to_numpy(), elig["feat_latency_s"].to_numpy()
    ds_dir = REPO_ROOT / inf_manifest["data"]["dataset_dir"]
    ds = pd.read_csv(ds_dir / f"{args.partition}.csv").set_index("instance_id").loc[elig["instance_id"]].reset_index()
    parsed = {i: pp for i, pp in zip(elig["instance_id"], elig["parsed_path"])}
    c_v = timing.time_fsm_verifier(ds, parsed, args.verifier_repeats)

    cert_cfg = policies["certification"]
    delta_i, alphas = cert_cfg["delta_per_policy"], cert_cfg["alphas"]
    results, decisions = [], []
    for pol in policies["policies"]:
        if pol["type"] == "single":
            r = route(p[pol["allow"][0]], pol["tau_allow"][0], pol["tau_disallow"])
        else:
            a, b = pol["allow"]
            r = route_joint(p[a], p[b], pol["tau_allow"][0], pol["tau_allow"][1], p[pol["disallow"]], pol["tau_disallow"])
        s = routing_summary(r, y, delta_i)
        cert = certify(s["escaped_failures"], s["n_allow"], alphas, delta_i)
        members = set(pol["allow"]) | {pol["disallow"]}
        cf = (c_feat if members != {"context_action"} else 0.0) + sum(probe_time[m] for m in members)
        cf_mean = float(np.mean(cf))
        verify = r == "VERIFY"
        t_always = c_gen.sum() + c_v.sum()
        t_policy = c_gen.sum() + np.sum(cf) + c_v[verify].sum()
        saved = s["simulator_calls_saved_frac"]
        results.append({
            "policy": pol["id"],
            **s,
            "allow_rate_upper_bound": cert["upper_bound"],
            "certified": cert["certified"],
            "marginal_escaped_rate": s["escaped_failures"] / len(y),
            "marginal_escaped_rate_upper_bound": cp_upper(s["escaped_failures"], len(y), delta_i),
            "router_overhead_per_cand_s": cf_mean,
            "fsm_time_saved_s": float(t_always - t_policy),
            "breakeven_verifier_cost_s": cf_mean / saved if saved > 0 else None,
            "whatif_saved_per_cand_at_cstr_cv_s": saved * CSTR_LEGACY_VERIFIER_S - cf_mean,
            "whatif_saved_pct_at_cstr_cv": 100 * (saved * CSTR_LEGACY_VERIFIER_S - cf_mean) / CSTR_LEGACY_VERIFIER_S,
        })
        decisions.append(pd.DataFrame({"instance_id": elig["instance_id"], "graph_hash": elig["graph_hash"],
                                       "num_nodes": elig["num_nodes"], "y_fail": y, "policy": pol["id"], "route": r}))

    out = {
        "DRY_RUN_NOT_A_CERTIFICATION": bool(args.dry_run_pre_freeze_data),
        "disclosure": disclosure,
        "policy_set": policies.get("policy_set", "main"),
        "partition": args.partition,
        "n_generated": n_generated,
        "format_failures_rejected_by_cheap_checks": fmt_fail,
        "n_eligible": int(len(y)),
        "failure_prevalence": float(y.mean()),
        "delta_family": cert_cfg["delta_family"],
        "delta_per_policy": delta_i,
        "alphas": alphas,
        "results": results,
        "measured_costs": {"C_gen_mean_s": float(c_gen.mean()), "C_feat_mean_s": float(c_feat.mean()),
                           "C_v_fsm_mean_s": float(c_v.mean()), "probe_scoring_median_s": probe_time},
        "guarantee": ("With probability >= 1 - delta_family over the certification sample, no declared policy is "
                      "certified at a level below its true ALLOW failure rate, for the declared IID population."),
    }
    write_json(run_dir / "certification_results.json", out)
    pd.concat(decisions).to_csv(run_dir / "cert_routing_decisions.csv", index=False)

    lines = [
        f"# Certification of frozen routing policies (FSM, partition `{args.partition}`)",
        "",
        f"Generated {n_generated}; rejected by cheap format checks {fmt_fail}; routed {len(y)}; failure prevalence {y.mean():.3f}.",
        f"Exact one-sided bounds at δ = {cert_cfg['delta_family']} / {cert_cfg['m_policies']} = {delta_i} per policy (Bonferroni).",
        *( [f"**Disclosure:** {disclosure}"] if disclosure else [] ),
        "Certified at α ⇔ ALLOW set non-empty and upper bound ≤ α.",
        "",
        "| Policy | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through | ALLOW failure rate | Upper bound | Certified α=0.05 | Certified α=0.02 | Marginal escaped rate [UB] | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ar = "–" if r["allow_failure_rate"] is None else f"{r['allow_failure_rate']:.4f}"
        ub = "–" if r["allow_rate_upper_bound"] is None else f"{r['allow_rate_upper_bound']:.4f}"
        lines.append(
            f"| {r['policy']} | {r['n_allow']} | {r['n_verify']} | {r['n_disallow']} | {r['simulator_calls_saved_frac']:.1%} | "
            f"{r['escaped_failures']} | {ar} | {ub} | {'yes' if r['certified']['0.05'] else 'no'} | "
            f"{'yes' if r['certified']['0.02'] else 'no'} | {r['marginal_escaped_rate']:.4f} [{r['marginal_escaped_rate_upper_bound']:.4f}] | "
            f"{r['lost_viable']} | {r['fsm_time_saved_s']:+.1f} | {r['whatif_saved_per_cand_at_cstr_cv_s']:+.2f} s "
            f"({r['whatif_saved_pct_at_cstr_cv']:+.0f}%) |"
        )
    (run_dir / "certification_table.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(
        frozen_dir=str(frozen_dir.relative_to(REPO_ROOT)), inference_dir=str(inf_dir.relative_to(REPO_ROOT)),
        frozen_sha256=fm["sha256"], partition=args.partition,
    ))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
