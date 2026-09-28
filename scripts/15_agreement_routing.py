"""
Agreement routing (option 2): ALLOW only if two probes are both confident.

For each prespecified probe pair (a, b):
  - (tau_a, tau_b) chosen on dev_thr to maximise ALLOW count with zero ALLOW failures
    (point rule, alpha = 0);
  - DISALLOW from probe b alone, tau_high on dev_thr with valid-rate <= beta (point);
  - frozen and applied to test_iid (exploratory pilot test).
Single-probe alpha = 0 rows are included for comparison. Time accounting reuses the
measured costs of a scripts/14 run (same identity: saved = avoided C_v - overhead).

Example:
    python -m scripts.15_agreement_routing --baseline_dir outputs/fsm_baseline/<run> \
        --time_dir outputs/fsm_time_savings/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import json

import numpy as np
import pandas as pd

from src.evaluation.routing_v2 import (
    random_baseline,
    route,
    route_joint,
    routing_summary,
    select_joint_allow,
    select_tau_high,
    select_tau_low,
)
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

COL = {
    "context_action": "p_context_action_platt",
    "token_confidence": "p_token_confidence_platt",
    "attention": "p_attention_platt",
    "attention+token_confidence": "p_attention+token_confidence_platt",
    "all_internal": "p_all_internal_platt",
    "context_action+all_internal": "p_context_action+all_internal_platt",
}
# Prespecified before looking at agreement results: an observable baseline paired with
# an internal probe, plus one internal/internal pair.
PAIRS = [
    ("context_action", "attention"),
    ("context_action", "attention+token_confidence"),
    ("context_action", "all_internal"),
    ("token_confidence", "attention"),
]
SINGLES = ["context_action", "attention+token_confidence", "all_internal", "context_action+all_internal"]
BETA = 0.10
CSTR_LEGACY_VERIFIER_S = 6.95


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, required=True)
    p.add_argument("--time_dir", type=str, required=True, help="scripts/14 run with measured costs")
    p.add_argument("--out_root", type=str, default="outputs/fsm_agreement_routing")
    p.add_argument("--tag", type=str, default="pilot")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = (REPO_ROOT / args.baseline_dir).resolve()
    time_cfg = json.loads(((REPO_ROOT / args.time_dir).resolve() / "run_config.json").read_text())
    measured = time_cfg["measured"]
    preds = pd.read_csv(base_dir / "predictions.csv")
    run_dir = make_run_dir(REPO_ROOT / args.out_root, args.tag)

    dev = preds[preds.partition == "dev_thr"].reset_index(drop=True)
    test = preds[preds.partition == "test_iid"].reset_index(drop=True)
    y_dev, y_test = dev["y"].to_numpy(), test["y"].to_numpy()

    def overhead(members):
        feat = measured["C_feat_mean_s"] if any(m != "context_action" for m in members) else 0.0
        return feat + sum(measured["probe_scoring_median_s"][m] for m in members)

    rows, decisions = [], []

    def add(policy, members, r_test, thresholds):
        s = routing_summary(r_test, y_test)
        rb = random_baseline(y_test, s["n_allow"], s["n_disallow"])
        cf = overhead(members)
        saved = s["simulator_calls_saved_frac"]
        rows.append({
            "policy": policy, **thresholds, **s,
            "random_escaped_failures_mean": rb["escaped_failures_mean"],
            "router_overhead_per_cand_s": cf,
            "fsm_time_saved_s": s["n"] * (saved * measured["C_v_fsm_mean_s"] - cf),
            "breakeven_verifier_cost_s": cf / saved if saved > 0 else None,
            "whatif_saved_per_cand_at_cstr_cv_s": saved * CSTR_LEGACY_VERIFIER_S - cf,
            "whatif_saved_pct_at_cstr_cv": 100 * (saved * CSTR_LEGACY_VERIFIER_S - cf) / CSTR_LEGACY_VERIFIER_S,
        })
        decisions.append(pd.DataFrame({"instance_id": test["instance_id"], "y_fail": y_test, "policy": policy, "route": r_test}))

    for m in SINGLES:
        lo = select_tau_low(dev[COL[m]].to_numpy(), y_dev, 0.0, "point")
        hi = select_tau_high(dev[COL[m]].to_numpy(), y_dev, BETA, "point")
        add(f"single: {m}", [m], route(test[COL[m]].to_numpy(), lo, hi), {"tau_a": lo, "tau_b": None, "tau_high": hi})

    for a, b in PAIRS:
        ta, tb, n_dev = select_joint_allow(dev[COL[a]].to_numpy(), dev[COL[b]].to_numpy(), y_dev, 0.0, "point")
        hi = select_tau_high(dev[COL[b]].to_numpy(), y_dev, BETA, "point")
        r = route_joint(test[COL[a]].to_numpy(), test[COL[b]].to_numpy(), ta, tb, test[COL[b]].to_numpy(), hi)
        add(f"agree: {a} & {b}", [a, b], r, {"tau_a": ta, "tau_b": tb, "tau_high": hi, "dev_n_allow": n_dev})

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "agreement_routing.csv", index=False)
    pd.concat(decisions).to_csv(run_dir / "agreement_decisions_test.csv", index=False)

    lines = [
        "# Agreement routing vs single-probe routing (pilot, exploratory test_iid, n = 500)",
        "",
        "ALLOW thresholds chosen on dev_thr for zero ALLOW failures (point rule, α = 0); DISALLOW β = 0.10 (point) "
        "from the single probe / the second probe of a pair. Thresholds frozen before test.",
        f"What-if = routing shares × CSTR legacy verifier cost ({CSTR_LEGACY_VERIFIER_S} s/call) − measured FSM overhead; not a CSTR result.",
        "",
        "| Policy | ALLOW (dev) | ALLOW | VERIFY | DISALLOW | Calls saved | Failures let through [95% UB rate] | Random | Lost valid | FSM time saved (s) | What-if at 6.95 s/call |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        ub = "–" if pd.isna(r.allow_failure_rate_cp95_upper) else f"{r.allow_failure_rate_cp95_upper:.3f}"
        dev_n = "" if pd.isna(r.get("dev_n_allow")) else int(r.dev_n_allow)
        lines.append(
            f"| {r.policy} | {dev_n} | {r.n_allow} | {r.n_verify} | {r.n_disallow} | {r.simulator_calls_saved_frac:.0%} | "
            f"{r.escaped_failures} [{ub}] | {r.random_escaped_failures_mean:.1f} | {r.lost_viable} | {r.fsm_time_saved_s:+.1f} | "
            f"{r.whatif_saved_per_cand_at_cstr_cv_s:+.2f} s ({r.whatif_saved_pct_at_cstr_cv:+.0f}%) |"
        )
    (run_dir / "agreement_routing.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(
        baseline_dir=str(base_dir.relative_to(REPO_ROOT)), time_dir=args.time_dir, pairs=PAIRS, singles=SINGLES,
        allow_rule="point, alpha=0, joint search on dev_thr", beta=BETA, test_status="exploratory (pilot)",
    ))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
