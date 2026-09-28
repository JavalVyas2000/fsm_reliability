"""
Three-way routing table: ALLOW (no simulator) / VERIFY (call simulator) / DISALLOW (reject,
no simulator), from frozen probe predictions (Stage 1 output).

Thresholds are selected on dev_thr only, under two rules:
    point : empirical rate on dev_thr <= target
    ucb   : exact one-sided 95% upper bound on dev_thr <= target   (conservative)
and then applied unchanged to test_iid (exploratory for the pilot).

Example:
    python -m scripts.13_routing_table --baseline_dir outputs/fsm_baseline/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.routing_v2 import (
    random_baseline,
    route,
    routing_summary,
    select_tau_high,
    select_tau_low,
)
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

FAMILIES = {
    # name in table -> column in predictions.csv
    "neg_mean_logprob (untrained)": "score_neg_mean_selected_logprob",
    "token_entropy (untrained)": "score_mean_token_entropy",
    "context_action": "p_context_action_platt",
    "token_confidence": "p_token_confidence_platt",
    "attention": "p_attention_platt",
    "attention+token_confidence": "p_attention+token_confidence_platt",
    "all_internal": "p_all_internal_platt",
    "context_action+all_internal": "p_context_action+all_internal_platt",
}
# Added automatically when the probe run includes the pre-registered grounding family.
GROUNDING_FAMILIES = {
    "grounding": "p_grounding_platt",
    "context_action+grounding": "p_context_action+grounding_platt",
    "context_action+all_internal+grounding": "p_context_action+all_internal+grounding_platt",
}
DEFAULT_SETTINGS = ["0.05:0.05", "0.10:0.10", "0.20:0.20"]  # alpha_allow:beta_disallow
RULES = ["ucb", "point"]
DELTA = 0.05

GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#b9b8b2"
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, default="outputs/fsm_selective_verification")
    p.add_argument("--tag", type=str, default="pilot")
    p.add_argument("--n_rep", type=int, default=2000)
    p.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS, help="alpha_allow:beta_disallow pairs")
    return p.parse_args()


def curve(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Descriptive one-sided curves over all cut points (test data; not for threshold selection)."""
    o = np.argsort(p, kind="stable")
    ps, ys = p[o], y[o]
    n = len(y)
    k = np.arange(1, n + 1)
    allow_fail = np.cumsum(ys) / k
    od = np.argsort(-p, kind="stable")
    disallow_valid = np.cumsum(1 - y[od]) / k
    return pd.DataFrame({
        "k": k,
        "allow_frac": k / n,
        "allow_threshold": ps,
        "allow_failure_rate": allow_fail,
        "disallow_frac": k / n,
        "disallow_threshold": p[od],
        "disallow_valid_rate": disallow_valid,
    })


def plot_bars(rows: pd.DataFrame, out: Path, title: str):
    import matplotlib.pyplot as plt

    rows = rows.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_, r) in enumerate(rows.iterrows()):
        left = 0.0
        for frac, color, label in [(r.frac_allow, GOOD, "ALLOW"), (r.frac_verify, NEUTRAL, "VERIFY"), (r.frac_disallow, CRITICAL, "DISALLOW")]:
            if frac > 0:
                ax.barh(i, frac, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
                if frac >= 0.07:
                    ax.text(left + frac / 2, i, f"{frac:.0%}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if color != NEUTRAL else INK)
            left += frac
    ax.set_yticks(range(len(rows)), rows["family"], fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of test candidates", fontsize=8.5, color=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8, left=False)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=GOOD, label="ALLOW (no simulator)"), Patch(color=NEUTRAL, label="VERIFY (simulator)"),
                 Patch(color=CRITICAL, label="DISALLOW (reject, no simulator)")],
        fontsize=7.5, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0 + 0.9 / (len(rows) + 2)), labelcolor=INK,
    )
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fmt_rate(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main():
    args = parse_args()
    base_dir = (REPO_ROOT / args.baseline_dir).resolve()
    preds = pd.read_csv(base_dir / "predictions.csv")
    summary = json.loads((base_dir / "dataset_summary.json").read_text())
    run_dir = make_run_dir(REPO_ROOT / args.out_root, args.tag)
    SETTINGS = [tuple(float(x) for x in s.split(":")) for s in args.settings]

    families = {k: v for k, v in {**FAMILIES, **GROUNDING_FAMILIES}.items() if v in preds.columns}
    dev = preds[preds.partition == "dev_thr"].reset_index(drop=True)
    test = preds[preds.partition == "test_iid"].reset_index(drop=True)
    y_dev, y_test = dev["y"].to_numpy(), test["y"].to_numpy()

    rows, per_example, curves = [], [], []
    for fam, col in families.items():
        p_dev, p_test = dev[col].to_numpy(), test[col].to_numpy()
        c = curve(p_test, y_test)
        c.insert(0, "family", fam)
        curves.append(c)
        for rule in RULES:
            for alpha, beta in SETTINGS:
                lo = select_tau_low(p_dev, y_dev, alpha, rule, DELTA)
                hi = select_tau_high(p_dev, y_dev, beta, rule, DELTA)
                dev_s = routing_summary(route(p_dev, lo, hi), y_dev, DELTA)
                r_test = route(p_test, lo, hi)
                s = routing_summary(r_test, y_test, DELTA)
                rb = random_baseline(y_test, s["n_allow"], s["n_disallow"], args.n_rep)
                rows.append({
                    "family": fam, "rule": rule, "alpha_allow": alpha, "beta_disallow": beta,
                    "tau_low": lo, "tau_high": max(hi, lo),
                    "dev_frac_allow": dev_s["frac_allow"], "dev_frac_disallow": dev_s["frac_disallow"],
                    **s,
                    "random_escaped_failures_mean": rb["escaped_failures_mean"],
                    "random_escaped_failures_p97.5": rb["escaped_failures_p97.5"],
                    "random_lost_viable_mean": rb["lost_viable_mean"],
                    "random_lost_viable_p97.5": rb["lost_viable_p97.5"],
                })
                per_example.append(pd.DataFrame({
                    "instance_id": test["instance_id"], "graph_hash": test["graph_hash"],
                    "group": test["num_nodes"] if "num_nodes" in test else test["family"],
                    "y_fail": y_test, "family": fam, "rule": rule, "alpha_allow": alpha, "beta_disallow": beta,
                    "p_fail": p_test, "route": r_test,
                }))

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "routing_table.csv", index=False)
    pd.concat(per_example).to_csv(run_dir / "routing_decisions_test.csv", index=False)
    pd.concat(curves).to_csv(run_dir / "risk_verification_curve.csv", index=False)

    # Human-readable tables.
    test_part = summary["partitions"]["test_iid"]
    lines = [
        "# FSM routing table (pilot, Qwen2.5-3B-Instruct, exploratory test_iid)",
        "",
        "Routes: **ALLOW** = execute without simulator; **VERIFY** = call simulator; "
        "**DISALLOW** = reject without simulator. p = predicted failure probability.",
        f"Test candidates: {test_part['n_total']} generated, {test_part['n_total'] - test_part['n_eligible']} rejected by "
        f"cheap format checks, {test_part['n_eligible']} routed; failure prevalence {test_part['prevalence_fail']:.3f}.",
        "Thresholds chosen on dev_thr only, then frozen. `ucb` = exact 95% upper bound on dev_thr within target; "
        "`point` = empirical dev_thr rate within target. Random = same ALLOW/DISALLOW counts assigned at random (mean over "
        f"{args.n_rep} draws).",
        "",
    ]
    for rule in RULES:
        for alpha, beta in SETTINGS:
            t = table[(table.rule == rule) & (table.alpha_allow == alpha) & (table.beta_disallow == beta)]
            lines += [
                f"## rule = {rule}, ALLOW failure target α = {alpha}, DISALLOW valid target β = {beta}",
                "",
                "| Family | ALLOW | VERIFY | DISALLOW | Simulator calls saved | Escaped failures (ALLOW & fail) | ALLOW failure rate [95% UB] | Random escaped | Lost valid (DISALLOW & valid) | DISALLOW valid rate [95% UB] | Random lost |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for _, r in t.iterrows():
                lines.append(
                    f"| {r.family} | {r.n_allow} ({r.frac_allow:.0%}) | {r.n_verify} ({r.frac_verify:.0%}) | "
                    f"{r.n_disallow} ({r.frac_disallow:.0%}) | {r.simulator_calls_saved_frac:.0%} | {r.escaped_failures} | "
                    f"{fmt_rate(r.allow_failure_rate)} [{fmt_rate(r.allow_failure_rate_cp95_upper)}] | "
                    f"{r['random_escaped_failures_mean']:.1f} | {r.lost_viable} | "
                    f"{fmt_rate(r.disallow_valid_rate)} [{fmt_rate(r.disallow_valid_rate_cp95_upper)}] | "
                    f"{r['random_lost_viable_mean']:.1f} |"
                )
            lines.append("")
    (run_dir / "routing_table.md").write_text("\n".join(lines), encoding="utf-8")

    for rule in RULES:
        for alpha, beta in SETTINGS:
            t = table[(table.rule == rule) & (table.alpha_allow == alpha) & (table.beta_disallow == beta)]
            plot_bars(t, run_dir / f"routing_bars_{rule}_a{alpha:.2f}_b{beta:.2f}.pdf",
                      f"Routing on test_iid ({rule}; α = {alpha}, β = {beta}; thresholds from dev_thr)")

    write_json(run_dir / "run_config.json", build_manifest(
        baseline_dir=str(base_dir.relative_to(REPO_ROOT)),
        families=families, settings=SETTINGS, rules=RULES, delta=DELTA,
        threshold_selection="dev_thr only; frozen before test_iid",
        test_status="exploratory (pilot)",
    ))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
