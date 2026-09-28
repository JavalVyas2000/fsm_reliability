"""
Routing table with computational time saved as the metric (protocol v1.0, section 8 / Stage 7 identity).

Per test candidate i, measured on this machine:
    C_gen_i   generation time                       (paid by every policy)
    C_f_i     router overhead = feature pass + probe scoring (0 for always-verify;
              probe scoring only for context_action, which needs no model internals)
    C_v_i     verifier time (FSM: path-validity check, timed here)

    T_always  = sum_i C_gen_i + sum_i C_v_i
    T_policy  = sum_i C_gen_i + sum_i C_f_i + sum_{i in VERIFY} C_v_i
    saved     = T_always - T_policy = sum_{i not VERIFY} C_v_i - sum_i C_f_i

DISALLOW is counted as a saved verifier call; the cost of re-proposing after a
rejection is an episode-level effect and is not included here.

A sensitivity analysis replaces C_v with a constant c and reports saved(c) and the
break-even verifier cost c* = mean(C_f) / share of calls saved. The CSTR legacy
verifier cost (6.95 s/call, Stage 0 audit, n = 21, same CPU) is marked for reference;
it is NOT a CSTR routing result.

Example:
    python -m scripts.14_time_savings --routing_dir outputs/fsm_selective_verification/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fsm_dataset_v2 import load_graph
from src.data.labels import is_valid_path
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

baseline = importlib.import_module("scripts.12_fit_fsm_baseline")

CSTR_LEGACY_VERIFIER_S = 6.95  # Stage 0 audit: mean of 21 rollout_validate_setpoints calls
INTERNAL_FAMILIES = {
    "neg_mean_logprob (untrained)", "token_entropy (untrained)", "token_confidence", "attention",
    "attention+token_confidence", "all_internal", "context_action+all_internal",
    # Grounding reuses the attention rows of the same feature pass (per-line sums add ~ms);
    # charged the full measured feature-pass time, as the other internal families.
    "grounding", "context_action+grounding", "context_action+all_internal+grounding",
}
PROBE_GROUP = {  # routing family -> Stage 1 feature group whose probe is timed
    "context_action": "context_action",
    "token_confidence": "token_confidence",
    "attention": "attention",
    "attention+token_confidence": "attention+token_confidence",
    "all_internal": "all_internal",
    "context_action+all_internal": "context_action+all_internal",
    "grounding": "grounding",
    "context_action+grounding": "context_action+grounding",
    "context_action+all_internal+grounding": "context_action+all_internal+grounding",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--routing_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, default="outputs/fsm_time_savings")
    p.add_argument("--tag", type=str, default="pilot")
    p.add_argument("--verifier_repeats", type=int, default=2000)
    p.add_argument("--probe_rows", type=int, default=200)
    p.add_argument("--cstr_verifier_timing", type=str, default=None,
                   help="CSTR: JSON {instance_id: seconds} from sequential re-timing (scripts/24); "
                        "otherwise the (parallel, inflated) labelling wall times are used")
    return p.parse_args()


def time_fsm_verifier(test_df: pd.DataFrame, answers: dict, repeats: int) -> np.ndarray:
    """Per-candidate wall time of the FSM validity check (mean over `repeats` calls)."""
    out = np.zeros(len(test_df))
    for j, (_, row) in enumerate(test_df.iterrows()):
        g = load_graph(row)
        path = answers[row["instance_id"]]
        s, t = int(row["start"]), int(row["goal"])
        t0 = time.perf_counter()
        for _ in range(repeats):
            is_valid_path(g, path, s, t)
        out[j] = (time.perf_counter() - t0) / repeats
    return out


def time_probes(inf_dir: Path, n_rows: int, domain: str = "fsm") -> dict:
    """Refit each Stage 1 probe exactly as in script 12 and time single-row scoring on test rows."""
    df = baseline.load_records(inf_dir)
    hidden, _ = baseline.load_hidden(inf_dir)
    groups = baseline.feature_groups(df, domain)
    elig = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()].copy()
    elig["y"] = elig["candidate_invalid"].astype(int)
    tr, te = elig[elig.partition == "train"], elig[elig.partition == "test_iid"].head(n_rows)
    out = {}
    for fam, g in PROBE_GROUP.items():
        if g not in groups:
            continue
        spec = groups[g]
        Xtr = baseline.build_matrix(tr, spec["scalar"], spec["hidden"], hidden)
        Xte = baseline.build_matrix(te, spec["scalar"], spec["hidden"], hidden)
        pipe = baseline.make_pipeline(len(spec["scalar"]), Xtr.shape[1]).fit(Xtr, tr["y"])
        times = []
        for i in range(len(Xte)):
            t0 = time.perf_counter()
            pipe.predict_proba(Xte[i : i + 1])
            times.append(time.perf_counter() - t0)
        out[fam] = float(np.median(times))
    return out


def plot_sensitivity(rows: pd.DataFrame, cf: dict, fsm_cv: float, out: Path, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    c = np.logspace(-7, 2, 600)
    for i, (_, r) in enumerate(rows.iterrows()):
        saved = r.simulator_calls_saved_frac * c - cf[r.family]
        ax.plot(c, saved, color=SERIES[i], linewidth=2, label=f"{r.family} ({r.simulator_calls_saved_frac:.0%} calls saved)")
    ax.axhline(0, color=MUTED, linewidth=1)
    for x, lab in [(fsm_cv, "measured verifier\n(this run)"), (CSTR_LEGACY_VERIFIER_S, "CSTR rollout\n(Stage 0 audit)")]:
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
        ax.text(x * 1.3, 0.02, lab, fontsize=7, color=INK_2, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Verifier cost per call (s)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("Time saved per candidate vs always-verify (s)", fontsize=8.5, color=INK_2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    args = parse_args()
    routing_dir = (REPO_ROOT / args.routing_dir).resolve()
    routing_cfg = json.loads((routing_dir / "run_config.json").read_text())
    base_dir = REPO_ROOT / routing_cfg["baseline_dir"]
    base_cfg = json.loads((base_dir / "run_config.json").read_text())
    inf_dir = REPO_ROOT / base_cfg["inference_dir"]
    ds_dir = REPO_ROOT / base_cfg["inference_manifest"]["data"]["dataset_dir"]
    run_dir = make_run_dir(REPO_ROOT / args.out_root, args.tag)

    table = pd.read_csv(routing_dir / "routing_table.csv")
    decisions = pd.read_csv(routing_dir / "routing_decisions_test.csv")

    recs = {}
    with open(inf_dir / "records.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["partition"] == "test_iid":
                recs[r["instance_id"]] = r
    test_ids = decisions.drop_duplicates("instance_id")["instance_id"].tolist()
    c_gen = np.array([recs[i]["gen_latency_s"] for i in test_ids])
    c_feat = np.array([recs[i]["feat_latency_s"] for i in test_ids])

    domain = base_cfg["fitting"].get("domain", "fsm")
    if domain == "fsm":
        test_df = pd.read_csv(ds_dir / "test_iid.csv").set_index("instance_id").loc[test_ids].reset_index()
        answers = {i: recs[i]["parsed_path"] for i in test_ids}
        c_v = time_fsm_verifier(test_df, answers, args.verifier_repeats)
        c_v_source = f"FSM path check, mean of {args.verifier_repeats} calls"
    elif args.cstr_verifier_timing:
        tmap = json.loads((REPO_ROOT / args.cstr_verifier_timing).read_text())
        c_v = np.array([tmap[i] for i in test_ids])
        c_v_source = f"CSTR legacy verifier, sequential re-timing ({args.cstr_verifier_timing})"
    else:
        c_v = np.array([recs[i]["verifier_wall_s"] for i in test_ids])
        c_v_source = "CSTR legacy verifier, labelling wall time (parallel workers; inflated)"
    probe_s = time_probes(inf_dir, args.probe_rows, domain)
    probe_s.setdefault("neg_mean_logprob (untrained)", 0.0)
    probe_s.setdefault("token_entropy (untrained)", 0.0)

    idx = {iid: k for k, iid in enumerate(test_ids)}
    rows = []
    for _, r in table.iterrows():
        d = decisions[(decisions.family == r.family) & (decisions.rule == r.rule)
                      & (decisions.alpha_allow == r.alpha_allow) & (decisions.beta_disallow == r.beta_disallow)]
        order = d["instance_id"].map(idx).to_numpy()
        verify = (d["route"] == "VERIFY").to_numpy()
        cf_i = (c_feat[order] if r.family in INTERNAL_FAMILIES else 0.0) + probe_s[r.family]
        t_always = c_gen[order].sum() + c_v[order].sum()
        t_policy = c_gen[order].sum() + np.sum(cf_i) + c_v[order][verify].sum()
        cf_mean = float(np.mean(cf_i))
        frac_saved = r.simulator_calls_saved_frac
        rows.append({
            **r.to_dict(),
            "C_gen_mean_s": float(c_gen[order].mean()),
            "C_f_mean_s": cf_mean,
            "C_v_mean_s": float(c_v[order].mean()),
            "T_always_s": float(t_always),
            "T_policy_s": float(t_policy),
            "time_saved_s": float(t_always - t_policy),
            "time_saved_pct_of_always": float(100 * (t_always - t_policy) / t_always),
            "verifier_time_saved_s": float(c_v[order][~verify].sum()),
            "router_overhead_s": float(np.sum(cf_i)),
            "breakeven_verifier_cost_s": cf_mean / frac_saved if frac_saved > 0 else None,
            "whatif_saved_per_candidate_at_cstr_cv_s": frac_saved * CSTR_LEGACY_VERIFIER_S - cf_mean,
            "whatif_saved_pct_of_verification_at_cstr_cv": 100 * (frac_saved * CSTR_LEGACY_VERIFIER_S - cf_mean) / CSTR_LEGACY_VERIFIER_S,
        })
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "time_savings.csv", index=False)

    cf_by_family = out.groupby("family")["C_f_mean_s"].first().to_dict()
    n_test = len(test_ids)
    if domain == "fsm":
        lines = [
            f"# FSM routing table with computational time saved (pilot, exploratory test_iid, n = {n_test})",
            "",
            f"Measured per candidate (means): generation {c_gen.mean():.3f} s; feature pass {c_feat.mean():.3f} s; "
            f"FSM verifier (path check) {c_v.mean()*1e6:.1f} µs. Probe scoring (median, single row): "
            + ", ".join(f"{k} {v*1e3:.2f} ms" for k, v in probe_s.items() if v > 0) + ".",
            "Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates). "
            "Re-proposal cost after DISALLOW is not included.",
            "",
            "**FSM (measured):** the path check costs microseconds, so any router overhead exceeds the verification it avoids — "
            "time saved is negative for every policy. FSM demonstrates routing quality, not time savings.",
            "",
            f"**What-if column:** FSM routing fractions and measured FSM overhead applied to the CSTR legacy verifier cost "
            f"({CSTR_LEGACY_VERIFIER_S} s/call, Stage 0 audit). This is arithmetic, not a CSTR result.",
            "",
        ]
        time_col = f"FSM time saved (s, {n_test} cands)"
    else:
        lines = [
            f"# CSTR routing table with computational time saved (pilot, exploratory test_iid, n = {n_test})",
            "",
            f"Measured per candidate (means): generation {c_gen.mean():.2f} s; feature pass {c_feat.mean():.3f} s; "
            f"verifier {c_v.mean():.2f} s ({c_v_source}). Probe scoring (median, single row): "
            + ", ".join(f"{k} {v*1e3:.2f} ms" for k, v in probe_s.items() if v > 0) + ".",
            "Time saved = (verifier time avoided on ALLOW + DISALLOW) − (router overhead on all candidates), all measured. "
            "Re-proposal cost after DISALLOW is not included. The last column repeats the arithmetic at the Stage 0 "
            f"reference cost ({CSTR_LEGACY_VERIFIER_S} s/call) for comparison only.",
            "",
        ]
        time_col = f"CSTR time saved (s, {n_test} cands, measured)"
    settings = out[["alpha_allow", "beta_disallow"]].drop_duplicates().itertuples(index=False)
    settings = list(settings)
    for rule in ["point", "ucb"]:
        for a, b in settings:
            t = out[(out.rule == rule) & (out.alpha_allow == a) & (out.beta_disallow == b)]
            lines += [
                f"## rule = {rule}, ALLOW failure target α = {a}, DISALLOW valid target β = {b}",
                "",
                f"| Family | ALLOW | VERIFY | DISALLOW | Calls saved | Escaped failures | Lost valid | {time_col} | Router overhead (s) | Break-even verifier cost (s) | At 6.95 s/call (s per cand, % of verification) |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for _, r in t.iterrows():
                be = "–" if r.breakeven_verifier_cost_s is None or np.isnan(r.breakeven_verifier_cost_s) else f"{r.breakeven_verifier_cost_s:.3f}"
                lines.append(
                    f"| {r.family} | {r.n_allow} | {r.n_verify} | {r.n_disallow} | {r.simulator_calls_saved_frac:.0%} | "
                    f"{r.escaped_failures} | {r.lost_viable} | {r.time_saved_s:+.2f} | {r.router_overhead_s:.2f} | {be} | "
                    f"{r.whatif_saved_per_candidate_at_cstr_cv_s:+.2f} ({r.whatif_saved_pct_of_verification_at_cstr_cv:+.0f}%) |"
                )
            lines.append("")
    (run_dir / "time_savings.md").write_text("\n".join(lines), encoding="utf-8")

    for rule, (a, b) in [(r, ab) for r in ("point", "ucb") for ab in settings]:
        sel = out[(out.rule == rule) & (out.alpha_allow == a) & (out.beta_disallow == b)
                  & out.family.isin(["context_action", "token_confidence", "attention+token_confidence", "context_action+all_internal"])]
        plot_sensitivity(sel, cf_by_family, float(c_v.mean()), run_dir / f"time_saved_vs_verifier_cost_{rule}_a{a:.2f}_b{b:.2f}.pdf",
                         f"Time saved vs verifier cost ({rule}, α = {a}, β = {b}; {domain.upper()} routing and overhead)")

    write_json(run_dir / "run_config.json", build_manifest(
        routing_dir=str(routing_dir.relative_to(REPO_ROOT)),
        inference_dir=str(inf_dir.relative_to(REPO_ROOT)),
        measured={"C_gen_mean_s": float(c_gen.mean()), "C_feat_mean_s": float(c_feat.mean()),
                  "C_v_fsm_mean_s": float(c_v.mean()), "C_v_mean_s": float(c_v.mean()), "C_v_source": c_v_source,
                  "domain": domain, "probe_scoring_median_s": probe_s,
                  "verifier_repeats": args.verifier_repeats},
        cstr_reference_verifier_s=CSTR_LEGACY_VERIFIER_S,
        accounting="saved = sum_{ALLOW,DISALLOW} C_v - sum_all C_f; C_f = feature pass (internal families) + probe scoring",
    ))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
