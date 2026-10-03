"""
All paper tables and figures from saved results (no model calls).

Outputs: paper_outputs/tables/*.md|csv, paper_outputs/figures/*.png|pdf, paper_outputs/manifest.json.

Sources: the frozen skip-the-validator runs (scripts/39-40), the grounding-augmented records
(scripts/27/31) and the closed-loop run (scripts/42).

Example:
    python -m scripts.44_paper_outputs
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.utils.manifest import REPO_ROOT, build_manifest, write_json  # noqa: E402

OUT = REPO_ROOT / "paper_outputs"
CERT = REPO_ROOT / "outputs/certification"
# validated categorical slots 1-5 (dataviz reference palette, light mode) and text tokens
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"

FSM = {  # model -> (freeze on pilot, evaluated on fresh cert2; pilot aug for mechanism)
    "Qwen2.5-3B": ("*_fsm_qwen25-3b_skip_ucb_cert2", "outputs/fsm_grounding/20260926_150747_20260923_205804_qwen25-3b-instruct_pilot3000/aug"),
    "Llama-3.2-3B": ("*_fsm_llama-32-3b_skip_ucb_cert2", "outputs/fsm_grounding/20260926_152900_20260924_000726_llama-32-3b-instruct_pilot3000/aug"),
    "Qwen2.5-1.5B": ("*_fsm_qwen25-15b_skip_ucb_cert2", "outputs/fsm_grounding/20260926_154919_20260924_014439_qwen25-15b-instruct_pilot3000/aug"),
    "SmolLM2-1.7B": ("*_fsm_smollm2-17b_skip_ucb_cert2", "outputs/fsm_grounding/20260926_160357_20260924_032202_smollm2-17b-instruct_pilot3000/aug"),
}
CSTR = {
    "Qwen2.5-1.5B": ("*_cstr_v4_qwen25-15b_skip_ucb", "outputs/cstr_grounding/20261002_040625_20261001_115207_qwen25-15b-instruct_v4_v31_r0/aug"),
    "Qwen2.5-3B*": ("*_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY", "outputs/cstr_grounding/20260929_204745_20260928_201301_qwen25-3b-instruct_v4_v31_r0/aug"),
    "Qwen2.5-7B (4-bit)": ("*_cstr_v4_qwen25-7b_skip_ucb", "outputs/cstr_grounding/20261003_022728_20261002_083124_qwen25-7b-instruct_v4_v31_r0/aug"),
    "Llama-3.2-3B": ("*_cstr_v4_llama-32-3b_skip_ucb", "outputs/cstr_grounding/20261001_065645_20260929_214136_llama-32-3b-instruct_v4_v31_r0/aug"),
}
CLOSED_LOOP = "outputs/cstr_closed_loop/20261003_033144_qwen25-3b"

# signal label (as frozen) -> (display name, family)
SIGNAL_NAMES = {
    "task context + proposed path": ("Observables", "obs"), "plant readings + proposed change": ("Observables", "obs"),
    "plant readings only": ("Plant readings only", "obs"),
    "token confidence": ("Token confidence", "tok"),
    "attention by prompt region": ("Attention (regions)", "int"),
    "region-based attention grounding": ("Region grounding", "int"),
    "hidden states": ("Hidden states", "int"),
    "all internals": ("All internals", "int"),
    "all internals + grounding": ("All internals + grounding", "int"),
    "context + path + grounding": ("Obs. + grounding", "comb"), "readings + change + grounding": ("Obs. + grounding", "comb"),
    "context + path + all internals": ("Obs. + all internals", "comb"), "readings + change + all internals": ("Obs. + all internals", "comb"),
    "context + path + all internals + grounding": ("Obs. + internals + grounding", "comb"),
    "readings + change + all internals + grounding": ("Obs. + internals + grounding", "comb"),
}
ORDER = ["Observables", "Plant readings only", "Obs. + grounding", "Obs. + all internals", "Obs. + internals + grounding",
         "Region grounding", "Attention (regions)", "Hidden states", "All internals", "All internals + grounding", "Token confidence"]
FAMILY_COLOR = {"obs": S1, "comb": S2, "int": S3, "tok": S4}
FAMILY_NAME = {"obs": "Observables", "comb": "Observables + internals", "int": "Internals alone", "tok": "Token confidence"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def latest(pattern):
    hits = sorted(CERT.glob(pattern))
    if not hits:
        raise SystemExit(f"no run matches {pattern}")
    return hits[-1]


def eval_rows(d, part):
    ev = json.loads((d / f"eval_{part}.json").read_text())
    dec_path = d / f"decisions_{part}.csv"
    dec = pd.read_csv(dec_path) if dec_path.exists() else None
    base = json.loads((d / "rules.json").read_text()).get("baseline", "plant readings + proposed change")
    ub_key = next(k for k in ev["rows"][0] if k.startswith("cp_upper"))
    rows = []
    rng = np.random.default_rng(0)
    for r in ev["rows"]:
        name, fam = SIGNAL_NAMES[r["signal"]]
        row = {"signal": name, "family": fam, "tolerance": r["tolerance"], "n": r["n"], "failure_rate": ev["failure_rate"],
               "skipped_pct": r["skipped_pct"], "accepted": r["accepted"], "accepted_failures": r["accepted_failures"],
               "cp_upper": r[ub_key], "certified": r["certified"], "rejected": r["rejected"], "rejected_good": r["rejected_good"],
               "delta_pts": np.nan, "delta_lo": np.nan, "delta_hi": np.nan}
        if dec is not None and r["signal"] != base:
            col, bcol = f"decision::{r['tolerance']:.2f}::{r['signal']}", f"decision::{r['tolerance']:.2f}::{base}"
            dv = (dec[col] != "validate").to_numpy(float) - (dec[bcol] != "validate").to_numpy(float)
            boots = dv[rng.integers(0, len(dv), (2000, len(dv)))].mean(1) * 100
            row.update(delta_pts=100 * dv.mean(), delta_lo=np.percentile(boots, 2.5), delta_hi=np.percentile(boots, 97.5))
        elif r["signal"] == base:
            row.update(delta_pts=0.0, delta_lo=0.0, delta_hi=0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def table_md(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def cell(r):
    if pd.isna(r.skipped_pct):
        return ""
    acc = f"{r.accepted_failures}/{r.accepted}" if r.accepted else "0"
    d = "" if r.signal == "Observables" or pd.isna(r.delta_pts) else f"; Δ {r.delta_pts:+.1f} [{r.delta_lo:+.1f}, {r.delta_hi:+.1f}]"
    return f"{r.skipped_pct:.1f}% (acc {acc}{', cert' if r.certified else ''}; rej-good {r.rejected_good}{d})"


def skip_tables(domain, models, part):
    frames = []
    for model, (pat, _) in models.items():
        df = eval_rows(latest(pat), part)
        df.insert(0, "model", model)
        frames.append(df)
    T = pd.concat(frames, ignore_index=True)
    T.to_csv(OUT / "tables" / f"skip_{domain}_{part}.csv", index=False)
    md = [f"# {domain.upper()}: validator calls skipped per signal ({part})", "",
          "Cell: share of validator calls skipped (accepted: failures/accepted; cert = certified failure bound <= X; "
          "rej-good = good proposals rejected unchecked; Δ = change vs observables, points, paired bootstrap 95% CI).", ""]
    for x in (0.10, 0.05):
        sub = T[np.isclose(T.tolerance, x)].copy()
        sub["cell"] = sub.apply(cell, axis=1)
        piv = sub.pivot(index="signal", columns="model", values="cell").reindex([o for o in ORDER if o in set(sub.signal)])
        piv = piv[list(models)]
        md += [f"## X = {x:.0%}", "", "| signal | " + " | ".join(piv.columns) + " |", "|---" * (len(piv.columns) + 1) + "|"]
        md += ["| " + s + " | " + " | ".join("" if pd.isna(v) else v for v in r) + " |" for s, r in piv.iterrows()]
        md.append("")
    (OUT / "tables" / f"skip_{domain}_{part}.md").write_text("\n".join(md), encoding="utf-8")
    return T


def fig_skip(T, models, domain, title):
    sub = T[np.isclose(T.tolerance, 0.10)]
    sigs = [o for o in ORDER if o in set(sub.signal)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 0.32 * len(sigs) + 1.4), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        style(ax)
        d = sub[sub.model == model].set_index("signal").reindex(sigs)
        y = np.arange(len(sigs))[::-1]
        ax.barh(y, d.skipped_pct, height=0.62, color=[FAMILY_COLOR[f] for f in d.family], edgecolor=SURFACE, linewidth=2)
        for yi, (_, r) in zip(y, d.iterrows()):
            tag = "  ✓" if r.certified else ""
            ax.text(r.skipped_pct + 1, yi, f"{r.skipped_pct:.0f}%{tag}", va="center", fontsize=7, color=INK2)
        ax.set_xlim(0, 105)
        ax.set_title(model, fontsize=9, color=INK, loc="left")
        ax.set_yticks(y)
        ax.set_yticklabels(sigs, fontsize=8, color=INK)
        ax.set_xlabel("validator calls skipped (%)", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, FAMILY_NAME.values(), loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left", y=1.08)
    fig.text(0.01, -0.02, "✓ = failure rate among unchecked accepts certified ≤ 10% (Clopper–Pearson, Bonferroni).",
             fontsize=7, color=MUTED)
    save(fig, f"skip_{domain}")


def fig_delta(T_fsm, T_cstr):
    focus = [("Obs. + grounding", S1), ("Obs. + all internals", S2), ("Obs. + internals + grounding", S3)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharex=True)
    for ax, (T, title) in zip(axes, [(T_fsm, "FSM (fresh certification set)"), (T_cstr, "CSTR (certification set)")]):
        style(ax)
        sub = T[np.isclose(T.tolerance, 0.10)]
        models = list(dict.fromkeys(sub.model))
        for k, (sig, col) in enumerate(focus):
            for i, m in enumerate(models):
                r = sub[(sub.model == m) & (sub.signal == sig)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = len(models) - 1 - i + (1 - k) * 0.22
                if pd.isna(r.delta_pts):  # no per-proposal decisions saved: point difference only
                    obs = sub[(sub.model == m) & (sub.signal == "Observables")].skipped_pct.iloc[0]
                    r = r.copy()
                    r["delta_pts"] = r.skipped_pct - obs
                if not pd.isna(r.delta_lo):
                    ax.plot([r.delta_lo, r.delta_hi], [y, y], color=col, linewidth=2, solid_capstyle="round")
                ax.plot(r.delta_pts, y, "o", color=col, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=sig if i == 0 else None)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(range(len(models))[::-1])
        ax.set_yticklabels(models, fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Δ validator calls skipped vs observables (points)", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.01, -0.04, "Tolerance X = 10%. Bars: paired bootstrap 95% CI. CSTR Qwen2.5-3B: exploratory re-analysis, point only.",
             fontsize=7, color=MUTED)
    save(fig, "delta_vs_observables")


def mechanism():
    rows = []
    for domain, models, feat in (("FSM", FSM, "grd_min_share"), ("CSTR", CSTR, "g31_min_share")):
        for model, (_, aug) in models.items():
            ys, xs = [], []
            for line in open(REPO_ROOT / aug / "records.jsonl", encoding="utf-8"):
                r = json.loads(line)
                if r.get("partition") not in ("train", "dev_cal", "dev_thr") or r.get("round", 0) != 0:
                    continue
                if r.get("schema_valid") != 1 or r.get("candidate_invalid") is None:
                    continue
                v = (r.get("features") or {}).get(feat)
                if v is None or not np.isfinite(v):
                    continue
                ys.append(int(r["candidate_invalid"]))
                xs.append(-float(v))
            rows.append({"domain": domain, "model": model, "feature": feat, "n": len(ys),
                         "auroc_low_grounding_predicts_failure": roc_auc_score(ys, xs)})
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "tables" / "mechanism_grounding_auroc.csv", index=False)
    (OUT / "tables" / "mechanism_grounding_auroc.md").write_text(
        "# Does low attention on the deciding prompt region predict failure? (train + dev, answer level)\n\n"
        + table_md(M.round(3), ["domain", "model", "feature", "n", "auroc_low_grounding_predicts_failure"],
                   ["domain", "model", "feature", "n", "AUROC"]), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    style(ax)
    labels, vals, cols = [], [], []
    for _, r in M.iterrows():
        labels.append(f"{r.domain}: {r.model}")
        vals.append(r.auroc_low_grounding_predicts_failure)
        cols.append(S1 if r.domain == "FSM" else S2)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, np.array(vals) - 0.5, left=0.5, height=0.6, color=cols, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + (0.01 if v >= 0.5 else -0.01), yi, f"{v:.2f}", va="center", ha="left" if v >= 0.5 else "right",
                fontsize=7, color=INK2)
    ax.axvline(0.5, color=INK2, linewidth=1)
    ax.set_xlim(0.25, 0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel("AUROC: low grounding → failure (0.5 = no signal)", fontsize=8, color=INK2)
    ax.set_title("Does low attention on the deciding prompt region predict failure?", fontsize=9, color=INK, loc="left")
    save(fig, "mechanism_grounding")
    return M


def closed_loop():
    d = REPO_ROOT / CLOSED_LOOP
    e = pd.DataFrame([json.loads(l) for l in open(d / "episodes.jsonl", encoding="utf-8")])
    e["executed_failure"] = e.outcome.str.startswith("executed_failure")
    pol = [("always_validate", "Always validate", S1), ("observables_probe", "Observables probe", S2),
           ("combined_probe", "Obs. + internals + grounding probe", S3), ("random_matched", "Random routing", S4),
           ("never_validate", "Never validate", S5)]
    g = e.groupby("policy").agg(recovered=("recovered", "mean"), executed_failure=("executed_failure", "sum"),
                                calls=("validator_calls", "mean"), proposals=("proposals", "mean"),
                                t_gen=("t_generation_s", "mean"), t_int=("t_internals_s", "mean"),
                                t_grd=("t_grounding_s", "mean"), t_val=("t_validator_s", "mean"))
    g = g.reindex([p for p, _, _ in pol])
    g.to_csv(OUT / "tables" / "closed_loop.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, (col, xl, fmt) in zip(axes, [("recovered", "episodes recovered (%)", lambda v: f"{100 * v:.1f}%"),
                                         ("calls", "validator calls per episode", lambda v: f"{v:.2f}"),
                                         ("executed_failure", "failing actions executed unchecked (of 400)", lambda v: f"{v:.0f}")]):
        style(ax)
        vals = g[col] * (100 if col == "recovered" else 1)
        y = np.arange(len(pol))[::-1]
        ax.barh(y, vals, height=0.6, color=[c for _, _, c in pol], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, g[col]):
            ax.text((100 * v if col == "recovered" else v) + vals.max() * 0.02, yi, fmt(v), va="center", fontsize=7, color=INK2)
        ax.set_xlim(0, vals.max() * 1.25)
        ax.set_xlabel(xl, fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([n for _, n, _ in pol], fontsize=8, color=INK)
    fig.suptitle("Closed-loop CSTR, 400 fresh episodes, Qwen2.5-3B", fontsize=10, color=INK, x=0.01, ha="left", y=1.04)
    save(fig, "closed_loop")

    # Verifier-cost sweep: decisions do not depend on verifier latency (plant paused), so total time is
    # generation + feature passes + validator calls x C_v for any validator cost C_v.
    cv = np.linspace(0, 60, 121)
    measured = e.t_validator_s.sum() / max(1, e.validator_calls.sum())
    sweep = {}
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for p, name, c in pol:
        r = g.loc[p]
        t = r.t_gen + r.t_int + r.t_grd + r.calls * cv
        sweep[p] = t
        ax.plot(cv, t, color=c, linewidth=2, label=name)
    ax.axvline(measured, color=MUTED, linewidth=1, linestyle="--")
    ax.text(measured + 0.5, 22, f"measured {measured:.1f} s", fontsize=7, color=MUTED, va="bottom")
    ax.axvline(30, color=MUTED, linewidth=1, linestyle=":")
    ax.text(30.5, 22, "30 s simulation", fontsize=7, color=MUTED, va="bottom")
    ax.set_xlabel("validator cost per call (s)", fontsize=8, color=INK2)
    ax.set_ylabel("compute per episode (s)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Total compute vs validator cost (same decisions, re-costed)", fontsize=9, color=INK, loc="left")
    save(fig, "compute_vs_validator_cost")
    rows = []
    base = g.loc["always_validate"]
    for p, name, _ in pol:
        r = g.loc[p]
        fixed = (r.t_gen + r.t_int + r.t_grd) - (base.t_gen + base.t_int + base.t_grd)
        dcalls = base.calls - r.calls
        breakeven = fixed / dcalls if dcalls > 0 else np.nan
        rows.append({"policy": name, "recovered_pct": round(100 * r.recovered, 1), "validator_calls_per_ep": round(r.calls, 2),
                     "unchecked_failing_executions": int(r.executed_failure),
                     "compute_s_at_measured": round(r.t_gen + r.t_int + r.t_grd + r.calls * measured, 1),
                     "compute_s_at_30s": round(r.t_gen + r.t_int + r.t_grd + r.calls * 30, 1),
                     "compute_s_at_60s": round(r.t_gen + r.t_int + r.t_grd + r.calls * 60, 1),
                     "breakeven_validator_s_vs_always": round(breakeven, 1) if p not in ("always_validate", "never_validate") else ""})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "tables" / "closed_loop_cost.csv", index=False)
    (OUT / "tables" / "closed_loop_cost.md").write_text(
        f"# Closed loop: outcomes and compute vs validator cost (measured validator {measured:.1f} s/call)\n\n"
        + table_md(C, list(C.columns), ["policy", "recovered %", "validator calls/ep", "unchecked failing executions",
                                        f"compute s/ep @ {measured:.1f} s", "@ 30 s", "@ 60 s", "break-even validator s"]),
        encoding="utf-8")
    return g, C


def datasets():
    rows = []
    for domain, models in (("FSM", FSM), ("CSTR", CSTR)):
        for model, (pat, aug) in models.items():
            d = latest(pat)
            for part in ("test_iid", "cert"):
                p = d / f"eval_{part}.json"
                if not p.exists():
                    continue
                ev = json.loads(p.read_text())
                rows.append({"domain": domain, "model": model, "set": "cert2 (fresh)" if domain == "FSM" else part,
                             "eligible": ev["n_eligible"], "format_failures": ev["n_format_failures"],
                             "failure_rate": round(ev["failure_rate"], 3)})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "tables" / "datasets.csv", index=False)
    (OUT / "tables" / "datasets.md").write_text("# Evaluation sets\n\n" + table_md(
        D, list(D.columns), ["domain", "model", "set", "eligible proposals", "format failures", "failure rate"]), encoding="utf-8")
    return D


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    D = datasets()
    T_fsm = skip_tables("fsm", FSM, "cert")
    T_cstr = skip_tables("cstr", CSTR, "cert")
    skip_tables("cstr", {k: v for k, v in CSTR.items()}, "test_iid")
    fig_skip(T_fsm, list(FSM), "fsm", "FSM: validator calls skipped per signal (fresh certification set, X = 10%)")
    fig_skip(T_cstr, list(CSTR), "cstr", "CSTR: validator calls skipped per signal (certification set, X = 10%)")
    fig_delta(T_fsm, T_cstr)
    M = mechanism()
    g, C = closed_loop()
    write_json(OUT / "manifest.json", build_manifest(
        sources={"fsm": {k: str(latest(v[0]).relative_to(REPO_ROOT)) for k, v in FSM.items()},
                 "cstr": {k: str(latest(v[0]).relative_to(REPO_ROOT)) for k, v in CSTR.items()},
                 "closed_loop": CLOSED_LOOP},
        notes={"cstr_qwen25-3b": "exploratory UCB re-analysis (its test_iid/cert had been used under Amendment 2)"}))
    print(D.to_string(index=False))
    print(M.round(3).to_string(index=False))
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
