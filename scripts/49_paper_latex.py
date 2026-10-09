"""
Build the paper's LaTeX tables (paper/tables/*.tex) from saved results and copy the figures (paper/figures/).
No number in the manuscript's tables is typed by hand: re-run after any result changes.

Sources: paper_outputs/tables/*.csv (scripts/44, 46, 47) and the closed-loop runs (scripts/42).

Example:
    python -m scripts.49_paper_latex
"""
from __future__ import annotations

import importlib
import json
import shutil

import numpy as np
import pandas as pd

from src.utils.manifest import REPO_ROOT

PAPER = REPO_ROOT / "paper"
TAB = REPO_ROOT / "paper_outputs" / "tables"
FIG = REPO_ROOT / "paper_outputs" / "figures"
CL = REPO_ROOT / "outputs" / "cstr_closed_loop"
S47 = importlib.import_module("scripts.47_closed_loop_costs")

FSM_MODELS = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B (4-bit)", "Llama-3.2-3B", "SmolLM2-1.7B"]
CSTR_MODELS = ["Qwen2.5-1.5B", "Qwen2.5-3B*", "Qwen2.5-7B (4-bit)", "Llama-3.2-3B", "SmolLM2-1.7B"]
SIGNALS = ["Observables", "Obs. + grounding", "Obs. + all internals", "Obs. + internals + grounding",
           "Region grounding", "Attention (regions)", "Hidden states", "All internals", "All internals + grounding",
           "Token confidence"]
SHORT = {"Qwen2.5-1.5B": "Qwen-1.5B", "Qwen2.5-3B": "Qwen-3B", "Qwen2.5-3B*": "Qwen-3B$^\\dagger$",
         "Qwen2.5-7B (4-bit)": "Qwen-7B", "Llama-3.2-3B": "Llama-3B", "SmolLM2-1.7B": "SmolLM-1.7B"}
CL_RUNS = {  # model -> run directory suffixes merged by episode (main run first)
    "Qwen2.5-1.5B": ["_qwen25-15b", "_qwen25-15b_internals"],
    "Qwen2.5-3B": ["_qwen25-3b", "_qwen25-3b_internals"],
    "Qwen2.5-7B (4-bit)": ["_qwen25-7b", "_qwen25-7b_r0rule"],
    "Llama-3.2-3B": ["_llama-32-3b"],
    "SmolLM2-1.7B": ["_smollm2-17b", "_smollm2-17b_r0rule"],
}
CL_POLICIES = [("always_validate", "Always validate"), ("observables_probe", "Observables"),
               ("observables_probe_r0", "Observables, R0"), ("combined_probe", "Obs.+int.+grd."),
               ("combined_probe_r0", "Obs.+int.+grd., R0"), ("internals_probe", "Internals only"),
               ("internals_probe_r0", "Internals only, R0"), ("random_matched", "Random (matched)"),
               ("never_validate", "Never validate")]


def esc(s):
    return s.replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write(name, body):
    (PAPER / "tables" / f"{name}.tex").write_text(body, encoding="utf-8")


def table_skip(domain, models, caption, label):
    T = pd.read_csv(TAB / f"skip_{domain}_cert.csv")
    T = T[(T.tolerance - 0.10).abs() < 1e-9]
    models = [m for m in models if m in set(T.model)]
    lines = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * len(models) + "}", "\\toprule",
             "Signal & " + " & ".join(SHORT[m] for m in models) + " \\\\", "\\midrule"]
    for sig in SIGNALS:
        cells = []
        for m in models:
            r = T[(T.model == m) & (T.signal == sig)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            c = f"{r.skipped_pct:.0f}"
            if r.certified:
                c = f"\\textbf{{{c}}}"
            if sig != "Observables" and not np.isnan(r.delta_lo) and (r.delta_lo > 0 or r.delta_hi < 0):
                c += "$^{+}$" if r.delta_lo > 0 else "$^{-}$"
            cells.append(c)
        lines.append(esc(sig) + " & " + " & ".join(cells) + " \\\\")
        if sig in ("Observables", "Obs. + internals + grounding"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(f"skip_{domain}", "\n".join(lines))


def table_datasets():
    D = pd.read_csv(TAB / "datasets.csv")
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Evaluation sets used for the confirmatory results (one candidate per graph or episode). "
             "FSM: fresh certification set, never used before evaluation. CSTR: sealed test and certification "
             "partitions, each evaluated once.}", "\\label{tab:datasets}",
             "\\begin{tabular}{llrrr}", "\\toprule", "Domain & Model & Set & Eligible & Failure rate \\\\", "\\midrule"]
    for _, r in D.iterrows():
        lines.append(f"{r.domain} & {SHORT.get(r.model, r.model)} & {esc(str(r.set))} & {r.eligible} & {r.failure_rate:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("datasets", "\n".join(lines))


def table_mechanism():
    M = pd.read_csv(TAB / "mechanism_grounding_auroc.csv")
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Does low attention on the prompt region that decides the answer predict failure? AUROC of the "
             "negative minimum grounding share for answer failure (train + development data; 0.5 = no signal).}",
             "\\label{tab:mechanism}", "\\begin{tabular}{lrr}", "\\toprule", "Model & FSM & CSTR \\\\", "\\midrule"]
    for m in FSM_MODELS:
        f = M[(M.domain == "FSM") & (M.model == m)].auroc_low_grounding_predicts_failure
        c = M[(M.domain == "CSTR") & (M.model.isin([m, m + "*"]))].auroc_low_grounding_predicts_failure
        lines.append(f"{SHORT[m]} & {f.iloc[0]:.2f} & {c.iloc[0]:.2f} \\\\" if len(f) and len(c) else
                     f"{SHORT[m]} & {f.iloc[0] if len(f) else '--'} & {c.iloc[0] if len(c) else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("mechanism", "\n".join(lines))


def closed_loop_frames():
    out = {}
    for model, suffixes in CL_RUNS.items():
        dirs = []
        for suf in suffixes:
            cands = sorted(p for p in CL.glob(f"*{suf}") if p.name.endswith(suf) and "ABORTED" not in p.name and "SMOKE" not in p.name)
            if cands and (cands[-1] / "episodes.jsonl").exists():
                dirs.append(str(cands[-1].relative_to(REPO_ROOT)))
        if dirs:
            e, _ = S47.load(",".join(dirs))
            out[model] = e
    return out


def table_closed_loop(frames):
    rng = np.random.default_rng(0)
    lines = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Closed-loop CSTR recovery on 400 fresh episodes per model. $\\Delta$: paired difference to "
             "always-validate (percentage points, bootstrap 95\\% CI over episodes). Unsafe: failing actions executed "
             "without validation. Calls: validator calls avoided relative to always-validate. R0: the probe may skip "
             "validation only for first proposals (retries are always validated).}",
             "\\label{tab:closedloop}", "\\begin{tabular}{llrrrr}", "\\toprule",
             "Model & Policy & Recovered (\\%) & $\\Delta$ recovered & Unsafe & Calls avoided \\\\", "\\midrule"]
    for model, e in frames.items():
        e = e.copy()
        e["unsafe"] = e.outcome.str.startswith("executed_failure")
        base = e[e.policy == "always_validate"].set_index("episode_id")
        idx = rng.integers(0, len(base), (2000, len(base)))
        first = True
        for p, name in CL_POLICIES:
            sub = e[e.policy == p].set_index("episode_id")
            if sub.empty:
                continue
            sub = sub.reindex(base.index)
            dv = (sub.recovered.astype(float) - base.recovered.astype(float)).to_numpy()
            b = dv[idx].mean(1) * 100
            d = "--" if p == "always_validate" else f"{100 * dv.mean():+.1f} [{np.percentile(b, 2.5):+.1f}, {np.percentile(b, 97.5):+.1f}]"
            calls = "--" if p == "always_validate" else f"{100 * (1 - sub.validator_calls.mean() / base.validator_calls.mean()):.0f}\\%"
            lines.append(f"{SHORT[model] if first else ''} & {name} & {100 * sub.recovered.mean():.1f} & {d} & "
                         f"{int(sub.unsafe.sum())} & {calls} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    write("closed_loop", "\n".join(lines))


def table_costs():
    C = pd.read_csv(TAB / "closed_loop_costs_all.csv")
    keep = ["Observables probe", "Obs. + internals + grounding probe", "Internals-only probe",
            "Observables probe, first-proposal-only", "Obs. + internals + grounding, first-proposal-only",
            "Internals-only, first-proposal-only"]
    lines = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Compute per episode relative to always-validate (positive = saving). The logged closed-loop "
             "decisions do not depend on validator latency (the plant is paused during decisions), so the same "
             "decisions are re-costed exactly for a validator taking 30\\,s or 60\\,s per call. Break-even: validator "
             "cost per call above which the policy saves time.}", "\\label{tab:costs}",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "Model & Policy & Measured & 30\\,s & 60\\,s & Break-even (s) \\\\", "\\midrule"]
    for model in C.model.unique():
        sub = C[(C.model == model) & C.policy.isin(keep)]
        first = True
        for _, r in sub.iterrows():
            be = "--" if pd.isna(r.breakeven_validator_s) else ("any" if r.breakeven_validator_s <= 0 else f"{r.breakeven_validator_s:.0f}")
            lines.append(f"{SHORT[model] if first else ''} & {esc(r.policy)} & {r.saving_pct_measured:+.0f}\\% & "
                         f"{r.saving_pct_30s:+.0f}\\% & {r.saving_pct_60s:+.0f}\\% & {be} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    write("costs", "\n".join(lines))


def table_shift():
    R = pd.read_csv(TAB / "shift_all.csv")
    R = R[(R.tolerance - 0.10).abs() < 1e-9]
    sigs = ["Observables", "Obs. + all internals", "All internals", "Token confidence"]
    shifts = ["control", "feed", "cooling", "family (pooled)"]
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{AUROC for failure on the target part of each pre-registered shift (probes fitted on the "
             "source part only). Control: random split of the same size.}", "\\label{tab:shift}",
             "\\begin{tabular}{llrrrr}", "\\toprule", "Model & Signal & Control & Feed & Cooling & Fault family \\\\", "\\midrule"]
    for model in R.model.unique():
        first = True
        for sig in sigs:
            vals = []
            for sh in shifts:
                v = R[(R.model == model) & (R.signal_name == sig) & (R["shift"] == sh)].auroc
                vals.append(f"{v.iloc[0]:.2f}" if len(v) and not pd.isna(v.iloc[0]) else "--")
            lines.append(f"{SHORT.get(model, model) if first else ''} & {esc(sig)} & " + " & ".join(vals) + " \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    write("shift", "\n".join(lines))


def main():
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    for f in FIG.glob("*.pdf"):
        shutil.copy2(f, PAPER / "figures" / f.name)
    table_datasets()
    table_skip("fsm", FSM_MODELS, "FSM: share of validator calls skipped (\\%) per signal on the fresh certification "
               "set (tolerance $X=10\\%$). Bold: failure rate among unchecked accepts certified $\\le 10\\%$ "
               "(one-sided Clopper--Pearson, Bonferroni over signals). $^{+}$/$^{-}$: significantly more/fewer calls "
               "skipped than observables (paired bootstrap 95\\% CI).", "tab:skip-fsm")
    table_skip("cstr", CSTR_MODELS, "CSTR: share of validator calls skipped (\\%) per signal on the certification "
               "partition ($X=10\\%$); notation as in Table~\\ref{tab:skip-fsm}. $^\\dagger$Qwen-3B: exploratory "
               "re-analysis under the bound-based rule (its sealed sets had been used once before the rule change).",
               "tab:skip-cstr")
    table_mechanism()
    frames = closed_loop_frames()
    table_closed_loop(frames)
    table_costs()
    table_shift()
    (PAPER / "tables" / "MANIFEST.json").write_text(json.dumps(
        {"closed_loop_runs": {m: CL_RUNS[m] for m in frames}, "sources": "paper_outputs/tables, outputs/cstr_closed_loop"},
        indent=2), encoding="utf-8")
    print("paper tables:", sorted(p.name for p in (PAPER / "tables").glob("*.tex")))


if __name__ == "__main__":
    main()
