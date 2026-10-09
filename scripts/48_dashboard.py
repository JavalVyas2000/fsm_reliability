"""
Regenerate DASHBOARD.md: campaign status (from outputs/data_queue.log and the running jobs' logs), live closed-loop
results per model, and the headline skip-the-validator results. Run by the git pre-commit hook on every commit;
also runnable by hand. Never raises: on any error the section shows the error instead.

Example:
    python -m scripts.48_dashboard
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "data_queue.log"
CL = ROOT / "outputs" / "cstr_closed_loop"
TABLES = ROOT / "paper_outputs" / "tables"
POLICY_ORDER = ["always_validate", "observables_probe", "combined_probe", "internals_probe", "random_matched", "never_validate"]
POLICY_NAME = {"always_validate": "Always validate", "observables_probe": "Observables probe",
               "combined_probe": "Obs. + internals + grounding", "internals_probe": "Internals only",
               "random_matched": "Random routing", "never_validate": "Never validate"}
CL_MODELS = [("Qwen2.5-1.5B", "_qwen25-15b"), ("Qwen2.5-3B", "_qwen25-3b"), ("Llama-3.2-3B", "_llama-32-3b"),
             ("Qwen2.5-7B (4-bit)", "_qwen25-7b"), ("SmolLM2-1.7B", "_smollm2-17b")]
# Planned steps after the closed loops: (label, START marker, END marker)
PLANNED = [
    ("Llama-3.2-3B closed loop (resumed)", "[llama resume] START", "[llama resume] END"),
    ("FSM Qwen2.5-7B: pilot inference", "[model set] START fsm_qwen7_pilot", "[model set] END   fsm_qwen7_pilot "),
    ("FSM Qwen2.5-7B: pilot grounding", "[model set] START fsm_qwen7_pilot_grounding", "[model set] END   fsm_qwen7_pilot_grounding"),
    ("FSM Qwen2.5-7B: cert2 inference", "[model set] START fsm_qwen7_cert2", "[model set] END   fsm_qwen7_cert2 "),
    ("FSM Qwen2.5-7B: cert2 grounding", "[model set] START fsm_qwen7_cert2_grounding", "[model set] END   fsm_qwen7_cert2_grounding"),
    ("FSM Qwen2.5-7B: freeze + cert evaluation", "[model set] START fsm_qwen7_freeze", "[model set] END   fsm_qwen7_cert2_eval"),
    ("CSTR SmolLM2-1.7B: 100-episode pilot (gate)", "[model set] START cstr_smollm_pilot", "SmolLM2 CSTR pilot:"),
    ("CSTR SmolLM2-1.7B: first collection attempt (crashed at 1073/5000, worker died)", "[model set] START cstr_smollm_collect", "[model set] END   cstr_smollm_collect"),
    ("CSTR SmolLM2-1.7B: resumed collection (after internals-only runs)", "[smollm resume] START cstr_smollm_collect_resume", "[smollm resume] START cstr_smollm_grounding_full"),
    ("CSTR SmolLM2-1.7B: grounding, freeze, test + cert evaluation", "[smollm resume] START cstr_smollm_grounding_full", "[smollm resume] SMOLLM DONE"),
    ("Closed loop: SmolLM2-1.7B (all six policies)", "[smollm closed loop] START", "[smollm closed loop] END"),
    ("Closed loop, internals only: Qwen2.5-3B", "[internals closed loop] START qwen25-3b_internals", "[internals closed loop] END   qwen25-3b_internals"),
    ("Closed loop, internals only: Qwen2.5-1.5B", "[internals closed loop] START qwen25-15b_internals", "[internals closed loop] END   qwen25-15b_internals"),
]


def safe(fn):
    try:
        return fn()
    except Exception:  # the dashboard must never block a commit
        return ["", "_Section failed:_", "```", traceback.format_exc(limit=2), "```", ""]


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def log_lines():
    return LOG.read_text(encoding="utf-8-sig", errors="replace").splitlines() if LOG.exists() else []


def latest_run(suffix):
    runs = sorted(p for p in CL.glob(f"*{suffix}") if "SMOKE" not in p.name and "ABORTED" not in p.name
                  and "internals" not in p.name) if CL.exists() else []
    return runs[-1] if runs else None


def progress(log_path):
    """Last '[i/400] ... | m min' line of a closed-loop log -> (done, total, minutes, min/episode recent)."""
    if not log_path.exists():
        return None
    pts = re.findall(r"^\[(\d+)/(\d+)\].*\|\s*([\d.]+) min", log_path.read_text(encoding="utf-8", errors="replace"), flags=re.M)
    if not pts:
        return None
    pts = [(int(a), int(b), float(c)) for a, b, c in pts]
    i, n, m = pts[-1]
    seg = [p for p in pts if p[2] <= m][-30:]
    rate = (seg[-1][2] - seg[0][2]) / max(1, seg[-1][0] - seg[0][0]) if len(seg) > 1 else None
    return i, n, m, rate


def status_section():
    lines = log_lines()
    text = "\n".join(lines)
    out = ["## Campaign status", "", "| Step | Status |", "|---|---|"]
    for label, start, end in PLANNED:
        if end in text:
            st = "✅ done"
            if "gate" in label.lower():
                m = re.findall(r"SmolLM2 CSTR pilot: (.*)", text)
                st += f" ({m[-1]})" if m else ""
        elif start in text:
            st = "⏳ running"
            if "Llama" in label:
                p = progress(ROOT / "outputs" / "queue_closed_loop_llama-32-3b.log")
                if p:
                    i, n, m, rate = p
                    eta = ""
                    if rate:
                        eta_t = dt.datetime.now() + dt.timedelta(minutes=(n - i) * rate)
                        eta = f", ~{rate:.1f} min/episode, ETA {eta_t:%a %d %b %H:%M}"
                    st += f" — episode {i}/{n}{eta}"
            elif "SmolLM2-1.7B (all six" in label:
                p = progress(ROOT / "outputs" / "queue_closed_loop_smollm2-17b.log")
                if p:
                    i, n, m, rate = p
                    eta = ""
                    if rate:
                        eta_t = dt.datetime.now() + dt.timedelta(minutes=(n - i) * rate)
                        eta = f", ~{rate:.1f} min/episode, ETA {eta_t:%a %d %b %H:%M}"
                    st += f" — episode {i}/{n}{eta}"
            elif "internals only" in label:
                tag = "qwen25-3b_internals" if "3B" in label else "qwen25-15b_internals"
                p = progress(ROOT / "outputs" / f"queue_closed_loop_{tag}.log")
                if p:
                    st += f" — episode {p[0]}/{p[1]}"
        else:
            st = "🕓 waiting"
        out.append(f"| {label} | {st} |")
    if "gate not met" in text:
        out.append("| SmolLM2 CSTR full run | ⏭️ skipped (pilot gate not met, prereg Amendment 4) |")
    fails = [l for l in lines if re.search(r"\(exit [1-9]\)|STOP|SKIP", l)]
    out += ["", "<details><summary>Last 8 queue-log lines" + (f" · {len(fails)} failure/skip line(s) in the log" if fails else "")
            + "</summary>", "", "```", *lines[-8:], "```", "", "</details>", ""]
    return out


def closed_loop_section():
    out = ["## Closed-loop CSTR (400 fresh episodes per model; live)", "",
           "Δ = change vs always-validate on the same episodes. Failing executed = failing actions executed without validation.", ""]
    for model, suffix in CL_MODELS:
        run = latest_run(suffix)
        if run is None or not (run / "episodes.jsonl").exists():
            out += [f"**{model}** — not started", ""]
            continue
        e = pd.DataFrame([json.loads(l) for l in open(run / "episodes.jsonl", encoding="utf-8")])
        extra = sorted(CL.glob(f"*{suffix}_internals"))
        if extra and (extra[-1] / "episodes.jsonl").exists():
            x = pd.DataFrame([json.loads(l) for l in open(extra[-1] / "episodes.jsonl", encoding="utf-8")])
            e = pd.concat([e, x[~x.policy.isin(e.policy.unique())]], ignore_index=True)
        n_ep = e[e.policy == "always_validate"].episode_id.nunique()
        e = e[e.episode_id.isin(e[e.policy == "always_validate"].episode_id)]
        e["unsafe"] = e.outcome.str.startswith("executed_failure")
        g = e.groupby("policy").agg(n=("episode_id", "nunique"), rec=("recovered", "mean"), unsafe=("unsafe", "sum"),
                                    calls=("validator_calls", "mean"), t=("t_total_s", "mean"))
        g = g.reindex([p for p in POLICY_ORDER if p in g.index])
        b = g.loc["always_validate"]
        done = "complete" if n_ep >= 400 else f"**in progress: {n_ep}/400 episodes**"
        out += [f"**{model}** — {done} (`{run.name}`)", "",
                "| Policy | Episodes | Recovered | Failing executed | Validator calls / ep (Δ) | Compute / ep |",
                "|---|---|---|---|---|---|"]
        for p, r in g.iterrows():
            d = "" if p == "always_validate" else f" ({100 * (r.calls / b.calls - 1):+.0f}%)" if b.calls else ""
            dr = "" if p == "always_validate" else f" ({100 * (r.rec - b.rec):+.1f})"
            out.append(f"| {POLICY_NAME[p]} | {int(r.n)} | {100 * r.rec:.1f}%{dr} | {int(r.unsafe)} | {r.calls:.2f}{d} | {r.t:.0f} s |")
        out.append("")
    return out


def skip_section():
    out = ["## Validator calls skipped per signal (certification sets, X = 10%)", "",
           "`*` = failure rate among unchecked accepts certified ≤ 10%. Full tables: `paper_outputs/tables/`.", ""]
    order = ["Observables", "Obs. + grounding", "Obs. + all internals", "Obs. + internals + grounding", "Region grounding",
             "Hidden states", "All internals", "All internals + grounding", "Token confidence"]
    for f, title in (("skip_fsm_cert.csv", "FSM (fresh cert2)"), ("skip_cstr_cert.csv", "CSTR (cert)")):
        p = TABLES / f
        if not p.exists():
            out += [f"_{f} not generated yet_", ""]
            continue
        T = pd.read_csv(p)
        T = T[(T.tolerance - 0.10).abs() < 1e-9]
        T["c"] = T.apply(lambda r: f"{r.skipped_pct:.0f}%{'*' if r.certified else ''} ({int(r.accepted_failures)}/{int(r.accepted)})", axis=1)
        piv = T.pivot(index="signal", columns="model", values="c").reindex([o for o in order if o in set(T.signal)])
        piv = piv[list(dict.fromkeys(T.model))]
        out += [f"**{title}** — cell: calls skipped (failing / accepted unchecked)", "",
                "| Signal | " + " | ".join(piv.columns) + " |", "|---" * (len(piv.columns) + 1) + "|"]
        out += ["| " + s + " | " + " | ".join("" if pd.isna(v) else v for v in r) + " |" for s, r in piv.iterrows()]
        out.append("")
    return out


def findings_section():
    return ["## Headline findings so far", "",
            "- **FSM (fresh cert2, 4 models):** adding internals to observables skips +6 to +15 points more validator calls "
            "for 3 of 4 models at a certified failure rate; region grounding carries the gain for Llama and Qwen2.5-1.5B.",
            "- **CSTR (4 models):** grounding never adds to observables; internals help only Qwen2.5-3B (accepts) and Llama "
            "(fewer good proposals rejected); internals alone skip 10–53% with 0 failing actions let through.",
            "- **Shift:** internals are not more robust than observables to plant / fault shift (`docs/cstr_shift_results.md`).",
            "- **Closed loop:** trained screens avoid 44–90% of validator calls with far fewer unsafe executions than random "
            "skipping; Qwen2.5-7B's combined probe fails on retries (accept unchecked only on first proposals fixes it); "
            "time savings grow with validator cost (`paper_outputs/tables/closed_loop_costs_all.md`).", ""]


def figures_section():
    figs = [("closed_loop_compute_all", "Compute vs validator cost (closed loop)"), ("delta_vs_observables", "Gain over observables"),
            ("skip_fsm", "FSM calls skipped per signal"), ("skip_cstr", "CSTR calls skipped per signal"),
            ("mechanism_grounding", "Grounding vs failure"), ("shift_auroc_change", "Shift: AUROC change")]
    out = ["## Figures", ""]
    for f, cap in figs:
        if (ROOT / "paper_outputs" / "figures" / f"{f}.png").exists():
            out += [f"**{cap}**", "", f"![{cap}](paper_outputs/figures/{f}.png)", ""]
    out += ["## Documents", "",
            "- Pre-registration: `docs/cstr_v4_prereg.md`, `docs/cstr_closed_loop_prereg.md`, `docs/cstr_shift_prereg.md`",
            "- Results: `docs/skip_validator_results.md`, `docs/cstr_closed_loop_results.md`, `docs/cstr_shift_results.md`, "
            "`docs/cstr_v4_results.md`, `docs/grounding_results.md`", ""]
    return out


def main():
    head = git("log", "-1", "--format=%h %s")
    md = ["# Selective verification — campaign dashboard", "",
          f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/48_dashboard.py` (git pre-commit hook). "
          f"Previous commit: `{head}`._", ""]
    for section in (status_section, closed_loop_section, findings_section, skip_section, figures_section):
        md += safe(section)
    (ROOT / "DASHBOARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("DASHBOARD.md updated")


if __name__ == "__main__":
    main()
