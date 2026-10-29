"""
Write the data files behind the manuscript's pgfplots figures (paper/data/*.dat) from saved results.
No number in a paper figure is typed by hand: re-run after any result changes (after scripts/44, 47 and 49).

Sources: paper_outputs/tables/{skip_fsm_cert,skip_cstr_cert,mechanism_grounding_auroc,closed_loop_costs_all}.csv

Example:
    python -m scripts.50_paper_figure_data
"""
from __future__ import annotations

import pandas as pd

from src.utils.manifest import REPO_ROOT

TAB = REPO_ROOT / "paper_outputs" / "tables"
OUT = REPO_ROOT / "paper" / "data"

# model -> (x position, short label); x positions are shared by every per-model figure
MODELS = {"Qwen2.5-1.5B": (1, "Qwen-1.5B"), "Qwen2.5-3B": (2, "Qwen-3B"), "Qwen2.5-3B*": (2, "Qwen-3B"),
          "Qwen2.5-7B (4-bit)": (3, "Qwen-7B"), "Llama-3.2-3B": (4, "Llama-3B"), "SmolLM2-1.7B": (5, "SmolLM-1.7B")}
COMBOS = {"Obs. + grounding": "grd", "Obs. + all internals": "int", "Obs. + internals + grounding": "both"}
COMPOSITION = {"Observables": "obs", "All internals + grounding": "int", "Obs. + internals + grounding": "comb"}


def write(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / f"{name}.dat", sep=" ", index=False, float_format="%.3f")


def skip_frames():
    out = {}
    for dom in ("fsm", "cstr"):
        T = pd.read_csv(TAB / f"skip_{dom}_cert.csv")
        out[dom] = T[(T.tolerance - 0.10).abs() < 1e-9]
    return out


def delta(skip):
    """Change in validator calls skipped against observables, with paired-bootstrap CI (one file per domain)."""
    for dom, T in skip.items():
        rows = []
        for m, (x, _) in MODELS.items():
            r = {"x": x}
            for sig, key in COMBOS.items():
                s = T[(T.model == m) & (T.signal == sig)]
                if s.empty or pd.isna(s.iloc[0].delta_pts):
                    break
                s = s.iloc[0]
                r[f"{key}"] = s.delta_pts
                r[f"{key}_lo"] = s.delta_pts - s.delta_lo
                r[f"{key}_hi"] = s.delta_hi - s.delta_pts
            else:
                rows.append(r)
        write(f"delta_{dom}", pd.DataFrame(rows))


def composition(skip):
    """Unchecked accepts and unchecked rejects as a share of all proposals, per signal set."""
    for dom, T in skip.items():
        rows = []
        for m, (x, _) in MODELS.items():
            r = {"x": x}
            for sig, key in COMPOSITION.items():
                s = T[(T.model == m) & (T.signal == sig)]
                if s.empty:
                    break
                s = s.iloc[0]
                r[f"{key}_acc"] = 100 * s.accepted / s.n
                r[f"{key}_rej"] = 100 * s.rejected / s.n
                r[f"{key}_accfail"] = int(s.accepted_failures)
            else:
                rows.append(r)
        write(f"composition_{dom}", pd.DataFrame(rows))


def mechanism():
    M = pd.read_csv(TAB / "mechanism_grounding_auroc.csv")
    rows = []
    for m, (x, _) in MODELS.items():
        f = M[(M.domain == "FSM") & (M.model == m)]
        c = M[(M.domain == "CSTR") & M.model.isin([m, m + "*"])]
        if len(f) and len(c):
            rows.append({"x": x, "fsm": f.auroc_low_grounding_predicts_failure.iloc[0],
                         "cstr": c.auroc_low_grounding_predicts_failure.iloc[0]})
    write("mechanism", pd.DataFrame(rows))


def closed_loop():
    """Safety-savings plane (one row per model x policy) and the linear compute model per model."""
    C = pd.read_csv(TAB / "closed_loop_costs_all.csv")
    kind = {"Always validate": "av", "Never validate": "nv", "Random routing": "rand",
            "Observables probe": "obs", "Obs. + internals + grounding probe": "comb", "Internals-only probe": "int",
            "Observables probe, first-proposal-only": "obs_r0",
            "Obs. + internals + grounding, first-proposal-only": "comb_r0",
            "Internals-only, first-proposal-only": "int_r0"}
    C = C[C.policy.isin(kind)].copy()
    C["kind"] = C.policy.map(kind)
    C["x"] = C.model.map(lambda m: MODELS[m][0])
    C["unsafe_pct"] = 100 * C.unsafe_executions / C.episodes
    C["fixed_s"] = C.gen_s + C.internals_s + C.grounding_s
    for k, sub in C.groupby("kind"):
        write(f"plane_{k}", sub[["x", "calls_avoided_pct", "unsafe_pct", "unsafe_executions", "recovered_pct"]])
    # R0 arrows: main run -> first-proposal-only run, for the policies where the rule was run
    arrows = []
    for base in ("obs", "comb", "int"):
        for m, sub in C.groupby("model"):
            a, b = sub[sub.kind == base], sub[sub.kind == base + "_r0"]
            if len(a) and len(b) and a.unsafe_executions.iloc[0] != b.unsafe_executions.iloc[0]:
                arrows.append({"x0": a.calls_avoided_pct.iloc[0], "y0": a.unsafe_pct.iloc[0],
                               "x1": b.calls_avoided_pct.iloc[0], "y1": b.unsafe_pct.iloc[0]})
    write("plane_arrows", pd.DataFrame(arrows))
    # compute per episode C(c) = fixed + n_val * c, using the deployed (first-proposal-only) variant where it was run
    for m, sub in C.groupby("model"):
        x = MODELS[m][0]
        rows = {}
        for k in ("av", "obs", "comb", "int"):
            r0 = sub[sub.kind == k + "_r0"]
            r = r0 if len(r0) else sub[sub.kind == k]
            rows[k] = (r.fixed_s.iloc[0], r.validator_calls_per_ep.iloc[0], r.measured_validator_s_per_call.iloc[0])
        grid = [0, 10, 20, 30, 40, 50, 60]
        df = pd.DataFrame({"c": grid, **{k: [f + n * c for c in grid] for k, (f, n, _) in rows.items()}})
        write(f"cost_m{x}", df)
        be = []
        for k in ("obs", "comb", "int"):
            (fa, na, _), (fk, nk, _) = rows["av"], rows[k]
            if na > nk:
                c = (fk - fa) / (na - nk)
                if 0 <= c <= 60:
                    be.append({"c": c, "C": fa + na * c})
        write(f"breakeven_m{x}", pd.DataFrame(be, columns=["c", "C"]))
        top = 1.05 * df[["av", "obs", "comb", "int"]].to_numpy().max()  # vertical marker spans the panel
        write(f"measured_m{x}", pd.DataFrame({"c": [rows["av"][2]] * 2, "C": [0, top]}))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    skip = skip_frames()
    delta(skip)
    composition(skip)
    mechanism()
    closed_loop()
    print("figure data:", sorted(p.name for p in OUT.glob("*.dat")))


if __name__ == "__main__":
    main()
