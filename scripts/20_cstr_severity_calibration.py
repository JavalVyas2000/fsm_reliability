"""
No-LLM severity calibration for CSTR fault families (protocol v1.0, section 5.3).

For a grid of severities per family (several noise seeds each), run the real monitoring
loop to the first trigger and roll out a fixed grid of setpoint actions with the
deterministic legacy verifier. A severity is "actionable" when
    (a) the no-change action fails for most seeds, and
    (b) at least one grid action passes for most seeds,
i.e. recovery is possible, but only with a real setpoint change. Passing grid actions are
witnesses of recoverability under the declared conditions only; a failed search does not
prove unrecoverability.

Example:
    python -m scripts.20_cstr_severity_calibration --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

GRIDS = {
    "fouling": [{"fouling_max": v, "fouling_tau": 2000.0} for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
    "pump_degrade": [{"pump_degrade_factor": v} for v in (0.8, 0.65, 0.5, 0.35, 0.2, 0.1)],
    "cool_stuck_closed": [{"stuck_opening": v} for v in (0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)],
    "outlet_block": [{"outlet_block_factor": v} for v in (0.8, 0.6, 0.45, 0.3, 0.2, 0.1)],
    "leak": [{"leak_k": v} for v in (0.001, 0.002, 0.003, 0.005, 0.008)],
}
SEEDS = (101, 202, 303)
ONSET = 2000.0
NOM_FIN = 2.0 / 60.0
ACTIONS = (
    [{"T_sp": T, "L_sp": 10.0, "Fin_sp": round(f, 4)} for T in (310.0, 309.0)
     for f in (NOM_FIN, 0.0283, 0.0233, 0.0183, 0.0133, 0.0083)]
    + [{"T_sp": 310.0, "L_sp": L, "Fin_sp": round(f, 4)} for L in (9.0, 8.0) for f in (NOM_FIN, 0.0233)]
)


def _task(args):
    from src.cstr.episodes import EpisodeSpec, run_to_trigger, verify

    family, params, seed = args
    spec = EpisodeSpec(f"cal_{family}_{json.dumps(params, sort_keys=True)}_{seed}", family, params, ONSET, seed)
    snap = run_to_trigger(spec)
    base = {"family": family, "severity": json.dumps(params, sort_keys=True), "seed": seed,
            "triggered": snap.triggered, "t_trigger": snap.t_trigger,
            "trigger_violations": snap.state.get("violated_params") if snap.triggered else None,
            "control_zone": snap.state.get("control_zone") if snap.triggered else None}
    if not snap.triggered:
        return [dict(base, action=None)]
    rows = []
    for a in ACTIONS:
        r = verify(snap, a)
        is_nochange = a["T_sp"] == 310.0 and a["L_sp"] == 10.0 and abs(a["Fin_sp"] - round(NOM_FIN, 4)) < 1e-9
        rows.append(dict(base, action=json.dumps(a), no_change=is_nochange, label_status=r["label_status"],
                         verifier_pass=r["verifier_pass"], fail_reason=r["fail_reason"],
                         time_to_safe=r["metrics"].get("time_to_safe"), unsafe_fraction=r["metrics"].get("unsafe_fraction"),
                         n_steps=r["n_steps"], wall_s=r["wall_s"]))
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out_root", type=str, default="outputs/cstr_calibration")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = make_run_dir(REPO_ROOT / args.out_root, "severity_grid")
    tasks = [(f, p, s) for f, grid in GRIDS.items() for p in grid for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(_task, tasks) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "calibration_rows.csv", index=False)

    summ = []
    for (fam, sev), g in df.groupby(["family", "severity"], sort=False):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            if not gs["triggered"].iloc[0]:
                per_seed.append({"triggered": False})
                continue
            nc = gs[gs["no_change"] == True]  # noqa: E712
            acts = gs[gs["no_change"] == False]  # noqa: E712
            per_seed.append({
                "triggered": True,
                "nochange_fail": bool((nc["verifier_pass"] == False).all()),  # noqa: E712
                "any_pass": bool((acts["verifier_pass"] == True).any()),  # noqa: E712
                "pass_frac": float((acts["verifier_pass"] == True).mean()),  # noqa: E712
            })
        trig = [x for x in per_seed if x["triggered"]]
        n = len(per_seed)
        row = {"family": fam, "severity": sev, "seeds": n, "triggered": len(trig)}
        if trig:
            row.update(
                nochange_fail_rate=np.mean([x["nochange_fail"] for x in trig]),
                any_pass_rate=np.mean([x["any_pass"] for x in trig]),
                mean_pass_frac_of_grid=np.mean([x["pass_frac"] for x in trig]),
            )
            row["actionable"] = bool(len(trig) >= 2 and row["nochange_fail_rate"] > 0.5 and row["any_pass_rate"] > 0.5)
        else:
            row["actionable"] = False
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(run_dir / "calibration_summary.csv", index=False)

    lines = ["# CSTR severity calibration (no LLM; legacy verifier, deterministic replay)", "",
             f"Seeds per severity: {len(SEEDS)}; onset {ONSET} s; {len(ACTIONS)} grid actions (incl. no-change). "
             "Actionable = no-change fails for > 50% of triggered seeds AND some grid action passes for > 50%.", "",
             "| family | severity | triggered/seeds | no-change fail rate | any grid pass rate | mean pass share of grid | actionable |",
             "|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        f = lambda k: "–" if pd.isna(r.get(k)) else f"{r[k]:.2f}"
        lines.append(f"| {r.family} | {r.severity} | {r.triggered}/{r.seeds} | {f('nochange_fail_rate')} | {f('any_pass_rate')} | "
                     f"{f('mean_pass_frac_of_grid')} | {'yes' if r.actionable else 'no'} |")
    (run_dir / "calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(grids=GRIDS, seeds=SEEDS, onset=ONSET, actions=ACTIONS,
                                                          workers=args.workers, wall_s=round(time.time() - t0, 1)))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
