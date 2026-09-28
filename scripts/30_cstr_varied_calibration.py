"""
No-LLM severity calibration for CSTR with a varied nominal plant and varied fault onset.

For each fault family and severity, several random plants are drawn (plant ranges and
onset range from the ranges file; T_sp fixed at 310 K). Each episode is run to its first
trigger and a small grid of actions RELATIVE TO THAT EPISODE'S OWN SETPOINTS is verified:
    no change, feed x0.9 (the typical LLM move observed), x0.8, x0.65, x0.5, x0.35,
    and feed x0.9 with the level setpoint lowered by 1.
Reported per severity: trigger rate, pre-fault false-alarm rate, no-change fail rate,
pass rate of the x0.9 move, and any-grid-pass rate. Severity ranges for the varied-plant
dataset are then chosen where no-change mostly fails, some correction passes, and the
typical x0.9 move passes only sometimes (outcome depends on the instance).

Example:
    python -m scripts.30_cstr_varied_calibration --ranges configs/cstr_severity_ranges_v3_draft.json --workers 16
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
    "fouling": [{"fouling_max": v} for v in (0.5, 0.55, 0.6, 0.65, 0.7, 0.8)],
    "pump_degrade": [{"pump_degrade_factor": v} for v in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25)],
    "cool_stuck_closed": [{"stuck_opening": v} for v in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25)],
    "outlet_block": [{"outlet_block_factor": v} for v in (0.6, 0.55, 0.5, 0.45, 0.4, 0.35)],
}
ACTIONS = [("no_change", 1.0, 0.0), ("feed_x0.9", 0.9, 0.0), ("feed_x0.8", 0.8, 0.0), ("feed_x0.65", 0.65, 0.0),
           ("feed_x0.5", 0.5, 0.0), ("feed_x0.35", 0.35, 0.0), ("feed_x0.9_L-1", 0.9, -1.0)]


def _task(args):
    from src.cstr.episodes import EpisodeSpec, current_setpoints, run_to_trigger, verify

    fam, params, plant, onset, seed = args
    spec = EpisodeSpec(f"cal_{fam}_{seed}", fam, params, onset, seed, plant=plant)
    s = run_to_trigger(spec)
    base = {"family": fam, "severity": json.dumps(params, sort_keys=True), "seed": seed, "onset_s": onset,
            **{f"plant_{k}": v for k, v in plant.items()}, "triggered": s.triggered,
            "pre_fault_trigger": s.pre_fault_trigger, "t_trigger": s.t_trigger}
    if not s.triggered or s.pre_fault_trigger:
        return base
    cur = current_setpoints(s)
    for name, fx, dl in ACTIONS:
        a = {"T_sp": cur["T_sp"], "L_sp": cur["L_sp"] + dl, "Fin_sp": cur["Fin_sp"] * fx}
        base[name] = verify(s, a)["verifier_pass"]
    return base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranges", type=str, required=True, help="JSON with 'plant' ranges and 'onset_s'")
    p.add_argument("--plants_per_severity", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260926)
    return p.parse_args()


def main():
    args = parse_args()
    ranges = json.loads((REPO_ROOT / args.ranges).read_text())
    rng = np.random.default_rng(args.seed)
    tasks = []
    for fam, grid in GRIDS.items():
        for params in grid:
            p = dict(params)
            if fam == "fouling":
                p["fouling_tau"] = 2000.0
            for _ in range(args.plants_per_severity):
                plant = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in sorted(ranges["plant"].items())}
                tasks.append((fam, p, plant, float(rng.uniform(*ranges["onset_s"])), int(rng.integers(0, 2**31 - 1))))
    run_dir = make_run_dir(REPO_ROOT / "outputs/cstr_calibration", "varied_plant")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_task, tasks)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "calibration_rows.csv", index=False)

    names = [n for n, _, _ in ACTIONS]
    summ = []
    for (fam, sev), g in df.groupby(["family", "severity"], sort=False):
        ok = g[g.triggered & ~g.pre_fault_trigger]
        row = {"family": fam, "severity": sev, "n": len(g), "triggered": int(g.triggered.sum()),
               "pre_fault_trigger": int(g.pre_fault_trigger.sum())}
        if len(ok):
            row.update(nochange_fail=float((ok["no_change"] == False).mean()),  # noqa: E712
                       x09_pass=float((ok["feed_x0.9"] == True).mean()),  # noqa: E712
                       any_pass=float(ok[names[1:]].eq(True).any(axis=1).mean()),
                       mean_pass_share=float(ok[names[1:]].eq(True).mean(axis=1).mean()))
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(run_dir / "calibration_summary.csv", index=False)
    lines = ["# CSTR varied-plant severity calibration (no LLM)", "",
             f"{args.plants_per_severity} random plants per severity (onset {ranges['onset_s']} s; plant ranges from {args.ranges}). "
             "Actions are relative to each episode's own setpoints.", "",
             "| family | severity | triggered/n | pre-fault alarms | no-change fail | x0.9 feed passes | any grid pass | mean pass share |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        f = lambda k: "–" if pd.isna(r.get(k)) else f"{r[k]:.2f}"
        lines.append(f"| {r.family} | {r.severity} | {r.triggered}/{r.n} | {r.pre_fault_trigger} | {f('nochange_fail')} | "
                     f"{f('x09_pass')} | {f('any_pass')} | {f('mean_pass_share')} |")
    (run_dir / "calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(ranges_file=args.ranges, ranges=ranges, grids=GRIDS,
                                                          actions=ACTIONS, plants_per_severity=args.plants_per_severity,
                                                          seed=args.seed, wall_s=round(time.time() - t0, 1)))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
