"""
No-LLM severity calibration against the model's own empirical moves (prompt v3.1).

Goal (docs/cstr_v4_ranges.md): choose severity ranges where the model's first proposals pass
often enough to train and certify an ACCEPT decision (target 30-50%), while doing nothing
still mostly fails. The model's pass rate is a property of the dataset design here, not a goal
for the model: the prompt is not changed.

Moves: the first proposals of a completed v3.1 run, each expressed relative to its episode's
own setpoints at the trigger (dT_sp and dL_sp absolute, Fin_sp as a ratio). For every
calibration episode (random plant, onset and the grid severity) the episode is run to its first
trigger, then verified with:
    no change; a fixed random subset of the empirical moves; feed x0.65 and x0.5 (fixability check).
The marginal move distribution ignores how the model's move depends on the snapshot, so the
chosen ranges are then checked with the real model on fresh episodes.

Example:
    python -m scripts.34_cstr_move_calibration --moves_run outputs/cstr_collect/<v3.1 run> --workers 20
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
    "fouling": [{"fouling_max": v, "fouling_tau": 2000.0} for v in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)],
    "pump_degrade": [{"pump_degrade_factor": v} for v in (0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35)],
    "cool_stuck_closed": [{"stuck_opening": v} for v in (0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3)],
    "outlet_block": [{"outlet_block_factor": v} for v in (0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35)],
}
FIX_CHECKS = {"feed_x0.65": 0.65, "feed_x0.5": 0.5}


def empirical_moves(run_dir):
    moves = []
    for line in open(run_dir / "records.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["round"] != 0 or r.get("schema_valid") != 1 or not r.get("action"):
            continue
        a, ca = r["action"], r["context_action"]
        moves.append({"source": r["instance_id"], "dT_sp": a["T_sp"] - 310.0,
                      "dL_sp": a["L_sp"] - ca["L_sp_current"], "Fin_ratio": a["Fin_sp"] / ca["Fin_sp_current"]})
    return moves


def _task(args):
    from src.cstr.episodes import ACTION_BOUNDS, EpisodeSpec, current_setpoints, run_to_trigger, verify

    fam, params, plant, onset, seed, moves = args
    s = run_to_trigger(EpisodeSpec(f"cal_{fam}_{seed}", fam, params, onset, seed, plant=plant))
    row = {"family": fam, "severity": json.dumps(params, sort_keys=True), "seed": seed, "onset_s": onset,
           **{f"plant_{k}": v for k, v in plant.items()}, "triggered": s.triggered,
           "pre_fault_trigger": s.pre_fault_trigger, "t_trigger": s.t_trigger}
    if not s.triggered or s.pre_fault_trigger:
        return row
    cur = current_setpoints(s)
    clip = lambda k, v: float(min(max(v, ACTION_BOUNDS[k][0]), ACTION_BOUNDS[k][1]))  # noqa: E731
    row["no_change"] = verify(s, cur)["verifier_pass"]
    for name, fx in FIX_CHECKS.items():
        row[name] = verify(s, {**cur, "Fin_sp": cur["Fin_sp"] * fx})["verifier_pass"]
    passes = []
    for m in moves:
        a = {"T_sp": clip("T_sp", cur["T_sp"] + m["dT_sp"]), "L_sp": clip("L_sp", cur["L_sp"] + m["dL_sp"]),
             "Fin_sp": clip("Fin_sp", cur["Fin_sp"] * m["Fin_ratio"])}
        passes.append(bool(verify(s, a)["verifier_pass"]))
    row["move_pass_share"] = float(np.mean(passes))
    row["move_passes"] = json.dumps(passes)
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moves_run", type=str, required=True, help="completed collection run with v3.1 first proposals")
    p.add_argument("--ranges", type=str, default="configs/cstr_severity_ranges_v3.json", help="plant and onset ranges")
    p.add_argument("--plants_per_severity", type=int, default=10)
    p.add_argument("--moves_per_episode", type=int, default=20)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260930)
    return p.parse_args()


def main():
    args = parse_args()
    ranges = json.loads((REPO_ROOT / args.ranges).read_text())
    moves_all = empirical_moves(REPO_ROOT / args.moves_run)
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(moves_all), size=min(args.moves_per_episode, len(moves_all)), replace=False))
    moves = [moves_all[i] for i in pick]
    tasks = []
    for fam, grid in GRIDS.items():
        for params in grid:
            for _ in range(args.plants_per_severity):
                plant = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in sorted(ranges["plant"].items())}
                tasks.append((fam, dict(params), plant, float(rng.uniform(*ranges["onset_s"])),
                              int(rng.integers(0, 2**31 - 1)), moves))
    run_dir = make_run_dir(REPO_ROOT / "outputs/cstr_calibration", "v31_moves")
    pd.DataFrame(moves_all).to_csv(run_dir / "empirical_moves_all.csv", index=False)
    pd.DataFrame(moves).to_csv(run_dir / "empirical_moves_used.csv", index=False)
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(tasks)} episodes, {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows).sort_values(["family", "severity", "seed"])
    df.to_csv(run_dir / "calibration_rows.csv", index=False)

    summ = []
    for (fam, sev), g in df.groupby(["family", "severity"], sort=False):
        ok = g[g.triggered & ~g.pre_fault_trigger]
        row = {"family": fam, "severity": sev, "n": len(g), "triggered": int(g.triggered.sum()),
               "pre_fault_trigger": int(g.pre_fault_trigger.sum()), "usable": len(ok)}
        if len(ok):
            nc = ok["no_change"].eq(True)
            row.update(nochange_pass=float(nc.mean()), move_pass=float(ok["move_pass_share"].mean()),
                       move_pass_when_nochange_fails=float(ok.loc[~nc, "move_pass_share"].mean()) if (~nc).any() else np.nan,
                       fix_x065=float(ok["feed_x0.65"].eq(True).mean()), fix_x05=float(ok["feed_x0.5"].eq(True).mean()))
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(run_dir / "calibration_summary.csv", index=False)
    f = lambda v: "–" if pd.isna(v) else f"{v:.2f}"  # noqa: E731
    lines = ["# CSTR severity calibration against the model's empirical v3.1 moves (no LLM)", "",
             f"{args.plants_per_severity} random plants per severity; {len(moves)} empirical moves per episode "
             f"(from {args.moves_run}); plant and onset ranges from {args.ranges}.", "",
             "| family | severity | usable/n | no-change passes | v3.1-like move passes | ... when no-change fails | feed x0.65 passes | feed x0.5 passes |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        lines.append(f"| {r.family} | {r.severity} | {r.usable}/{r.n} | {f(r.get('nochange_pass'))} | {f(r.get('move_pass'))} | "
                     f"{f(r.get('move_pass_when_nochange_fails'))} | {f(r.get('fix_x065'))} | {f(r.get('fix_x05'))} |")
    (run_dir / "calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(
        moves_run=args.moves_run, ranges_file=args.ranges, grids=GRIDS, fix_checks=FIX_CHECKS,
        plants_per_severity=args.plants_per_severity, moves_per_episode=len(moves), seed=args.seed,
        wall_s=round(time.time() - t0, 1)))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
