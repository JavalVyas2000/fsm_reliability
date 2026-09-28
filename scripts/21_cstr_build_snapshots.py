"""
Build a CSTR snapshot dataset: sample episode specs from frozen severity ranges, run each
to its first action trigger (CPU, parallel) and capture the exact first-proposal prompt.

Per episode, independently drawn: fault family, continuous severity, onset time,
fouling time constant (fouling only) and simulator noise seed. Episodes are the
independence unit; partitions are disjoint by construction. Episodes whose monitor never
triggers before the SHUTDOWN phase are recorded and excluded from the population.

Outputs (data/v2/<tag>_seed<root>/):
    specs.csv                  every drawn spec, triggered flag, trigger time
    snapshots_<partition>.pkl  Snapshot objects (triggered episodes only)
    prompts_<partition>.jsonl  messages + prompt hashes
    kg_context.ttl             frozen KG context used in every prompt
    diversity_report.json      prompt / snapshot diversity checks
    dataset_manifest.json

Example:
    python -m scripts.21_cstr_build_snapshots --ranges configs/cstr_severity_ranges_v1.json \
        --root_seed 20260925 --tag cstr_pilot --sizes train=150 dev_cal=50 dev_thr=50 test_iid=50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.utils.manifest import REPO_ROOT, build_manifest, sha256_file, write_json

PARTITIONS = ("train", "dev_cal", "dev_thr", "cert", "test_iid")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranges", type=str, required=True)
    p.add_argument("--root_seed", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--sizes", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--kg_source", choices=["graphdb", "file"], default="graphdb")
    p.add_argument("--kg_file", type=str, default=None)
    p.add_argument("--oversample", type=float, default=1.4)
    return p.parse_args()


def draw_spec(rng: np.random.Generator, ranges: dict, partition: str, idx: int):
    from src.cstr.episodes import EpisodeSpec

    fams = sorted(ranges["families"])
    fam = fams[int(rng.integers(len(fams)))]
    fr = ranges["families"][fam]
    params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in sorted(fr["params"].items())}
    onset = float(rng.uniform(*ranges["onset_s"]))
    seed = int(rng.integers(0, 2**31 - 1))
    # Optional varied nominal plant (ranges v3+); drawn after the v2 fields so v2 datasets
    # regenerate identically.
    plant = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in sorted(ranges.get("plant", {}).items())}
    return EpisodeSpec(f"cstr_{partition}_{idx:05d}", fam, params, onset, seed, plant=plant)


def _run(spec):
    from src.cstr.episodes import run_to_trigger

    return run_to_trigger(spec)


def main():
    args = parse_args()
    from src.cstr.car_bridge import car_provenance
    from src.cstr.episodes import build_messages, fetch_kg_context_live, set_kg_context, spec_to_dict

    ranges = json.loads((REPO_ROOT / args.ranges).read_text())
    sizes = {k: int(v) for k, v in (s.split("=") for s in args.sizes)}
    out_dir = REPO_ROOT / "data/v2" / f"{args.tag}_seed{args.root_seed}"
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    kg = fetch_kg_context_live() if args.kg_source == "graphdb" else (REPO_ROOT / args.kg_file).read_text(encoding="utf-8")
    (out_dir / "kg_context.ttl").write_bytes(kg.encode("utf-8"))
    set_kg_context(kg)

    children = dict(zip(PARTITIONS, np.random.SeedSequence(args.root_seed).spawn(len(PARTITIONS))))
    all_rows, report, t0 = [], {}, time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for part in PARTITIONS:
            need = sizes.get(part, 0)
            if need <= 0:
                continue
            rng = np.random.default_rng(children[part])
            kept, idx = [], 0
            while len(kept) < need:
                batch = [draw_spec(rng, ranges, part, idx + i) for i in range(max(8, int((need - len(kept)) * args.oversample)))]
                idx += len(batch)
                snaps = pool.map(_run, batch)  # order preserved -> deterministic selection
                for s in snaps:
                    usable = s.triggered and not getattr(s, "pre_fault_trigger", False)
                    all_rows.append({**spec_to_dict(s.spec), "partition": part, "triggered": s.triggered,
                                     "pre_fault_trigger": getattr(s, "pre_fault_trigger", False),
                                     "t_trigger": s.t_trigger, "sim_wall_s": s.wall_s,
                                     "selected": usable and len(kept) < need})
                    if usable and len(kept) < need:
                        kept.append(s)
            prompts = []
            for s in kept:
                msgs = build_messages(s)
                prompts.append({"episode_id": s.spec.episode_id, "messages": msgs,
                                "system_sha256": hashlib.sha256(msgs[0]["content"].encode()).hexdigest(),
                                "user_sha256": hashlib.sha256(msgs[1]["content"].encode()).hexdigest()})
            with open(out_dir / f"snapshots_{part}.pkl", "wb") as f:
                pickle.dump(kept, f)
            with open(out_dir / f"prompts_{part}.jsonl", "w", encoding="utf-8") as f:
                for p in prompts:
                    f.write(json.dumps(p) + "\n")
            fam_counts = Counter(s.spec.family for s in kept)
            report[part] = {
                "n": len(kept),
                "families": dict(fam_counts),
                "distinct_system_prompts": len({p["system_sha256"] for p in prompts}),
                "distinct_user_prompts": len({p["user_sha256"] for p in prompts}),
                "distinct_trigger_times": len({s.t_trigger for s in kept}),
                "distinct_snapshot_tuples": len({(round(s.state["sim_last"]["T_meas"], 6), round(s.state["sim_last"]["L_meas"], 6),
                                                  round(s.state["sim_last"]["u_cool"], 6)) for s in kept}),
                "trigger_reasons": dict(Counter(s.state.get("violated_params") for s in kept).most_common(8)),
            }
    specs = pd.DataFrame(all_rows)
    specs.to_csv(out_dir / "specs.csv", index=False)
    write_json(out_dir / "diversity_report.json", report)
    files = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.suffix in (".csv", ".pkl", ".jsonl", ".ttl")}
    write_json(out_dir / "dataset_manifest.json", build_manifest(dataset={
        "type": "cstr_snapshots", "root_seed": args.root_seed, "sizes": sizes, "ranges_file": args.ranges, "ranges": ranges,
        "kg_source": args.kg_source, "kg_sha256": hashlib.sha256(kg.encode("utf-8")).hexdigest(),
        "n_drawn": len(specs), "n_triggered": int(specs["triggered"].sum()),
        "trigger_rate_by_family": specs.groupby("family")["triggered"].mean().round(3).to_dict(),
        "files_sha256": files, "car": car_provenance(), "wall_s": round(time.time() - t0, 1),
        "checks": {"episodes_disjoint": bool(specs["episode_id"].is_unique)},
    }))
    print(json.dumps({"out_dir": str(out_dir), "diversity": report}, indent=2))


if __name__ == "__main__":
    main()
