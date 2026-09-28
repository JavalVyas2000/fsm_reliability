"""
Sequential (single-process) wall-time measurement of the CSTR legacy verifier for the
candidates of one partition, for the time accounting (scripts/14 --cstr_verifier_timing).

Labelling (scripts/23) runs many rollouts in parallel, which inflates per-call wall time
through CPU contention. Here each admissible first proposal of the partition is verified
once, one at a time, on an otherwise idle machine; the verdict must match the label.

Example:
    python -m scripts.24_cstr_verifier_timing --labeled_dir outputs/cstr_labeled/<run> --partition test_iid
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np

from src.cstr.episodes import verify
from src.utils.manifest import REPO_ROOT, build_manifest, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled_dir", type=str, required=True)
    p.add_argument("--partition", type=str, default="test_iid")
    return p.parse_args()


def main():
    args = parse_args()
    lab = (REPO_ROOT / args.labeled_dir).resolve()
    man = json.loads((lab / "run_manifest.json").read_text())
    ds = REPO_ROOT / man["data"]["dataset_dir"]
    snaps = {s.spec.episode_id: s for s in pickle.load(open(ds / f"snapshots_{args.partition}.pkl", "rb"))}
    recs = [json.loads(l) for l in open(lab / "records.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["partition"] == args.partition and r.get("label_status") == "known"
            and r.get("round", 0) == 0 and r.get("repeat_of_round") is None]
    times, cpu, mismatches = {}, {}, []
    t0 = time.time()
    for r in recs:
        snap = snaps[r.get("graph_hash", r["instance_id"]) if r.get("round") is not None else r["instance_id"]]
        res = verify(snap, r["action"])
        times[r["instance_id"]] = res["wall_s"]
        cpu[r["instance_id"]] = res["cpu_s"]
        if res["verifier_pass"] != r["verifier_pass"]:
            mismatches.append(r["instance_id"])
    out = lab / f"verifier_timing_sequential_{args.partition}.json"
    write_json(out, times)
    write_json(lab / f"verifier_cpu_sequential_{args.partition}.json", cpu)
    v = np.array(list(times.values()))
    write_json(lab / f"verifier_timing_sequential_{args.partition}_manifest.json", build_manifest(
        partition=args.partition, n=len(v), mean_s=float(v.mean()), median_s=float(np.median(v)),
        p95_s=float(np.percentile(v, 95)), min_s=float(v.min()), max_s=float(v.max()),
        cpu_mean_s=float(np.mean(list(cpu.values()))),
        verdict_mismatches_vs_labels=mismatches, total_wall_s=round(time.time() - t0, 1)))
    print(json.dumps({"out": str(out), "n": len(v), "mean_s": round(float(v.mean()), 2), "median_s": round(float(np.median(v)), 2),
                      "mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()
