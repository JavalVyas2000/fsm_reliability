"""
Label CSTR first proposals with the deterministic legacy verifier (CPU, parallel).

Every schema-valid (admissible) proposal is verified, whatever a routing policy would do.
Per candidate the verifier wall time is recorded; that is C_v in the time accounting.
The no-change verdict of the same snapshot is stored for offline analysis only (it is
privileged information and never a feature).

Writes a new directory (outputs/cstr_labeled/<run>/) with the merged records.jsonl
(candidate_invalid = verifier FAIL), a copy of the hidden shards and a manifest, so the
shared fitting/routing scripts can consume it unchanged.

Example:
    python -m scripts.23_cstr_label --inference_dir outputs/cstr_inference/<run> --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
import time

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

NO_CHANGE = {"T_sp": 310.0, "L_sp": 10.0, "Fin_sp": 2.0 / 60.0}


def _verify(args):
    from src.cstr.episodes import verify

    snap, action = args
    r = verify(snap, action)
    nc = verify(snap, NO_CHANGE)
    return snap.spec.episode_id, r, nc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_dir", type=str, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out_root", type=str, default="outputs/cstr_labeled")
    return p.parse_args()


def main():
    args = parse_args()
    inf = (REPO_ROOT / args.inference_dir).resolve()
    man = json.loads((inf / "run_manifest.json").read_text())
    ds = REPO_ROOT / man["data"]["dataset_dir"]
    recs = [json.loads(l) for l in open(inf / "records.jsonl", encoding="utf-8") if l.strip()]
    snaps = {}
    for part in {r["partition"] for r in recs}:
        for s in pickle.load(open(ds / f"snapshots_{part}.pkl", "rb")):
            snaps[s.spec.episode_id] = s
    tasks = [(snaps[r["instance_id"]], r["action"]) for r in recs if r.get("schema_valid") == 1]
    run_dir = make_run_dir(REPO_ROOT / args.out_root, inf.name)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = {eid: (r, nc) for eid, r, nc in pool.imap_unordered(_verify, tasks)}

    n_known = 0
    with open(run_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            res = results.get(rec["instance_id"])
            if res is None:
                rec.update(label_status="not_verified_format_failure", verifier_pass=None, candidate_invalid=None,
                           verifier_wall_s=None, verifier_steps=None)
            else:
                r, nc = res
                known = r["label_status"] == "known"
                n_known += known
                rec.update(
                    label_status=r["label_status"], verifier_pass=r["verifier_pass"],
                    candidate_invalid=(None if not known else int(not r["verifier_pass"])),
                    verifier_fail_reason=r["fail_reason"], verifier_metrics=r["metrics"],
                    verifier_wall_s=r["wall_s"], verifier_steps=r["n_steps"],
                    offline_nochange_pass=nc["verifier_pass"],
                )
            f.write(json.dumps(rec) + "\n")
    shutil.copytree(inf / "hidden", run_dir / "hidden")
    shutil.copy(inf / "run_manifest.json", run_dir / "inference_manifest.json")
    man_out = dict(man)
    man_out["labeling"] = {"inference_dir": str(inf.relative_to(REPO_ROOT)), "verifier": "legacy rollout_validate_setpoints, deterministic replay",
                           "n_records": len(recs), "n_verified": len(results), "n_known": n_known,
                           "workers": args.workers, "wall_s": round(time.time() - t0, 1)}
    write_json(run_dir / "run_manifest.json", man_out)
    write_json(run_dir / "label_manifest.json", build_manifest(**man_out["labeling"]))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
