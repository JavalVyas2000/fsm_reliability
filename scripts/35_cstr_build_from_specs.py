"""
Simulate the episodes of a spec spreadsheet (scripts/33) to their first monitor trigger and
save the snapshots used by scripts/25_cstr_collect.

An episode is unusable when the monitor never triggers before SHUTDOWN, or triggers before the
fault onset (a false alarm of the nominal plant). Its slot is redrawn with the SAME fault family
(so the family balance and split sizes are kept) from the same ranges file, with a seed derived
from (replace_seed, split, slot, attempt), until the slot is usable. Every dropped draw is kept
in the `replacements` sheet.

Outputs (<out_dir>/):
    cstr_episodes.xlsx         final episodes per split (+ t_trigger, attempts) and replacements
    snapshots_<split>.pkl      Snapshot objects, in spreadsheet order
    kg_context.ttl             KG context used by every prompt
    dataset_manifest.json      provenance (read by scripts/25)
A split whose snapshots file already exists is skipped, so an interrupted build can be resumed.

Example:
    python -m scripts.35_cstr_build_from_specs --specs data/cstr/cstr_episode_specs_v4.xlsx \
        --ranges configs/cstr_severity_ranges_v4.json --out_dir data/cstr/episodes_v4 --workers 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import time

import numpy as np
import pandas as pd

from src.cstr.specs import draw_episode, row_to_spec
from src.utils.manifest import REPO_ROOT, build_manifest, sha256_file, write_json

MAX_ATTEMPTS = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", type=str, required=True)
    p.add_argument("--ranges", type=str, required=True, help="ranges file the specs were drawn from (for redraws)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--kg_file", type=str, default="data/v2/cstr_varied_seed20260928/kg_context.ttl")
    p.add_argument("--replace_seed", type=int, default=20261001)
    p.add_argument("--workers", type=int, default=20)
    return p.parse_args()


def _run(row):
    from src.cstr.episodes import run_to_trigger

    s = run_to_trigger(row_to_spec(row))
    return s, bool(s.triggered and not s.pre_fault_trigger)


def main():
    args = parse_args()
    ranges = json.loads((REPO_ROOT / args.ranges).read_text())
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    kg = (REPO_ROOT / args.kg_file).read_text(encoding="utf-8")
    (out / "kg_context.ttl").write_bytes(kg.encode("utf-8"))
    sheets = pd.read_excel(REPO_ROOT / args.specs, sheet_name=None)
    used_seeds = set(int(x) for s in args.splits for x in sheets[s]["noise_seed"])

    t0 = time.time()
    finals, dropped = {}, []
    state_path = out / "build_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for si, split in enumerate(args.splits):
            if (out / f"snapshots_{split}.pkl").exists() and split in state:
                finals[split] = pd.DataFrame(state[split]["rows"])
                dropped += state[split]["dropped"]
                print(f"{split}: already built, skipped", flush=True)
                continue
            rows = sheets[split].to_dict("records")
            for r in rows:
                r["attempts"] = 1
            snaps = [None] * len(rows)
            todo = list(range(len(rows)))
            split_dropped = []
            while todo:
                results = pool.map(_run, [rows[i] for i in todo], chunksize=4)
                again = []
                for i, (s, ok) in zip(todo, results):
                    if ok:
                        snaps[i] = s
                        rows[i]["t_trigger"] = s.t_trigger
                        continue
                    split_dropped.append({**{k: v for k, v in rows[i].items() if k != "t_trigger"},
                                          "why": "pre_fault_trigger" if s.triggered else "no_trigger"})
                    if rows[i]["attempts"] >= MAX_ATTEMPTS:
                        raise RuntimeError(f"{rows[i]['episode_id']}: no usable draw in {MAX_ATTEMPTS} attempts")
                    rng = np.random.default_rng(np.random.SeedSequence([args.replace_seed, si, i, rows[i]["attempts"]]))
                    new = draw_episode(rng, ranges, rows[i]["family"])
                    seed = int(rng.integers(0, 2**31 - 1))
                    while seed in used_seeds:
                        seed = int(rng.integers(0, 2**31 - 1))
                    used_seeds.add(seed)
                    rows[i] = {**rows[i], **new, "noise_seed": seed, "attempts": rows[i]["attempts"] + 1}
                    again.append(i)
                print(f"{split}: {len(todo) - len(again)}/{len(todo)} usable this pass, {len(again)} redrawn, "
                      f"{time.time() - t0:.0f} s", flush=True)
                todo = again
            with open(out / f"snapshots_{split}.pkl", "wb") as f:
                pickle.dump(snaps, f)
            finals[split] = pd.DataFrame(rows)
            dropped += split_dropped
            state[split] = {"rows": rows, "dropped": split_dropped}
            state_path.write_text(json.dumps(state, default=float))

    with pd.ExcelWriter(out / "cstr_episodes.xlsx", engine="openpyxl") as xw:
        for split, df in finals.items():
            df.to_excel(xw, sheet_name=split, index=False)
        pd.DataFrame(dropped).to_excel(xw, sheet_name="replacements", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
    allr = pd.concat(finals.values(), ignore_index=True)
    drop = pd.DataFrame(dropped)
    files = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.suffix in (".pkl", ".ttl", ".xlsx")}
    write_json(out / "dataset_manifest.json", build_manifest(dataset={
        "type": "cstr_snapshots_from_specs", "root_seed": args.replace_seed, "specs_file": args.specs,
        "specs_sha256": sha256_file(REPO_ROOT / args.specs), "ranges_file": args.ranges, "ranges": ranges,
        "sizes": {s: len(d) for s, d in finals.items()},
        "kg_file": args.kg_file, "kg_sha256": hashlib.sha256(kg.encode("utf-8")).hexdigest(),
        "redraws": int(len(drop)), "redraws_by_family": drop["family"].value_counts().to_dict() if len(drop) else {},
        "redraw_reasons": drop["why"].value_counts().to_dict() if len(drop) else {},
        "unusable_rate_first_draw_by_family": (
            allr.groupby("family")["attempts"].apply(lambda a: float((a > 1).mean())).round(3).to_dict()),
        "checks": {"episode_ids_unique": bool(allr["episode_id"].is_unique),
                   "noise_seeds_unique": bool(allr["noise_seed"].is_unique),
                   "family_counts": allr.groupby(["split", "family"]).size().unstack().to_dict()},
        "files_sha256": files, "wall_s": round(time.time() - t0, 1),
    }))
    print(json.dumps({"out_dir": str(out), "sizes": {s: len(d) for s, d in finals.items()}, "redraws": len(drop)}))


if __name__ == "__main__":
    main()
