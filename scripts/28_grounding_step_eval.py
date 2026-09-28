"""
H1 of docs/grounding_attention_prereg.md: are invalid path steps less grounded (lower
attention share on the adjacency line of their source node) than valid steps?

Reports train/dev partitions by default; test_iid only with --include_test (to be used
once, after the dev results are written down).

Example:
    python -m scripts.28_grounding_step_eval --grounding_dirs outputs/fsm_grounding/<run1> outputs/fsm_grounding/<run2>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

TAGS = ["L025", "L050", "L075", "L100"]


def grouped_boot_auroc(y, s, g, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grounding_dirs", nargs="+", required=True)
    p.add_argument("--include_test", action="store_true")
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    parts = ["train", "dev_cal", "dev_thr"] + (["test_iid"] if args.include_test else [])
    run_dir = make_run_dir(REPO_ROOT / "outputs/fsm_grounding_eval", "step_h1" + ("_with_test" if args.include_test else "_dev"))
    results, lines = {}, ["# H1: step-level grounding (share of graph attention on the source node's adjacency line)", "",
                          f"Partitions: {', '.join(parts)}. Score = −share (higher = more likely invalid). "
                          "AUROC with 95% bootstrap CI grouped by instance.", "",
                          "| Model | Partition set | Steps (invalid) | Mean share valid / invalid | AUROC −share (layer mean) [CI] | Best single layer | AUROC −share_hmax | top-1 rate valid / invalid |",
                          "|---|---|---|---|---|---|---|---|"]
    for gd in args.grounding_dirs:
        gdir = REPO_ROOT / gd
        man = json.loads((gdir / "run_manifest.json").read_text())
        inf = REPO_ROOT / man["inference_dir"]
        part_of = {json.loads(l)["instance_id"]: json.loads(l)["partition"] for l in open(inf / "records.jsonl", encoding="utf-8")}
        y, share, hmax, top1, grp, per_layer = [], [], [], [], [], {t: [] for t in TAGS}
        for line in open(gdir / "grounding.jsonl", encoding="utf-8"):
            g = json.loads(line)
            if g["status"] != "ok" or part_of[g["instance_id"]] not in parts:
                continue
            for s in g["steps"]:
                if not s["u_in_graph"]:
                    continue
                y.append(1 - s["valid"])
                share.append(np.nanmean([s[f"{t}_share"] for t in TAGS]))
                hmax.append(np.nanmean([s[f"{t}_share_hmax"] for t in TAGS]))
                top1.append(np.nanmean([s[f"{t}_top1"] for t in TAGS]))
                for t in TAGS:
                    per_layer[t].append(s[f"{t}_share"])
                grp.append(g["instance_id"])
        y, share, hmax, top1, grp = map(np.array, (y, share, hmax, top1, grp))
        auc = roc_auc_score(y, -share)
        lo, hi = grouped_boot_auroc(y, -share, grp, args.n_boot)
        layer_auc = {t: roc_auc_score(y, -np.array(per_layer[t])) for t in TAGS}
        best = max(layer_auc, key=layer_auc.get)
        res = {"model": man["model"], "n_steps": int(len(y)), "n_invalid": int(y.sum()),
               "mean_share_valid": float(share[y == 0].mean()), "mean_share_invalid": float(share[y == 1].mean()),
               "auroc_neg_share": float(auc), "ci95": [lo, hi], "auroc_by_layer": layer_auc,
               "auroc_neg_share_hmax": float(roc_auc_score(y, -hmax)),
               "top1_rate_valid": float(top1[y == 0].mean()), "top1_rate_invalid": float(top1[y == 1].mean())}
        results[gd] = res
        lines.append(f"| {man['model']} | {'+'.join(parts)} | {len(y)} ({int(y.sum())}) | {res['mean_share_valid']:.3f} / "
                     f"{res['mean_share_invalid']:.3f} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {best} {layer_auc[best]:.3f} | "
                     f"{res['auroc_neg_share_hmax']:.3f} | {res['top1_rate_valid']:.2f} / {res['top1_rate_invalid']:.2f} |")
    write_json(run_dir / "h1_results.json", results)
    (run_dir / "h1_results.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "run_config.json", build_manifest(grounding_dirs=args.grounding_dirs, partitions=parts,
                                                          prereg="docs/grounding_attention_prereg.md"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
