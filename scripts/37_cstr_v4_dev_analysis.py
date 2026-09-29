"""
Dev-only analyses of docs/cstr_v4_prereg.md that scripts/12 does not produce:
    H2  paired dAUROC vs context_action within each no-change stratum (dev_thr)
    H3  AUROC of -g31_Fin_physics_share for "against the physics" proposals (train + dev)
test_iid and cert are not read.

Example:
    python -m scripts.37_cstr_v4_dev_analysis --probe_dir outputs/fsm_baseline/<run> --aug_dir outputs/cstr_grounding/<run>/aug
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics_v2 import bootstrap_delta_auroc
from src.utils.manifest import REPO_ROOT, build_manifest, write_json

OPEN = ("train", "dev_cal", "dev_thr")
COMPARE = ["context_action+all_internal+grounding_v31", "context_action+token_confidence", "context_action+hidden",
           "context_action+grounding_v31", "context_action+all_internal"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe_dir", type=str, required=True)
    p.add_argument("--aug_dir", type=str, required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    probe, aug = REPO_ROOT / args.probe_dir, REPO_ROOT / args.aug_dir
    recs = []
    for line in open(aug / "records.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["partition"] not in OPEN or r["round"] != 0 or r.get("schema_valid") != 1:
            continue
        a, ca, f = r["action"], r["context_action"], r["features"]
        recs.append({"instance_id": r["instance_id"], "partition": r["partition"], "fail": int(not r["verifier_pass"]),
                     "nochange_pass": r.get("offline_nochange_pass"),
                     "against": bool(a["Fin_sp"] > ca["Fin_sp_current"] * (1 + 1e-9) and ca["u_cool"] >= 0.95 and ca["T_meas"] > 310.0),
                     "g31_Fin_physics_share": f.get("g31_Fin_physics_share")})
    R = pd.DataFrame(recs)
    preds = pd.read_csv(probe / "predictions.csv")
    assert not preds["partition"].isin(["test_iid", "cert"]).any(), "probe run is not sealed"
    out = {"H2": {}, "H3": {}}

    d = preds[preds["partition"] == "dev_thr"].merge(R[["instance_id", "nochange_pass"]], on="instance_id")
    for stratum, g in d.groupby("nochange_pass"):
        key = "nochange_passes" if stratum else "nochange_fails"
        y, grp = g["y"].to_numpy(), g["graph_hash"].to_numpy()
        base = g["p_context_action_raw"].to_numpy()
        out["H2"][key] = {"n": int(len(g)), "failure_rate": float(y.mean()),
                          "auroc_context_action": float(roc_auc_score(y, base))}
        for name in COMPARE:
            out["H2"][key][f"{name} - context_action"] = bootstrap_delta_auroc(
                y, g[f"p_{name}_raw"].to_numpy(), base, grp, args.n_boot, 0)

    h = R.dropna(subset=["g31_Fin_physics_share"])
    for label, sub in (("all proposals (pre-registered)", h), ("failing proposals only (sensitivity)", h[h.fail == 1])):
        out["H3"][label] = {
            "n": int(len(sub)), "n_against": int(sub.against.sum()),
            "mean_share_against": float(sub.loc[sub.against, "g31_Fin_physics_share"].mean()),
            "mean_share_other": float(sub.loc[~sub.against, "g31_Fin_physics_share"].mean()),
            "auroc_neg_share": float(roc_auc_score(sub.against.astype(int), -sub["g31_Fin_physics_share"])),
        }
    out["H3"]["against_pass_rate"] = float(1 - h.loc[h.against, "fail"].mean())
    write_json(probe / "dev_analysis_h2_h3.json", {**out, "manifest": build_manifest(probe_dir=args.probe_dir, aug_dir=args.aug_dir)})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
