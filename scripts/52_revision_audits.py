"""
Revision item 7 (docs/revision_prereg.md): (7a) teacher-forced argmax agreement of every grounding run behind the
paper; (7b) CSTR grounding-region sensitivity, i.e. the AUROC of low grounding for failure under four region definitions.

Outputs (outputs/revision/<ts>_audits/): tf_agreement.csv, region_sensitivity.csv, manifest.json.

Example:
    python -m scripts.52_revision_audits
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import importlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
S51 = importlib.import_module("scripts.51_probe_sensitivity")
CERT = REPO_ROOT / "outputs" / "certification"
REGION_DEFS = {"deciding regions (pre-registered)": "share", "physics section": "physics_share",
               "setpoint-action section": "acts_share", "whole snapshot": "snap_share"}


def grounding_dirs(freeze):
    """The grounding runs behind a freeze: open partitions and certification set."""
    fz = CERT / freeze
    opn = json.loads((fz / "freeze_manifest.json").read_text())["aug_dir"]
    crt = json.loads((fz / "eval_cert.json").read_text())["manifest"]["aug_dir"]
    return {"development": REPO_ROOT / opn, "certification": REPO_ROOT / crt}


def tf_agreement():
    rows = []
    for (domain, model), freeze in S51.FREEZES.items():
        for part, aug in grounding_dirs(freeze).items():
            m = json.loads((aug.parent / "run_manifest.json").read_text())
            v = m.get("tf_argmax_agreement", m.get("teacher_forced_argmax_agreement"))
            rows.append({"domain": domain, "model": model, "data": part, "tf_argmax_agreement": v,
                         "grounding_run": aug.parent.name})
    return pd.DataFrame(rows)


def region_sensitivity():
    rows = []
    for (domain, model), freeze in S51.FREEZES.items():
        if domain != "cstr":
            continue
        df = S51.eligible(S12.load_records(grounding_dirs(freeze)["development"]), ["train", "dev_cal", "dev_thr"])
        y = df["y"].to_numpy()
        for name, suffix in REGION_DEFS.items():
            cols = [f"g31_{sp}_{suffix}" for sp in ("Fin", "L", "T")]
            score = -df[cols].min(axis=1).to_numpy()  # low grounding on any setpoint -> predicted failure
            ok = ~np.isnan(score)
            rows.append({"model": model, "region_definition": name, "n": int(ok.sum()),
                         "auroc_low_grounding_predicts_failure": roc_auc_score(y[ok], score[ok])})
    return pd.DataFrame(rows)


def main():
    out = make_run_dir(REPO_ROOT / "outputs" / "revision", "audits")
    tf, rs = tf_agreement(), region_sensitivity()
    tf.to_csv(out / "tf_agreement.csv", index=False)
    rs.to_csv(out / "region_sensitivity.csv", index=False)
    write_json(out / "manifest.json", build_manifest(prereg="docs/revision_prereg.md (item 7)"))
    pd.set_option("display.width", 200)
    print(tf.drop(columns="grounding_run").to_string(index=False))
    print(rs.pivot(index="model", columns="region_definition", values="auroc_low_grounding_predicts_failure").round(3))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
