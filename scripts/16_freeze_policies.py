"""
Freeze the declared routing policies BEFORE any certification data exists (Stage 3).

Refits the Stage 1 probes exactly as scripts/12 (deterministic), checks they reproduce the
stored Platt predictions, selects thresholds on dev_thr as scripts/13 and 15 do, and saves:
    policies.json   declared policy set, thresholds, feature lists, alpha levels, delta
    probes.joblib   fitted pipelines + Platt calibrators
    sha256 of both files (in freeze_manifest.json)

Example:
    python -m scripts.16_freeze_policies --baseline_dir outputs/fsm_baseline/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.certification import bonferroni
from src.evaluation.routing_v2 import select_joint_allow, select_tau_high, select_tau_low
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, sha256_file, write_json

baseline = importlib.import_module("scripts.12_fit_fsm_baseline")

PROBES = ["context_action", "attention+token_confidence", "all_internal"]
POLICIES = [
    {"id": "P1_all_internal", "type": "single", "allow": ["all_internal"], "disallow": "all_internal"},
    {"id": "P2_attention_token", "type": "single", "allow": ["attention+token_confidence"], "disallow": "attention+token_confidence"},
    {"id": "P3_agree_context_all_internal", "type": "agree", "allow": ["context_action", "all_internal"], "disallow": "all_internal"},
    {"id": "P4_context_action_baseline", "type": "single", "allow": ["context_action"], "disallow": "context_action"},
]
# Grounding policy set (declared 2026-09-26, after the pre-registered grounding analysis on
# pilot data; frozen before grounding features are computed on the certification sets).
POLICY_SETS = {
    "main": (PROBES, POLICIES),
    "grounding": (
        ["context_action", "grounding", "context_action+grounding", "context_action+all_internal+grounding"],
        [
            {"id": "G1_context_grounding", "type": "single", "allow": ["context_action+grounding"], "disallow": "context_action+grounding"},
            {"id": "G2_context_allinternal_grounding", "type": "single", "allow": ["context_action+all_internal+grounding"],
             "disallow": "context_action+all_internal+grounding"},
            {"id": "G3_grounding_only", "type": "single", "allow": ["grounding"], "disallow": "grounding"},
            {"id": "G4_context_action_baseline", "type": "single", "allow": ["context_action"], "disallow": "context_action"},
        ],
    ),
    # CSTR (declared 2026-09-26 on the main dataset's train/dev partitions; cert sealed).
    "cstr": (
        ["context_action", "context_action+all_internal", "all_internal", "context_action+token_confidence"],
        [
            {"id": "C1_context_action", "type": "single", "allow": ["context_action"], "disallow": "context_action"},
            {"id": "C2_context_allinternal", "type": "single", "allow": ["context_action+all_internal"], "disallow": "context_action+all_internal"},
            {"id": "C3_all_internal", "type": "single", "allow": ["all_internal"], "disallow": "all_internal"},
            {"id": "C4_context_token", "type": "single", "allow": ["context_action+token_confidence"], "disallow": "context_action+token_confidence"},
        ],
    ),
}
ALLOW_ALPHA_SELECTION = 0.0  # point rule on dev_thr
BETA = 0.10
CERT_ALPHAS = [0.05, 0.02]
DELTA = 0.05


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, default="outputs/certification")
    p.add_argument("--tag", type=str, default="frozen_policies")
    p.add_argument("--policy_set", choices=list(POLICY_SETS), default="main")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = (REPO_ROOT / args.baseline_dir).resolve()
    base_cfg = json.loads((base_dir / "run_config.json").read_text())
    inf_dir = REPO_ROOT / base_cfg["inference_dir"]
    stored = pd.read_csv(base_dir / "predictions.csv")
    run_dir = make_run_dir(REPO_ROOT / args.out_root, args.tag)

    df = baseline.load_records(inf_dir)
    hidden, hidden_keys = baseline.load_hidden(inf_dir)
    fit_cfg = base_cfg.get("fitting", {})
    domain = fit_cfg.get("domain", "fsm")
    groups = baseline.feature_groups(df, domain)
    # Reproduce scripts/12 eligibility exactly (cert excluded; CSTR round filters).
    df = df[df["partition"].isin(baseline.PARTS)]
    elig = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()].copy()
    if "round" in elig.columns:
        keep = pd.Series(True, index=elig.index)
        if fit_cfg.get("train_rounds", "all") == "0":
            keep &= ~((elig["partition"] == "train") & (elig["round"] != 0))
        if fit_cfg.get("eval_rounds", "all") == "0":
            keep &= ~((elig["partition"] != "train") & (elig["round"] != 0))
        elig = elig[keep]
    elig["y"] = elig["candidate_invalid"].astype(int)
    split = {p: elig[elig.partition == p] for p in baseline.PARTS}

    probe_names, policy_list = POLICY_SETS[args.policy_set]
    probes, platt_dev_thr, repro = {}, {}, {}
    for name in probe_names:
        spec = groups[name]
        X = {p: baseline.build_matrix(split[p], spec["scalar"], spec["hidden"], hidden) for p in baseline.PARTS}
        pipe = baseline.make_pipeline(len(spec["scalar"]), X["train"].shape[1]).fit(X["train"], split["train"]["y"])
        raw_cal = pipe.predict_proba(X["dev_cal"])[:, 1]
        platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(raw_cal)[:, None], split["dev_cal"]["y"])
        probes[name] = {"pipeline": pipe, "platt": platt, "scalar": spec["scalar"], "hidden": spec["hidden"]}
        p_thr = platt.predict_proba(logit(pipe.predict_proba(X["dev_thr"])[:, 1])[:, None])[:, 1]
        platt_dev_thr[name] = p_thr
        ref = stored.set_index("instance_id").loc[split["dev_thr"]["instance_id"], f"p_{name}_platt"].to_numpy()
        repro[name] = float(np.max(np.abs(ref - p_thr)))
        if repro[name] > 1e-9:
            raise SystemExit(f"Refit does not reproduce stored predictions for {name}: max diff {repro[name]}")

    y_thr = split["dev_thr"]["y"].to_numpy()
    frozen = []
    for pol in policy_list:
        entry = dict(pol)
        if pol["type"] == "single":
            p = platt_dev_thr[pol["allow"][0]]
            entry["tau_allow"] = [select_tau_low(p, y_thr, ALLOW_ALPHA_SELECTION, "point")]
        else:
            a, b = pol["allow"]
            ta, tb, _ = select_joint_allow(platt_dev_thr[a], platt_dev_thr[b], y_thr, ALLOW_ALPHA_SELECTION, "point")
            entry["tau_allow"] = [ta, tb]
        entry["tau_disallow"] = select_tau_high(platt_dev_thr[pol["disallow"]], y_thr, BETA, "point")
        frozen.append(entry)

    policies = {
        "declared_before_certification_data": True,
        "domain": domain,
        "fitting": {k: fit_cfg.get(k) for k in ("train_rounds", "eval_rounds")},
        "policy_set": args.policy_set,
        "policies": frozen,
        "selection": {"allow": f"point rule, alpha={ALLOW_ALPHA_SELECTION} on dev_thr", "disallow": f"point rule, beta={BETA} on dev_thr"},
        "certification": {"alphas": CERT_ALPHAS, "delta_family": DELTA, "m_policies": len(policy_list),
                          "delta_per_policy": bonferroni(DELTA, len(policy_list)), "bound": "clopper_pearson_one_sided_upper",
                          "unit": "one candidate per graph (graph-disjoint cert partition)"},
        "probe_outputs": "Platt-calibrated failure probability",
        "hidden_keys": hidden_keys,
    }
    write_json(run_dir / "policies.json", policies)
    joblib.dump(probes, run_dir / "probes.joblib")
    write_json(run_dir / "freeze_manifest.json", build_manifest(
        baseline_dir=str(base_dir.relative_to(REPO_ROOT)), inference_dir=str(inf_dir.relative_to(REPO_ROOT)),
        reproduction_max_abs_diff=repro,
        sha256={"policies.json": sha256_file(run_dir / "policies.json"), "probes.joblib": sha256_file(run_dir / "probes.joblib")},
    ))
    print(json.dumps({"run_dir": str(run_dir), "policies": [{k: v for k, v in f.items() if k in ("id", "tau_allow", "tau_disallow")} for f in frozen], "repro": repro}, indent=2))


if __name__ == "__main__":
    main()
