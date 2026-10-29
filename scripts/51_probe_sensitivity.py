"""
Sensitivity of the skip-the-validator results to the probe class and to validator-derived observables
(docs/probe_sensitivity_prereg.md, paper item 3a).

Per model and domain, on the partitions of the confirmatory freeze (outputs/certification/*_skip_ucb*):
fit each signal set with the confirmatory logistic pipeline and with a fixed, untuned gradient-boosting probe
(train), Platt-calibrate (dev_cal), choose the bound-based ACCEPT and the REJECT thresholds (dev_cal + dev_thr),
and evaluate once on the certification set (FSM cert2, CSTR cert). The logistic arm must reproduce the confirmatory
certification results exactly (sanity check, printed).

Outputs (outputs/sensitivity/<ts>_probe_class/): results.csv (one row per model x signal x probe x tolerance),
deltas.csv (pre-registered paired comparisons), sanity.csv, manifest.json.

Example:
    python -m scripts.51_probe_sensitivity
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, sha256_file, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
S39 = importlib.import_module("scripts.39_cstr_freeze_skip_rules")
S40 = importlib.import_module("scripts.40_cstr_evaluate_skip_rules")

PREREG = "docs/probe_sensitivity_prereg.md"
CERT = REPO_ROOT / "outputs" / "certification"
FREEZES = {  # (domain, model) -> confirmatory freeze directory
    ("fsm", "Qwen2.5-1.5B"): "20261002_094835_fsm_qwen25-15b_skip_ucb_cert2",
    ("fsm", "Qwen2.5-3B"): "20261002_094657_fsm_qwen25-3b_skip_ucb_cert2",
    ("fsm", "Qwen2.5-7B (4-bit)"): "20261007_090218_fsm_qwen25-7b_skip_ucb_cert2",
    ("fsm", "Llama-3.2-3B"): "20261002_094741_fsm_llama-32-3b_skip_ucb_cert2",
    ("fsm", "SmolLM2-1.7B"): "20261002_094915_fsm_smollm2-17b_skip_ucb_cert2",
    ("cstr", "Qwen2.5-1.5B"): "20261002_094601_cstr_v4_qwen25-15b_skip_ucb",
    ("cstr", "Qwen2.5-3B*"): "20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY",
    ("cstr", "Qwen2.5-7B (4-bit)"): "20261003_164452_cstr_v4_qwen25-7b_skip_ucb",
    ("cstr", "Llama-3.2-3B"): "20261001_112113_cstr_v4_llama-32-3b_skip_ucb",
    ("cstr", "SmolLM2-1.7B"): "20261009_153848_cstr_v4_smollm2-17b_skip_ucb",
}
GRD = {"fsm": "grounding", "cstr": "grounding_v31"}
VALIDATOR_DERIVED = ["shortest_length", "answer_endpoints_match"]  # FSM observables that reuse validator logic
# confirmatory label of each signal set, for the sanity check against the frozen certification results
CONFIRMATORY = {
    "fsm": {"observables": "task context + proposed path", "internals only": "all internals + grounding",
            "combined": "context + path + all internals + grounding"},
    "cstr": {"observables": "plant readings + proposed change", "internals only": "all internals + grounding",
             "combined": "readings + change + all internals + grounding"},
}
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
           early_stopping=False, random_state=0)  # fixed in the pre-registration; never tuned
TOLERANCES = (0.10, 0.05)


def signal_sets(groups, domain):
    """Pre-registered signal sets: name -> (scalar columns, uses hidden states)."""
    obs = groups["context_action"]["scalar"]
    intl = groups[f"all_internal+{GRD[domain]}"]
    comb = groups[f"context_action+all_internal+{GRD[domain]}"]
    sets = {"observables": (obs, False), "internals only": (intl["scalar"], True), "combined": (comb["scalar"], True)}
    if domain == "fsm":
        vf = [c for c in obs if c not in VALIDATOR_DERIVED]
        sets["observables without BFS"] = ([c for c in obs if c != "shortest_length"], False)
        sets["validator-free observables"] = (vf, False)
        sets["validator-free combined"] = (vf + [c for c in intl["scalar"] if c not in obs], True)
    return sets


def make_probe(kind, n_scalar, n_total):
    pipe = S12.make_pipeline(n_scalar, n_total)
    if kind == "gbm":
        pipe.steps[-1] = ("gbm", HistGradientBoostingClassifier(**GBM))
    return pipe


def eligible(df, partitions):
    if "round" not in df.columns:
        df["round"] = 0
    df = df[df["partition"].isin(partitions) & (df["round"] == 0)]
    df = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()].copy()
    df["y"] = df["candidate_invalid"].astype(int)
    return df.reset_index(drop=True)


def run_model(domain, model, freeze_name, n_boot):
    fz = CERT / freeze_name
    open_aug = REPO_ROOT / json.loads((fz / "freeze_manifest.json").read_text())["aug_dir"]
    cert_aug = REPO_ROOT / json.loads((fz / "eval_cert.json").read_text())["manifest"]["aug_dir"]
    opn = eligible(S12.load_records(open_aug), S39.OPEN)
    crt = eligible(S12.load_records(cert_aug), ["cert"])
    hidden = {**S12.load_hidden(open_aug)[0], **S12.load_hidden(cert_aug)[0]}
    split = {p: opn[opn["partition"] == p] for p in S39.OPEN}
    dev_y = pd.concat([split["dev_cal"], split["dev_thr"]])["y"].to_numpy()
    y = crt["y"].to_numpy()
    sets = signal_sets(S12.feature_groups(opn, domain), domain)
    K = 2 * len(sets)  # Bonferroni over the screens of this analysis
    rows, masks = [], {}
    for name, (cols, use_h) in sets.items():
        X = {p: S12.build_matrix(split[p], cols, use_h, hidden) for p in S39.OPEN}
        Xc = S12.build_matrix(crt, cols, use_h, hidden)
        for kind in ("logistic", "gbm"):
            pipe = make_probe(kind, len(cols), X["train"].shape[1]).fit(X["train"], split["train"]["y"])
            platt = LogisticRegression(C=1e6, max_iter=5000).fit(
                S39.logit(pipe.predict_proba(X["dev_cal"])[:, 1])[:, None], split["dev_cal"]["y"])
            cal = lambda M: platt.predict_proba(S39.logit(pipe.predict_proba(M)[:, 1])[:, None])[:, 1]  # noqa: E731
            dev_risk = cal(np.concatenate([X["dev_cal"], X["dev_thr"]]))
            cert_risk = cal(Xc)
            t_rej = S39.reject_threshold(dev_risk, dev_y, 0.95)
            for x in TOLERANCES:
                t_acc = S39.accept_threshold_ucb(dev_risk, dev_y, x, 0.05)
                acc, rej = S40.decide(cert_risk, t_acc, t_rej)
                masks[(name, kind, x)] = acc | rej
                k, n = int((acc & (y == 1)).sum()), int(acc.sum())
                ub = S40.cp_upper(k, n, 0.05 / K)
                rows.append({"domain": domain, "model": model, "signal": name, "probe": kind, "tolerance": x,
                             "n": len(y), "skipped_pct": 100 * (acc | rej).mean(), "accepted": n,
                             "accepted_failures": k, "cp_upper": ub, "certified": bool(n > 0 and ub <= x),
                             "rejected": int(rej.sum()), "rejected_good": int((rej & (y == 0)).sum())})
    # pre-registered paired comparisons (C1-C4)
    pairs = [("combined", "logistic", "observables", "logistic", "C1"), ("combined", "gbm", "observables", "gbm", "C1"),
             ("observables", "gbm", "observables", "logistic", "C2"), ("combined", "gbm", "combined", "logistic", "C2")]
    if domain == "fsm":
        for kind in ("logistic", "gbm"):
            pairs += [("validator-free observables", kind, "observables", kind, "C3"),
                      ("observables without BFS", kind, "observables", kind, "C3"),
                      ("validator-free combined", kind, "validator-free observables", kind, "C4")]
    idx = np.random.default_rng(0).integers(0, len(y), size=(n_boot, len(y)))
    deltas = []
    for x in TOLERANCES:
        for a, ka, b, kb, tag in pairs:
            d = masks[(a, ka, x)].astype(float) - masks[(b, kb, x)].astype(float)
            boots = d[idx].mean(axis=1) * 100
            deltas.append({"domain": domain, "model": model, "comparison": tag, "tolerance": x,
                           "a": f"{a} [{ka}]", "b": f"{b} [{kb}]", "delta_pts": 100 * d.mean(),
                           "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))})
    # sanity: the logistic arm against the frozen confirmatory certification results
    conf = pd.read_csv(fz / "eval_cert.csv")
    sanity = []
    for name, label in CONFIRMATORY[domain].items():
        for x in TOLERANCES:
            mine = next(r["skipped_pct"] for r in rows if r["signal"] == name and r["probe"] == "logistic" and r["tolerance"] == x)
            ref = conf[(conf["signal"] == label) & ((conf["tolerance"] - x).abs() < 1e-9)]["skipped_pct"].iloc[0]
            sanity.append({"domain": domain, "model": model, "signal": name, "tolerance": x,
                           "reproduced_pct": mine, "confirmatory_pct": ref, "match": abs(mine - ref) < 1e-6})
    return rows, deltas, sanity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--only", type=str, default=None, help="restrict to one domain (fsm or cstr) for a quick run")
    args = p.parse_args()
    out = make_run_dir(REPO_ROOT / "outputs" / "sensitivity", "probe_class")
    R, D, S = [], [], []
    for (domain, model), fz in FREEZES.items():
        if args.only and domain != args.only:
            continue
        print(f"[{domain}] {model} ...", flush=True)
        r, d, s = run_model(domain, model, fz, args.n_boot)
        R += r; D += d; S += s
        print(pd.DataFrame(s).to_string(index=False), flush=True)
    pd.DataFrame(R).to_csv(out / "results.csv", index=False)
    pd.DataFrame(D).to_csv(out / "deltas.csv", index=False)
    pd.DataFrame(S).to_csv(out / "sanity.csv", index=False)
    write_json(out / "manifest.json", build_manifest(prereg={"file": PREREG, "sha256": sha256_file(REPO_ROOT / PREREG)},
                                                     freezes=FREEZES and {f"{k[0]}|{k[1]}": v for k, v in FREEZES.items()},
                                                     gbm=GBM))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
