"""
Freeze the skip-the-validator rules of docs/cstr_v4_prereg.md (Amendment 2) before any test
data is read. Only train, dev_cal and dev_thr records are loaded.

Per signal: probe fit on train and Platt calibration on dev_cal (as scripts/12); on dev_cal +
dev_thr pooled Platt risks, the ACCEPT threshold for each tolerance X (largest threshold whose
accepted set has failure rate <= X) and the REJECT threshold (smallest threshold whose rejected
set is >= --min_reject_fail failures).

Outputs (outputs/certification/<ts>_<tag>/): probes.joblib, rules.json, dev_routing.csv,
freeze_manifest.json (hashes checked by scripts/40).

Example:
    python -m scripts.39_cstr_freeze_skip_rules --aug_dir outputs/cstr_grounding/<run>/aug --tag cstr_v4_qwen25-3b_skip
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
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, sha256_file, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
OPEN = ("train", "dev_cal", "dev_thr")
SIGNALS_CSTR = {  # label -> scripts/12 feature group
    "plant readings only": "context_only_no_action",
    "plant readings + proposed change": "context_action",
    "token confidence": "token_confidence",
    "attention by prompt region": "attention",
    "region-based attention grounding": "grounding_v31",
    "hidden states": "hidden",
    "all internals": "all_internal",
    "all internals + grounding": "all_internal+grounding_v31",
    "readings + change + all internals + grounding": "context_action+all_internal+grounding_v31",
    # Amendment 3: isolate grounding and the other internals within the combined probe
    "readings + change + grounding": "context_action+grounding_v31",
    "readings + change + all internals": "context_action+all_internal",
}
SIGNALS_FSM = {
    "task context + proposed path": "context_action",
    "token confidence": "token_confidence",
    "attention by prompt region": "attention",
    "region-based attention grounding": "grounding",
    "hidden states": "hidden",
    "all internals": "all_internal",
    "all internals + grounding": "all_internal+grounding",
    "context + path + all internals + grounding": "context_action+all_internal+grounding",
    "context + path + grounding": "context_action+grounding",
    "context + path + all internals": "context_action+all_internal",
}
SIGNALS = {"cstr": SIGNALS_CSTR, "fsm": SIGNALS_FSM}
BASELINE = {"cstr": "plant readings + proposed change", "fsm": "task context + proposed path"}
TOLERANCES = (0.10, 0.05)


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def accept_threshold(p, y, max_fail):
    """Largest t with failure rate <= max_fail among risk <= t (None if no such t)."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    rate = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    best = None
    for i in range(len(ps)):
        if (i == len(ps) - 1 or ps[i + 1] > ps[i]) and rate[i] <= max_fail:
            best = float(ps[i])
    return best


def accept_threshold_ucb(p, y, max_fail, delta_sel):
    """Largest t whose accepted set has a one-sided Clopper-Pearson upper bound (level 1 - delta_sel)
    on its failure rate <= max_fail (Amendment 3). None if no such t."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    k = np.cumsum(ys)
    best = None
    for i in range(len(ps)):
        if not (i == len(ps) - 1 or ps[i + 1] > ps[i]):
            continue
        n = i + 1
        ub = 1.0 if k[i] >= n else float(beta.ppf(1 - delta_sel, k[i] + 1, n - k[i]))
        if ub <= max_fail:
            best = float(ps[i])
    return best


def reject_threshold(p, y, min_fail):
    """Smallest t with failure share >= min_fail among risk >= t (None if no such t)."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    best = None
    for i in range(len(ps)):
        if (i == len(ps) - 1 or ps[i + 1] < ps[i]) and prec[i] >= min_fail:
            best = float(ps[i])
    return best


def load_open(aug_dir):
    df = S12.load_records(aug_dir)
    if "round" not in df.columns:  # FSM: one candidate per graph
        df["round"] = 0
    df = df[df["partition"].isin(OPEN) & (df["round"] == 0)]
    df = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()].copy()
    df["y"] = df["candidate_invalid"].astype(int)
    return df.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aug_dir", type=str, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--min_reject_fail", type=float, default=0.95)
    p.add_argument("--domain", choices=["cstr", "fsm"], default="cstr")
    p.add_argument("--threshold_rule", choices=["point", "ucb"], default="point",
                   help="ACCEPT threshold from the dev point estimate (Amendment 2) or from its "
                        "Clopper-Pearson upper bound (Amendment 3)")
    p.add_argument("--delta_sel", type=float, default=0.05, help="level of the dev bound for --threshold_rule ucb")
    p.add_argument("--prereg", type=str, default="docs/cstr_v4_prereg.md")
    return p.parse_args()


def main():
    args = parse_args()
    aug = REPO_ROOT / args.aug_dir
    df = load_open(aug)
    hidden, hidden_keys = S12.load_hidden(aug)
    hidden = {k: v for k, v in hidden.items() if k in set(df["instance_id"])}
    groups = S12.feature_groups(df, args.domain)
    signals = SIGNALS[args.domain]
    split = {p: df[df["partition"] == p] for p in OPEN}
    dev = pd.concat([split["dev_cal"], split["dev_thr"]])
    out = make_run_dir(REPO_ROOT / "outputs/certification", args.tag)

    probes, rules, rows = {}, {}, []
    for label, name in signals.items():
        spec = groups[name]
        cols = spec["scalar"]
        X = {p: S12.build_matrix(split[p], cols, spec["hidden"], hidden) for p in OPEN}
        pipe = S12.make_pipeline(len(cols), X["train"].shape[1]).fit(X["train"], split["train"]["y"])
        platt = LogisticRegression(C=1e6, max_iter=5000).fit(
            logit(pipe.predict_proba(X["dev_cal"])[:, 1])[:, None], split["dev_cal"]["y"])
        risk = platt.predict_proba(logit(pipe.predict_proba(np.concatenate([X["dev_cal"], X["dev_thr"]]))[:, 1])[:, None])[:, 1]
        y = dev["y"].to_numpy()
        r = {"group": name, "scalar": cols, "hidden": spec["hidden"],
             "t_reject": reject_threshold(risk, y, args.min_reject_fail),
             **{f"t_accept_{int(x * 100):02d}": (accept_threshold(risk, y, x) if args.threshold_rule == "point"
                                                  else accept_threshold_ucb(risk, y, x, args.delta_sel))
                for x in TOLERANCES}}
        rules[label] = r
        probes[label] = {"pipeline": pipe, "platt": platt}
        for x in TOLERANCES:
            ta, tr = r[f"t_accept_{int(x * 100):02d}"], r["t_reject"]
            acc = risk <= ta if ta is not None else np.zeros(len(y), bool)
            rej = (risk >= tr) & ~acc if tr is not None else np.zeros(len(y), bool)
            rows.append({"signal": label, "tolerance": x, "dev_n": len(y), "accepted": int(acc.sum()),
                         "accepted_failures": int((acc & (y == 1)).sum()), "rejected": int(rej.sum()),
                         "rejected_good": int((rej & (y == 0)).sum()),
                         "skipped_pct": round(100 * (acc.sum() + rej.sum()) / len(y), 1)})

    joblib.dump(probes, out / "probes.joblib")
    write_json(out / "rules.json", {"domain": args.domain, "baseline": BASELINE[args.domain], "tolerances": TOLERANCES,
                                    "threshold_rule": args.threshold_rule, "delta_sel": args.delta_sel,
                                    "min_reject_fail": args.min_reject_fail, "rules": rules})
    pd.DataFrame(rows).to_csv(out / "dev_routing.csv", index=False)
    write_json(out / "freeze_manifest.json", build_manifest(
        aug_dir=args.aug_dir, partitions_loaded=list(OPEN), n_by_partition={p: int(len(split[p])) for p in OPEN},
        prereg={"file": args.prereg, "sha256": sha256_file(REPO_ROOT / args.prereg)},
        hidden_keys=hidden_keys, signals=signals, domain=args.domain,
        files_sha256={f: sha256_file(out / f) for f in ("probes.joblib", "rules.json", "dev_routing.csv")}))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Frozen: {out}")


if __name__ == "__main__":
    main()
