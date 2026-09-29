"""
Evaluate the frozen skip-the-validator rules (scripts/39) once on a sealed partition
(docs/cstr_v4_prereg.md, Amendment 2).

Per signal and tolerance X:
- share of validator calls skipped (ACCEPT + REJECT);
- failures executed without validation, with a one-sided Clopper-Pearson upper bound on their
  rate among ACCEPTs at delta = 0.05 / n_signals (certified at X if bound <= X);
- good proposals rejected without validation.
On test_iid also the paired bootstrap 95% CI of the difference in calls skipped versus
"plant readings + proposed change".

A partition can be evaluated only once per frozen run (a marker file is written).

Example:
    python -m scripts.40_cstr_evaluate_skip_rules --frozen_dir outputs/certification/<frozen> \
        --aug_dir outputs/cstr_grounding/<run>/aug --partition test_iid
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

from src.utils.manifest import REPO_ROOT, build_manifest, sha256_file, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
BASELINE = "plant readings + proposed change"


def cp_upper(k, n, delta):
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def decide(risk, t_acc, t_rej):
    acc = risk <= t_acc if t_acc is not None else np.zeros(len(risk), bool)
    rej = (risk >= t_rej) & ~acc if t_rej is not None else np.zeros(len(risk), bool)
    return acc, rej


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen_dir", type=str, required=True)
    p.add_argument("--aug_dir", type=str, required=True)
    p.add_argument("--partition", choices=["test_iid", "cert", "dev_thr"], required=True,
                   help="dev_thr = dry run on open data (in-sample for the thresholds; no marker written)")
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    fz = REPO_ROOT / args.frozen_dir
    man = json.loads((fz / "freeze_manifest.json").read_text())
    for f, h in man["files_sha256"].items():
        if sha256_file(fz / f) != h:
            raise SystemExit(f"frozen file changed since the freeze: {f}")
    marker = fz / f"examined_{args.partition}.json"
    if marker.exists() and args.partition != "dev_thr":
        raise SystemExit(f"{args.partition} was already evaluated for this freeze ({marker})")
    rules_all = json.loads((fz / "rules.json").read_text())
    rules, tolerances = rules_all["rules"], rules_all["tolerances"]
    baseline = rules_all.get("baseline", BASELINE)
    probes = joblib.load(fz / "probes.joblib")
    delta = 0.05 / len(rules)

    aug = REPO_ROOT / args.aug_dir
    df_all = S12.load_records(aug)
    if "round" not in df_all.columns:  # FSM: one candidate per graph
        df_all["round"] = 0
    df_all = df_all[(df_all["partition"] == args.partition) & (df_all["round"] == 0)]
    n_format = int((df_all["schema_valid"] != 1).sum())
    df = df_all[(df_all["schema_valid"] == 1) & (df_all["feature_status"] == "ok") & df_all["candidate_invalid"].notna()].copy()
    if df.empty:
        raise SystemExit(f"no eligible {args.partition} records in {aug}")
    y = df["candidate_invalid"].astype(int).to_numpy()
    hidden, _ = S12.load_hidden(aug)
    hidden = {k: v for k, v in hidden.items() if k in set(df["instance_id"])}

    risks = {}
    for label, r in rules.items():
        X = S12.build_matrix(df, r["scalar"], r["hidden"], hidden)
        pr = probes[label]
        risks[label] = pr["platt"].predict_proba(logit(pr["pipeline"].predict_proba(X)[:, 1])[:, None])[:, 1]

    rows, skip_masks = [], {}
    for x in tolerances:
        key = f"t_accept_{int(x * 100):02d}"
        for label, r in rules.items():
            acc, rej = decide(risks[label], r[key], r["t_reject"])
            skip_masks[(x, label)] = acc | rej
            k, n = int((acc & (y == 1)).sum()), int(acc.sum())
            ub = cp_upper(k, n, delta)
            rows.append({"tolerance": x, "signal": label, "n": len(y), "skipped_pct": 100 * (acc | rej).mean(),
                         "accepted": n, "accepted_failures": k, "accepted_failure_rate": (k / n) if n else np.nan,
                         f"cp_upper_delta_{delta:.4f}": ub, "certified": bool(n > 0 and ub <= x),
                         "rejected": int(rej.sum()), "rejected_good": int((rej & (y == 0)).sum()),
                         "good_rejected_pct_of_good": 100 * (rej & (y == 0)).sum() / max(1, (y == 0).sum()),
                         "validated": int((~acc & ~rej).sum())})
    R = pd.DataFrame(rows)

    if args.partition in ("test_iid", "dev_thr"):
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(y), size=(args.n_boot, len(y)))
        deltas = []
        for x in tolerances:
            base = skip_masks[(x, baseline)].astype(float)
            for label in rules:
                if label == baseline:
                    continue
                d = skip_masks[(x, label)].astype(float) - base
                boots = d[idx].mean(axis=1) * 100
                deltas.append({"tolerance": x, "signal": label, "delta_skipped_pts": 100 * d.mean(),
                               "ci95_lo": float(np.percentile(boots, 2.5)), "ci95_hi": float(np.percentile(boots, 97.5))})
        pd.DataFrame(deltas).to_csv(fz / f"eval_{args.partition}_delta_vs_baseline.csv", index=False)
    R.to_csv(fz / f"eval_{args.partition}.csv", index=False)
    write_json(fz / f"eval_{args.partition}.json", {
        "partition": args.partition, "n_eligible": int(len(y)), "n_format_failures": n_format,
        "failure_rate": float(y.mean()), "delta_bonferroni": delta, "rows": rows,
        "manifest": build_manifest(frozen_dir=args.frozen_dir, aug_dir=args.aug_dir)})
    if args.partition != "dev_thr":
        write_json(marker, {"partition": args.partition, "aug_dir": args.aug_dir, "n": int(len(df))})
    pd.set_option("display.width", 250)
    print(f"{args.partition}: n={len(y)} eligible, {n_format} format failures, failure rate {y.mean():.3f}")
    print(R.round(3).to_string(index=False))
    if args.partition in ("test_iid", "dev_thr"):
        print(pd.DataFrame(deltas).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
