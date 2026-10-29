"""
Revision item 4 (docs/revision_prereg.md): stability of the screens across development splits.

Per model and domain, the open data (train + dev_cal + dev_thr) are re-partitioned at random 10 times into the
original partition sizes, stratified by graph size (path planning) or fault family (CSTR). Each split refits the
linear probe (train), Platt-calibrates (dev_cal) and rethresholds with the confirmatory rules (dev_cal + dev_thr);
each split is evaluated once on the model's certification set. Signal sets: observables, internals only, combined.

Outputs (outputs/revision/<ts>_split_stability/): splits.csv (one row per model x signal x seed x tolerance),
summary.csv, manifest.json.

Example:
    python -m scripts.53_split_stability                    # path planning cert2, CSTR cert
    python -m scripts.53_split_stability --cstr_cert cert2  # CSTR on the fresh cert2 set (once it exists)
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
S39 = importlib.import_module("scripts.39_cstr_freeze_skip_rules")
S40 = importlib.import_module("scripts.40_cstr_evaluate_skip_rules")
S51 = importlib.import_module("scripts.51_probe_sensitivity")
SETS = ("observables", "internals only", "combined")
SEEDS = range(1, 11)
STRATUM = {"fsm": "num_nodes", "cstr": "family"}


def resplit(opn, domain, seed):
    """Random re-partition of the open data into the original partition sizes, stratified."""
    rng = np.random.default_rng(seed)
    sizes = opn["partition"].value_counts()
    out = opn.copy()
    for _, g in opn.groupby(STRATUM[domain]):
        idx = rng.permutation(g.index.to_numpy())
        share = (sizes / sizes.sum()).reindex(S39.OPEN)
        cuts = np.round(np.cumsum(share.to_numpy()) * len(idx)).astype(int)
        for part, chunk in zip(S39.OPEN, np.split(idx, cuts[:-1])):
            out.loc[chunk, "partition"] = part
    return out


def cert_dir(freeze, which):
    fz = S51.CERT / freeze
    if which == "cert":
        return REPO_ROOT / json.loads((fz / "eval_cert.json").read_text())["manifest"]["aug_dir"], "cert"
    return REPO_ROOT / json.loads((fz / "eval_cert2.json").read_text())["manifest"]["aug_dir"], "cert2"


def run_model(domain, model, freeze, cstr_cert):
    fz = S51.CERT / freeze
    open_aug = REPO_ROOT / json.loads((fz / "freeze_manifest.json").read_text())["aug_dir"]
    c_aug, c_part = cert_dir(freeze, "cert" if domain == "fsm" else cstr_cert)
    opn = S51.eligible(S12.load_records(open_aug), S39.OPEN)
    crt = S51.eligible(S12.load_records(c_aug), [c_part])
    hidden = {**S12.load_hidden(open_aug)[0], **S12.load_hidden(c_aug)[0]}
    sets = {k: v for k, v in S51.signal_sets(S12.feature_groups(opn, domain), domain).items() if k in SETS}
    y = crt["y"].to_numpy()
    K = 10 if domain == "fsm" else 11  # Bonferroni as in the confirmatory certification
    rows = []
    for seed in SEEDS:
        df = resplit(opn, domain, seed)
        split = {p: df[df["partition"] == p] for p in S39.OPEN}
        dev_y = pd.concat([split["dev_cal"], split["dev_thr"]])["y"].to_numpy()
        for name, (cols, use_h) in sets.items():
            X = {p: S12.build_matrix(split[p], cols, use_h, hidden) for p in S39.OPEN}
            pipe = S12.make_pipeline(len(cols), X["train"].shape[1]).fit(X["train"], split["train"]["y"])
            platt = LogisticRegression(C=1e6, max_iter=5000).fit(
                S39.logit(pipe.predict_proba(X["dev_cal"])[:, 1])[:, None], split["dev_cal"]["y"])
            cal = lambda M: platt.predict_proba(S39.logit(pipe.predict_proba(M)[:, 1])[:, None])[:, 1]  # noqa: E731
            dev_risk = cal(np.concatenate([X["dev_cal"], X["dev_thr"]]))
            risk = cal(S12.build_matrix(crt, cols, use_h, hidden))
            t_rej = S39.reject_threshold(dev_risk, dev_y, 0.95)
            for x in (0.10, 0.05):
                acc, rej = S40.decide(risk, S39.accept_threshold_ucb(dev_risk, dev_y, x, 0.05), t_rej)
                k, n = int((acc & (y == 1)).sum()), int(acc.sum())
                ub = S40.cp_upper(k, n, 0.05 / K)
                rows.append({"domain": domain, "model": model, "cert_set": c_part, "signal": name, "seed": seed,
                             "tolerance": x, "skipped_pct": 100 * (acc | rej).mean(), "accepted": n,
                             "accepted_failures": k, "certified": bool(n > 0 and ub <= x)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cstr_cert", choices=["cert", "cert2"], default="cert")
    p.add_argument("--only", choices=["fsm", "cstr"], default=None)
    args = p.parse_args()
    out = make_run_dir(REPO_ROOT / "outputs" / "revision", f"split_stability_{args.cstr_cert}")
    rows = []
    for (domain, model), fz in S51.FREEZES.items():
        if args.only and domain != args.only:
            continue
        print(f"[{domain}] {model}", flush=True)
        rows += run_model(domain, model, fz, args.cstr_cert)
    R = pd.DataFrame(rows)
    R.to_csv(out / "splits.csv", index=False)
    S = (R.groupby(["domain", "model", "signal", "tolerance"])
          .agg(mean=("skipped_pct", "mean"), sd=("skipped_pct", "std"), min=("skipped_pct", "min"),
               max=("skipped_pct", "max"), certified_splits=("certified", "sum"), n_splits=("seed", "nunique"))
          .reset_index())
    S.to_csv(out / "summary.csv", index=False)
    write_json(out / "manifest.json", build_manifest(prereg="docs/revision_prereg.md (item 4)", seeds=list(SEEDS),
                                                     cstr_cert=args.cstr_cert))
    pd.set_option("display.width", 220)
    print(S[S.tolerance == 0.10].round(1).to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
