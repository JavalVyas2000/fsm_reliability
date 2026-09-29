"""
How many validator calls can each signal skip? (development data only; test_iid/cert not read)

For each probe of a sealed scripts/12 run, thresholds are chosen on dev_cal and applied to dev_thr:
    ACCEPT (skip the validator, execute): risk <= t_acc, the largest threshold whose accepted set on
        dev_cal has a failure rate <= --max_accept_fail (and, strict variant, zero failures);
    REJECT (skip the validator, send back to the model): risk >= t_rej, the smallest threshold whose
        rejected set on dev_cal is >= --min_reject_fail failures;
    everything else is validated.
Reported on dev_thr: share of validator calls skipped, failures executed without validation,
good proposals rejected without validation.

Example:
    python -m scripts.38_cstr_skip_validator --probe_dir outputs/fsm_baseline/<sealed run>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.manifest import REPO_ROOT, write_json

SIGNALS = [
    ("plant readings only (no action)", "context_only_no_action"),
    ("plant readings + proposed change", "context_action"),
    ("token confidence", "token_confidence"),
    ("attention by prompt region", "attention"),
    ("region-based attention grounding", "grounding_v31"),
    ("hidden states", "hidden"),
    ("all internals", "all_internal"),
    ("all internals + grounding", "all_internal+grounding_v31"),
    ("readings+change + all internals + grounding", "context_action+all_internal+grounding_v31"),
]


def accept_threshold(p, y, max_fail):
    """Largest t such that the proposals with risk <= t have failure rate <= max_fail (None if none)."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    fails = np.cumsum(ys)
    n = np.arange(1, len(ys) + 1)
    ok = [(i, ps[i]) for i in range(len(ps)) if fails[i] / n[i] <= max_fail and (i == len(ps) - 1 or ps[i + 1] > ps[i])]
    return ok[-1][1] if ok else None


def reject_threshold(p, y, min_fail):
    """Smallest t such that the proposals with risk >= t are at least min_fail failures (None if none)."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = [ps[i] for i in range(len(ps)) if prec[i] >= min_fail and (i == len(ps) - 1 or ps[i + 1] < ps[i])]
    return ok[-1] if ok else None


def route(p, y, t_acc, t_rej):
    acc = p <= t_acc if t_acc is not None else np.zeros_like(y, bool)
    rej = (p >= t_rej) & ~acc if t_rej is not None else np.zeros_like(y, bool)
    return {"skipped_pct": 100 * (acc.sum() + rej.sum()) / len(y), "accepted": int(acc.sum()),
            "failures_executed": int((acc & (y == 1)).sum()), "rejected": int(rej.sum()),
            "good_rejected": int((rej & (y == 0)).sum()), "validated": int(len(y) - acc.sum() - rej.sum())}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe_dir", type=str, required=True)
    p.add_argument("--max_accept_fail", type=float, default=0.05)
    p.add_argument("--min_reject_fail", type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    d = REPO_ROOT / args.probe_dir
    preds = pd.read_csv(d / "predictions.csv")
    assert not preds["partition"].isin(["test_iid", "cert"]).any(), "probe run is not sealed"
    cal, thr = preds[preds.partition == "dev_cal"], preds[preds.partition == "dev_thr"]
    yc, yt = cal["y"].to_numpy(), thr["y"].to_numpy()
    rows = []
    for label, name in SIGNALS:
        pc, pt = cal[f"p_{name}_platt"].to_numpy(), thr[f"p_{name}_platt"].to_numpy()
        t_rej = reject_threshold(pc, yc, args.min_reject_fail)
        for variant, max_fail in ((f"accept <= {args.max_accept_fail:.0%} failures", args.max_accept_fail), ("accept 0 failures", 0.0)):
            t_acc = accept_threshold(pc, yc, max_fail)
            r = route(pt, yt, t_acc, t_rej)
            acc_only = route(pt, yt, t_acc, None)
            rows.append({"signal": label, "variant": variant, **r, "skipped_by_accept_pct": acc_only["skipped_pct"]})
    R = pd.DataFrame(rows)
    R.to_csv(d / "skip_validator_dev.csv", index=False)
    write_json(d / "skip_validator_dev.json", {"rule": vars(args), "dev_thr_n": int(len(yt)),
                                               "dev_thr_failure_rate": float(yt.mean()), "rows": rows})
    pd.set_option("display.width", 200)
    print(f"dev_thr: n={len(yt)}, failure rate {yt.mean():.2f}")
    print(R.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
