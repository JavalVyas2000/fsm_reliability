"""
Skip-the-validator under plant and fault shift (docs/cstr_shift_prereg.md).

For each model and shift (feed, cooling, family-k x4, control): probes and thresholds are fitted on
the source part only (60/20/20 train/dev_cal/dev_thr, seeded), then evaluated once on the target
part: AUROC, validator calls skipped, failures among unchecked accepts, good proposals rejected,
and the paired difference in calls skipped against observables.

Example:
    python -m scripts.45_cstr_shift_eval
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
S39 = importlib.import_module("scripts.39_cstr_freeze_skip_rules")

G = "outputs/cstr_grounding"
MODELS = {  # model -> (main aug run: train/dev/test_iid, pattern of the run that also has cert)
    "Qwen2.5-1.5B": (f"{G}/20261002_040625_20261001_115207_qwen25-15b-instruct_v4_v31_r0", "*qwen25-15b-instruct_v4_v31_r0_withcert"),
    "Qwen2.5-3B": (f"{G}/20260929_204745_20260928_201301_qwen25-3b-instruct_v4_v31_r0", "*qwen25-3b-instruct_v4_v31_r0_withcert"),
    "Qwen2.5-7B (4-bit)": (f"{G}/20261003_022728_20261002_083124_qwen25-7b-instruct_v4_v31_r0", "*qwen25-7b-instruct_v4_v31_r0_withcert"),
    "Llama-3.2-3B": (f"{G}/20261001_065645_20260929_214136_llama-32-3b-instruct_v4_v31_r0", "*llama-32-3b-instruct_v4_v31_r0_withcert"),
}
BASE = "plant readings + proposed change"
TOL = (0.10, 0.05)
FAMILIES = ("fouling", "pump_degrade", "cool_stuck_closed", "outlet_block")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--models", nargs="*", default=None)
    return p.parse_args()


def load(main, cert_pattern):
    df = S12.load_records(REPO_ROOT / main / "aug")
    cert_dir = sorted((REPO_ROOT / G).glob(cert_pattern))[-1]
    dc = S12.load_records(cert_dir / "aug")
    df = pd.concat([df, dc[dc["partition"] == "cert"]], ignore_index=True)
    if "round" in df.columns:
        df = df[df["round"] == 0]
    df = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()].copy()
    df["y"] = df["candidate_invalid"].astype(int)
    hidden, _ = S12.load_hidden(REPO_ROOT / main / "aug")
    return df.drop_duplicates("instance_id").reset_index(drop=True), hidden


def shifts(eps, rng):
    med_feed, med_ua = eps["Fin_sp"].median(), eps["UA"].median()
    out = {"feed": (eps.Fin_sp <= med_feed, eps.Fin_sp > med_feed),
           "cooling": (eps.UA <= med_ua, eps.UA > med_ua)}
    for f in FAMILIES:
        out[f"family-{f}"] = (eps.family != f, eps.family == f)
    half = pd.Series(rng.permutation(len(eps)) < len(eps) // 2, index=eps.index)
    out["control"] = (half, ~half)
    return {k: (set(eps.episode_id[s]), set(eps.episode_id[t])) for k, (s, t) in out.items()}, {"feed_median": med_feed, "UA_median": med_ua}


def main():
    args = parse_args()
    run_dir = make_run_dir(REPO_ROOT / "outputs/cstr_shift_v4", "shift")
    eps = pd.concat([v for k, v in pd.read_excel(REPO_ROOT / "data/cstr/episodes_v4/cstr_episodes.xlsx", sheet_name=None).items()
                     if k in ("train", "val", "test")], ignore_index=True)
    split_ids, medians = shifts(eps, np.random.default_rng(args.seed))
    rows, deltas = [], []
    t0 = time.time()
    for model, (main_aug, cert_pat) in MODELS.items():
        if args.models and model not in args.models:
            continue
        df, hidden = load(main_aug, cert_pat)
        groups = S12.feature_groups(df, "cstr")
        for shift, (src, tgt) in split_ids.items():
            s_df = df[df.graph_hash.isin(src)].copy()
            t_df = df[df.graph_hash.isin(tgt)].copy()
            r = np.random.default_rng([args.seed, zlib.crc32(shift.encode())]).random(len(s_df))
            parts = {"train": s_df[r < 0.6], "dev_cal": s_df[(r >= 0.6) & (r < 0.8)], "dev_thr": s_df[r >= 0.8]}
            dev = pd.concat([parts["dev_cal"], parts["dev_thr"]])
            yt = t_df.y.to_numpy()
            skip_masks = {}
            for label, name in S39.SIGNALS_CSTR.items():
                spec = groups[name]
                cols = spec["scalar"]
                X = {p: S12.build_matrix(d, cols, spec["hidden"], hidden) for p, d in parts.items()}
                Xt = S12.build_matrix(t_df, cols, spec["hidden"], hidden)
                pipe = S12.make_pipeline(len(cols), X["train"].shape[1]).fit(X["train"], parts["train"].y)
                platt = LogisticRegression(C=1e6, max_iter=5000).fit(
                    S39.logit(pipe.predict_proba(X["dev_cal"])[:, 1])[:, None], parts["dev_cal"].y)
                cal = lambda Z: platt.predict_proba(S39.logit(pipe.predict_proba(Z)[:, 1])[:, None])[:, 1]  # noqa: E731
                rdev = cal(np.concatenate([X["dev_cal"], X["dev_thr"]]))
                rt = cal(Xt)
                t_rej = S39.reject_threshold(rdev, dev.y.to_numpy(), 0.95)
                auroc = roc_auc_score(yt, rt) if 0 < yt.sum() < len(yt) else np.nan
                for x in TOL:
                    t_acc = S39.accept_threshold_ucb(rdev, dev.y.to_numpy(), x, 0.05)
                    acc = rt <= t_acc if t_acc is not None else np.zeros(len(yt), bool)
                    rej = (rt >= t_rej) & ~acc if t_rej is not None else np.zeros(len(yt), bool)
                    skip_masks[(x, label)] = acc | rej
                    rows.append({"model": model, "shift": shift, "signal": label, "tolerance": x,
                                 "n_source_train": len(parts["train"]), "n_target": len(yt),
                                 "target_failure_rate": float(yt.mean()), "auroc": auroc,
                                 "skipped_pct": 100 * (acc | rej).mean(), "accepted": int(acc.sum()),
                                 "accepted_failures": int((acc & (yt == 1)).sum()),
                                 "accepted_failure_rate": float((acc & (yt == 1)).sum() / acc.sum()) if acc.sum() else np.nan,
                                 "rejected": int(rej.sum()), "rejected_good": int((rej & (yt == 0)).sum())})
            idx = np.random.default_rng(0).integers(0, len(yt), (args.n_boot, len(yt)))
            for x in TOL:
                base = skip_masks[(x, BASE)].astype(float)
                for label in S39.SIGNALS_CSTR:
                    if label == BASE:
                        continue
                    dv = skip_masks[(x, label)].astype(float) - base
                    b = dv[idx].mean(1) * 100
                    deltas.append({"model": model, "shift": shift, "signal": label, "tolerance": x,
                                   "delta_skipped_pts": 100 * dv.mean(), "ci_lo": np.percentile(b, 2.5),
                                   "ci_hi": np.percentile(b, 97.5)})
            print(f"{model} | {shift}: source {len(s_df)}, target {len(t_df)} ({yt.mean():.2f} fail) | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
        pd.DataFrame(deltas).to_csv(run_dir / "deltas.csv", index=False)
    write_json(run_dir / "run_config.json", build_manifest(prereg="docs/cstr_shift_prereg.md", medians=medians, seed=args.seed,
                                                          models=MODELS, signals=S39.SIGNALS_CSTR, tolerances=TOL))
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
