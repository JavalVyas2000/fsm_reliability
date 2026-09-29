"""
Stage 1 FSM baseline: fit per-family failure probes on frozen inference records.

Target: candidate_invalid (positive = failure) on schema-valid candidates.
Fitting: train only (imputation, scaling, PCA, logistic regression).
Calibration: dev_cal (none / Platt / isotonic).  Secondary threshold: dev_thr.
test_iid of the pilot is EXPLORATORY.

Example:
    python -m scripts.12_fit_fsm_baseline --inference_dir outputs/fsm_inference/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise);
# src.features imports torch transitively.
import torch  # noqa: F401,I001

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics_v2 import (
    bootstrap_auroc_ci,
    bootstrap_delta_auroc,
    predictive_metrics,
    reliability_table,
)
from src.features.context_action import FSM_CONTEXT_ACTION, FSM_CONTEXT_ACTION_NO_BFS
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json

PARTS = ["train", "dev_cal", "dev_thr", "test_iid"]
ATT_REGIONS = ["system", "instruction", "example", "graph", "query", "generated"]
PCA_COMPONENTS = 64
LR_C = 1.0
SEED = 0

# Series colours: validated categorical slots 1-4 (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, default="outputs/fsm_baseline")
    p.add_argument("--tag", type=str, default="pilot")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--domain", choices=["fsm", "cstr"], default="fsm")
    p.add_argument("--train_rounds", choices=["all", "0"], default="all",
                   help="CSTR reprompt data: which rounds to fit on (train partition)")
    p.add_argument("--eval_rounds", choices=["all", "0"], default="all",
                   help="CSTR reprompt data: which rounds dev_cal/dev_thr/test use (0 = first proposals)")
    p.add_argument("--seal_test", action="store_true",
                   help="fit and calibrate as usual but compute, write and print nothing for test_iid "
                        "(pre-registered designs: dev results are written down before test is examined)")
    return p.parse_args()


# ----------------------------------------------------------------------------- data

def load_records(run_dir: Path) -> pd.DataFrame:
    rows = []
    with open(run_dir / "records.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            flat = {k: v for k, v in r.items() if k not in ("features", "context_action", "generated_ids", "region_token_counts")}
            flat.update(r.get("context_action", {}))
            flat.update(r.get("features", {}))
            rows.append(flat)
    return pd.DataFrame(rows)


def load_hidden(run_dir: Path) -> tuple[Dict[str, np.ndarray], List[str]]:
    vecs: Dict[str, np.ndarray] = {}
    keys: List[str] = []
    for shard in sorted((run_dir / "hidden").glob("shard_*.npz")):
        z = np.load(shard)
        keys = sorted(k for k in z.files if k.startswith("hid_"))
        mat = np.concatenate([z[k].astype(np.float32) for k in keys], axis=1)
        for iid, v in zip(z["instance_id"], mat):
            vecs[str(iid)] = v
    return vecs, keys


def feature_groups(df: pd.DataFrame, domain: str = "fsm") -> Dict[str, Dict[str, list]]:
    tok = sorted(c for c in df.columns if c.startswith("tok_"))
    # All region masses except `template` (masses sum to 1), plus entropies. For FSM this is
    # exactly the ATT_REGIONS selection used in Stage 1.
    att = sorted(
        c for c in df.columns
        if c.startswith("att_") and "_template_" not in c
        and (c.endswith(("_mean", "_hmax", "_entropy", "_entropy_norm")))
    )
    if domain == "fsm":
        ctx, ablation = FSM_CONTEXT_ACTION, ("context_action_noBFS", FSM_CONTEXT_ACTION_NO_BFS)
    else:
        from src.cstr.llm_io import CSTR_CONTEXT_ACTION, CSTR_ROUND_FEATURES, CSTR_SETPOINT_FEATURES

        ctx = CSTR_CONTEXT_ACTION + [c for c in CSTR_ROUND_FEATURES + CSTR_SETPOINT_FEATURES
                                     if c in df.columns and df[c].nunique() > 1]
        ablation = ("context_only_no_action", [c for c in CSTR_CONTEXT_ACTION if not c.startswith("d")])
    FSM = ctx  # local alias used below
    base = {
        "context_action": {"scalar": ctx, "hidden": False},
        ablation[0]: {"scalar": ablation[1], "hidden": False},
        "token_confidence": {"scalar": tok, "hidden": False},
        "attention": {"scalar": att, "hidden": False},
        "hidden": {"scalar": [], "hidden": True},
        "attention+token_confidence": {"scalar": att + tok, "hidden": False},
        "all_internal": {"scalar": att + tok, "hidden": True},
        "context_action+token_confidence": {"scalar": FSM + tok, "hidden": False},
        "context_action+attention": {"scalar": FSM + att, "hidden": False},
        "context_action+hidden": {"scalar": FSM, "hidden": True},
        "context_action+all_internal": {"scalar": FSM + att + tok, "hidden": True},
    }
    # Pre-registered attention-grounding family (docs/grounding_attention_prereg.md); only
    # present when scripts/27 has merged grd_* features into the records.
    grd = sorted(c for c in df.columns if c.startswith("grd_"))
    if grd:
        base.update({
            "grounding": {"scalar": grd, "hidden": False},
            "grounding_primary_only": {"scalar": ["grd_min_share"], "hidden": False},
            "grounding+token_confidence": {"scalar": grd + tok, "hidden": False},
            "attention+grounding": {"scalar": att + grd, "hidden": False},
            "context_action+grounding": {"scalar": FSM + grd, "hidden": False},
            "context_action+all_internal+grounding": {"scalar": FSM + att + tok + grd, "hidden": True},
            "all_internal+grounding": {"scalar": att + tok + grd, "hidden": True},
        })
    # Pre-registered CSTR field-grounding family (docs/grounding_cstr_prereg.md).
    fld = sorted(c for c in df.columns if c.startswith("fld_"))
    if fld:
        base.update({
            "field_grounding": {"scalar": fld, "hidden": False},
            "field_grounding_primary_only": {"scalar": ["fld_min_share"], "hidden": False},
            "context_action+field_grounding": {"scalar": FSM + fld, "hidden": False},
            "context_action+all_internal+field_grounding": {"scalar": FSM + att + tok + fld, "hidden": True},
        })
    # Pre-registered v3.1 grounding family (docs/cstr_v4_prereg.md; policies P2-P4).
    g31 = sorted(c for c in df.columns if c.startswith("g31_"))
    if g31:
        base.update({
            "grounding_v31": {"scalar": g31, "hidden": False},
            "grounding_v31_primary_only": {"scalar": ["g31_min_share"], "hidden": False},
            "context_action+grounding_v31": {"scalar": FSM + g31, "hidden": False},
            "context_action+all_internal+grounding_v31": {"scalar": FSM + att + tok + g31, "hidden": True},
            "all_internal+grounding_v31": {"scalar": att + tok + g31, "hidden": True},
        })
    return base


def build_matrix(df: pd.DataFrame, scalar: List[str], use_hidden: bool, hidden: Dict[str, np.ndarray]):
    X = df[scalar].to_numpy(dtype=np.float64) if scalar else np.zeros((len(df), 0))
    if use_hidden:
        H = np.stack([hidden[i] for i in df["instance_id"]]).astype(np.float64)
        X = np.concatenate([X, H], axis=1)
    return X


def make_pipeline(n_scalar: int, n_total: int) -> Pipeline:
    transformers = []
    if n_scalar:
        transformers.append(
            ("scalar", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
             list(range(n_scalar)))
        )
    if n_total > n_scalar:
        transformers.append(
            ("hidden", Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED))]),
             list(range(n_scalar, n_total)))
        )
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("lr", LogisticRegression(C=LR_C, max_iter=5000)),
    ])


# ----------------------------------------------------------------------------- summaries

def dataset_summary(df: pd.DataFrame, domain: str = "fsm") -> Dict:
    group_col = "num_nodes" if domain == "fsm" else "family"
    out = {}
    for part in PARTS:
        d = df[df["partition"] == part]
        if d.empty:
            continue
        elig = d[(d["schema_valid"] == 1) & (d["feature_status"] == "ok") & d["candidate_invalid"].notna()]
        by_n = {}
        for n, g in elig.groupby(group_col):
            by_n[str(n)] = {"n": int(len(g)), "prevalence_fail": float(g["candidate_invalid"].mean())}
        out[part] = {
            "n_total": int(len(d)),
            "parse_fail": int((d["parse_success"] == 0).sum()),
            "schema_fail": int(((d["parse_success"] == 1) & (d["schema_valid"] == 0)).sum()),
            "format_failure_reasons": d["format_failure_reason"].value_counts(dropna=True).to_dict(),
            "stop_reason": d["stop_reason"].value_counts(dropna=False).to_dict(),
            "feature_status": d["feature_status"].value_counts(dropna=False).to_dict(),
            "n_eligible": int(len(elig)),
            "n_fail": int(elig["candidate_invalid"].sum()),
            "prevalence_fail": float(elig["candidate_invalid"].mean()) if len(elig) else None,
            "suboptimal_valid_rate": float(elig["suboptimal_valid"].mean()) if len(elig) and "suboptimal_valid" in elig else None,
            "unknown_label": int(((d["schema_valid"] == 1) & d["candidate_invalid"].isna()).sum()),
            "all_response_fail_or_format_rate": float(1 - ((d["schema_valid"] == 1) & (d["candidate_invalid"] == 0)).mean()),
            f"by_{group_col}": by_n,
        }
    return out


def cost_summary(df: pd.DataFrame) -> Dict:
    ok = df[df["feature_status"] == "ok"]
    def s(col):
        v = ok[col].dropna().to_numpy()
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "n": int(len(v))}
    return {
        "gen_latency_s": s("gen_latency_s"),
        "feat_latency_s": s("feat_latency_s"),
        "peak_mem_gb_max": float(ok["peak_mem_bytes"].max() / 1e9),
        "prompt_tokens": s("prompt_tokens"),
        "answer_token_count": s("answer_token_count"),
        "gen_vs_tf_logprob_maxdiff": s("gen_vs_tf_logprob_maxdiff"),
        "greedy_consistent_rate": float(ok["greedy_consistent"].mean()),
        **({"verifier_wall_s": s("verifier_wall_s")} if "verifier_wall_s" in ok and ok["verifier_wall_s"].notna().any() else {}),
    }


def feature_availability(df: pd.DataFrame, groups: Dict) -> Dict:
    cols = sorted({c for g in groups.values() for c in g["scalar"]})
    out = {}
    for part in PARTS:
        d = df[(df["partition"] == part) & (df["schema_valid"] == 1)]
        if d.empty:
            continue
        nan = {c: int(d[c].isna().sum()) for c in cols if d[c].isna().any()}
        out[part] = {"n_schema_valid": int(len(d)), "nan_counts": nan}
    return out


def overlap_audit(df: pd.DataFrame, dataset_dir: Path) -> Dict:
    hashes = {p: set(df.loc[df["partition"] == p, "graph_hash"]) for p in PARTS}
    pairs = {}
    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1:]:
            pairs[f"{a}|{b}"] = len(hashes[a] & hashes[b])
    ds_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    return {
        "pairwise_graph_overlap": pairs,
        "dataset_generation_checks": {k: v for k, v in ds_manifest["dataset"]["checks"].items() if k != "excluded_files"},
        "instance_id_unique": bool(df["instance_id"].is_unique),
    }


# ----------------------------------------------------------------------------- plots

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_auroc(metrics: Dict, out: Path, split: str):
    import matplotlib.pyplot as plt

    names = [g for g in metrics["groups"] if metrics["groups"][g][split]["raw"]["auroc"] is not None]
    names = sorted(names, key=lambda g: metrics["groups"][g][split]["raw"]["auroc"])
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(names) + 1.0), facecolor=SURFACE)
    _style(ax)
    for i, g in enumerate(names):
        m = metrics["groups"][g][split]["raw"]
        ci = metrics["groups"][g][split]["auroc_ci"]
        ax.plot([ci["ci95_lo"], ci["ci95_hi"]], [i, i], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(m["auroc"], i, "o", color=SERIES[0], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(ci["ci95_hi"] + 0.005, i, f"{m['auroc']:.3f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(names)), names, fontsize=8, color=INK)
    ax.set_xlabel(f"AUROC for failure ({split}, 95% bootstrap CI)", fontsize=8.5, color=INK_2)
    ax.set_title("Failure prediction by feature family", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_reliability(tables: Dict[str, Dict], out: Path, title: str):
    import matplotlib.pyplot as plt

    # Small multiples: one family per panel; bins with < 5 samples are omitted from
    # the plot (all bins remain in metrics.json reliability tables).
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (name, t) in zip(axes.flat, tables.items()):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        pts = [(mp, ff) for mp, ff, n in zip(t["mean_p"], t["frac_fail"], t["n"]) if n >= 5]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, color=SERIES[0], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Predicted failure probability (Platt)", fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed failure rate", fontsize=8, color=INK_2)
    fig.suptitle(title + "  (dashed: perfect calibration; bins with n < 5 hidden)", fontsize=9.5, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    inf_dir = (REPO_ROOT / args.inference_dir).resolve()
    inf_manifest = json.loads((inf_dir / "run_manifest.json").read_text())
    dataset_dir = REPO_ROOT / inf_manifest["data"]["dataset_dir"]
    run_dir = make_run_dir(REPO_ROOT / args.out_root, args.tag)

    df = load_records(inf_dir)
    hidden, hidden_keys = load_hidden(inf_dir)
    groups = feature_groups(df, args.domain)

    # The certification partition is never loaded into fitting or outputs before the freeze.
    df = df[df["partition"].isin(PARTS)].reset_index(drop=True)
    elig = df[(df["schema_valid"] == 1) & (df["feature_status"] == "ok") & df["candidate_invalid"].notna()].copy()
    if "round" in elig.columns:
        keep = pd.Series(True, index=elig.index)
        if args.train_rounds == "0":
            keep &= ~((elig["partition"] == "train") & (elig["round"] != 0))
        if args.eval_rounds == "0":
            keep &= ~((elig["partition"] != "train") & (elig["round"] != 0))
        elig = elig[keep]
    elig["y"] = elig["candidate_invalid"].astype(int)
    split = {p: elig[elig["partition"] == p] for p in PARTS}
    missing = [p for p in PARTS if split[p].empty]
    if missing:
        raise SystemExit(f"Missing partitions in inference records: {missing}")

    eval_parts = ["dev_cal", "dev_thr"] + ([] if args.seal_test else ["test_iid"])
    report_parts = [p for p in ("dev_thr", "test_iid") if p in eval_parts]
    group_col = "num_nodes" if args.domain == "fsm" else "family"
    preds = elig[["instance_id", "partition", "graph_hash", group_col, "y"]].copy()
    metrics: Dict = {"target": "candidate_invalid", "positive_class": "failure", "groups": {}, "score_baselines": {}}

    for name, spec in groups.items():
        cols = spec["scalar"]
        Xs = {p: build_matrix(split[p], cols, spec["hidden"], hidden) for p in PARTS}
        pipe = make_pipeline(len(cols), Xs["train"].shape[1])
        pipe.fit(Xs["train"], split["train"]["y"])
        raw = {p: pipe.predict_proba(Xs[p])[:, 1] for p in PARTS}

        logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
        platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(raw["dev_cal"])[:, None], split["dev_cal"]["y"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw["dev_cal"], split["dev_cal"]["y"])
        cal = {
            "raw": raw,
            "platt": {p: platt.predict_proba(logit(raw[p])[:, None])[:, 1] for p in PARTS},
            "isotonic": {p: iso.predict(raw[p]) for p in PARTS},
        }

        # Secondary classification threshold: max balanced accuracy on dev_thr (Platt probabilities).
        y_thr, p_thr = split["dev_thr"]["y"].to_numpy(), cal["platt"]["dev_thr"]
        cands = np.unique(p_thr)
        bal = [0.5 * (((p_thr >= t) & (y_thr == 1)).sum() / max(1, y_thr.sum())
                      + ((p_thr < t) & (y_thr == 0)).sum() / max(1, (1 - y_thr).sum())) for t in cands]
        tau = float(cands[int(np.argmax(bal))])

        gm: Dict = {"n_features_scalar": len(cols), "uses_hidden": spec["hidden"], "threshold_dev_thr_platt": tau}
        for p in eval_parts:
            y = split[p]["y"].to_numpy()
            entry = {c: predictive_metrics(y, cal[c][p]) for c in cal}
            entry["auroc_ci"] = bootstrap_auroc_ci(y, raw[p], n_boot=args.n_boot, seed=SEED)
            yhat = cal["platt"][p] >= tau
            tp, fp = int((yhat & (y == 1)).sum()), int((yhat & (y == 0)).sum())
            fn, tn = int((~yhat & (y == 1)).sum()), int((~yhat & (y == 0)).sum())
            entry["at_threshold"] = {
                "positive_class": "failure", "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall_fail": tp / max(1, tp + fn), "precision_fail": tp / max(1, tp + fp),
                "specificity": tn / max(1, tn + fp),
            }
            entry["reliability_platt"] = reliability_table(y, cal["platt"][p])
            gm[p] = entry
        metrics["groups"][name] = gm
        for c in cal:
            for p in PARTS:
                preds.loc[split[p].index, f"p_{name}_{c}"] = cal[c][p]

    # Untrained single-score baselines (higher = riskier).
    score_defs = {
        "neg_mean_selected_logprob": lambda d: -d["tok_mean_logprob"],
        "mean_token_entropy": lambda d: d["tok_mean_entropy"],
    }
    for sname, fn in score_defs.items():
        metrics["score_baselines"][sname] = {}
        for p in report_parts:
            y = split[p]["y"].to_numpy()
            s = fn(split[p]).to_numpy()
            metrics["score_baselines"][sname][p] = {
                "auroc": predictive_metrics(y, (s - s.min()) / (np.ptp(s) + 1e-12))["auroc"],
                "auroc_ci": bootstrap_auroc_ci(y, s, n_boot=args.n_boot, seed=SEED),
            }
            preds.loc[split[p].index, f"score_{sname}"] = s

    # Paired incremental value vs context_action (the gate) and vs token confidence.
    comparisons = {}
    for p in report_parts:
        y = split[p]["y"].to_numpy()
        g = split[p]["graph_hash"].to_numpy()
        base_ca = preds.loc[split[p].index, "p_context_action_raw"].to_numpy()
        base_tok = preds.loc[split[p].index, "p_token_confidence_raw"].to_numpy()
        comparisons[p] = {}
        for name in groups:
            if name == "context_action":
                continue
            pa = preds.loc[split[p].index, f"p_{name}_raw"].to_numpy()
            comparisons[p][f"{name} - context_action"] = bootstrap_delta_auroc(y, pa, base_ca, g, args.n_boot, SEED)
        extra = [g for g in ("grounding", "grounding+token_confidence", "attention+grounding", "field_grounding", "grounding_v31") if g in groups]
        for name in ["attention", "hidden", "attention+token_confidence", "all_internal"] + extra:
            pa = preds.loc[split[p].index, f"p_{name}_raw"].to_numpy()
            comparisons[p][f"{name} - token_confidence"] = bootstrap_delta_auroc(y, pa, base_tok, g, args.n_boot, SEED)
    metrics["paired_delta_auroc"] = comparisons
    metrics["stage1_gate_test_iid"] = {
        k: v for k, v in comparisons.get("test_iid", {}).items()
        if k.startswith(("token_confidence -", "attention -", "hidden -", "attention+token_confidence -", "all_internal -", "context_action+"))
        and k.endswith("- context_action")
    }

    # Outputs.
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "dataset_summary.json", {"partitions": dataset_summary(df, args.domain), "cost": cost_summary(df)})
    write_json(run_dir / "feature_availability.json", feature_availability(df, groups))
    write_json(run_dir / "split_overlap_audit.json", overlap_audit(df, dataset_dir))
    if args.seal_test:
        preds = preds[preds["partition"] != "test_iid"]
    preds.to_csv(run_dir / "predictions.csv", index=False)
    write_json(
        run_dir / "run_config.json",
        build_manifest(
            inference_dir=str(inf_dir.relative_to(REPO_ROOT) if inf_dir.is_relative_to(REPO_ROOT) else inf_dir),
            inference_manifest=inf_manifest,
            fitting={
                "domain": args.domain, "train_rounds": args.train_rounds, "eval_rounds": args.eval_rounds,
                "certification_partition": "excluded (sealed until policy freeze)",
                "target": "candidate_invalid", "eligible": "schema_valid & feature_status == ok & known label",
                "model": f"LogisticRegression(C={LR_C})", "preprocessing": "median impute + standard scale (train)",
                "hidden": f"standard scale + PCA({PCA_COMPONENTS}) fit on train", "hidden_keys": hidden_keys,
                "calibration": "none / Platt / isotonic fit on dev_cal",
                "threshold": "max balanced accuracy on dev_thr (Platt)", "bootstrap": args.n_boot, "seed": SEED,
                "feature_groups": {k: {"scalar": v["scalar"], "hidden": v["hidden"]} for k, v in groups.items()},
                "test_iid_status": "sealed (not evaluated)" if args.seal_test else "evaluated",
            },
        ),
    )
    for p in report_parts:
        plot_auroc(metrics, run_dir / f"auroc_by_group_{p}.pdf", p)
        tables = {g: metrics["groups"][g][p]["reliability_platt"] for g in
                  ["context_action", "token_confidence", "attention", "context_action+all_internal"]}
        plot_reliability(tables, run_dir / f"reliability_{p}.pdf", f"Reliability ({p})")

    gate = metrics["stage1_gate_test_iid"]
    print(json.dumps({"run_dir": str(run_dir), "stage1_gate_test_iid": gate}, indent=2))


if __name__ == "__main__":
    main()
