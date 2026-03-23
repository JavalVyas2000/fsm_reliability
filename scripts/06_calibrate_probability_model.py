from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


DEFAULT_EXCLUDE = {
    "valid_path",
    "is_correct",
    "correct",
    "label",
    "target",
    "parsed_prediction",
    "prediction_text",
    "raw_output",
    "gold_path",
    "prompt",
    "sample_id",
    "id",
    "prob_correct",
    "prob_wrong",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train risk model on train, calibrate/tune on val, report once on test."
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--target", type=str, default="valid_path")
    parser.add_argument(
        "--feature_group",
        type=str,
        default="all",
        choices=[
            "all",
            "token_confidence_only",
            "pooled_attention_only",
            "region_attention_only",
        ],
    )
    parser.add_argument(
        "--parsed_only",
        action="store_true",
        help="Keep only rows where parsed_prediction looks valid if such a column exists.",
    )
    parser.add_argument(
        "--parsed_col",
        type=str,
        default="parsed_prediction",
        help="Column used with --parsed_only if present.",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=None,
        help="Optional fixed low-risk threshold on P(wrong). If omitted, chosen on val.",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=None,
        help="Optional fixed high-risk threshold on P(wrong). If omitted, chosen on val.",
    )
    parser.add_argument(
        "--target_accept_precision",
        type=float,
        default=0.98,
        help="Used only when low threshold is auto-selected on val.",
    )
    parser.add_argument(
        "--target_discard_purity",
        type=float,
        default=0.85,
        help="Used only when high threshold is auto-selected on val.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def safe_to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def load_df(path: str, target: str, parsed_only: bool, parsed_col: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    df = df[df[target].notna()].copy()
    df[target] = df[target].astype(int)

    if parsed_only and parsed_col in df.columns:
        col = df[parsed_col]
        if pd.api.types.is_bool_dtype(col):
            df = df[col].copy()
        else:
            df = df[col.notna() & (col.astype(str).str.strip() != "")].copy()

    return df.reset_index(drop=True)


def choose_feature_columns(df: pd.DataFrame, target: str, feature_group: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in DEFAULT_EXCLUDE and c != target]

    if feature_group == "all":
        return cols

    def has_any(name: str, patterns: tuple[str, ...]) -> bool:
        name = name.lower()
        return any(p in name for p in patterns)

    if feature_group == "token_confidence_only":
        patterns = (
            "logprob",
            "token_entropy",
            "selected_logprob",
            "confidence",
            "num_generated_tokens",
        )
        filtered = [c for c in cols if has_any(c, patterns) and "attention" not in c.lower()]
    elif feature_group == "pooled_attention_only":
        patterns = (
            "attention_entropy",
            "attention_maxprob",
            "mean_attention",
            "min_attention",
            "max_attention",
            "layer_",
            "attn",
        )
        filtered = [
            c for c in cols
            if has_any(c, patterns)
            and "region" not in c.lower()
            and "cross_region" not in c.lower()
        ]
    elif feature_group == "region_attention_only":
        patterns = (
            "region",
            "attention_region",
            "attn_region",
            "cross_region",
        )
        filtered = [c for c in cols if has_any(c, patterns)]
    else:
        raise ValueError(f"Unknown feature_group: {feature_group}")

    if not filtered:
        raise ValueError(
            f"No columns matched feature_group='{feature_group}'. "
            f"Available numeric columns include: {cols[:25]}"
        )
    return filtered


def fit_model(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def calibrate_on_val(base_model: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> IsotonicRegression:
    raw_val = base_model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    return calibrator


def predict_prob_correct(base_model: Pipeline, calibrator: IsotonicRegression, X: pd.DataFrame) -> np.ndarray:
    raw = base_model.predict_proba(X)[:, 1]
    prob = calibrator.predict(raw)
    return np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)


def summarize_split(name: str, y_true: np.ndarray, prob_correct: np.ndarray) -> dict:
    row = {
        "split": name,
        "n_samples": int(len(y_true)),
        "base_correct_rate": float(np.mean(y_true)),
        "base_wrong_rate": float(1.0 - np.mean(y_true)),
        "auroc_correct": np.nan,
        "auprc_correct": np.nan,
        "brier_correct": np.nan,
        "ece_correct": np.nan,
    }

    try:
        if len(np.unique(y_true)) > 1:
            row["auroc_correct"] = float(roc_auc_score(y_true, prob_correct))
            row["auprc_correct"] = float(average_precision_score(y_true, prob_correct))
            row["brier_correct"] = float(brier_score_loss(y_true, prob_correct))
        row["ece_correct"] = float(expected_calibration_error(y_true, prob_correct))
    except Exception as e:
        print(f"[warn] failed to summarize split '{name}': {e}")

    return row


def auto_select_thresholds(
    y_val: np.ndarray,
    prob_wrong_val: np.ndarray,
    target_accept_precision: float,
    target_discard_purity: float,
) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0, 1001)

    chosen_low = 0.2
    best_accept_coverage = -1.0
    for thr in grid:
        mask = prob_wrong_val <= thr
        if not np.any(mask):
            continue
        precision = float(np.mean(y_val[mask]))
        coverage = float(np.mean(mask))
        if precision >= target_accept_precision and coverage > best_accept_coverage:
            chosen_low = float(thr)
            best_accept_coverage = coverage

    chosen_high = 0.8
    best_discard_coverage = -1.0
    for thr in grid:
        mask = prob_wrong_val >= thr
        if not np.any(mask):
            continue
        purity = float(np.mean(1 - y_val[mask]))
        coverage = float(np.mean(mask))
        if purity >= target_discard_purity and coverage > best_discard_coverage:
            chosen_high = float(thr)
            best_discard_coverage = coverage

    if chosen_high < chosen_low:
        chosen_low, chosen_high = min(chosen_low, 0.3), max(chosen_high, 0.7)

    return chosen_low, chosen_high


def route_samples(prob_wrong: np.ndarray, low_thr: float, high_thr: float) -> np.ndarray:
    return np.where(
        prob_wrong <= low_thr,
        "accept",
        np.where(prob_wrong >= high_thr, "discard", "validate"),
    )


def routing_table(y_true: np.ndarray, prob_wrong: np.ndarray, low_thr: float, high_thr: float) -> pd.DataFrame:
    route = route_samples(prob_wrong, low_thr, high_thr)
    rows = []
    for label in ["accept", "validate", "discard"]:
        mask = route == label
        n = int(np.sum(mask))
        if n == 0:
            rows.append(
                {
                    "route": label,
                    "count": 0,
                    "fraction": 0.0,
                    "empirical_correct": np.nan,
                    "empirical_wrong": np.nan,
                    "wilson_low": np.nan,
                    "wilson_high": np.nan,
                }
            )
            continue

        if label == "discard":
            success_count = int(np.sum(1 - y_true[mask]))
        else:
            success_count = int(np.sum(y_true[mask]))

        lo, hi = wilson_interval(success_count, n)
        rows.append(
            {
                "route": label,
                "count": n,
                "fraction": float(np.mean(mask)),
                "empirical_correct": float(np.mean(y_true[mask])),
                "empirical_wrong": float(np.mean(1 - y_true[mask])),
                "wilson_low": lo,
                "wilson_high": hi,
            }
        )
    return pd.DataFrame(rows)


def reliability_table(y_true: np.ndarray, prob_correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob_correct, bins) - 1
    rows = []
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_left": bins[b],
                "bin_right": bins[b + 1],
                "count": int(np.sum(mask)),
                "mean_pred_correct": float(np.mean(prob_correct[mask])),
                "empirical_correct": float(np.mean(y_true[mask])),
                "empirical_wrong": float(np.mean(1 - y_true[mask])),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["bin_left", "bin_right", "count", "mean_pred_correct", "empirical_correct", "empirical_wrong"]
        )
    return pd.DataFrame(rows)


def plot_reliability(rel_df: pd.DataFrame, out_path: Path, title: str):
    if rel_df.empty:
        print(f"[warn] reliability dataframe empty, skipping plot: {out_path}")
        return
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(rel_df["mean_pred_correct"], rel_df["empirical_correct"], marker="o")
    plt.xlabel("Predicted probability of correctness")
    plt.ylabel("Empirical correctness")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def plot_hist(prob_correct: np.ndarray, y_true: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(7, 5))
    if np.any(y_true == 1):
        plt.hist(prob_correct[y_true == 1], bins=20, alpha=0.7, label="Correct")
    if np.any(y_true == 0):
        plt.hist(prob_correct[y_true == 0], bins=20, alpha=0.7, label="Incorrect")
    plt.xlabel("Predicted probability of correctness")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def save_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    prob_correct: np.ndarray,
    low_thr: float,
    high_thr: float,
    out_path: Path,
):
    out = df.copy()
    out["y_true"] = y_true
    out["prob_correct"] = prob_correct
    out["prob_wrong"] = 1.0 - prob_correct
    out["route"] = route_samples(out["prob_wrong"].values, low_thr, high_thr)
    safe_to_csv(out, out_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output_dir resolved to: {output_dir}")

    train_df = load_df(args.train_path, args.target, args.parsed_only, args.parsed_col)
    val_df = load_df(args.val_path, args.target, args.parsed_only, args.parsed_col)
    test_df = load_df(args.test_path, args.target, args.parsed_only, args.parsed_col)

    print(f"[info] loaded train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")

    feature_cols = choose_feature_columns(train_df, args.target, args.feature_group)
    print(f"[info] selected {len(feature_cols)} feature columns")

    safe_to_csv(pd.DataFrame({"feature_column": feature_cols}), output_dir / "feature_columns.csv")

    X_train = train_df[feature_cols]
    y_train = train_df[args.target].values.astype(int)

    X_val = val_df[feature_cols]
    y_val = val_df[args.target].values.astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[args.target].values.astype(int)

    base_model = fit_model(X_train, y_train)
    calibrator = calibrate_on_val(base_model, X_val, y_val)

    prob_correct_train = predict_prob_correct(base_model, calibrator, X_train)
    prob_correct_val = predict_prob_correct(base_model, calibrator, X_val)
    prob_correct_test = predict_prob_correct(base_model, calibrator, X_test)

    prob_wrong_val = 1.0 - prob_correct_val
    low_thr = args.low_threshold
    high_thr = args.high_threshold

    if low_thr is None or high_thr is None:
        auto_low, auto_high = auto_select_thresholds(
            y_val=y_val,
            prob_wrong_val=prob_wrong_val,
            target_accept_precision=args.target_accept_precision,
            target_discard_purity=args.target_discard_purity,
        )
        if low_thr is None:
            low_thr = auto_low
        if high_thr is None:
            high_thr = auto_high

    thresholds_df = pd.DataFrame(
        [
            {
                "low_risk_threshold_prob_wrong": float(low_thr),
                "high_risk_threshold_prob_wrong": float(high_thr),
                "target_accept_precision": float(args.target_accept_precision),
                "target_discard_purity": float(args.target_discard_purity),
            }
        ]
    )
    safe_to_csv(thresholds_df, output_dir / "thresholds.csv")

    summary_rows = [
        summarize_split("train", y_train, prob_correct_train),
        summarize_split("val", y_val, prob_correct_val),
        summarize_split("test", y_test, prob_correct_test),
    ]
    summary_df = pd.DataFrame(summary_rows)
    safe_to_csv(summary_df, output_dir / "summary_metrics.csv")

    val_routing = routing_table(y_val, 1.0 - prob_correct_val, low_thr, high_thr)
    test_routing = routing_table(y_test, 1.0 - prob_correct_test, low_thr, high_thr)
    safe_to_csv(val_routing, output_dir / "routing_table_val.csv")
    safe_to_csv(test_routing, output_dir / "routing_table_test.csv")

    rel_train = reliability_table(y_train, prob_correct_train)
    rel_val = reliability_table(y_val, prob_correct_val)
    rel_test = reliability_table(y_test, prob_correct_test)

    safe_to_csv(rel_train, output_dir / "reliability_train.csv")
    safe_to_csv(rel_val, output_dir / "reliability_val.csv")
    safe_to_csv(rel_test, output_dir / "reliability_test.csv")

    plot_reliability(rel_val, output_dir / "reliability_val.png", "Validation reliability")
    plot_reliability(rel_test, output_dir / "reliability_test.png", "Test reliability")
    plot_hist(prob_correct_train, y_train, output_dir / "prob_correct_train.png", "Train predicted correctness")
    plot_hist(prob_correct_val, y_val, output_dir / "prob_correct_val.png", "Validation predicted correctness")
    plot_hist(prob_correct_test, y_test, output_dir / "prob_correct_test.png", "Test predicted correctness")

    save_predictions(train_df, y_train, prob_correct_train, low_thr, high_thr, output_dir / "train_predictions.csv")
    save_predictions(val_df, y_val, prob_correct_val, low_thr, high_thr, output_dir / "val_predictions.csv")
    save_predictions(test_df, y_test, prob_correct_test, low_thr, high_thr, output_dir / "test_predictions.csv")

    metadata = {
        "train_path": str(Path(args.train_path).expanduser().resolve()),
        "val_path": str(Path(args.val_path).expanduser().resolve()),
        "test_path": str(Path(args.test_path).expanduser().resolve()),
        "target": args.target,
        "feature_group": args.feature_group,
        "parsed_only": args.parsed_only,
        "parsed_col": args.parsed_col,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "selected_low_risk_threshold_prob_wrong": float(low_thr),
        "selected_high_risk_threshold_prob_wrong": float(high_thr),
        "output_dir": str(output_dir),
        "threshold_selection_note": (
            "Trained logistic model on train. Fitted isotonic calibrator on val. "
            "Thresholds chosen on val unless fixed by user. Final report should use test only."
        ),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {output_dir / 'run_metadata.json'}")

    print("\nSummary metrics:")
    print(summary_df.to_string(index=False))

    print("\nValidation routing table:")
    print(val_routing.to_string(index=False))

    print("\nTest routing table:")
    print(test_routing.to_string(index=False))

    print(f"\nChosen thresholds on P(wrong): low={low_thr:.3f}, high={high_thr:.3f}")

    print("\nFiles now present in output_dir:")
    for p in sorted(output_dir.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()