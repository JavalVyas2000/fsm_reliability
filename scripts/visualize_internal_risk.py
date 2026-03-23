from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize internal feature collection and risk behavior."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to logged feature CSV."
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="valid_path",
        help="Ground-truth correctness column (1 correct, 0 incorrect)."
    )
    parser.add_argument(
        "--prob_correct_col",
        type=str,
        default=None,
        help="Optional existing probability-of-correctness column."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/feature_visualization",
        help="Directory to save plots and tables."
    )
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=12,
        help="Number of most correlated numeric features to visualize."
    )
    parser.add_argument(
        "--low_risk_threshold",
        type=float,
        default=0.2,
        help="Risk below this is accepted directly."
    )
    parser.add_argument(
        "--high_risk_threshold",
        type=float,
        default=0.6,
        help="Risk above this is discarded."
    )
    return parser.parse_args()


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)


def make_reliability_table(y_true: np.ndarray, prob_correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob_correct, bins) - 1

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) == 0:
            continue

        rows.append({
            "bin_left": bins[b],
            "bin_right": bins[b + 1],
            "count": int(np.sum(mask)),
            "mean_pred_correct": float(np.mean(prob_correct[mask])),
            "empirical_correct": float(np.mean(y_true[mask])),
            "empirical_wrong": float(1.0 - np.mean(y_true[mask])),
        })
    return pd.DataFrame(rows)


def plot_reliability_diagram(rel_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(rel_df["mean_pred_correct"], rel_df["empirical_correct"], marker="o")
    plt.xlabel("Predicted probability of correctness")
    plt.ylabel("Empirical correctness")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_probability_histograms(prob_correct: np.ndarray, y_true: np.ndarray, output_path: Path):
    plt.figure(figsize=(7, 5))
    plt.hist(prob_correct[y_true == 1], bins=20, alpha=0.7, label="Correct")
    plt.hist(prob_correct[y_true == 0], bins=20, alpha=0.7, label="Incorrect")
    plt.xlabel("Predicted probability of correctness")
    plt.ylabel("Count")
    plt.title("Predicted correctness distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_risk_histogram(prob_wrong: np.ndarray, output_path: Path):
    plt.figure(figsize=(7, 5))
    plt.hist(prob_wrong, bins=20)
    plt.xlabel("Predicted risk = P(wrong)")
    plt.ylabel("Count")
    plt.title("Predicted risk distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {target_col, "prob_correct", "prob_wrong"}
    return [c for c in numeric_cols if c not in excluded]


def rank_features_by_correlation(df: pd.DataFrame, target_col: str, feature_cols: list[str], top_k: int) -> list[str]:
    corrs = []
    y = df[target_col].astype(float)

    for col in feature_cols:
        x = df[col]
        if x.nunique(dropna=True) < 2:
            continue
        corr = x.corr(y)
        if pd.notna(corr):
            corrs.append((col, abs(corr)))

    corrs.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in corrs[:top_k]]


def plot_top_feature_distributions(df: pd.DataFrame, target_col: str, top_features: list[str], output_dir: Path):
    for col in top_features:
        plt.figure(figsize=(7, 5))
        plt.hist(df.loc[df[target_col] == 1, col].dropna(), bins=20, alpha=0.7, label="Correct")
        plt.hist(df.loc[df[target_col] == 0, col].dropna(), bins=20, alpha=0.7, label="Incorrect")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"{col}: correct vs incorrect")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_dist_{col}.png", dpi=200)
        plt.close()


def make_routing_table(prob_wrong: np.ndarray, y_true: np.ndarray, low_thr: float, high_thr: float) -> pd.DataFrame:
    route = np.where(
        prob_wrong < low_thr,
        "accept",
        np.where(prob_wrong < high_thr, "validate", "discard")
    )

    rows = []
    for label in ["accept", "validate", "discard"]:
        mask = route == label
        if np.sum(mask) == 0:
            rows.append({
                "route": label,
                "count": 0,
                "fraction": 0.0,
                "empirical_correct": np.nan,
                "empirical_wrong": np.nan,
            })
            continue

        rows.append({
            "route": label,
            "count": int(np.sum(mask)),
            "fraction": float(np.mean(mask)),
            "empirical_correct": float(np.mean(y_true[mask])),
            "empirical_wrong": float(1.0 - np.mean(y_true[mask])),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in CSV.")

    df = df.copy()
    df = df[df[args.target_col].notna()]
    df[args.target_col] = df[args.target_col].astype(int)

    # Use existing probability column if present, otherwise fit a small logistic model.
    if args.prob_correct_col is not None and args.prob_correct_col in df.columns:
        df["prob_correct"] = df[args.prob_correct_col].astype(float)
    elif "prob_correct" in df.columns:
        df["prob_correct"] = df["prob_correct"].astype(float)
    else:
        feature_cols = select_numeric_features(df, args.target_col)
        if not feature_cols:
            raise ValueError("No numeric feature columns found to fit probability model.")

        X = df[feature_cols].copy()
        y = df[args.target_col].values

        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)

        # Fit on part of this file for visualization only.
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_imp, y, df.index.values, test_size=0.3, random_state=42, stratify=y
        )

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)

        prob_correct = clf.predict_proba(X_imp)[:, 1]
        df["prob_correct"] = prob_correct

    df["prob_correct"] = np.clip(df["prob_correct"], 0.0, 1.0)
    df["prob_wrong"] = 1.0 - df["prob_correct"]

    y_true = df[args.target_col].values
    prob_correct = df["prob_correct"].values
    prob_wrong = df["prob_wrong"].values

    # Metrics
    summary = {
        "n_samples": len(df),
        "base_correct_rate": float(np.mean(y_true)),
        "base_wrong_rate": float(1.0 - np.mean(y_true)),
        "auroc_correct": float(roc_auc_score(y_true, prob_correct)),
        "auprc_correct": float(average_precision_score(y_true, prob_correct)),
        "brier_correct": float(brier_score_loss(y_true, prob_correct)),
        "ece_correct": float(expected_calibration_error(y_true, prob_correct, n_bins=10)),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary_metrics.csv", index=False)

    # Reliability
    rel_df = make_reliability_table(y_true, prob_correct, n_bins=10)
    rel_df.to_csv(output_dir / "reliability_table.csv", index=False)
    plot_reliability_diagram(rel_df, output_dir / "reliability_diagram.png")

    # Histograms
    plot_probability_histograms(prob_correct, y_true, output_dir / "prob_correct_by_label.png")
    plot_risk_histogram(prob_wrong, output_dir / "risk_histogram.png")

    # Routing
    routing_df = make_routing_table(
        prob_wrong,
        y_true,
        low_thr=args.low_risk_threshold,
        high_thr=args.high_risk_threshold,
    )
    routing_df.to_csv(output_dir / "routing_table.csv", index=False)

    # Feature distributions
    feature_cols = select_numeric_features(df, args.target_col)
    top_features = rank_features_by_correlation(
        df,
        target_col=args.target_col,
        feature_cols=feature_cols,
        top_k=args.top_k_features,
    )
    pd.DataFrame({"top_features": top_features}).to_csv(output_dir / "top_features.csv", index=False)
    plot_top_feature_distributions(df, args.target_col, top_features, output_dir)

    print(f"Saved outputs to: {output_dir}")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nRouting table:")
    print(routing_df.to_string(index=False))


if __name__ == "__main__":
    main()