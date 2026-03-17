from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.probability_model import (
    build_reliability_table,
    build_selective_table,
    build_selective_table_by_coverage,
    get_feature_groups,
    train_probability_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeated train/val/test resampling for calibrated probability model on one logged feature table."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument(
        "--target",
        type=str,
        default="valid_path",
        choices=["valid_path", "optimal_path"],
    )
    parser.add_argument(
        "--feature_group",
        type=str,
        default="pooled_attention_only",
    )
    parser.add_argument(
        "--parsed_only",
        action="store_true",
        help="Restrict to parse_success == 1.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of repeated random splits.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed; split i uses base_seed + i.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction for test split.",
    )
    parser.add_argument(
        "--val_size_within_trainval",
        type=float,
        default=0.1875,
        help="Fraction of trainval used for val. 0.1875 gives ~65/15/20 overall.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/probability_model_repeated",
    )
    return parser.parse_args()


def _maybe_filter_parsed(df: pd.DataFrame, parsed_only: bool) -> pd.DataFrame:
    if parsed_only:
        if "parse_success" not in df.columns:
            raise ValueError("parse_success column not found.")
        df = df[df["parse_success"] == 1].copy()
    return df


def _split_single_dataframe(df: pd.DataFrame, target: str, seed: int, test_size: float, val_size_within_trainval: float):
    from sklearn.model_selection import train_test_split

    work = df.dropna(subset=[target]).copy()
    y = work[target].astype(int)

    if y.nunique() < 2:
        raise ValueError(f"Target '{target}' has fewer than 2 classes.")

    stratify = y if y.value_counts().min() >= 2 else None

    trainval_df, test_df = train_test_split(
        work,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    y_trainval = trainval_df[target].astype(int)
    stratify_tv = y_trainval if y_trainval.value_counts().min() >= 2 else None

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_size_within_trainval,
        random_state=seed,
        stratify=stratify_tv,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _summarize_repeat(
    repeat_idx: int,
    seed: int,
    result,
) -> dict:
    row = dict(result.__dict__)
    row["repeat_idx"] = repeat_idx
    row["seed"] = seed
    return row


def main():
    args = parse_args()

    df = pd.read_csv(args.input_path)
    df = _maybe_filter_parsed(df, args.parsed_only)

    groups = get_feature_groups(df)

    if args.feature_group not in groups:
        raise ValueError(
            f"Unknown feature group '{args.feature_group}'. "
            f"Available: {list(groups.keys())}"
        )

    feature_cols = groups[args.feature_group]
    if len(feature_cols) == 0:
        raise ValueError(f"Feature group '{args.feature_group}' is empty.")

    print(f"Loaded rows: {len(df)}")
    print(f"Target: {args.target}")
    print(f"Feature group: {args.feature_group}")
    print(f"Requested features: {len(feature_cols)}")
    print(f"Repeats: {args.n_repeats}")
    print(df[args.target].value_counts(dropna=False).sort_index())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    all_pred_rows = []
    all_reliability_rows = []
    all_selective_rows = []
    all_selective_cov_rows = []
    used_feature_cols_final = None

    for repeat_idx in range(args.n_repeats):
        seed = args.base_seed + repeat_idx

        train_df, val_df, test_df = _split_single_dataframe(
            df=df,
            target=args.target,
            seed=seed,
            test_size=args.test_size,
            val_size_within_trainval=args.val_size_within_trainval,
        )

        print("\n" + "=" * 100)
        print(f"Repeat {repeat_idx + 1}/{args.n_repeats} | seed={seed}")
        print(f"Train rows: {len(train_df)}")
        print(f"Val rows  : {len(val_df)}")
        print(f"Test rows : {len(test_df)}")

        result, pred_df, used_feature_cols, model = train_probability_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target=args.target,
        )

        result.feature_group = args.feature_group
        used_feature_cols_final = used_feature_cols

        y_true = pred_df["y_true"].to_numpy()
        y_prob = pred_df["y_prob"].to_numpy()

        reliability_df = build_reliability_table(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=10,
        )

        selective_df = build_selective_table(
            y_true=y_true,
            y_prob=y_prob,
            thresholds=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        )

        selective_cov_df = build_selective_table_by_coverage(
            y_true=y_true,
            y_prob=y_prob,
            coverages=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        )

        summary_row = _summarize_repeat(repeat_idx=repeat_idx, seed=seed, result=result)
        all_summary_rows.append(summary_row)

        pred_df = pred_df.copy()
        pred_df["repeat_idx"] = repeat_idx
        pred_df["seed"] = seed
        all_pred_rows.append(pred_df)

        reliability_df = reliability_df.copy()
        reliability_df["repeat_idx"] = repeat_idx
        reliability_df["seed"] = seed
        all_reliability_rows.append(reliability_df)

        selective_df = selective_df.copy()
        selective_df["repeat_idx"] = repeat_idx
        selective_df["seed"] = seed
        all_selective_rows.append(selective_df)

        selective_cov_df = selective_cov_df.copy()
        selective_cov_df["repeat_idx"] = repeat_idx
        selective_cov_df["seed"] = seed
        all_selective_cov_rows.append(selective_cov_df)

        print("\nRepeat summary")
        print(pd.DataFrame([summary_row]).to_string(index=False))

    summary_df = pd.DataFrame(all_summary_rows)
    pred_all_df = pd.concat(all_pred_rows, ignore_index=True)
    reliability_all_df = pd.concat(all_reliability_rows, ignore_index=True)
    selective_all_df = pd.concat(all_selective_rows, ignore_index=True)
    selective_cov_all_df = pd.concat(all_selective_cov_rows, ignore_index=True)

    metric_cols = ["accuracy", "roc_auc", "brier_score", "ece"]
    agg_summary = summary_df[metric_cols].agg(["mean", "std"]).T.reset_index()
    agg_summary.columns = ["metric", "mean", "std"]

    # Aggregate selective tables by threshold / coverage
    selective_agg = (
        selective_all_df.groupby("threshold")[["coverage", "selective_accuracy", "avg_confidence_of_accepted"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    selective_agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in selective_agg.columns.to_flat_index()
    ]

    selective_cov_agg = (
        selective_cov_all_df.groupby("target_coverage")[["actual_coverage", "selective_accuracy", "avg_confidence_of_accepted"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    selective_cov_agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in selective_cov_agg.columns.to_flat_index()
    ]

    stem = f"{args.target}_{args.feature_group}"
    summary_path = output_dir / f"summary_{stem}.csv"
    aggregate_summary_path = output_dir / f"aggregate_summary_{stem}.csv"
    pred_path = output_dir / f"predictions_{stem}.csv"
    reliability_path = output_dir / f"reliability_{stem}.csv"
    selective_path = output_dir / f"selective_{stem}.csv"
    selective_cov_path = output_dir / f"selective_by_coverage_{stem}.csv"
    selective_agg_path = output_dir / f"aggregate_selective_{stem}.csv"
    selective_cov_agg_path = output_dir / f"aggregate_selective_by_coverage_{stem}.csv"
    used_features_path = output_dir / f"used_features_{stem}.txt"

    summary_df.to_csv(summary_path, index=False)
    agg_summary.to_csv(aggregate_summary_path, index=False)
    pred_all_df.to_csv(pred_path, index=False)
    reliability_all_df.to_csv(reliability_path, index=False)
    selective_all_df.to_csv(selective_path, index=False)
    selective_cov_all_df.to_csv(selective_cov_path, index=False)
    selective_agg.to_csv(selective_agg_path, index=False)
    selective_cov_agg.to_csv(selective_cov_agg_path, index=False)

    if used_feature_cols_final is not None:
        with used_features_path.open("w", encoding="utf-8") as f:
            for col in used_feature_cols_final:
                f.write(col + "\n")

    print("\n" + "=" * 100)
    print("Aggregate summary (mean ± std)")
    print(agg_summary.to_string(index=False))

    print("\nAggregate selective acceptance table (fixed thresholds)")
    print(selective_agg.to_string(index=False))

    print("\nAggregate selective acceptance table (top-confidence coverage)")
    print(selective_cov_agg.to_string(index=False))

    print(f"\nSaved per-repeat summary to: {summary_path.resolve()}")
    print(f"Saved aggregate summary to: {aggregate_summary_path.resolve()}")
    print(f"Saved predictions to: {pred_path.resolve()}")
    print(f"Saved reliability table to: {reliability_path.resolve()}")
    print(f"Saved selective table to: {selective_path.resolve()}")
    print(f"Saved coverage-based selective table to: {selective_cov_path.resolve()}")
    print(f"Saved aggregate selective table to: {selective_agg_path.resolve()}")
    print(f"Saved aggregate coverage-based selective table to: {selective_cov_agg_path.resolve()}")
    print(f"Saved used features to: {used_features_path.resolve()}")


if __name__ == "__main__":
    main()