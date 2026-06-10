from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_GROUPS = [
    "token_only",
    "pooled_attention_only",
    "region_summary_only",
    "attention_only_all",
    "internal_only_combined",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multiseed probability-model summaries.")
    parser.add_argument("--base_output_dir", type=str, default="outputs/multiseed")
    parser.add_argument("--feature_groups", type=str, nargs="+", default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--output_path", type=str, default="outputs/multiseed/aggregate_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    base_output_dir = Path(args.base_output_dir)
    rows = []

    for seed_dir in sorted(base_output_dir.glob("seed_*")):
        seed_name = seed_dir.name.replace("seed_", "")
        try:
            seed = int(seed_name)
        except ValueError:
            continue

        for fg in args.feature_groups:
            summary_path = seed_dir / fg / "summary_metrics.csv"
            if not summary_path.exists():
                print(f"Missing: {summary_path}")
                continue

            df = pd.read_csv(summary_path)
            test = df[df["split"] == "test"]
            if len(test) != 1:
                print(f"Unexpected test rows in: {summary_path}")
                continue

            row = test.iloc[0].to_dict()
            rows.append(
                {
                    "seed": seed,
                    "feature_group": fg,
                    "n_samples": row.get("n_samples"),
                    "accuracy": row.get("base_correct_rate"),
                    "roc_auc": row.get("auroc_correct"),
                    "auprc": row.get("auprc_correct"),
                    "brier_score": row.get("brier_correct"),
                    "ece": row.get("ece_correct"),
                }
            )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("No summary files found to aggregate.")

    numeric_cols = ["accuracy", "roc_auc", "auprc", "brier_score", "ece"]
    agg = all_df.groupby("feature_group")[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in agg.columns.to_flat_index()]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_path.with_name("all_seed_summaries.csv"), index=False)
    agg.to_csv(output_path, index=False)

    print("\nAll seed summaries")
    print(all_df.to_string(index=False))
    print("\nAggregate mean ± std")
    print(agg.to_string(index=False))
    print(f"\nSaved per-seed table to: {output_path.with_name('all_seed_summaries.csv').resolve()}")
    print(f"Saved aggregate table to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
