from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.training.evaluate_feature_groups import evaluate_feature_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate offline feature groups for FSM reliability.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/pilot_features_100_regions.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="valid_path",
        choices=["valid_path", "optimal_path", "parse_success"],
    )
    parser.add_argument(
        "--parsed_only",
        action="store_true",
        help="Restrict to parse_success == 1 before training/evaluation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/tables/feature_group_results.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_path)

    if args.parsed_only:
        if "parse_success" not in df.columns:
            raise ValueError("parse_success column not found.")
        df = df[df["parse_success"] == 1].copy()

    print(f"Loaded rows: {len(df)}")
    print(f"Target: {args.target}")
    if args.target in df.columns:
        print(df[args.target].value_counts(dropna=False).sort_index())

    results = evaluate_feature_groups(df=df, target=args.target)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nFeature-group results")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()