from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_FEATURE_GROUPS = [
    "token_only",
    "pooled_attention_only",
    "region_summary_only",
    "attention_only_all",
    "internal_only_combined",
]


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("RUNNING:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiseed FSM reliability pipeline.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--train_samples", type=int, default=3000)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--num_nodes", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--edge_prob", type=float, default=0.30)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--feature_groups", type=str, nargs="+", default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--base_data_dir", type=str, default="data")
    parser.add_argument("--base_output_dir", type=str, default="outputs/multiseed")
    parser.add_argument("--parsed_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path.cwd()
    base_data_dir = repo_root / args.base_data_dir
    base_output_dir = repo_root / args.base_output_dir

    for seed in args.seeds:
        print(f"\n\n########## SEED {seed} ##########")

        raw_dir = base_data_dir / f"raw_seed_{seed}"
        processed_dir = base_data_dir / f"processed_seed_{seed}"
        seed_output_dir = base_output_dir / f"seed_{seed}"

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Generate dataset. build_default_dataset expects num_nodes_list, not num_nodes.
        run_cmd(
            [
                sys.executable,
                "-c",
                (
                    "from src.data.generate_fsm_dataset import build_default_dataset; "
                    f"build_default_dataset(output_dir=r'{raw_dir.as_posix()}', "
                    f"train_samples={args.train_samples}, "
                    f"val_samples={args.val_samples}, "
                    f"test_samples={args.test_samples}, "
                    f"num_nodes_list={list(args.num_nodes)!r}, "
                    f"edge_prob={args.edge_prob}, "
                    f"seed={seed})"
                ),
            ]
        )

        # 2) Extract features for each split.
        for split, n in [
            ("train", args.train_samples),
            ("val", args.val_samples),
            ("test", args.test_samples),
        ]:
            run_cmd(
                [
                    sys.executable,
                    "scripts/03_extract_pilot_features.py",
                    "--model_name",
                    args.model_name,
                    "--data_path",
                    str(raw_dir / f"{split}.csv"),
                    "--num_samples",
                    str(n),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--output_path",
                    str(processed_dir / f"features_{split}.csv"),
                ]
            )

        # 3) Run probability model once per feature group.
        # Use a separate output directory per feature group to avoid overwriting summary_metrics.csv.
        for feature_group in args.feature_groups:
            output_dir = seed_output_dir / feature_group
            cmd = [
                sys.executable,
                "scripts/06_calibrate_probability_model.py",
                "--train_path",
                str(processed_dir / "features_train.csv"),
                "--val_path",
                str(processed_dir / "features_val.csv"),
                "--test_path",
                str(processed_dir / "features_test.csv"),
                "--target",
                "valid_path",
                "--feature_group",
                feature_group,
                "--output_dir",
                str(output_dir),
            ]
            if args.parsed_only:
                cmd.append("--parsed_only")
            run_cmd(cmd)

    print("\nDone. All seeds completed.")


if __name__ == "__main__":
    main()
