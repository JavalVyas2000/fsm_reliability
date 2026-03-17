from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--edge_prob", type=float, default=0.30)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument(
        "--feature_groups",
        type=str,
        nargs="+",
        default=["difficulty_only", "token_only", "pooled_attention_only", "full_small_combined"],
    )
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
        output_dir = base_output_dir / f"seed_{seed}"

        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Generate dataset
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
                    f"num_nodes={args.num_nodes}, "
                    f"edge_prob={args.edge_prob}, "
                    f"seed={seed})"
                ),
            ]
        )

        # 2) Extract train features
        run_cmd(
            [
                sys.executable,
                "scripts/03_extract_pilot_features.py",
                "--model_name",
                args.model_name,
                "--data_path",
                str(raw_dir / "train.csv"),
                "--num_samples",
                str(args.train_samples),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--output_path",
                str(processed_dir / "features_train.csv"),
            ]
        )

        # 3) Extract val features
        run_cmd(
            [
                sys.executable,
                "scripts/03_extract_pilot_features.py",
                "--model_name",
                args.model_name,
                "--data_path",
                str(raw_dir / "val.csv"),
                "--num_samples",
                str(args.val_samples),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--output_path",
                str(processed_dir / "features_val.csv"),
            ]
        )

        # 4) Extract test features
        run_cmd(
            [
                sys.executable,
                "scripts/03_extract_pilot_features.py",
                "--model_name",
                args.model_name,
                "--data_path",
                str(raw_dir / "test.csv"),
                "--num_samples",
                str(args.test_samples),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--output_path",
                str(processed_dir / "features_test.csv"),
            ]
        )

        # 5) Run probability model for each feature group
        for feature_group in args.feature_groups:
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