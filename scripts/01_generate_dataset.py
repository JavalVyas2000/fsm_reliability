from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.generate_fsm_dataset import build_default_dataset


if __name__ == "__main__":
    build_default_dataset(
        output_dir="data/raw_10node_large",
        train_samples=3000,
        val_samples=500,
        test_samples=500,
        num_nodes=10,
        edge_prob=0.30,
        seed=42,
    )