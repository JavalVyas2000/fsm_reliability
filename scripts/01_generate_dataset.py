from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.generate_fsm_dataset import build_default_dataset


if __name__ == "__main__":
    build_default_dataset(
        output_dir="data/raw_multi_node_large",
        train_samples=3000,
        val_samples=500,
        test_samples=500,
        num_nodes_list=[5, 10, 15, 20],
        edge_prob={
            5: 0.35,
            10: 0.30,
            15: 0.24,
            20: 0.20,
        },
        seed=42,
    )