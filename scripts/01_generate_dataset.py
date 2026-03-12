from src.data.generate_fsm_dataset import build_default_dataset


if __name__ == "__main__":
    build_default_dataset(
        output_dir="data/raw",
        train_samples=1000,
        val_samples=200,
        test_samples=200,
        num_nodes=10,
        edge_prob=0.30,
        seed=42,
    )