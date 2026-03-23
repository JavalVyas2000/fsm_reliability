from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = df[["y_true", "y_prob"]].copy()
    out["y_true"] = out["y_true"].astype(int)
    out["y_prob"] = out["y_prob"].astype(float)
    return out


def compute_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    coverages: np.ndarray | None = None,
) -> pd.DataFrame:
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)

    order = np.argsort(-y_prob)  # highest confidence first
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    n = len(y_true_sorted)
    rows = []

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        yt = y_true_sorted[:k]
        yp = y_prob_sorted[:k]

        selective_accuracy = float(np.mean(yt))
        risk = float(1.0 - selective_accuracy)
        avg_confidence = float(np.mean(yp))
        threshold_used = float(np.min(yp))

        rows.append(
            {
                "coverage": float(k / n),
                "kept": int(k),
                "threshold_used": threshold_used,
                "avg_confidence": avg_confidence,
                "selective_accuracy": selective_accuracy,
                "risk": risk,
            }
        )

    return pd.DataFrame(rows)


def compute_three_band_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    high_frac: float = 0.30,
    low_frac: float = 0.30,
) -> pd.DataFrame:
    if high_frac + low_frac >= 1.0:
        raise ValueError("high_frac + low_frac must be < 1.0")

    order = np.argsort(-y_prob)  # highest confidence first
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    n = len(y_true_sorted)
    high_k = int(round(high_frac * n))
    low_k = int(round(low_frac * n))
    mid_k = n - high_k - low_k

    bands = {
        "Accept / low risk": slice(0, high_k),
        "Review / medium risk": slice(high_k, high_k + mid_k),
        "Reject or verify / high risk": slice(high_k + mid_k, n),
    }

    rows = []
    for name, s in bands.items():
        yt = y_true_sorted[s]
        yp = y_prob_sorted[s]

        if len(yt) == 0:
            continue

        rows.append(
            {
                "band": name,
                "count": int(len(yt)),
                "fraction": float(len(yt) / n),
                "avg_predicted_reliability": float(np.mean(yp)),
                "empirical_valid_rate": float(np.mean(yt)),
                "empirical_risk": float(1.0 - np.mean(yt)),
                "min_score": float(np.min(yp)),
                "max_score": float(np.max(yp)),
            }
        )

    return pd.DataFrame(rows)


def make_single_plots(
    rc_df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    title_prefix: str,
) -> None:
    # Risk–coverage
    plt.figure(figsize=(7.2, 5.0))
    plt.plot(rc_df["coverage"], rc_df["risk"], marker="o", linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - selective accuracy)")
    plt.title(f"{title_prefix}: Risk–coverage")
    plt.xlim(0.1, 1.0)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_risk_coverage.png", dpi=300)
    plt.close()

    # Selective accuracy–coverage
    plt.figure(figsize=(7.2, 5.0))
    plt.plot(rc_df["coverage"], rc_df["selective_accuracy"], marker="o", linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Selective accuracy")
    plt.title(f"{title_prefix}: Selective accuracy–coverage")
    plt.xlim(0.1, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_selective_accuracy_coverage.png", dpi=300)
    plt.close()


def make_comparison_plots(
    all_rc: dict[str, pd.DataFrame],
    output_dir: Path,
    title_prefix: str,
) -> None:
    # Comparison risk–coverage
    plt.figure(figsize=(8.6, 5.6))
    for label, rc_df in all_rc.items():
        plt.plot(rc_df["coverage"], rc_df["risk"], marker="o", linewidth=2, label=label)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - selective accuracy)")
    plt.title(f"{title_prefix}: Risk–coverage comparison")
    plt.xlim(0.1, 1.0)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_risk_coverage.png", dpi=300)
    plt.close()

    # Comparison selective accuracy–coverage
    plt.figure(figsize=(8.6, 5.6))
    for label, rc_df in all_rc.items():
        plt.plot(
            rc_df["coverage"],
            rc_df["selective_accuracy"],
            marker="o",
            linewidth=2,
            label=label,
        )
    plt.xlabel("Coverage")
    plt.ylabel("Selective accuracy")
    plt.title(f"{title_prefix}: Selective accuracy–coverage comparison")
    plt.xlim(0.1, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_selective_accuracy_coverage.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate risk–coverage and selective-accuracy plots from prediction CSVs."
    )
    parser.add_argument("--combined_path", type=str, default="")
    parser.add_argument("--token_path", type=str, default="")
    parser.add_argument("--pooled_path", type=str, default="")
    parser.add_argument("--region_path", type=str, default="")
    parser.add_argument("--title", type=str, default="Model")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "Combined": args.combined_path,
        "Token only": args.token_path,
        "Pooled attention": args.pooled_path,
        "Region summary": args.region_path,
    }

    all_rc = {}
    all_band_rows = []
    all_key_rows = []

    key_coverages = [0.1, 0.3, 0.5, 0.7, 1.0]

    for label, path_str in file_map.items():
        if not path_str:
            continue

        path = Path(path_str)
        df = load_predictions(path)

        y_true = df["y_true"].to_numpy()
        y_prob = df["y_prob"].to_numpy()

        rc_df = compute_risk_coverage(y_true, y_prob)
        band_df = compute_three_band_summary(y_true, y_prob)

        stem = label.lower().replace(" ", "_")
        rc_df.to_csv(output_dir / f"{stem}_risk_coverage.csv", index=False)
        band_df.to_csv(output_dir / f"{stem}_decision_bands.csv", index=False)

        make_single_plots(rc_df, output_dir, stem, f"{args.title} - {label}")

        all_rc[label] = rc_df

        band_tmp = band_df.copy()
        band_tmp.insert(0, "feature_group", label)
        all_band_rows.append(band_tmp)

        for cov in key_coverages:
            idx = (rc_df["coverage"] - cov).abs().idxmin()
            row = rc_df.loc[idx]
            all_key_rows.append(
                {
                    "feature_group": label,
                    "coverage": round(float(row["coverage"]), 3),
                    "threshold_used": round(float(row["threshold_used"]), 3),
                    "avg_confidence": round(float(row["avg_confidence"]), 3),
                    "selective_accuracy": round(float(row["selective_accuracy"]), 3),
                    "risk": round(float(row["risk"]), 3),
                }
            )

    if len(all_rc) >= 2:
        make_comparison_plots(all_rc, output_dir, args.title)

    if all_band_rows:
        all_bands_df = pd.concat(all_band_rows, ignore_index=True)
        all_bands_df.to_csv(output_dir / "all_decision_bands.csv", index=False)

    if all_key_rows:
        all_key_df = pd.DataFrame(all_key_rows)
        all_key_df.to_csv(output_dir / "all_key_coverages.csv", index=False)

        print("\nKey coverage summary")
        print(all_key_df.to_string(index=False))

    if all_band_rows:
        print("\nThree-band decision summary")
        print(all_bands_df.to_string(index=False))

    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()