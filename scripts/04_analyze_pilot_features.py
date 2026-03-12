from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze pilot FSM reliability features.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/pilot_features.csv",
        help="Path to pilot feature CSV.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="valid_path",
        choices=["valid_path", "optimal_path", "parse_success"],
        help="Target label to analyze.",
    )
    parser.add_argument(
        "--parsed_only",
        action="store_true",
        help="If set, restrict analysis to parse_success == 1.",
    )
    return parser.parse_args()


def safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())


def safe_std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return float("nan")
    return float(vals.std(ddof=1))


def cohens_d(x0: pd.Series, x1: pd.Series) -> float:
    a = pd.to_numeric(x0, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(x1, errors="coerce").dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return float("nan")

    return float((mean_b - mean_a) / math.sqrt(pooled))


def get_candidate_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "instance_id",
        "start",
        "goal",
        "generated_text",
        "parsed_path",
        "ground_truth_shortest_path",
        "output_format",
        "valid_path",
        "optimal_path",
        "parse_success",
    }

    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def print_basic_counts(df: pd.DataFrame, target: str) -> None:
    print("\nDataset summary")
    print(f"Rows: {len(df)}")
    if "parse_success" in df.columns:
        print(f"Parse success rate: {df['parse_success'].mean():.3f} ({int(df['parse_success'].sum())}/{len(df)})")
    if "valid_path" in df.columns:
        print(f"Valid path rate   : {df['valid_path'].mean():.3f} ({int(df['valid_path'].sum())}/{len(df)})")
    if "optimal_path" in df.columns:
        print(f"Optimal path rate : {df['optimal_path'].mean():.3f} ({int(df['optimal_path'].sum())}/{len(df)})")

    print(f"\nTarget distribution for '{target}':")
    print(df[target].value_counts(dropna=False).sort_index())


def print_output_format_breakdown(df: pd.DataFrame) -> None:
    if "output_format" not in df.columns:
        return

    print("\nOutput format breakdown")
    fmt_counts = df["output_format"].value_counts(dropna=False)
    print(fmt_counts)

    if "valid_path" in df.columns:
        print("\nOutput format vs valid_path")
        table = pd.crosstab(df["output_format"], df["valid_path"], margins=True)
        print(table)


def compare_feature_means(df: pd.DataFrame, target: str, top_k: int = 15) -> None:
    feature_cols = get_candidate_feature_columns(df)
    if len(feature_cols) == 0:
        print("\nNo numeric feature columns found.")
        return

    rows = []
    neg = df[df[target] == 0]
    pos = df[df[target] == 1]

    for col in feature_cols:
        m0 = safe_mean(neg[col])
        m1 = safe_mean(pos[col])
        d = cohens_d(neg[col], pos[col])
        rows.append(
            {
                "feature": col,
                "mean_target0": m0,
                "mean_target1": m1,
                "delta": m1 - m0 if not (math.isnan(m0) or math.isnan(m1)) else float("nan"),
                "cohens_d": d,
            }
        )

    out = pd.DataFrame(rows)
    out["abs_d"] = out["cohens_d"].abs()
    out = out.sort_values("abs_d", ascending=False)

    print(f"\nTop {top_k} features by |Cohen's d| for '{target}'")
    print(out[["feature", "mean_target0", "mean_target1", "delta", "cohens_d"]].head(top_k).to_string(index=False))


def try_logistic_probe(df: pd.DataFrame, target: str) -> None:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print("\nscikit-learn not installed; skipping logistic probe.")
        print(f"Reason: {e}")
        return

    feature_cols = get_candidate_feature_columns(df)
    if len(feature_cols) == 0:
        print("\nNo numeric feature columns available for logistic probe.")
        return

    work = df.copy()
    work = work.dropna(subset=[target])

    # Need both classes
    if work[target].nunique() < 2:
        print(f"\nTarget '{target}' has fewer than 2 classes; skipping logistic probe.")
        return

    X = work[feature_cols]
    y = work[target].astype(int)

    # Very small datasets can break stratification
    stratify = y if y.value_counts().min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.35,
            random_state=42,
            stratify=stratify,
        )
    except Exception as e:
        print("\nCould not create train/test split; skipping logistic probe.")
        print(f"Reason: {e}")
        return

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float("nan")

    print(f"\nTiny logistic probe for '{target}'")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Accuracy  : {acc:.3f}")
    print(f"ROC AUC   : {auc:.3f}" if not math.isnan(auc) else "ROC AUC   : nan")

    clf = pipe.named_steps["clf"]
    coef = clf.coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False)

    print("\nTop 15 logistic coefficients by absolute value")
    print(coef_df[["feature", "coefficient"]].head(15).to_string(index=False))


def print_core_feature_summary(df: pd.DataFrame) -> None:
    core_cols = [
        "mean_selected_logprob",
        "min_selected_logprob",
        "mean_token_entropy",
        "max_token_entropy",
        "mean_attention_entropy_all_layers",
        "mean_attention_maxprob_all_layers",
    ]
    present = [c for c in core_cols if c in df.columns]
    if not present:
        return

    print("\nCore feature summary")
    print(df[present].describe().T.to_string())


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.parsed_only:
        if "parse_success" not in df.columns:
            raise ValueError("--parsed_only requested but 'parse_success' column not found.")
        df = df[df["parse_success"] == 1].copy()

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    print(f"Loaded: {input_path.resolve()}")
    print_basic_counts(df, args.target)
    print_output_format_breakdown(df)
    print_core_feature_summary(df)
    compare_feature_means(df, args.target, top_k=15)
    try_logistic_probe(df, args.target)


if __name__ == "__main__":
    main()