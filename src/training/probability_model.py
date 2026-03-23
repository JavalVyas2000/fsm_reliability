from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProbabilityEvalResult:
    feature_group: str
    train_rows: int
    val_rows: int
    test_rows: int
    n_features: int
    accuracy: float
    roc_auc: float
    brier_score: float
    ece: float


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Internal-only feature groups.

    Excluded on purpose:
    - task metadata: num_nodes, num_edges, shortest_length
    - output-derived structure: path_length
    - region token counts: region_*_token_count

    Included:
    - token-confidence features
    - pooled attention summaries
    - region-attention summaries
    - optional richer layerwise region-attention features
    """
    cols = set(df.columns)

    # ---------------------------
    # Token-confidence features
    # ---------------------------
    token = [
        c for c in [
            "num_generated_tokens",
            "mean_selected_logprob",
            "min_selected_logprob",
            "max_selected_logprob",
            "mean_token_entropy",
            "max_token_entropy",
            "min_token_entropy",
        ]
        if c in cols
    ]

    # Strict version excludes num_generated_tokens
    token_strict = [
        c for c in [
            "mean_selected_logprob",
            "min_selected_logprob",
            "max_selected_logprob",
            "mean_token_entropy",
            "max_token_entropy",
            "min_token_entropy",
        ]
        if c in cols
    ]

    # ---------------------------
    # Pooled attention summaries
    # ---------------------------
    pooled_attention = [
        c for c in df.columns
        if (
            c.startswith("layer_")
            and (
                c.endswith("_mean_attention_entropy")
                or c.endswith("_mean_attention_maxprob")
            )
        )
        or c in {
            "num_attention_layers",
            "mean_attention_entropy_all_layers",
            "min_attention_entropy_all_layers",
            "max_attention_entropy_all_layers",
            "mean_attention_maxprob_all_layers",
            "min_attention_maxprob_all_layers",
            "max_attention_maxprob_all_layers",
        }
    ]

    # ---------------------------
    # Aggregate region summaries
    # ---------------------------
    region_summary = [
        c for c in [
            "mean_output_to_graph_attn_all_layers",
            "mean_output_to_start_attn_all_layers",
            "mean_output_to_goal_attn_all_layers",
            "mean_output_to_prompt_attn_all_layers",
            "mean_output_to_output_attn_all_layers",
            "mean_output_prompt_vs_output_attn_ratio_all_layers",
            "mean_output_goal_vs_start_attn_ratio_all_layers",
        ]
        if c in cols
    ]

    # ---------------------------
    # Rich layerwise region-attention features
    # Excludes region token counts
    # ---------------------------
    region_layerwise = [
        c for c in df.columns
        if (
            c.startswith("layer_")
            and (
                "_output_to_graph_attn" in c
                or "_output_to_start_attn" in c
                or "_output_to_goal_attn" in c
                or "_output_to_prompt_attn" in c
                or "_output_to_output_attn" in c
                or "_output_prompt_vs_output_attn_ratio" in c
                or "_output_goal_vs_start_attn_ratio" in c
            )
        )
    ]

    groups = {
        # Token groups
        "token_only": sorted(set(token)),
        "token_only_strict": sorted(set(token_strict)),

        # Attention groups
        "pooled_attention_only": sorted(set(pooled_attention)),
        "region_summary_only": sorted(set(region_summary)),
        "attention_only_all": sorted(set(pooled_attention + region_summary)),
        "region_attention_rich_only": sorted(set(region_layerwise)),

        # Fusion groups
        "token_plus_region_summary": sorted(set(token + region_summary)),
        "token_strict_plus_region_summary": sorted(set(token_strict + region_summary)),
        "internal_only_combined": sorted(set(token + pooled_attention + region_summary)),
        "internal_only_combined_strict": sorted(set(token_strict + pooled_attention + region_summary)),
        "internal_only_rich_attention": sorted(set(token + pooled_attention + region_layerwise)),
        "internal_only_rich_attention_strict": sorted(set(token_strict + pooled_attention + region_layerwise)),
    }

    return groups


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    if n == 0:
        return float("nan")

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def build_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "empirical_accuracy": np.nan,
                }
            )
            continue

        rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "count": count,
                "mean_confidence": float(y_prob[mask].mean()),
                "empirical_accuracy": float(y_true[mask].mean()),
            }
        )

    return pd.DataFrame(rows)


def build_selective_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    rows = []
    n = len(y_true)

    for thr in thresholds:
        mask = y_prob >= thr
        covered = int(mask.sum())
        coverage = covered / n if n > 0 else 0.0

        if covered == 0:
            selective_acc = np.nan
            avg_conf = np.nan
        else:
            selective_acc = float(y_true[mask].mean())
            avg_conf = float(y_prob[mask].mean())

        rows.append(
            {
                "threshold": thr,
                "covered": covered,
                "coverage": coverage,
                "selective_accuracy": selective_acc,
                "avg_confidence_of_accepted": avg_conf,
            }
        )

    return pd.DataFrame(rows)


def build_selective_table_by_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    coverages: List[float] | None = None,
) -> pd.DataFrame:
    if coverages is None:
        coverages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    n = len(y_true)

    if n == 0:
        return pd.DataFrame(
            columns=[
                "target_coverage",
                "actual_coverage",
                "covered",
                "threshold_used",
                "selective_accuracy",
                "avg_confidence_of_accepted",
            ]
        )

    order = np.argsort(-y_prob)

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        chosen_idx = order[:k]

        threshold_used = float(y_prob[chosen_idx].min())
        actual_coverage = k / n
        selective_acc = float(y_true[chosen_idx].mean())
        avg_conf = float(y_prob[chosen_idx].mean())

        rows.append(
            {
                "target_coverage": cov,
                "actual_coverage": actual_coverage,
                "covered": k,
                "threshold_used": threshold_used,
                "selective_accuracy": selective_acc,
                "avg_confidence_of_accepted": avg_conf,
            }
        )

    return pd.DataFrame(rows)


def _align_feature_columns(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> List[str]:
    common = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    return [c for c in feature_cols if c in common]


def train_probability_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_cols = _align_feature_columns(train_df, val_df, test_df, feature_cols)
    if len(feature_cols) == 0:
        raise ValueError("No common feature columns across train/val/test.")

    train_df = train_df.dropna(subset=[target]).copy()
    val_df = val_df.dropna(subset=[target]).copy()
    test_df = test_df.dropna(subset=[target]).copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target].astype(int)

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target].astype(int)

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target].astype(int)

    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Target must contain both classes in train, val, and test.")

    base_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )

    base_pipeline.fit(X_train, y_train)

    calibrator = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)

    y_prob = calibrator.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    ece = expected_calibration_error(y_test.to_numpy(), y_prob, n_bins=10)

    result = ProbabilityEvalResult(
        feature_group="",
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        n_features=len(feature_cols),
        accuracy=accuracy,
        roc_auc=roc_auc,
        brier_score=brier,
        ece=ece,
    )

    pred_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    return result, pred_df, feature_cols, calibrator


def evaluate_feature_groups(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_groups_to_run: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compare all internal-only feature groups on the same train/val/test split.

    This is useful as a one-shot sanity-check ablation. Your main repeated
    experiment can still be run via 06_calibrate_probability_model.py.
    """
    groups = get_feature_groups(train_df)

    if feature_groups_to_run is not None:
        missing_groups = [g for g in feature_groups_to_run if g not in groups]
        if missing_groups:
            raise ValueError(f"Unknown feature groups requested: {missing_groups}")
        groups = {k: v for k, v in groups.items() if k in feature_groups_to_run}

    rows = []

    for group_name, feature_cols in groups.items():
        if len(feature_cols) == 0:
            rows.append(
                {
                    "feature_group": group_name,
                    "train_rows": None,
                    "val_rows": None,
                    "test_rows": None,
                    "n_features": 0,
                    "accuracy": None,
                    "roc_auc": None,
                    "brier_score": None,
                    "ece": None,
                    "error": "empty feature group",
                }
            )
            continue

        try:
            result, _, used_feature_cols, _ = train_probability_model(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target=target,
            )

            result.feature_group = group_name

            rows.append(
                {
                    "feature_group": result.feature_group,
                    "train_rows": result.train_rows,
                    "val_rows": result.val_rows,
                    "test_rows": result.test_rows,
                    "n_features": len(used_feature_cols),
                    "accuracy": result.accuracy,
                    "roc_auc": result.roc_auc,
                    "brier_score": result.brier_score,
                    "ece": result.ece,
                    "error": None,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "feature_group": group_name,
                    "train_rows": None,
                    "val_rows": None,
                    "test_rows": None,
                    "n_features": len(feature_cols),
                    "accuracy": None,
                    "roc_auc": None,
                    "brier_score": None,
                    "ece": None,
                    "error": str(e),
                }
            )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            by=["roc_auc", "accuracy"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    return out