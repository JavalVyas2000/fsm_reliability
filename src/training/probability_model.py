from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EXCLUDE = {
    # targets / labels
    "valid_path",
    "optimal_path",
    "is_correct",
    "correct",
    "label",
    "target",
    "y_true",
    # ids / text / prompt-like columns
    "instance_id",
    "sample_id",
    "id",
    "prompt",
    "graph_text",
    "graph_json",
    "generated_text",
    "prediction_text",
    "raw_output",
    "output_format",
    "parse_mode",
    "ground_truth_shortest_path",
    "gold_path",
    # parsed outputs / posthoc predictions
    "parsed_prediction",
    "parsed_path",
    "prob_correct",
    "prob_wrong",
    "route",
    "y_prob",
    "y_pred",
    # metadata we do NOT want as learned inputs
    "split",
    "start",
    "goal",
    "num_nodes",
    "edge_prob",
    "num_edges",
    "shortest_length",
    "path_length",
    "length_gap",
    "max_new_tokens_used",
    # parse-related flags / quasi-labels
    "parse_success",
    "strict_json_success",
    # region token counts
    "prompt_token_count",
    "full_token_count",
    "region_prompt_token_count",
    "region_full_token_count",
    "region_graph_token_count",
    "region_start_token_count",
    "region_goal_token_count",
    "region_output_token_count",
}


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


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    if n == 0:
        return float("nan")

    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Internal-only feature groups.

    Excluded on purpose:
    - task metadata: num_nodes, num_edges, shortest_length, etc.
    - output-derived structure: path_length, length_gap, etc.
    - region token counts

    Included:
    - token-confidence features
    - pooled attention summaries
    - region-attention summaries
    - optional richer layerwise region-attention features
    """
    cols = set(df.columns)

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
        "token_only": sorted(set(token)),
        "token_only_strict": sorted(set(token_strict)),
        "pooled_attention_only": sorted(set(pooled_attention)),
        "region_summary_only": sorted(set(region_summary)),
        "attention_only_all": sorted(set(pooled_attention + region_summary)),
        "region_attention_rich_only": sorted(set(region_layerwise)),
        "token_plus_region_summary": sorted(set(token + region_summary)),
        "token_strict_plus_region_summary": sorted(set(token_strict + region_summary)),
        "internal_only_combined": sorted(set(token + pooled_attention + region_summary)),
        "internal_only_combined_strict": sorted(set(token_strict + pooled_attention + region_summary)),
        "internal_only_rich_attention": sorted(set(token + pooled_attention + region_layerwise)),
        "internal_only_rich_attention_strict": sorted(set(token_strict + pooled_attention + region_layerwise)),
    }

    return groups


def choose_feature_columns(
    df: pd.DataFrame,
    target: str,
    feature_group: str = "internal_only_combined",
) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    allowed_numeric = [
        c for c in numeric_cols
        if c not in DEFAULT_EXCLUDE and c != target
    ]

    if feature_group == "all":
        return sorted(allowed_numeric)

    groups = get_feature_groups(df)
    if feature_group not in groups:
        raise ValueError(f"Unknown feature_group: {feature_group}")

    filtered = [c for c in groups[feature_group] if c in allowed_numeric]

    if not filtered:
        raise ValueError(
            f"No columns matched feature_group='{feature_group}'. "
            f"Available allowed numeric columns include: {allowed_numeric[:40]}"
        )

    return filtered


def build_selective_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

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


def fit_probability_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_isotonic_calibrator(
    base_model: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> IsotonicRegression:
    raw_val = base_model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, y_val)
    return calibrator


def predict_calibrated_prob(
    base_model: Pipeline,
    calibrator: IsotonicRegression,
    X: pd.DataFrame,
) -> np.ndarray:
    raw = base_model.predict_proba(X)[:, 1]
    prob = calibrator.predict(raw)
    return np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)


def train_probability_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
) -> Tuple[ProbabilityEvalResult, pd.DataFrame, List[str], Tuple[Pipeline, IsotonicRegression]]:
    feature_cols = _align_feature_columns(train_df, val_df, test_df, feature_cols)
    if len(feature_cols) == 0:
        raise ValueError("No common feature columns across train/val/test.")

    train_df = train_df.dropna(subset=[target]).copy()
    val_df = val_df.dropna(subset=[target]).copy()
    test_df = test_df.dropna(subset=[target]).copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target].astype(int).to_numpy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target].astype(int).to_numpy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Target must contain both classes in train, val, and test.")

    base_model = fit_probability_model(X_train, y_train)
    calibrator = fit_isotonic_calibrator(base_model, X_val, y_val)

    y_prob = predict_calibrated_prob(base_model, calibrator, X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else float("nan")
    brier = float(brier_score_loss(y_test, y_prob))
    ece = expected_calibration_error(y_test, y_prob, n_bins=10)

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

    pred_df = test_df.copy()
    pred_df["y_true"] = y_test
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df["prob_correct"] = y_prob
    pred_df["prob_wrong"] = 1.0 - y_prob

    return result, pred_df, feature_cols, (base_model, calibrator)


def evaluate_feature_groups(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_groups_to_run: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compare all internal-only feature groups on the same train/val/test split.

    This is useful as a one-shot sanity-check ablation.
    Main repeated runs can still be driven by 06_calibrate_probability_model.py.
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