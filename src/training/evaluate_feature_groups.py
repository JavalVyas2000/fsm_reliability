from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class EvalResult:
    feature_group: str
    n_rows: int
    n_features: int
    train_size: int
    test_size: int
    accuracy: float
    roc_auc: float


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

    # Token-confidence features
    # "num_generated_tokens" is somewhat output-structure-ish, so we expose both:
    # token_only           -> includes it
    # token_only_strict    -> excludes it
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

    # Pooled attention summaries
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

    # Region-attention summaries only (aggregate summaries, no token counts)
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

    # Richer layerwise region-attention features
    # Keep only true attention-derived columns, exclude region_*_token_count
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
        # Token families
        "token_only": sorted(set(token)),
        "token_only_strict": sorted(set(token_strict)),

        # Attention families
        "pooled_attention_only": sorted(set(pooled_attention)),
        "region_summary_only": sorted(set(region_summary)),
        "attention_only_all": sorted(set(pooled_attention + region_summary)),
        "region_attention_rich_only": sorted(set(region_layerwise)),

        # Fusion families
        "token_plus_region_summary": sorted(set(token + region_summary)),
        "token_strict_plus_region_summary": sorted(set(token_strict + region_summary)),
        "internal_only_combined": sorted(set(token + pooled_attention + region_summary)),
        "internal_only_combined_strict": sorted(set(token_strict + pooled_attention + region_summary)),
        "internal_only_rich_attention": sorted(set(token + pooled_attention + region_layerwise)),
        "internal_only_rich_attention_strict": sorted(set(token_strict + pooled_attention + region_layerwise)),
    }

    return groups


def run_logistic_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    random_state: int = 42,
) -> EvalResult:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    work = work.dropna(subset=[target])

    if len(feature_cols) == 0:
        raise ValueError("feature_cols is empty.")

    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = work[feature_cols].copy()
    y = work[target].astype(int)

    if y.nunique() < 2:
        raise ValueError(f"Target '{target}' has fewer than 2 classes.")

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=random_state,
        stratify=stratify,
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    return EvalResult(
        feature_group="",
        n_rows=len(work),
        n_features=len(feature_cols),
        train_size=len(X_train),
        test_size=len(X_test),
        accuracy=acc,
        roc_auc=auc,
    )


def evaluate_feature_groups(
    df: pd.DataFrame,
    target: str,
    random_state: int = 42,
    min_features: int = 1,
) -> pd.DataFrame:
    """
    Quick internal-only ablation runner.

    Parameters
    ----------
    df : pd.DataFrame
        Logged feature table.
    target : str
        Target column, e.g. 'valid_path' or 'optimal_path'.
    random_state : int
        Train/test split seed.
    min_features : int
        Skip groups with fewer than this many available features.
    """
    groups = get_feature_groups(df)
    results = []

    for group_name, feature_cols in groups.items():
        feature_cols = [c for c in feature_cols if c in df.columns]

        if len(feature_cols) < min_features:
            continue

        try:
            res = run_logistic_eval(
                df=df,
                feature_cols=feature_cols,
                target=target,
                random_state=random_state,
            )
            res.feature_group = group_name
            results.append(res.__dict__)
        except Exception as e:
            results.append(
                {
                    "feature_group": group_name,
                    "n_rows": int(df[target].notna().sum()) if target in df.columns else len(df),
                    "n_features": len(feature_cols),
                    "train_size": None,
                    "test_size": None,
                    "accuracy": None,
                    "roc_auc": None,
                    "error": str(e),
                }
            )

    out = pd.DataFrame(results)
    if len(out) > 0 and "roc_auc" in out.columns:
        out = out.sort_values(
            by=["roc_auc", "accuracy"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    return out