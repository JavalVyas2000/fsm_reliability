from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
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
    cols = set(df.columns)

    difficulty = [
        c for c in ["num_nodes", "num_edges", "shortest_length", "path_length"]
        if c in cols
    ]

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
            "mean_attention_entropy_all_layers",
            "min_attention_entropy_all_layers",
            "max_attention_entropy_all_layers",
            "mean_attention_maxprob_all_layers",
            "min_attention_maxprob_all_layers",
            "max_attention_maxprob_all_layers",
        }
    ]

    region_attention = [
        c for c in df.columns
        if (
            "_output_to_graph_attn" in c
            or "_output_to_start_attn" in c
            or "_output_to_goal_attn" in c
            or "_output_to_prompt_attn" in c
            or "_output_to_output_attn" in c
            or "_output_prompt_vs_output_attn_ratio" in c
            or "_output_goal_vs_start_attn_ratio" in c
        )
        or c in {
            "mean_output_to_graph_attn_all_layers",
            "mean_output_to_start_attn_all_layers",
            "mean_output_to_goal_attn_all_layers",
            "mean_output_to_prompt_attn_all_layers",
            "mean_output_to_output_attn_all_layers",
            "mean_output_prompt_vs_output_attn_ratio_all_layers",
            "mean_output_goal_vs_start_attn_ratio_all_layers",
            "region_prompt_token_count",
            "region_full_token_count",
            "region_graph_token_count",
            "region_start_token_count",
            "region_goal_token_count",
            "region_output_token_count",
        }
    ]

    groups = {
        "difficulty_only": sorted(set(difficulty)),
        "token_only": sorted(set(token)),
        "pooled_attention_only": sorted(set(pooled_attention)),
        "region_attention_only": sorted(set(region_attention)),
        "token_plus_region": sorted(set(token + region_attention)),
        "difficulty_plus_token": sorted(set(difficulty + token)),
        "difficulty_plus_region": sorted(set(difficulty + region_attention)),
        "full_combined": sorted(set(difficulty + token + pooled_attention + region_attention)),
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

    X = work[feature_cols].copy()
    y = work[target].astype(int)

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
) -> pd.DataFrame:
    groups = get_feature_groups(df)
    results = []

    for group_name, feature_cols in groups.items():
        if len(feature_cols) == 0:
            continue

        res = run_logistic_eval(
            df=df,
            feature_cols=feature_cols,
            target=target,
        )
        res.feature_group = group_name
        results.append(res.__dict__)

    out = pd.DataFrame(results)
    if len(out) > 0:
        out = out.sort_values(["roc_auc", "accuracy"], ascending=False).reset_index(drop=True)
    return out