from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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


def _safe_auc(y_true, probs) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, probs))


def _build_joint_stratify_labels(work: pd.DataFrame, y: pd.Series) -> Optional[pd.Series]:
    """
    Build a joint stratification label so random train/test splits preserve both:
    - class balance
    - pooled node-size composition

    Falls back to y if the joint groups are too small.
    """
    if "num_nodes" not in work.columns:
        return y if y.value_counts().min() >= 2 else None

    joint = work["num_nodes"].astype(str) + "__" + y.astype(str)

    if joint.value_counts().min() >= 2:
        return joint

    return y if y.value_counts().min() >= 2 else None


def run_logistic_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    random_state: int = 42,
    test_size: float = 0.35,
    stratify_by_num_nodes: bool = True,
) -> EvalResult:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
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

    if stratify_by_num_nodes:
        stratify = _build_joint_stratify_labels(work, y)
    else:
        stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
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
    auc = _safe_auc(y_test, probs)

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
    stratify_by_num_nodes: bool = True,
) -> pd.DataFrame:
    """
    Quick internal-only ablation runner on one pooled dataframe.

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
    stratify_by_num_nodes : bool
        Preserve mixed node-size composition in random train/test splitting when possible.
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
                stratify_by_num_nodes=stratify_by_num_nodes,
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


def evaluate_per_num_nodes(
    df: pd.DataFrame,
    target: str,
    random_state: int = 42,
    min_features: int = 1,
) -> pd.DataFrame:
    """
    Optional analysis helper: run the same ablation separately for each node count.

    This does NOT change the main pooled-model workflow. It is only for reporting.
    """
    if "num_nodes" not in df.columns:
        raise ValueError("num_nodes column not found in dataframe.")

    outputs = []
    for n in sorted(df["num_nodes"].dropna().unique()):
        sub = df[df["num_nodes"] == n].copy()
        res = evaluate_feature_groups(
            df=sub,
            target=target,
            random_state=random_state,
            min_features=min_features,
            stratify_by_num_nodes=False,
        )
        res["eval_num_nodes"] = int(n)
        outputs.append(res)

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)