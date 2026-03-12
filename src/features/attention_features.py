from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def _entropy_from_attention_probs(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy over the last dimension of attention probabilities.

    attn shape expected:
        [..., seq_len]
    """
    eps = 1e-12
    attn = attn.clamp(min=eps)
    return -(attn * torch.log(attn)).sum(dim=-1)


def extract_attention_features(
    attentions: Optional[List[torch.Tensor]],
) -> Dict[str, Optional[float]]:
    """
    Extract pooled attention features from a list of attention tensors.

    attentions:
        List of tensors, one per layer.
        Typical shape per layer:
            [batch_size, num_heads, seq_len, seq_len]

    Returns:
        Compact pooled features suitable for pilot experiments.
    """
    features: Dict[str, Optional[float]] = {
        "num_attention_layers": 0,
        "mean_attention_entropy_all_layers": None,
        "min_attention_entropy_all_layers": None,
        "max_attention_entropy_all_layers": None,
        "mean_attention_maxprob_all_layers": None,
        "min_attention_maxprob_all_layers": None,
        "max_attention_maxprob_all_layers": None,
    }

    if not attentions:
        return features

    layer_mean_entropies: List[float] = []
    layer_mean_maxprobs: List[float] = []

    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn shape: [1, heads, seq_len, seq_len]
        if layer_attn is None:
            continue

        # Remove batch dimension
        attn = layer_attn[0]  # [heads, seq_len, seq_len]

        # Entropy per head/query position
        ent = _entropy_from_attention_probs(attn)  # [heads, seq_len]
        maxprob = attn.max(dim=-1).values  # [heads, seq_len]

        layer_mean_entropies.append(ent.mean().item())
        layer_mean_maxprobs.append(maxprob.mean().item())

        features[f"layer_{layer_idx}_mean_attention_entropy"] = ent.mean().item()
        features[f"layer_{layer_idx}_mean_attention_maxprob"] = maxprob.mean().item()

    if not layer_mean_entropies:
        return features

    features["num_attention_layers"] = len(layer_mean_entropies)
    features["mean_attention_entropy_all_layers"] = sum(layer_mean_entropies) / len(layer_mean_entropies)
    features["min_attention_entropy_all_layers"] = min(layer_mean_entropies)
    features["max_attention_entropy_all_layers"] = max(layer_mean_entropies)

    features["mean_attention_maxprob_all_layers"] = sum(layer_mean_maxprobs) / len(layer_mean_maxprobs)
    features["min_attention_maxprob_all_layers"] = min(layer_mean_maxprobs)
    features["max_attention_maxprob_all_layers"] = max(layer_mean_maxprobs)

    return features