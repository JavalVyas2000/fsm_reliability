from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def extract_token_confidence_features(
    generation_scores: Optional[List[torch.Tensor]],
    generated_ids: Optional[List[int]] = None,
) -> Dict[str, Optional[float]]:
    """
    Extract simple token-level confidence features from generation scores.

    generation_scores:
        A list of tensors, one per generated token step.
        Each tensor is shape [batch_size, vocab_size].

    generated_ids:
        The generated token ids corresponding to each step.

    Returns:
        Dictionary of pooled confidence features.
    """
    features: Dict[str, Optional[float]] = {
        "num_generated_tokens": 0,
        "mean_selected_logprob": None,
        "min_selected_logprob": None,
        "max_selected_logprob": None,
        "mean_token_entropy": None,
        "max_token_entropy": None,
        "min_token_entropy": None,
    }

    if not generation_scores or not generated_ids:
        return features

    if len(generation_scores) != len(generated_ids):
        # Mismatch can happen in some generation edge cases.
        n = min(len(generation_scores), len(generated_ids))
        generation_scores = generation_scores[:n]
        generated_ids = generated_ids[:n]

    selected_logprobs: List[float] = []
    entropies: List[float] = []

    for step_scores, token_id in zip(generation_scores, generated_ids):
        # step_scores shape: [1, vocab_size]
        logits = step_scores[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        selected_lp = log_probs[token_id].item()
        entropy = -(probs * log_probs).sum().item()

        selected_logprobs.append(selected_lp)
        entropies.append(entropy)

    if not selected_logprobs:
        return features

    features["num_generated_tokens"] = len(selected_logprobs)
    features["mean_selected_logprob"] = _safe_float(sum(selected_logprobs) / len(selected_logprobs))
    features["min_selected_logprob"] = _safe_float(min(selected_logprobs))
    features["max_selected_logprob"] = _safe_float(max(selected_logprobs))

    features["mean_token_entropy"] = _safe_float(sum(entropies) / len(entropies))
    features["max_token_entropy"] = _safe_float(max(entropies))
    features["min_token_entropy"] = _safe_float(min(entropies))

    return features