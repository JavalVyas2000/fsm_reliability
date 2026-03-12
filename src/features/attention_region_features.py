from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PromptRegions:
    prompt_token_count: int
    full_token_count: int
    graph_mask: torch.Tensor          # [seq_len] bool
    start_mask: torch.Tensor          # [seq_len] bool
    goal_mask: torch.Tensor           # [seq_len] bool
    prompt_mask: torch.Tensor         # [seq_len] bool
    output_mask: torch.Tensor         # [seq_len] bool


def _find_subsequence_positions(sequence: List[int], subseq: List[int]) -> List[Tuple[int, int]]:
    """
    Return all (start, end) token index spans where subseq appears in sequence.
    end is exclusive.
    """
    if not subseq or len(subseq) > len(sequence):
        return []

    spans = []
    n = len(sequence)
    m = len(subseq)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subseq:
            spans.append((i, i + m))
    return spans


def _char_span_to_token_mask(
    offsets: List[Tuple[int, int]],
    seq_len: int,
    char_start: int,
    char_end: int,
) -> torch.Tensor:
    """
    Convert a character span into a token mask using tokenizer offset mappings.
    Only applies to prompt tokens.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for idx, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        # overlap
        if not (e <= char_start or s >= char_end):
            mask[idx] = True
    return mask


def _find_graph_char_span(prompt_text: str) -> Optional[Tuple[int, int]]:
    """
    Assumes prompt contains:
        Graph:
        ...
        
        Task:
    """
    graph_key = "Graph:\n"
    task_key = "\n\nTask:"

    g = prompt_text.find(graph_key)
    if g == -1:
        return None

    graph_start = g + len(graph_key)
    task_start = prompt_text.find(task_key, graph_start)
    if task_start == -1:
        return None

    return graph_start, task_start


def _find_task_number_char_span(prompt_text: str, number: int, which: str) -> Optional[Tuple[int, int]]:
    """
    Try to locate the start/goal number inside the task sentence:
        Find one valid path from state X to state Y.
    """
    task_prefix = "Find one valid path from state "
    idx = prompt_text.find(task_prefix)
    if idx == -1:
        return None

    task_segment = prompt_text[idx : idx + 200]
    num_str = str(number)

    if which == "start":
        needle = f"from state {num_str}"
    elif which == "goal":
        needle = f"to state {num_str}"
    else:
        raise ValueError("which must be 'start' or 'goal'")

    local_idx = task_segment.find(needle)
    if local_idx == -1:
        return None

    abs_idx = idx + local_idx + len(needle) - len(num_str)
    return abs_idx, abs_idx + len(num_str)


def build_prompt_regions(
    tokenizer,
    prompt_text: str,
    full_input_ids: List[int],
    prompt_len: int,
    start: int,
    goal: int,
) -> PromptRegions:
    """
    Build boolean masks over the full prompt+output token sequence for:
    - graph region
    - start token in task sentence
    - goal token in task sentence
    - prompt tokens
    - output tokens
    """
    seq_len = len(full_input_ids)

    # Fast tokenizers provide offset mapping on raw prompt text.
    prompt_enc = tokenizer(
        prompt_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    prompt_input_ids = prompt_enc["input_ids"]
    offsets = prompt_enc["offset_mapping"]

    # Sometimes special tokenization lengths differ in edge cases; align defensively.
    usable_prompt_len = min(prompt_len, len(prompt_input_ids), seq_len)

    prompt_mask = torch.zeros(seq_len, dtype=torch.bool)
    prompt_mask[:usable_prompt_len] = True

    output_mask = torch.zeros(seq_len, dtype=torch.bool)
    output_mask[usable_prompt_len:] = True

    graph_mask = torch.zeros(seq_len, dtype=torch.bool)
    start_mask = torch.zeros(seq_len, dtype=torch.bool)
    goal_mask = torch.zeros(seq_len, dtype=torch.bool)

    graph_span = _find_graph_char_span(prompt_text)
    if graph_span is not None:
        g0, g1 = graph_span
        graph_mask[:usable_prompt_len] = _char_span_to_token_mask(
            offsets=offsets[:usable_prompt_len],
            seq_len=usable_prompt_len,
            char_start=g0,
            char_end=g1,
        )

    start_span = _find_task_number_char_span(prompt_text, start, which="start")
    if start_span is not None:
        s0, s1 = start_span
        start_mask[:usable_prompt_len] = _char_span_to_token_mask(
            offsets=offsets[:usable_prompt_len],
            seq_len=usable_prompt_len,
            char_start=s0,
            char_end=s1,
        )

    goal_span = _find_task_number_char_span(prompt_text, goal, which="goal")
    if goal_span is not None:
        g0, g1 = goal_span
        goal_mask[:usable_prompt_len] = _char_span_to_token_mask(
            offsets=offsets[:usable_prompt_len],
            seq_len=usable_prompt_len,
            char_start=g0,
            char_end=g1,
        )

    return PromptRegions(
        prompt_token_count=usable_prompt_len,
        full_token_count=seq_len,
        graph_mask=graph_mask,
        start_mask=start_mask,
        goal_mask=goal_mask,
        prompt_mask=prompt_mask,
        output_mask=output_mask,
    )


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def extract_attention_region_features(
    attentions: Optional[List[torch.Tensor]],
    regions: PromptRegions,
) -> Dict[str, Optional[float]]:
    """
    Compute attention-region features using generated/output tokens as queries.

    For each layer:
      - mean attention mass from output queries to graph tokens
      - mean attention mass from output queries to start token region
      - mean attention mass from output queries to goal token region
      - mean attention mass from output queries to prompt tokens
      - mean attention mass from output queries to output tokens

    Also returns pooled versions across layers.
    """
    features: Dict[str, Optional[float]] = {
        "region_prompt_token_count": float(regions.prompt_token_count),
        "region_full_token_count": float(regions.full_token_count),
        "region_graph_token_count": float(regions.graph_mask.sum().item()),
        "region_start_token_count": float(regions.start_mask.sum().item()),
        "region_goal_token_count": float(regions.goal_mask.sum().item()),
        "region_output_token_count": float(regions.output_mask.sum().item()),
        "mean_output_to_graph_attn_all_layers": None,
        "mean_output_to_start_attn_all_layers": None,
        "mean_output_to_goal_attn_all_layers": None,
        "mean_output_to_prompt_attn_all_layers": None,
        "mean_output_to_output_attn_all_layers": None,
        "mean_output_prompt_vs_output_attn_ratio_all_layers": None,
        "mean_output_goal_vs_start_attn_ratio_all_layers": None,
    }

    if not attentions:
        return features

    if regions.output_mask.sum().item() == 0:
        return features

    graph_vals = []
    start_vals = []
    goal_vals = []
    prompt_vals = []
    output_vals = []
    prompt_output_ratio_vals = []
    goal_start_ratio_vals = []

    output_query_idx = torch.where(regions.output_mask)[0]

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue

        # [1, heads, seq_len, seq_len] -> [heads, seq_len, seq_len]
        attn = layer_attn[0]

        # Restrict to output query positions: [heads, out_len, seq_len]
        out_attn = attn[:, output_query_idx, :]

        def masked_mass(mask: torch.Tensor) -> float:
            denom = mask.sum().item()
            if denom == 0:
                return 0.0
            # Sum probability mass onto masked key positions
            mass = out_attn[..., mask].sum(dim=-1)  # [heads, out_len]
            return float(mass.mean().item())

        graph_mass = masked_mass(regions.graph_mask)
        start_mass = masked_mass(regions.start_mask)
        goal_mass = masked_mass(regions.goal_mask)
        prompt_mass = masked_mass(regions.prompt_mask)
        output_mass = masked_mass(regions.output_mask)

        eps = 1e-8
        prompt_output_ratio = prompt_mass / (output_mass + eps)
        goal_start_ratio = goal_mass / (start_mass + eps)

        features[f"layer_{layer_idx}_output_to_graph_attn"] = graph_mass
        features[f"layer_{layer_idx}_output_to_start_attn"] = start_mass
        features[f"layer_{layer_idx}_output_to_goal_attn"] = goal_mass
        features[f"layer_{layer_idx}_output_to_prompt_attn"] = prompt_mass
        features[f"layer_{layer_idx}_output_to_output_attn"] = output_mass
        features[f"layer_{layer_idx}_output_prompt_vs_output_attn_ratio"] = prompt_output_ratio
        features[f"layer_{layer_idx}_output_goal_vs_start_attn_ratio"] = goal_start_ratio

        graph_vals.append(graph_mass)
        start_vals.append(start_mass)
        goal_vals.append(goal_mass)
        prompt_vals.append(prompt_mass)
        output_vals.append(output_mass)
        prompt_output_ratio_vals.append(prompt_output_ratio)
        goal_start_ratio_vals.append(goal_start_ratio)

    features["mean_output_to_graph_attn_all_layers"] = _safe_mean(graph_vals)
    features["mean_output_to_start_attn_all_layers"] = _safe_mean(start_vals)
    features["mean_output_to_goal_attn_all_layers"] = _safe_mean(goal_vals)
    features["mean_output_to_prompt_attn_all_layers"] = _safe_mean(prompt_vals)
    features["mean_output_to_output_attn_all_layers"] = _safe_mean(output_vals)
    features["mean_output_prompt_vs_output_attn_ratio_all_layers"] = _safe_mean(prompt_output_ratio_vals)
    features["mean_output_goal_vs_start_attn_ratio_all_layers"] = _safe_mean(goal_start_ratio_vals)

    return features