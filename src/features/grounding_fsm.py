"""
FSM attention grounding (pre-registered in docs/grounding_attention_prereg.md).

For each step u -> v of the answer path: where did the model look among the instance
graph's adjacency lines while writing node v? The line `u: [...]` decides whether the
step is legal.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.models.inference_v2 import layer_tag

LINE_RE = re.compile(r"^(\d+):")


def graph_line_spans(rendered: str, graph_span: Tuple[int, int], graph_text: str) -> Dict[int, Tuple[int, int]]:
    """Char span of each adjacency line of the instance graph inside the rendered prompt."""
    gs, ge = graph_span
    assert rendered[gs:ge] == graph_text, "graph region does not match graph_text"
    spans, pos = {}, gs
    for line in graph_text.split("\n"):
        m = LINE_RE.match(line)
        if m:
            spans[int(m.group(1))] = (pos, pos + len(line))
        pos += len(line) + 1
    return spans


def token_line_labels(offsets: Sequence[Tuple[int, int]], line_spans: Dict[int, Tuple[int, int]]) -> np.ndarray:
    """Per prompt token: the graph line (node id) with the largest char overlap, else -1."""
    out = np.full(len(offsets), -1, dtype=np.int64)
    for i, (a, b) in enumerate(offsets):
        best, best_ov = -1, 0
        for u, (s, e) in line_spans.items():
            ov = max(0, min(b, e) - max(a, s))
            if ov > best_ov:
                best, best_ov = u, ov
        out[i] = best
    return out


def path_node_token_rows(tokenizer, answer_ids: List[int], expected: List[int]) -> Optional[List[List[int]]]:
    """
    For each node of the answer path: the answer-token indices spelling that number.
    Returns None if the numbers found in the path array do not match the parsed path.
    """
    text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    ends = [len(tokenizer.decode(answer_ids[: i + 1], skip_special_tokens=True)) for i in range(len(answer_ids))]
    starts = [0] + ends[:-1]
    key = text.find('"path"')
    lb = text.find("[", key if key >= 0 else 0)
    rb = text.find("]", lb)
    if lb < 0 or rb < 0:
        return None
    nums = [(int(m.group()), lb + m.start(), lb + m.end()) for m in re.finditer(r"-?\d+", text[lb:rb])]
    if [n for n, _, _ in nums] != list(expected):
        return None
    rows = []
    for _, cs, ce in nums:
        idx = [i for i in range(len(answer_ids)) if min(ends[i], ce) - max(starts[i], cs) > 0]
        if not idx:
            return None
        rows.append(idx)
    return rows


def step_grounding(
    attn_by_block: Dict[int, torch.Tensor],
    blocks: Sequence[int],
    rel_layers: Sequence[float],
    key_lines: np.ndarray,
    node_rows: List[List[int]],
    path: List[int],
    adjacency: Dict[int, List[int]],
) -> List[Dict]:
    """
    Per-step measurements (share, share_hmax, top1) at each selected layer, plus, for the
    interpretable account, the top-attended line per layer and the layer-mean share of
    every graph line.
    """
    line_ids = sorted(int(u) for u in set(key_lines.tolist()) if u >= 0)
    steps = []
    for i in range(1, len(path)):
        u, v = path[i - 1], path[i]
        rows = node_rows[i]
        rec = {"u": u, "v": v, "valid": int(u in adjacency and v in adjacency[u]), "u_in_graph": u in line_ids}
        line_share_acc = {lid: [] for lid in line_ids}
        for r, l in zip(rel_layers, blocks):
            a = attn_by_block[l][0][:, rows, : len(key_lines)].float()  # [H, r, P] prompt keys only
            masses = {}
            for lid in line_ids:
                m = torch.as_tensor(key_lines == lid, device=a.device)
                masses[lid] = a[..., m].sum(-1)  # [H, r]
            tot = torch.stack(list(masses.values())).sum(0)  # [H, r]
            tag = layer_tag(r)
            if not rec["u_in_graph"] or float(tot.mean()) <= 0:
                rec[f"{tag}_share"] = rec[f"{tag}_share_hmax"] = rec[f"{tag}_top1"] = float("nan")
                continue
            mu = masses[u]
            rec[f"{tag}_share"] = float(mu.mean() / tot.mean())
            rec[f"{tag}_share_hmax"] = float((mu.mean(1) / tot.mean(1).clamp_min(1e-12)).max())
            means = {lid: float(ms.mean()) for lid, ms in masses.items()}
            top = max(means, key=means.get)
            rec[f"{tag}_top1"] = float(top == u)
            rec[f"{tag}_top_line"] = int(top)
            tm = float(tot.mean())
            for lid in line_ids:
                line_share_acc[lid].append(means[lid] / tm)
        rec["line_share_layer_mean"] = {str(lid): float(np.mean(v)) for lid, v in line_share_acc.items() if v}
        steps.append(rec)
    return steps


def candidate_features(steps: List[Dict], rel_layers: Sequence[float]) -> Dict[str, float]:
    """Pre-registered candidate-level grounding features (NaN if there is no step)."""
    tags = [layer_tag(r) for r in rel_layers]
    nan = float("nan")
    names = ["grd_min_share", "grd_mean_share", "grd_top1_frac", "grd_first_share", "grd_min_share_hmax"]
    names += [f"grd_{t}_min_share" for t in tags] + [f"grd_{t}_top1_frac" for t in tags]
    if not steps:
        return {n: nan for n in names}

    def arr(key):
        return np.array([s[key] for s in steps], dtype=float)

    share = np.nanmean(np.stack([arr(f"{t}_share") for t in tags]), axis=0)  # per step, mean over layers
    hmax = np.nanmean(np.stack([arr(f"{t}_share_hmax") for t in tags]), axis=0)
    top1 = np.nanmean(np.stack([arr(f"{t}_top1") for t in tags]), axis=0)
    f = {
        "grd_min_share": float(np.nanmin(share)) if np.isfinite(share).any() else nan,
        "grd_mean_share": float(np.nanmean(share)) if np.isfinite(share).any() else nan,
        "grd_top1_frac": float(np.nanmean(top1)) if np.isfinite(top1).any() else nan,
        "grd_first_share": float(share[0]) if np.isfinite(share[0]) else nan,
        "grd_min_share_hmax": float(np.nanmin(hmax)) if np.isfinite(hmax).any() else nan,
    }
    for t in tags:
        s, tp = arr(f"{t}_share"), arr(f"{t}_top1")
        f[f"grd_{t}_min_share"] = float(np.nanmin(s)) if np.isfinite(s).any() else nan
        f[f"grd_{t}_top1_frac"] = float(np.nanmean(tp)) if np.isfinite(tp).any() else nan
    return f
