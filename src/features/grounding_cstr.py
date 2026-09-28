"""
CSTR field grounding (pre-registered in docs/grounding_cstr_prereg.md).

When the model writes each setpoint number, how much of its prompt attention (excluding
template and generated tokens) goes to the snapshot fields and KG relations that should
decide that setpoint?
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.models.inference_v2 import layer_tag

SNAPSHOT_FIELDS = ["time", "phase", "T_meas", "L_meas", "u_valve", "u_pump", "u_cool", "anomaly_ratio",
                   "violated_params", "control_zone", "control_reasons", "current setpoints", "symptoms"]
KG_TAGS = {"PO6": ["PO6"], "PO7": ["PO7"], "PO8": ["PO8", "PO1"]}
DECISIVE = {
    "Fin_sp": ["field:u_cool", "field:T_meas", "field:current setpoints", "kg:PO8"],
    "L_sp": ["field:L_meas", "field:u_pump", "field:current setpoints", "kg:PO6"],
    "T_sp": ["field:T_meas", "field:u_cool", "field:current setpoints", "kg:PO7"],
}
SETPOINTS = ("Fin_sp", "L_sp", "T_sp")


def _line_span(text: str, pos: int) -> Tuple[int, int]:
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e < 0 else e)


def cstr_regions(rendered: str, system: str, user: str) -> Dict[str, List[Tuple[int, int]]]:
    """Named char regions: field:<name> lines, kg:<PO> relation lines, snapshot, context (system+KG)."""
    regions: Dict[str, List[Tuple[int, int]]] = {}
    snap_at = rendered.find("CURRENT SNAPSHOT:")
    fb_at = rendered.find("EVALUATION FEEDBACK", snap_at)
    regions["snapshot"] = [(snap_at, fb_at)]
    block = rendered[snap_at:fb_at]
    for f in SNAPSHOT_FIELDS:
        m = re.search(r"-\s*" + re.escape(f) + r":", block)
        if m:
            s, e = _line_span(rendered, snap_at + m.start())
            regions[f"field:{f}"] = [(s, e)]
    sys_at = rendered.find(system.strip()[:200])
    sys_end = sys_at + len(system.strip())
    regions["context"] = [(sys_at, sys_end)]
    for tag, keys in KG_TAGS.items():
        spans = set()
        for k in keys:
            for m in re.finditer(re.escape(k), rendered[sys_at:sys_end]):
                spans.add(_line_span(rendered, sys_at + m.start()))
        regions[f"kg:{tag}"] = sorted(spans)
    return regions


def token_masks(offsets: Sequence[Tuple[int, int]], regions: Dict[str, List[Tuple[int, int]]]) -> Dict[str, np.ndarray]:
    """Boolean mask over prompt tokens per region (any char overlap)."""
    a = np.array([o[0] for o in offsets]); b = np.array([o[1] for o in offsets])
    out = {}
    for name, spans in regions.items():
        m = np.zeros(len(offsets), dtype=bool)
        for s, e in spans:
            m |= (np.minimum(b, e) - np.maximum(a, s)) > 0
        out[name] = m
    return out


def setpoint_token_rows(tokenizer, answer_ids: List[int]) -> Optional[Dict[str, List[int]]]:
    """Answer-token indices of the number written for each setpoint key."""
    text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    ends = [len(tokenizer.decode(answer_ids[: i + 1], skip_special_tokens=True)) for i in range(len(answer_ids))]
    starts = [0] + ends[:-1]
    rows = {}
    for key in SETPOINTS:
        m = re.search(r'"' + key + r'"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)', text)
        if not m:
            return None
        cs, ce = m.span(1)
        idx = [i for i in range(len(answer_ids)) if min(ends[i], ce) - max(starts[i], cs) > 0]
        if not idx:
            return None
        rows[key] = idx
    return rows


def field_grounding(attn_by_block: Dict[int, torch.Tensor], blocks: Sequence[int], rel_layers: Sequence[float],
                    masks: Dict[str, np.ndarray], rows: Dict[str, List[int]], n_prompt: int) -> Dict:
    """Per-setpoint shares per layer, plus the layer-mean attention share of every named region."""
    denom_mask = masks["snapshot"] | masks["context"]
    res = {"per_setpoint": {}}
    for key in SETPOINTS:
        dec = np.zeros(n_prompt, dtype=bool)
        for r in DECISIVE[key]:
            if r in masks:
                dec |= masks[r]
        rec = {}
        region_acc = {n: [] for n in masks if n.startswith(("field:", "kg:"))}
        for rl, l in zip(rel_layers, blocks):
            a = attn_by_block[l][0][:, rows[key], :n_prompt].float()  # [H, r, P]
            tot = a[..., torch.as_tensor(denom_mask, device=a.device)].sum(-1).mean()
            tag = layer_tag(rl)
            if float(tot) <= 0:
                rec[f"{tag}_share"] = rec[f"{tag}_snap_share"] = float("nan")
                continue
            rec[f"{tag}_share"] = float(a[..., torch.as_tensor(dec & denom_mask, device=a.device)].sum(-1).mean() / tot)
            rec[f"{tag}_snap_share"] = float(a[..., torch.as_tensor(masks["snapshot"], device=a.device)].sum(-1).mean() / tot)
            for n in region_acc:
                region_acc[n].append(float(a[..., torch.as_tensor(masks[n], device=a.device)].sum(-1).mean() / tot))
        rec["region_share_layer_mean"] = {n: float(np.mean(v)) for n, v in region_acc.items() if v}
        res["per_setpoint"][key] = rec
    return res


def candidate_features(fg: Optional[Dict], rel_layers: Sequence[float]) -> Dict[str, float]:
    tags = [layer_tag(r) for r in rel_layers]
    nan = float("nan")
    names = ["fld_min_share"] + [f"fld_{k[:-3]}_share" for k in SETPOINTS] + [f"fld_{k[:-3]}_snap_share" for k in SETPOINTS]
    names += [f"fld_{t}_Fin_share" for t in tags]
    if not fg:
        return {n: nan for n in names}
    f = {}
    shares = []
    for key in SETPOINTS:
        rec = fg["per_setpoint"][key]
        s = float(np.nanmean([rec[f"{t}_share"] for t in tags]))
        f[f"fld_{key[:-3]}_share"] = s
        f[f"fld_{key[:-3]}_snap_share"] = float(np.nanmean([rec[f"{t}_snap_share"] for t in tags]))
        shares.append(s)
    f["fld_min_share"] = float(np.nanmin(shares))
    for t in tags:
        f[f"fld_{t}_Fin_share"] = fg["per_setpoint"]["Fin_sp"][f"{t}_share"]
    return f


# ----------------------------------------------------------------------------- prompt v3.1
# Pre-registered in docs/cstr_v4_prereg.md ("Grounding for prompt v3.1"). KG relation lines are
# taken from the Turtle block only, so the v3.1 instruction sections (which also name PO5-PO8)
# are separate regions.

DECISIVE_V31 = {
    "Fin_sp": ["field:u_cool", "field:T_meas", "field:current setpoints", "kg:PO8", "kg:PO5"],
    "L_sp": ["field:L_meas", "field:u_pump", "field:current setpoints", "kg:PO6", "kg:PO5"],
    "T_sp": ["field:T_meas", "field:u_cool", "field:current setpoints", "kg:PO7", "kg:PO5"],
}
KG_TAGS_V31 = {"PO5": ["PO5", "dT/dt", "dV/dt"], "PO6": ["PO6"], "PO7": ["PO7"], "PO8": ["PO8", "PO1"]}
SECTIONS_V31 = {"sec:acts": "# How each setpoint acts (KG)", "sec:physics": "# Plant physics (KG, PO5 dT/dt)"}
EXTRA_V31 = {"physics": "sec:physics", "acts": "sec:acts", "snap": "snapshot"}


def cstr_regions_v31(rendered: str, system: str, user: str) -> Dict[str, List[Tuple[int, int]]]:
    """v2 regions (snapshot, field:*, context) + v3.1 sections + KG lines inside the Turtle block."""
    regions = {k: v for k, v in cstr_regions(rendered, system, user).items() if not k.startswith("kg:")}
    sys_at = rendered.find(system.strip()[:200])
    sys_end = sys_at + len(system.strip())
    for name, header in SECTIONS_V31.items():
        s = rendered.find(header, sys_at, sys_end)
        if s < 0:
            raise ValueError(f"v3.1 section not found: {header}")
        e = rendered.find("\n# ", s + len(header), sys_end)
        regions[name] = [(s, e if e >= 0 else sys_end)]
    t0 = rendered.find("```turtle", sys_at, sys_end)
    t1 = rendered.find("```", t0 + len("```turtle"), sys_end) if t0 >= 0 else -1
    if t0 < 0 or t1 < 0:
        raise ValueError("KG Turtle block not found")
    for tag, keys in KG_TAGS_V31.items():
        spans = set()
        for k in keys:
            for m in re.finditer(re.escape(k), rendered[t0:t1]):
                spans.add(_statement_span(rendered, t0 + m.start(), t1))
        regions[f"kg:{tag}"] = sorted(spans)
    return regions


def _statement_span(text: str, pos: int, limit: int) -> Tuple[int, int]:
    """Whole Turtle statement containing `pos`: from its first line to the line ending in ' .'.
    (Amendment 1 of docs/cstr_v4_prereg.md: a line-based span missed the equation label line.)"""
    s = text.rfind(" .\n", 0, pos)
    s = max(s + 3 if s >= 0 else 0, text.rfind("```turtle\n", 0, pos) + len("```turtle\n"))
    while text[s] == "\n":
        s += 1
    e = text.find(" .\n", pos, limit)
    return s, (limit if e < 0 else e + 2)


def grounding_v31(attn_by_block: Dict[int, torch.Tensor], blocks: Sequence[int], rel_layers: Sequence[float],
                  masks: Dict[str, np.ndarray], rows: Dict[str, List[int]], n_prompt: int) -> Dict:
    """Per setpoint and layer: decisive-region share, and physics / acts / snapshot shares."""
    denom_mask = masks["snapshot"] | masks["context"]
    res = {"per_setpoint": {}}
    for key in SETPOINTS:
        dec = np.zeros(n_prompt, dtype=bool)
        for r in DECISIVE_V31[key]:
            if r in masks:
                dec |= masks[r]
        rec = {}
        for rl, l in zip(rel_layers, blocks):
            a = attn_by_block[l][0][:, rows[key], :n_prompt].float()  # [H, r, P]
            share = lambda m: a[..., torch.as_tensor(m & denom_mask, device=a.device)].sum(-1).mean()  # noqa: E731
            tot = share(denom_mask)
            tag = layer_tag(rl)
            names = ["share"] + [f"{n}_share" for n in EXTRA_V31]
            if float(tot) <= 0:
                rec.update({f"{tag}_{n}": float("nan") for n in names})
                continue
            rec[f"{tag}_share"] = float(share(dec) / tot)
            for n, region in EXTRA_V31.items():
                rec[f"{tag}_{n}_share"] = float(share(masks[region]) / tot)
        res["per_setpoint"][key] = rec
    return res


def candidate_features_v31(fg: Optional[Dict], rel_layers: Sequence[float]) -> Dict[str, float]:
    tags = [layer_tag(r) for r in rel_layers]
    kinds = ["share"] + [f"{n}_share" for n in EXTRA_V31]
    names = ["g31_min_share"] + [f"g31_{k[:-3]}_{kind}" for k in SETPOINTS for kind in kinds]
    if not fg:
        return {n: float("nan") for n in names}
    f = {}
    for key in SETPOINTS:
        rec = fg["per_setpoint"][key]
        for kind in kinds:
            f[f"g31_{key[:-3]}_{kind}"] = float(np.nanmean([rec[f"{t}_{kind}"] for t in tags]))
    f["g31_min_share"] = float(np.nanmin([f[f"g31_{k[:-3]}_share"] for k in SETPOINTS]))
    return f
