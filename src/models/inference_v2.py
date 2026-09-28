"""
Generation plus answer-aligned internal-signal extraction (protocol v1.0, sections 4 and 6).

Per candidate:
  1. Greedy generation with SDPA attention, repetition_penalty=1.0 so the chosen
     token is the argmax of the raw logits. Generation stops at the end of the
     first complete JSON object, at EOS, or at max_new_tokens.
  2. Answer span = generated tokens up to and including the token that closes the
     first JSON object (all non-EOS tokens if no object closes).
  3. Feature pass: the generation KV cache is cropped to the prompt minus its last
     token, and the "decision positions" [last prompt token, a_0, ..., a_{k-2}] are
     re-run with eager attention. Row i of this pass produces the logits for answer
     token a_i, so:
       - token features use these raw teacher-forced logits;
       - attention rows are exactly the answer-query rows against all keys
         (memory O(heads * k * T), never O(heads * T^2));
       - hidden states are the residual stream at the decision positions,
         i.e. the state *before* each answer token is consumed.
     hidden_states index 0 is the embedding output; index l (1..L) is after block l.
     Attention weights are captured by forward hooks on the selected blocks only, so the
     other blocks' attention matrices are never retained (about 0.5 GB saved for the
     4.8k-token CSTR prompt, which otherwise pushes an 8 GB GPU into shared memory).
"""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import DynamicCache, StoppingCriteria, StoppingCriteriaList

from src.evaluation.fsm_labels_v2 import find_first_json_object

TEMPLATE_REGION = "template"
FIXED_TEMPLATE_DATE = "23 Sep 2026"
GENERATED_REGION = "generated"


@dataclass
class InternalsConfig:
    max_new_tokens: int
    relative_layers: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    collect_attention: bool = True
    collect_hidden: bool = True
    feature_attn_implementation: str = "eager"
    generation_attn_implementation: str = "sdpa"
    # Pinned template variables. Llama-3.x templates otherwise insert today's date, which
    # would make the prompt depend on the run date. Ignored by templates that do not use them.
    chat_template_kwargs: Dict[str, str] = field(default_factory=lambda: {"date_string": FIXED_TEMPLATE_DATE})
    # Prompts longer than this are prefilled through the KV cache in chunks. A single long
    # SDPA prefill falls back to the math kernel on this platform (4.8k tokens: 10 GB, 60 s).
    # Prompts that fit in one chunk use plain generate, so short-prompt runs are unchanged.
    prefill_chunk: int = 1024
    # Optional KV cache of a prompt prefix shared by every candidate (e.g. the fixed CSTR
    # system prompt + KG context). Used only when the candidate's prompt starts with exactly
    # these token ids; the prefix is computed once, as a deployed system would cache it.
    prefix_cache: Optional["PrefixCache"] = None


class PrefixCache:
    """KV cache for a fixed token prefix, built once and copied per candidate."""

    @torch.no_grad()
    def __init__(self, model, prefix_ids: List[int], chunk: int = 1024):
        self.ids = list(prefix_ids)
        self.cache = DynamicCache(config=model.config)
        t = torch.tensor([self.ids], device=model.device)
        for s0 in range(0, len(self.ids), chunk):
            model(t[:, s0 : s0 + chunk], past_key_values=self.cache, use_cache=True, logits_to_keep=1)

    def matches(self, prompt_ids: List[int]) -> bool:
        n = len(self.ids)
        return len(prompt_ids) > n + 1 and prompt_ids[:n] == self.ids


def selected_layers(num_layers: int, relative: Sequence[float]) -> List[int]:
    """Block indices in 1..L (hidden_states indexing) for relative depths."""
    out = []
    for r in relative:
        idx = int(round(r * num_layers))
        out.append(min(max(idx, 1), num_layers))
    return out


def layer_tag(r: float) -> str:
    return f"L{int(round(r * 100)):03d}"


class JsonObjectStop(StoppingCriteria):
    """Stop (batch size 1) once the generated text contains a complete JSON object."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        done = find_first_json_object(text) is not None
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def token_region_labels(
    offsets: Sequence[Tuple[int, int]],
    region_spans: Dict[str, List[Tuple[int, int]]],
) -> List[str]:
    """Assign each prompt token to the region with the largest character overlap."""
    labels = []
    for a, b in offsets:
        best, best_ov = TEMPLATE_REGION, 0
        for name, spans in region_spans.items():
            ov = sum(max(0, min(b, e) - max(a, s)) for s, e in spans)
            if ov > best_ov:
                best, best_ov = name, ov
        labels.append(best)
    return labels


def _eos_ids(model, tokenizer) -> set:
    ids = model.generation_config.eos_token_id
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    out = {int(i) for i in ids if i is not None}
    if tokenizer.eos_token_id is not None:
        out.add(int(tokenizer.eos_token_id))
    return out


def _answer_token_count(tokenizer, gen_ids: List[int]) -> Tuple[int, Optional[Tuple[int, int]], str]:
    """(k, json_char_span, decoded_text_of_first_k_tokens)."""
    full = tokenizer.decode(gen_ids, skip_special_tokens=True)
    span = find_first_json_object(full)
    if span is None:
        return len(gen_ids), None, full
    end = span[1]
    for k in range(1, len(gen_ids) + 1):
        text_k = tokenizer.decode(gen_ids[:k], skip_special_tokens=True)
        if len(text_k) >= end and text_k[:end] == full[:end]:
            return k, span, text_k
    return len(gen_ids), span, full


def token_confidence_features(
    logits: torch.Tensor, target_ids: torch.Tensor, digit_mask: np.ndarray
) -> Dict[str, float]:
    """logits: [k, V] raw; target_ids: [k]. Features over all answer tokens and digit tokens."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    sel = lp.gather(1, target_ids[:, None]).squeeze(1)
    ent = -(lp.exp() * lp).sum(-1)
    top2 = torch.topk(lp, 2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    sel, ent, margin = (t.cpu().numpy() for t in (sel, ent, margin))

    def _stats(prefix: str, m: np.ndarray) -> Dict[str, float]:
        if m.sum() == 0:
            return {f"{prefix}{k}": np.nan for k in
                    ("mean_logprob", "min_logprob", "mean_entropy", "max_entropy", "min_top2_margin")}
        return {
            f"{prefix}mean_logprob": float(sel[m].mean()),
            f"{prefix}min_logprob": float(sel[m].min()),
            f"{prefix}mean_entropy": float(ent[m].mean()),
            f"{prefix}max_entropy": float(ent[m].max()),
            f"{prefix}min_top2_margin": float(margin[m].min()),
        }

    all_mask = np.ones(len(sel), dtype=bool)
    return {**_stats("tok_", all_mask), **_stats("tok_num_", digit_mask.astype(bool))}


def attention_features(
    attn: torch.Tensor,
    key_labels: np.ndarray,
    regions: Sequence[str],
    prompt_len_minus_1: int,
    tag: str,
) -> Dict[str, float]:
    """
    attn: [H, k, T] answer-query rows (T = prompt_len-1 + k). key_labels: region name per key.
    Region mass: sum of attention on the region's keys, averaged over heads and rows (mean)
    and the max over heads of the per-head row-averaged mass (hmax).
    """
    a = attn.float()
    H, k, T = a.shape
    feats: Dict[str, float] = {}
    for r in regions:
        mask = torch.as_tensor(key_labels == r, device=a.device)
        if not bool(mask.any()):
            feats[f"att_{tag}_{r}_mean"] = 0.0
            feats[f"att_{tag}_{r}_hmax"] = 0.0
            continue
        mass = a[..., mask].sum(-1)  # [H, k]
        feats[f"att_{tag}_{r}_mean"] = float(mass.mean())
        feats[f"att_{tag}_{r}_hmax"] = float(mass.mean(1).max())
    ent = -(a * torch.log(a.clamp_min(1e-12))).sum(-1)  # [H, k]
    visible = torch.arange(k, device=a.device, dtype=torch.float32) + prompt_len_minus_1 + 1
    feats[f"att_{tag}_entropy"] = float(ent.mean())
    feats[f"att_{tag}_entropy_norm"] = float((ent / torch.log(visible)[None, :]).mean())
    return feats


def _decoder_layers(model):
    return model.model.layers


@torch.no_grad()
def decision_pass(
    model,
    cache,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    cfg: InternalsConfig,
    attn_blocks: Optional[Sequence[int]] = None,
):
    """
    Re-run the decision positions on top of a KV cache holding the full prompt.
    The cache is cropped in place to prompt_len-1. prompt_ids: [1, P]; answer_ids: [k].
    Returns (output, attn): output has logits [1, k, V] and hidden_states per index
    [1, k, d]; attn maps block l (1-based) in `attn_blocks` to weights [1, H, k, P-1+k].
    """
    P = prompt_ids.shape[1]
    cache.crop(P - 1)
    attn: Dict[int, torch.Tensor] = {}
    hooks = []
    if cfg.collect_attention and attn_blocks:
        model.set_attn_implementation(cfg.feature_attn_implementation)
        layers = _decoder_layers(model)
        for l in attn_blocks:
            def hook(module, inputs, output, _l=l):
                attn[_l] = output[1].detach()
            hooks.append(layers[l - 1].self_attn.register_forward_hook(hook))
    try:
        inp = torch.cat([prompt_ids[:, P - 1 : P], answer_ids[None, :-1]], dim=1)
        out = model(
            inp,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=cfg.collect_hidden,
        )
    finally:
        for h in hooks:
            h.remove()
        model.set_attn_implementation(cfg.generation_attn_implementation)
    missing = [l for l in (attn_blocks or []) if cfg.collect_attention and (l not in attn or attn[l] is None)]
    if missing:
        raise RuntimeError(f"Attention weights not returned by blocks {missing}; eager attention required")
    return out, attn


@torch.no_grad()
def run_candidate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    region_spans_fn: Callable[[str], Dict[str, List[Tuple[int, int]]]],
    cfg: InternalsConfig,
) -> Dict:
    """Generate one answer and extract answer-aligned internals. region_spans_fn(rendered) -> spans."""
    device = model.device
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **cfg.chat_template_kwargs
    )
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids: List[int] = enc["input_ids"]
    P = len(prompt_ids)
    region_spans = region_spans_fn(rendered)
    prompt_labels = token_region_labels(enc["offset_mapping"], region_spans)
    regions = sorted(region_spans) + [TEMPLATE_REGION, GENERATED_REGION]

    ids = torch.tensor([prompt_ids], device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.set_attn_implementation(cfg.generation_attn_implementation)

    t0 = time.perf_counter()
    gen_kwargs = {}
    prefix_hit = cfg.prefix_cache is not None and cfg.prefix_cache.matches(prompt_ids)
    if prefix_hit or P - 1 > cfg.prefill_chunk:
        if prefix_hit:
            cache = copy.deepcopy(cfg.prefix_cache.cache)
            start = len(cfg.prefix_cache.ids)
        else:
            cache = DynamicCache(config=model.config)
            start = 0
        for s0 in range(start, P - 1, cfg.prefill_chunk):
            model(ids[:, s0 : min(s0 + cfg.prefill_chunk, P - 1)], past_key_values=cache, use_cache=True, logits_to_keep=1)
        gen_kwargs["past_key_values"] = cache
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        **gen_kwargs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        return_dict_in_generate=True,
        output_logits=True,
        stopping_criteria=StoppingCriteriaList([JsonObjectStop(tokenizer, P)]),
        pad_token_id=tokenizer.pad_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    gen_all = out.sequences[0, P:].tolist()
    eos = _eos_ids(model, tokenizer)
    gen_ids = []
    for t in gen_all:
        if t in eos:
            break
        gen_ids.append(t)
    hit_eos = len(gen_ids) < len(gen_all)

    rec: Dict = {
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_tokens": P,
        "chunked_prefill": P - 1 > cfg.prefill_chunk,
        "prefix_cache_hit": bool(prefix_hit),
        "generated_ids": gen_all,
        "generated_text": tokenizer.decode(gen_all, skip_special_tokens=True),
        "gen_latency_s": t_gen,
        "generated_token_count": len(gen_all),
    }

    k, json_span, answer_text = _answer_token_count(tokenizer, gen_ids) if gen_ids else (0, None, "")
    rec.update(answer_token_count=k, answer_text=answer_text)
    if json_span is not None:
        rec["stop_reason"] = "json_object"
    elif hit_eos:
        rec["stop_reason"] = "eos"
    else:
        rec["stop_reason"] = "max_new_tokens"

    if k == 0:
        rec["feature_status"] = "empty_answer"
        rec["feat_latency_s"] = 0.0
        rec["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        return rec

    ans = torch.tensor(gen_ids[:k], device=device)
    # Raw logits seen during generation for the answer tokens (sanity/consistency only).
    gen_logits = torch.stack(out.logits[:k])[:, 0].float()
    gen_sel = torch.log_softmax(gen_logits, -1).gather(1, ans[:, None]).squeeze(1)
    rec["greedy_consistent"] = bool((gen_logits.argmax(-1) == ans).all())

    t1 = time.perf_counter()
    layers = selected_layers(model.config.num_hidden_layers, cfg.relative_layers)
    fo, attn_by_block = decision_pass(model, out.past_key_values, ids, ans, cfg, attn_blocks=layers)

    tf_logits = fo.logits[0].float()
    tf_sel = torch.log_softmax(tf_logits, -1).gather(1, ans[:, None]).squeeze(1)
    rec["gen_vs_tf_logprob_maxdiff"] = float((tf_sel - gen_sel).abs().max())

    digit_mask = np.array(
        [any(ch.isdigit() for ch in tokenizer.decode([t])) for t in gen_ids[:k]], dtype=bool
    )
    feats: Dict[str, float] = token_confidence_features(tf_logits, ans, digit_mask)

    rec["selected_layers"] = layers
    hidden: Dict[str, np.ndarray] = {}

    if cfg.collect_attention:
        key_labels = np.array(prompt_labels[: P - 1] + [prompt_labels[P - 1]] + [GENERATED_REGION] * (k - 1))
        for r, l in zip(cfg.relative_layers, layers):
            feats.update(
                attention_features(attn_by_block[l][0], key_labels, regions, P - 1, layer_tag(r))
            )
    if cfg.collect_hidden:
        for r, l in zip(cfg.relative_layers, layers):
            h = fo.hidden_states[l][0].float()  # [k, d]
            hidden[f"hid_{layer_tag(r)}_last_prompt"] = h[0].cpu().numpy().astype(np.float16)
            hidden[f"hid_{layer_tag(r)}_mean_answer"] = h.mean(0).cpu().numpy().astype(np.float16)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rec["feat_latency_s"] = time.perf_counter() - t1
    rec["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    rec["feature_status"] = "ok"
    rec["features"] = feats
    rec["hidden"] = hidden
    rec["region_token_counts"] = {r: int(sum(1 for x in prompt_labels if x == r)) for r in regions}
    return rec
