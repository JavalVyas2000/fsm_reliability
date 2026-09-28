"""
Compute pre-registered attention-grounding features for an existing FSM inference run
(docs/grounding_attention_prereg.md). No regeneration: the stored answer tokens are
re-run teacher-forced with the same model, dtype, prompt and chat template.

Outputs (outputs/fsm_grounding/<run>/):
    grounding.jsonl   per candidate: status, per-step measurements, candidate features
    aug/              copy of the inference run with grd_* features merged into records
                      (consumable by scripts/12 as --inference_dir)
    run_manifest.json

Example:
    python -m scripts.27_fsm_grounding --inference_dir outputs/fsm_inference/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: I001

import argparse
import json
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DynamicCache

from src.data.fsm_dataset_v2 import load_graph
from src.features.grounding_fsm import (
    candidate_features,
    graph_line_spans,
    path_node_token_rows,
    step_grounding,
    token_line_labels,
)
from src.models.inference_v2 import InternalsConfig, decision_pass, selected_layers
from src.models.load_model import load_hf_model_and_tokenizer
from src.prompts.fsm_prompts_v2 import build_messages, region_char_spans
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, default="outputs/fsm_grounding")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N records only")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    inf = (REPO_ROOT / args.inference_dir).resolve()
    man = json.loads((inf / "run_manifest.json").read_text())
    model_id = man["model"]["hf_id"]
    ds_dir = REPO_ROOT / man["data"]["dataset_dir"]
    kwargs = man["prompt"].get("chat_template_kwargs") or {"date_string": "23 Sep 2026"}
    run_dir = make_run_dir(REPO_ROOT / args.out_root, inf.name)

    rows = {}
    for f in ds_dir.glob("*.csv"):
        for _, r in pd.read_csv(f).iterrows():
            rows[r["instance_id"]] = r
    recs = [json.loads(l) for l in open(inf / "records.jsonl", encoding="utf-8")]
    if args.limit:
        recs = recs[: args.limit]

    model, tok = load_hf_model_and_tokenizer(model_id, device_map="cuda", torch_dtype=man["model"].get("dtype", "bfloat16"),
                                             attn_implementation="sdpa", local_files_only=True)
    cfg = InternalsConfig(max_new_tokens=0, collect_hidden=False)
    blocks = selected_layers(model.config.num_hidden_layers, cfg.relative_layers)

    out, n_ok, agree_tok, total_tok = [], 0, 0, 0
    for rec in tqdm(recs):
        g = {"instance_id": rec["instance_id"], "status": None}
        path = rec.get("parsed_path")
        if rec.get("schema_valid") != 1 or rec.get("feature_status") != "ok" or not isinstance(path, list) or len(path) < 2:
            g["status"] = "no_step"
            g["features"] = candidate_features([], cfg.relative_layers)
            out.append(g)
            continue
        row = rows[rec["instance_id"]]
        start, goal = int(row["start"]), int(row["goal"])
        msgs = build_messages(row["graph_text"], start, goal)
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        P = len(enc["input_ids"])
        if P != rec["prompt_tokens"]:
            g["status"] = "prompt_length_mismatch"
            g["features"] = candidate_features([], cfg.relative_layers)
            out.append(g)
            continue
        k = int(rec["answer_token_count"])
        ans = rec["generated_ids"][:k]
        node_rows = path_node_token_rows(tok, ans, path)
        if node_rows is None:
            g["status"] = "path_token_alignment_failed"
            g["features"] = candidate_features([], cfg.relative_layers)
            out.append(g)
            continue
        spans = region_char_spans(rendered, row["graph_text"], start, goal)
        lines = graph_line_spans(rendered, spans["graph"][0], row["graph_text"])
        key_lines = token_line_labels(enc["offset_mapping"], lines)

        ids = torch.tensor([enc["input_ids"]], device=model.device)
        cache = DynamicCache(config=model.config)
        model(ids[:, : P - 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
        fo, attn = decision_pass(model, cache, ids, torch.tensor(ans, device=model.device), cfg, attn_blocks=blocks)
        pred = fo.logits[0].argmax(-1).tolist()
        agree_tok += sum(int(a == b) for a, b in zip(pred, ans))
        total_tok += len(ans)

        steps = step_grounding(attn, blocks, cfg.relative_layers, key_lines, node_rows, path, load_graph(row))
        g.update(status="ok", n_steps=len(steps), steps=steps, features=candidate_features(steps, cfg.relative_layers))
        out.append(g)
        n_ok += 1

    with open(run_dir / "grounding.jsonl", "w", encoding="utf-8") as f:
        for g in out:
            f.write(json.dumps(g) + "\n")

    # Augmented copy of the inference run for the shared fitting code.
    aug = run_dir / "aug"
    aug.mkdir()
    gmap = {g["instance_id"]: g["features"] for g in out}
    with open(aug / "records.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            rec = dict(rec)
            rec["features"] = {**rec.get("features", {}), **gmap[rec["instance_id"]]}
            f.write(json.dumps(rec) + "\n")
    shutil.copytree(inf / "hidden", aug / "hidden")
    m2 = dict(man)
    m2["grounding"] = {"source_inference_dir": str(inf.relative_to(REPO_ROOT)), "prereg": "docs/grounding_attention_prereg.md"}
    write_json(aug / "run_manifest.json", m2)

    status = pd.Series([g["status"] for g in out]).value_counts().to_dict()
    write_json(run_dir / "run_manifest.json", build_manifest(
        inference_dir=str(inf.relative_to(REPO_ROOT)), model=model_id, blocks=blocks, relative_layers=list(cfg.relative_layers),
        status_counts=status, teacher_forced_argmax_agreement=(agree_tok / total_tok if total_tok else None),
        prereg="docs/grounding_attention_prereg.md"))
    print(json.dumps({"run_dir": str(run_dir), "status": status,
                      "tf_argmax_agreement": round(agree_tok / total_tok, 4) if total_tok else None}))


if __name__ == "__main__":
    main()
