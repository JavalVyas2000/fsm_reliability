"""
Run FSM prompt-v2 inference with answer-aligned internals (protocol v1.0, Stage 1).

Resumable: re-running with --run_dir skips instances already recorded.

Outputs (in the run directory):
    records.jsonl         one record per instance: labels, answer, timings, features
    hidden/shard_*.npz    pooled hidden-state vectors keyed by instance_id
    run_manifest.json

Example:
    python -m scripts.11_run_fsm_inference_v2 --dataset_dir data/v2/fsm_pilot_seed20260923 \
        --partitions train --limit 200 --tag timing200
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: I001

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.fsm_dataset_v2 import load_graph
from src.evaluation.fsm_labels_v2 import label_answer
from src.features.context_action import fsm_context_action
from src.models.inference_v2 import InternalsConfig, run_candidate
from src.models.load_model import load_hf_model_and_tokenizer, resolved_revision
from src.prompts.fsm_prompts_v2 import (
    PROMPT_VERSION,
    build_messages,
    prompt_template_hash,
    region_char_spans,
)
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, sha256_file, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--partitions", nargs="+", default=["train", "dev_cal", "dev_thr", "test_iid"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--limit", type=int, default=None, help="first N instances per partition")
    p.add_argument("--no_attention", action="store_true")
    p.add_argument("--no_hidden", action="store_true")
    p.add_argument("--flush_every", type=int, default=100)
    p.add_argument("--out_root", type=str, default="outputs/fsm_inference")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--run_dir", type=str, default=None, help="resume an existing run")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = (REPO_ROOT / args.dataset_dir).resolve()
    if args.run_dir:
        run_dir = (REPO_ROOT / args.run_dir).resolve()
    else:
        model_tag = args.model.split("/")[-1].lower().replace(".", "")
        run_dir = make_run_dir(REPO_ROOT / args.out_root, f"{model_tag}_{args.tag}")
    (run_dir / "hidden").mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"

    done = set()
    if records_path.exists():
        with open(records_path, encoding="utf-8") as f:
            done = {json.loads(line)["instance_id"] for line in f if line.strip()}
    shard_idx = len(list((run_dir / "hidden").glob("shard_*.npz")))

    frames = []
    for part in args.partitions:
        df = pd.read_csv(dataset_dir / f"{part}.csv")
        frames.append(df.head(args.limit) if args.limit else df)
    todo = pd.concat(frames, ignore_index=True)
    todo = todo[~todo["instance_id"].isin(done)]

    model, tok = load_hf_model_and_tokenizer(
        args.model,
        device_map="cuda",
        torch_dtype=args.dtype,
        revision=args.revision,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    )
    cfg = InternalsConfig(
        max_new_tokens=args.max_new_tokens,
        collect_attention=not args.no_attention,
        collect_hidden=not args.no_hidden,
    )

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        ds_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
        write_json(
            manifest_path,
            build_manifest(
                model={
                    "hf_id": args.model,
                    "requested_revision": args.revision,
                    "resolved_revision": resolved_revision(model),
                    "dtype": args.dtype,
                    "num_hidden_layers": model.config.num_hidden_layers,
                    "tokenizer_class": type(tok).__name__,
                },
                prompt={"version": PROMPT_VERSION, "template_hash": prompt_template_hash(), "chat_template": True,
                        "chat_template_kwargs": cfg.chat_template_kwargs},
                decoding={
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": args.max_new_tokens,
                    "stop": "end_of_first_json_object | eos | max_new_tokens",
                    "generation_attn_implementation": cfg.generation_attn_implementation,
                    "feature_attn_implementation": cfg.feature_attn_implementation,
                },
                internals={
                    "relative_layers": list(cfg.relative_layers),
                    "collect_attention": cfg.collect_attention,
                    "collect_hidden": cfg.collect_hidden,
                    "decision_positions": "last prompt token + answer tokens except the last",
                },
                data={
                    "dataset_dir": str(dataset_dir.relative_to(REPO_ROOT)),
                    "partitions": args.partitions,
                    "limit": args.limit,
                    "dataset_files": ds_manifest["dataset"]["files"],
                    "root_seed": ds_manifest["dataset"]["root_seed"],
                },
            ),
        )

    buf_records, buf_hidden = [], []

    def flush():
        nonlocal shard_idx, buf_records, buf_hidden
        if not buf_records:
            return
        hid_ids = [r["instance_id"] for r, h in zip(buf_records, buf_hidden) if h]
        if hid_ids:
            keys = sorted(next(h for h in buf_hidden if h).keys())
            arrays = {k: np.stack([h[k] for h in buf_hidden if h]) for k in keys}
            np.savez(run_dir / "hidden" / f"shard_{shard_idx:05d}.npz", instance_id=np.array(hid_ids), **arrays)
            shard_idx += 1
        with open(records_path, "a", encoding="utf-8") as f:
            for r in buf_records:
                f.write(json.dumps(r) + "\n")
        buf_records, buf_hidden = [], []

    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        graph = load_graph(row)
        start, goal = int(row["start"]), int(row["goal"])
        messages = build_messages(row["graph_text"], start, goal)
        spans_fn = lambda rendered: region_char_spans(rendered, row["graph_text"], start, goal)

        try:
            out = run_candidate(model, tok, messages, spans_fn, cfg)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = {"feature_status": "oom", "answer_text": "", "generated_text": ""}

        labels = label_answer(
            out.get("answer_text", ""),
            graph,
            start,
            goal,
            json.loads(row["shortest_path"]),
            int(row["num_nodes"]),
        )
        ctx = fsm_context_action(
            int(row["num_nodes"]),
            int(row["num_edges"]),
            int(row["shortest_path_length"]),
            start,
            goal,
            labels["parsed_path"] if labels["schema_valid"] else None,
        )
        hidden = out.pop("hidden", {})
        record = {
            "instance_id": row["instance_id"],
            "partition": row["partition"],
            "graph_hash": row["graph_hash"],
            "num_nodes": int(row["num_nodes"]),
            **labels,
            **{k: v for k, v in out.items() if k != "features"},
            "context_action": ctx,
            "features": out.get("features", {}),
        }
        buf_records.append(record)
        buf_hidden.append(hidden)
        if len(buf_records) >= args.flush_every:
            flush()
    flush()
    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
