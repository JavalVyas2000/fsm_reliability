"""
CSTR first-proposal inference with answer-aligned internals (Stage 4).

For each snapshot the exact upstream action_propose prompt (captured in scripts/21) is
rendered with the model's chat template, the first proposal is generated greedily, and
token / attention / hidden features are extracted as in the FSM pipeline
(src/models/inference_v2.py). Labels are added separately by scripts/23.

Resumable with --run_dir. Outputs: records.jsonl, hidden/shard_*.npz, run_manifest.json.

Example:
    python -m scripts.22_cstr_inference --dataset_dir data/v2/cstr_pilot_seed20260925
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: I001

import argparse
import hashlib
import json
import pickle

import numpy as np
from tqdm import tqdm

from src.cstr.episodes import set_kg_context
from src.cstr.llm_io import PROMPT_VERSION as PROMPT_V1, context_action, parse_action, region_char_spans
from src.cstr.prompt_v2 import PROMPT_VERSION as PROMPT_V2, build_messages_v2
from src.models.inference_v2 import InternalsConfig, run_candidate
from src.models.load_model import load_hf_model_and_tokenizer, resolved_revision
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--partitions", nargs="+", default=["train", "dev_cal", "dev_thr", "test_iid"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--quantization", choices=["4bit"], default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no_attention", action="store_true")
    p.add_argument("--flush_every", type=int, default=25)
    p.add_argument("--out_root", type=str, default="outputs/cstr_inference")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--prompt_version", choices=["v1", "v2"], default="v1",
                   help="v1 = captured upstream action_propose prompt; v2 = src/cstr/prompt_v2.py")
    return p.parse_args()


def main():
    args = parse_args()
    ds = (REPO_ROOT / args.dataset_dir).resolve()
    model_tag = args.model.split("/")[-1].lower().replace(".", "") + ("_4bit" if args.quantization else "")
    run_dir = (REPO_ROOT / args.run_dir).resolve() if args.run_dir else make_run_dir(REPO_ROOT / args.out_root, f"{model_tag}_{args.tag}")
    (run_dir / "hidden").mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        done = {json.loads(l)["instance_id"] for l in open(rec_path, encoding="utf-8") if l.strip()}
    shard_idx = len(list((run_dir / "hidden").glob("shard_*.npz")))

    PROMPT_VERSION = PROMPT_V1 if args.prompt_version == "v1" else PROMPT_V2
    if args.prompt_version == "v2":
        set_kg_context((ds / "kg_context.ttl").read_bytes().decode("utf-8"))
    items = []
    for part in args.partitions:
        snaps = pickle.load(open(ds / f"snapshots_{part}.pkl", "rb"))
        prompts = {json.loads(l)["episode_id"]: json.loads(l) for l in open(ds / f"prompts_{part}.jsonl", encoding="utf-8")}
        sel = snaps[: args.limit] if args.limit else snaps
        for s in sel:
            if s.spec.episode_id in done:
                continue
            pr = prompts[s.spec.episode_id]
            if args.prompt_version == "v2":
                msgs = build_messages_v2(s)
                pr = {"messages": msgs, "user_sha256": hashlib.sha256(msgs[1]["content"].encode()).hexdigest(),
                      "system_sha256": hashlib.sha256(msgs[0]["content"].encode()).hexdigest()}
            items.append((part, s, pr))

    model, tok = load_hf_model_and_tokenizer(args.model, device_map="cuda", torch_dtype=args.dtype,
                                             attn_implementation="sdpa", local_files_only=args.local_files_only,
                                             quantization=args.quantization)
    cfg = InternalsConfig(max_new_tokens=args.max_new_tokens, collect_attention=not args.no_attention)
    if not (run_dir / "run_manifest.json").exists():
        dsm = json.loads((ds / "dataset_manifest.json").read_text())
        write_json(run_dir / "run_manifest.json", build_manifest(
            model={"hf_id": args.model, "resolved_revision": resolved_revision(model), "dtype": args.dtype,
                   "quantization": args.quantization,
                   "num_hidden_layers": model.config.num_hidden_layers},
            prompt={"version": PROMPT_VERSION, "chat_template": True, "chat_template_kwargs": cfg.chat_template_kwargs,
                    "kg_sha256": dsm["dataset"]["kg_sha256"]},
            decoding={"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens,
                      "stop": "end_of_first_json_object | eos | max_new_tokens"},
            internals={"relative_layers": list(cfg.relative_layers), "collect_attention": cfg.collect_attention,
                       "collect_hidden": cfg.collect_hidden},
            data={"dataset_dir": str(ds.relative_to(REPO_ROOT)), "partitions": args.partitions, "limit": args.limit,
                  "root_seed": dsm["dataset"]["root_seed"], "files_sha256": dsm["dataset"]["files_sha256"]},
        ))

    buf_r, buf_h = [], []

    def flush():
        nonlocal shard_idx, buf_r, buf_h
        if not buf_r:
            return
        ids = [r["instance_id"] for r, h in zip(buf_r, buf_h) if h]
        if ids:
            keys = sorted(next(h for h in buf_h if h).keys())
            np.savez(run_dir / "hidden" / f"shard_{shard_idx:05d}.npz", instance_id=np.array(ids),
                     **{k: np.stack([h[k] for h in buf_h if h]) for k in keys})
            shard_idx += 1
        with open(rec_path, "a", encoding="utf-8") as f:
            for r in buf_r:
                f.write(json.dumps(r) + "\n")
        buf_r, buf_h = [], []

    for part, snap, pr in tqdm(items):
        msgs = pr["messages"]
        try:
            out = run_candidate(model, tok, msgs, lambda rendered: region_char_spans(rendered, msgs), cfg)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = {"feature_status": "oom", "answer_text": "", "generated_text": ""}
        parsed = parse_action(out.get("answer_text", ""))
        hidden = out.pop("hidden", {})
        rec = {
            "instance_id": snap.spec.episode_id,
            "graph_hash": snap.spec.episode_id,  # independence unit (name kept for the shared fitting code)
            "partition": part,
            "family": snap.spec.family,
            "severity": snap.spec.params,
            "onset_s": snap.spec.onset_s,
            "t_trigger": snap.t_trigger,
            "prompt_version": PROMPT_VERSION,
            "user_prompt_sha256": pr["user_sha256"],
            "system_prompt_sha256": pr["system_sha256"],
            **{k: v for k, v in parsed.items()},
            **{k: v for k, v in out.items() if k != "features"},
            "context_action": context_action(snap.state, parsed["action"] if parsed["schema_valid"] else None),
            "features": out.get("features", {}),
        }
        buf_r.append(rec)
        buf_h.append(hidden)
        if len(buf_r) >= args.flush_every:
            flush()
    flush()
    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
