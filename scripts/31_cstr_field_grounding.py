"""
Compute pre-registered CSTR field-grounding features (docs/grounding_cstr_prereg.md) for
the first proposals of a scripts/25 collection run. Teacher-forced re-run of the stored
answer tokens (no regeneration), with the shared-prefix KV cache.

The cert partition is processed only with --include_cert (after a policy freeze).

Outputs (outputs/cstr_grounding/<run>/): grounding.jsonl, aug/ (records with fld_*
features merged; round-0 candidates of the processed partitions only), run_manifest.json.

Example:
    python -m scripts.31_cstr_field_grounding --collect_dir outputs/cstr_collect/<run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: I001

import argparse
import copy
import json
import pickle
import shutil

import numpy as np
from tqdm import tqdm

from src.cstr.episodes import set_kg_context
from src.cstr.prompt_v2 import build_messages_v2
from src.cstr.prompt_v3 import build_messages_v3
from src.features.grounding_cstr import (
    candidate_features, candidate_features_v31, cstr_regions, cstr_regions_v31, field_grounding, grounding_v31,
    setpoint_token_rows, token_masks,
)
from src.models.inference_v2 import FIXED_TEMPLATE_DATE, InternalsConfig, PrefixCache, decision_pass, selected_layers
from src.models.load_model import load_hf_model_and_tokenizer
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json


VARIANTS = {  # --prompt -> (prompt version, messages, regions, grounding, candidate features, prereg)
    "v2.1": ("cstr_prompt_v2.1", build_messages_v2, cstr_regions, field_grounding, candidate_features,
             "docs/grounding_cstr_prereg.md"),
    "v3.1": ("cstr_prompt_v3.1", lambda s: build_messages_v3(s, version="cstr_prompt_v3.1"), cstr_regions_v31,
             grounding_v31, candidate_features_v31, "docs/cstr_v4_prereg.md"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collect_dir", type=str, required=True)
    p.add_argument("--include_cert", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_root", type=str, default="outputs/cstr_grounding")
    p.add_argument("--prompt", choices=sorted(VARIANTS), default="v2.1",
                   help="must match the prompt of the collection run (v3.1: docs/cstr_v4_prereg.md features g31_*)")
    p.add_argument("--partitions", nargs="+", default=None, help="default: train dev_cal dev_thr test_iid (+cert)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cdir = (REPO_ROOT / args.collect_dir).resolve()
    man = json.loads((cdir / "run_manifest.json").read_text())
    ds = REPO_ROOT / man["data"]["dataset_dir"]
    version, build_messages, regions_fn, grounding_fn, features_fn, prereg = VARIANTS[args.prompt]
    if man["prompt"]["version"] != version:
        raise SystemExit(f"collection prompt {man['prompt']['version']} does not match --prompt {args.prompt}")
    parts = args.partitions or ["train", "dev_cal", "dev_thr", "test_iid"]
    parts = parts + (["cert"] if args.include_cert and "cert" not in parts else [])
    if "cert" in parts and not args.include_cert:
        raise SystemExit("cert is processed only with --include_cert (after a policy freeze)")
    run_dir = make_run_dir(REPO_ROOT / args.out_root, cdir.name + ("_withcert" if args.include_cert else ""))

    set_kg_context((ds / "kg_context.ttl").read_bytes().decode("utf-8"))
    snaps = {}
    for part in parts:
        for s in pickle.load(open(ds / f"snapshots_{part}.pkl", "rb")):
            snaps[s.spec.episode_id] = s
    recs = [json.loads(l) for l in open(cdir / "records.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["round"] == 0 and r["partition"] in parts]
    if args.limit:
        recs = recs[: args.limit]

    # same weights as the collection run (4-bit if it was quantized)
    model, tok = load_hf_model_and_tokenizer(man["model"]["hf_id"], device_map="cuda", torch_dtype="bfloat16",
                                             attn_implementation="sdpa", local_files_only=True,
                                             quantization=man["model"].get("quantization"))
    cfg = InternalsConfig(max_new_tokens=0, collect_hidden=False)
    blocks = selected_layers(model.config.num_hidden_layers, cfg.relative_layers)
    kwargs = {"date_string": FIXED_TEMPLATE_DATE}
    # shared prefix = rendered tokens before the user message (identical across episodes)
    m0 = build_messages(snaps[recs[0]["graph_hash"]])
    r0 = tok.apply_chat_template(m0, tokenize=False, add_generation_prompt=True, **kwargs)
    e0 = tok(r0, add_special_tokens=False, return_offsets_mapping=True)
    u_at = r0.find(m0[1]["content"].strip()[:200])
    n_pre = max(0, sum(1 for a, b in e0["offset_mapping"] if b <= u_at) - 4)
    pc = PrefixCache(model, e0["input_ids"][:n_pre])

    out, agree, total = [], 0, 0
    for rec in tqdm(recs):
        g = {"instance_id": rec["instance_id"], "status": None}
        if rec.get("schema_valid") != 1 or rec.get("feature_status") != "ok":
            g.update(status="not_eligible", features=features_fn(None, cfg.relative_layers)); out.append(g); continue
        msgs = build_messages(snaps[rec["graph_hash"]])
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        P = len(enc["input_ids"])
        if P != rec["prompt_tokens"] or enc["input_ids"][:n_pre] != pc.ids:
            g.update(status="prompt_mismatch", features=features_fn(None, cfg.relative_layers)); out.append(g); continue
        ans = rec["generated_ids"][: int(rec["answer_token_count"])]
        rows = setpoint_token_rows(tok, ans)
        if rows is None:
            g.update(status="setpoint_token_alignment_failed", features=features_fn(None, cfg.relative_layers)); out.append(g); continue
        masks = token_masks(enc["offset_mapping"], regions_fn(rendered, msgs[0]["content"], msgs[1]["content"]))
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        cache = copy.deepcopy(pc.cache)
        for s0 in range(n_pre, P - 1, 1024):
            model(ids[:, s0 : min(s0 + 1024, P - 1)], past_key_values=cache, use_cache=True, logits_to_keep=1)
        fo, attn = decision_pass(model, cache, ids, torch.tensor(ans, device=model.device), cfg, attn_blocks=blocks)
        pred = fo.logits[0].argmax(-1).tolist()
        agree += sum(int(a == b) for a, b in zip(pred, ans)); total += len(ans)
        fg = grounding_fn(attn, blocks, cfg.relative_layers, masks, rows, P)
        g.update(status="ok", grounding=fg, features=features_fn(fg, cfg.relative_layers))
        out.append(g)

    with open(run_dir / "grounding.jsonl", "w", encoding="utf-8") as f:
        for g in out:
            f.write(json.dumps(g) + "\n")
    aug = run_dir / "aug"; aug.mkdir()
    gmap = {g["instance_id"]: g["features"] for g in out}
    with open(aug / "records.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            rec = dict(rec); rec["features"] = {**rec.get("features", {}), **gmap[rec["instance_id"]]}
            f.write(json.dumps(rec) + "\n")
    shutil.copytree(cdir / "hidden", aug / "hidden")
    m2 = dict(man); m2["grounding"] = {"source_collect_dir": str(cdir.relative_to(REPO_ROOT)), "prereg": prereg,
                                       "rounds": "0 only", "partitions": parts}
    write_json(aug / "run_manifest.json", m2)
    status = {}
    for g in out:
        status[g["status"]] = status.get(g["status"], 0) + 1
    write_json(run_dir / "run_manifest.json", build_manifest(collect_dir=str(cdir.relative_to(REPO_ROOT)), partitions=parts,
                                                            status_counts=status, tf_argmax_agreement=(agree / total if total else None),
                                                            blocks=blocks, prereg=prereg, prompt=args.prompt))
    print(json.dumps({"run_dir": str(run_dir), "status": status, "tf_agreement": round(agree / total, 4) if total else None}))


if __name__ == "__main__":
    main()
