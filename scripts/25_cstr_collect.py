"""
CSTR data collection with the CAR propose -> verify -> reprompt loop (Stage 4 scale-up).

Per episode (snapshot at the first action trigger), as in the upstream graph:
    round 0 : first proposal (prompt v2.1 or v3, see --prompt)
    round r : after a failed verification, reprompt with the upstream `reprompting`
              feedback text and the previous proposal (up to --max_reprompts rounds)
    stop    : at the first passing proposal, after the last reprompt, on a format failure
              (the upstream graph aborts the episode on an unparseable answer), or when a
              reprompt exactly repeats an earlier proposal of the episode. The verifier is
              deterministic, so a repeat's verdict is known without a rollout; it is recorded
              with the copied label (repeat_of_round) and the episode ends. (Upstream instead
              nudges Fin_sp by 0.9x, which would make the verified action differ from the LLM's.)
Every proposal is a labelled candidate:  instance_id = <episode>_r<round>,
graph_hash = <episode> (the independence unit for grouping and certification).

Throughput: the fixed system prompt + KG context (~3.9k tokens, identical in every
episode) is prefilled once and reused through a KV-cache copy; the GPU generates while a
CPU process pool verifies; first proposals of all episodes are scheduled before any
reprompt round (round-major order), so a time-limited run yields complete round-0 data
first. Resumable with --run_dir; --max_hours stops scheduling new generations cleanly.

Outputs: records.jsonl (labelled), hidden/shard_*.npz, run_manifest.json.

Example:
    python -m scripts.25_cstr_collect --dataset_dir data/v2/cstr_main_seed20260927 --tag main3000 --max_hours 11
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: I001

import argparse
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from tqdm import tqdm

from src.cstr.car_bridge import car_provenance
from src.cstr.episodes import reprompt_feedback, set_kg_context, verify_job
from src.cstr.llm_io import context_action, parse_action, region_char_spans
from src.cstr import prompt_v2, prompt_v3
from src.models.inference_v2 import InternalsConfig, PrefixCache, run_candidate
from src.models.load_model import load_hf_model_and_tokenizer, resolved_revision
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, write_json


PROMPTS = {  # --prompt -> (version, message builder, feedback after a failed verification)
    "v2.1": (prompt_v2.PROMPT_VERSION, prompt_v2.build_messages_v2, reprompt_feedback),
    "v3": (prompt_v3.PROMPT_VERSION, prompt_v3.build_messages_v3, lambda snap, res, rnd: prompt_v3.neutral_feedback(res)),
    "v3.1": ("cstr_prompt_v3.1", lambda s, fb=None, prev=None: prompt_v3.build_messages_v3(s, fb, prev, "cstr_prompt_v3.1"),
             lambda snap, res, rnd: prompt_v3.neutral_feedback(res)),
}
FEEDBACK_DESC = {"v2.1": "upstream reprompting() text + previous proposal",
                 "v3": "verifier summary, fail reason and measured metrics (no hint) + previous proposal",
                 "v3.1": "verifier summary, fail reason and measured metrics (no hint) + previous proposal"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--partitions", nargs="+", default=["train", "dev_cal", "dev_thr", "cert", "test_iid"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_reprompts", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="first N episodes per partition")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--flush_every", type=int, default=20)
    p.add_argument("--out_root", type=str, default="outputs/cstr_collect")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--quantization", choices=["4bit"], default=None,
                   help="4bit: bitsandbytes NF4 weights with bf16 compute (models that do not fit in bf16)")
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="v2.1",
                   help="v2.1: upstream reprompt hints; v3: neutral prompt and neutral feedback")
    return p.parse_args()


def shared_prefix_ids(tok, messages, kwargs) -> list:
    """Token ids of the rendered prompt that precede the user message (identical across episodes)."""
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    user_at = rendered.find(messages[1]["content"].strip()[:200])
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    n = 0
    for (a, b) in enc["offset_mapping"]:
        if b <= user_at:
            n += 1
        else:
            break
    return enc["input_ids"][: max(0, n - 4)]  # stop a few tokens early: safe against boundary merges


def main():
    args = parse_args()
    t_start = time.time()
    ds = (REPO_ROOT / args.dataset_dir).resolve()
    model_tag = args.model.split("/")[-1].lower().replace(".", "")
    run_dir = (REPO_ROOT / args.run_dir).resolve() if args.run_dir else make_run_dir(REPO_ROOT / args.out_root, f"{model_tag}_{args.tag}")
    (run_dir / "hidden").mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"
    shard_idx = len(list((run_dir / "hidden").glob("shard_*.npz")))

    set_kg_context((ds / "kg_context.ttl").read_bytes().decode("utf-8"))
    snaps, part_of, order = {}, {}, []
    for part in args.partitions:
        lst = pickle.load(open(ds / f"snapshots_{part}.pkl", "rb"))
        lst = lst[: args.limit] if args.limit else lst
        for s in lst:
            snaps[s.spec.episode_id] = s
            part_of[s.spec.episode_id] = part
            order.append(s.spec.episode_id)
    rank = {e: i for i, e in enumerate(order)}

    # --- resume: rebuild per-episode progress from labelled records
    last, history = {}, {}
    if rec_path.exists():
        for line in open(rec_path, encoding="utf-8"):
            r = json.loads(line)
            e = r["graph_hash"]
            if r.get("action") is not None:
                history.setdefault(e, []).append((r["round"], _key(r["action"]), r))
            if e not in last or r["round"] > last[e]["round"]:
                last[e] = r
    heap = []
    for e in order:
        r = last.get(e)
        if r is None:
            heapq.heappush(heap, (0, rank[e], e, None, None))
        elif (r.get("schema_valid") == 1 and r.get("verifier_pass") is False and r["round"] < args.max_reprompts
              and r.get("repeat_of_round") is None):
            heapq.heappush(heap, (r["round"] + 1, rank[e], e, r["verifier_feedback"], r["action"]))

    model, tok = load_hf_model_and_tokenizer(args.model, device_map="cuda", torch_dtype="bfloat16",
                                             attn_implementation="sdpa", local_files_only=args.local_files_only,
                                             quantization=args.quantization)
    cfg = InternalsConfig(max_new_tokens=args.max_new_tokens)
    prompt_version, build_messages, make_feedback = PROMPTS[args.prompt]
    prefix = shared_prefix_ids(tok, build_messages(snaps[order[0]]), cfg.chat_template_kwargs)
    cfg.prefix_cache = PrefixCache(model, prefix)

    if not (run_dir / "run_manifest.json").exists():
        dsm = json.loads((ds / "dataset_manifest.json").read_text())
        write_json(run_dir / "run_manifest.json", build_manifest(
            model={"hf_id": args.model, "resolved_revision": resolved_revision(model), "dtype": "bfloat16",
                   "quantization": args.quantization,
                   "num_hidden_layers": model.config.num_hidden_layers},
            prompt={"version": prompt_version, "chat_template": True, "chat_template_kwargs": cfg.chat_template_kwargs,
                    "kg_sha256": dsm["dataset"]["kg_sha256"], "shared_prefix_tokens": len(prefix)},
            decoding={"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens,
                      "stop": "end_of_first_json_object | eos | max_new_tokens"},
            loop={"max_reprompts": args.max_reprompts, "stop": "first pass | last reprompt | format failure",
                  "feedback": FEEDBACK_DESC[args.prompt], "scheduling": "round-major"},
            internals={"relative_layers": list(cfg.relative_layers), "collect_attention": True, "collect_hidden": True},
            data={"dataset_dir": str(ds.relative_to(REPO_ROOT)), "partitions": args.partitions, "limit": args.limit,
                  "root_seed": dsm["dataset"]["root_seed"], "files_sha256": dsm["dataset"]["files_sha256"]},
            car=car_provenance(),
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

    def finish(rec, hidden, job_out):
        e, rnd = rec["graph_hash"], rec["round"]
        if rnd == 0:
            bar.update(1)
        if job_out is not None:
            res = job_out["result"]
            known = res["label_status"] == "known"
            rec.update(label_status=res["label_status"], verifier_pass=res["verifier_pass"],
                       candidate_invalid=(int(not res["verifier_pass"]) if known else None),
                       verifier_fail_reason=res["fail_reason"], verifier_summary=res["summary"],
                       verifier_metrics=res["metrics"], verifier_wall_s=res["wall_s"], verifier_steps=res["n_steps"])
            if "nochange" in job_out:
                rec["offline_nochange_pass"] = job_out["nochange"]["verifier_pass"]
            fb = None
            if known and not res["verifier_pass"]:
                fb = make_feedback(snaps[e], res, rnd + 1)
                if rnd < args.max_reprompts and not stop_flag:
                    heapq.heappush(heap, (rnd + 1, rank[e], e, fb, rec["action"]))
            rec["verifier_feedback"] = fb
        buf_r.append(rec)
        buf_h.append(hidden)
        progress_update(rec)
        if len(buf_r) >= args.flush_every:
            flush()

    stop_flag = False
    pending = {}
    n_gen = 0
    n_round0_total = len(order)
    n_round0_done = sum(1 for e in order if e in last)
    stats = {"cand": sum(len(v) for v in history.values()), "r0_pass": 0, "r0_known": 0, "ep_pass": 0}
    for e, r in last.items():
        stats["ep_pass"] += int(bool(r.get("verifier_pass")))
    for e, hs in history.items():
        r0 = [h[2] for h in hs if h[0] == 0]
        if r0 and r0[0].get("verifier_pass") is not None:
            stats["r0_known"] += 1
            stats["r0_pass"] += int(bool(r0[0]["verifier_pass"]))
    bar = tqdm(total=n_round0_total, initial=n_round0_done, desc="first proposals", unit="ep",
               dynamic_ncols=True, mininterval=5)
    gen_times = []

    def progress_update(rec):
        stats["cand"] += 1
        if rec["round"] == 0 and rec.get("verifier_pass") is not None:
            stats["r0_known"] += 1
            stats["r0_pass"] += int(bool(rec["verifier_pass"]))
        if rec.get("verifier_pass"):
            stats["ep_pass"] += 1
        q0 = sum(1 for h in heap if h[0] == 0)
        bar.set_postfix_str(
            f"cand {stats['cand']} | r0 pass {stats['r0_pass']}/{stats['r0_known']} | episodes solved {stats['ep_pass']} | "
            f"queue r0 {q0} reprompt {len(heap) - q0} | verifying {len(pending)} | "
            f"{(np.mean(gen_times[-50:]) if gen_times else 0):.1f} s/proposal", refresh=False)
        write_json(run_dir / "progress.json", {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "hours_this_session": round((time.time() - t_start) / 3600, 2),
            "first_proposals_done": bar.n, "first_proposals_total": n_round0_total, "candidates": stats["cand"],
            "round0_pass": stats["r0_pass"], "round0_labelled": stats["r0_known"], "episodes_solved": stats["ep_pass"],
            "queue_round0": q0, "queue_reprompt": len(heap) - q0, "verifying": len(pending),
            "sec_per_proposal_recent": round(float(np.mean(gen_times[-50:])) if gen_times else 0.0, 2),
        })
    # Verifier workers are single-threaded numerically; stop their BLAS/OpenMP pools from
    # spinning on extra cores and competing with the GPU generation loop.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        while heap or pending:
            done_now = [f for f in pending if f.done()]
            for f in done_now:
                rec, hidden = pending.pop(f)
                finish(rec, hidden, f.result())
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                stop_flag = True
            if heap and not stop_flag:
                rnd, _, e, fb, prev = heapq.heappop(heap)
                s = snaps[e]
                msgs = build_messages(s, fb, prev)
                try:
                    out = run_candidate(model, tok, msgs, lambda rendered: region_char_spans(rendered, msgs), cfg)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    out = {"feature_status": "oom", "answer_text": "", "generated_text": ""}
                n_gen += 1
                gen_times.append(out.get("gen_latency_s", 0.0) + (out.get("feat_latency_s") or 0.0))
                parsed = parse_action(out.get("answer_text", ""))
                hidden = out.pop("hidden", {})
                rec = {
                    "instance_id": f"{e}_r{rnd}", "graph_hash": e, "partition": part_of[e], "round": rnd,
                    "family": s.spec.family, "severity": s.spec.params, "onset_s": s.spec.onset_s, "t_trigger": s.t_trigger,
                    "prompt_version": prompt_version, "previous_action": prev,
                    "user_prompt_sha256": hashlib.sha256(msgs[1]["content"].encode()).hexdigest(),
                    **parsed, **{k: v for k, v in out.items() if k != "features"},
                    "context_action": context_action(s.state, parsed["action"] if parsed["schema_valid"] else None,
                                                     rnd, _prev_reason(fb)),
                    "features": out.get("features", {}),
                }
                earlier = [h for h in history.get(e, []) if parsed["action"] is not None and h[1] == _key(parsed["action"])]
                if parsed["action"] is not None:
                    history.setdefault(e, []).append((rnd, _key(parsed["action"]), rec))
                if parsed["schema_valid"] == 1 and out.get("feature_status") == "ok" and earlier:
                    src = earlier[0][2]
                    rec.update(label_status=src.get("label_status"), verifier_pass=src.get("verifier_pass"),
                               candidate_invalid=src.get("candidate_invalid"), verifier_fail_reason=src.get("verifier_fail_reason"),
                               verifier_summary=src.get("verifier_summary"), verifier_metrics=src.get("verifier_metrics"),
                               verifier_wall_s=None, verifier_steps=None, verifier_feedback=None,
                               repeat_of_round=earlier[0][0])
                    finish(rec, hidden, None)
                elif parsed["schema_valid"] == 1 and out.get("feature_status") == "ok":
                    fut = pool.submit(verify_job, s, parsed["action"], rnd == 0)
                    pending[fut] = (rec, hidden)
                else:
                    rec.update(label_status="not_verified_format_failure", verifier_pass=None, candidate_invalid=None)
                    finish(rec, hidden, None)
            elif pending:
                wait(list(pending), return_when=FIRST_COMPLETED)
            else:
                break
    flush()
    bar.close()
    print(json.dumps({"run_dir": str(run_dir), "generated_this_session": n_gen, "stopped_by_time": stop_flag,
                      "remaining_in_queue": len(heap), "hours": round((time.time() - t_start) / 3600, 2)}))


def _key(action):
    return tuple(round(float(action[k]), 9) for k in ("T_sp", "L_sp", "Fin_sp"))


def _prev_reason(feedback):
    if not feedback:
        return None
    for k in ("no_recovery", "slow_recovery", "unsafe_fraction"):
        if f"fail_reason={k}" in feedback:
            return k
    return None


if __name__ == "__main__":
    main()
