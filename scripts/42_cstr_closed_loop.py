"""
Closed-loop CSTR episodes with skip-the-validator routing (docs/cstr_closed_loop_prereg.md).

Per episode and policy, up to --max_proposals proposals (prompt v3.1):
    always_validate   validate every proposal; pass -> execute, fail -> re-prompt with neutral feedback
    never_validate    execute the first proposal
    observables_probe / combined_probe
                      frozen probe risk: ACCEPT -> execute, REJECT -> re-prompt unchecked, else validate
    random_matched    ACCEPT / REJECT / validate at the combined probe's dev rates
Executed outcome = CAR rollout validator on the plant copy with the plant's own future noise.
Every generated proposal is verified in the background (its verdict is the executed outcome or
the ground truth for an unchecked reject); a validator call is charged to a policy only when that
policy validates. Generations and verdicts are cached and shared across policies; measured times
are charged per policy.

Outputs (outputs/cstr_closed_loop/<run>/): steps.jsonl (one row per policy decision),
episodes.jsonl (one row per episode and policy), run_manifest.json. Resumable with --run_dir.

Example:
    python -m scripts.42_cstr_closed_loop --dataset_dir data/cstr/closedloop_v4 \
        --frozen_dir outputs/certification/<freeze> --tag qwen25-3b
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: I001

import argparse
import copy
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from src.cstr.episodes import set_kg_context
from src.cstr.llm_io import context_action, parse_action, region_char_spans
from src.cstr.prompt_v3 import build_messages_v3, neutral_feedback
from src.features.grounding_cstr import (
    candidate_features_v31, cstr_regions_v31, grounding_v31, setpoint_token_rows, token_masks,
)
from src.models.inference_v2 import InternalsConfig, PrefixCache, decision_pass, run_candidate, selected_layers
from src.models.load_model import load_hf_model_and_tokenizer, resolved_revision
from src.utils.manifest import REPO_ROOT, build_manifest, make_run_dir, sha256_file, write_json

S12 = importlib.import_module("scripts.12_fit_fsm_baseline")
S25 = importlib.import_module("scripts.25_cstr_collect")

VERSION = "cstr_prompt_v3.1"
POLICIES = ["always_validate", "never_validate", "observables_probe", "combined_probe", "random_matched", "internals_probe",
            # Amendment 4: same probes, but a retry (round > 0) is never accepted unchecked; it is validated instead
            "observables_probe_r0", "combined_probe_r0", "internals_probe_r0"]
PROBE_SIGNAL = {"observables_probe": "plant readings + proposed change",
                "combined_probe": "readings + change + all internals + grounding",
                # Amendment 2 of docs/cstr_closed_loop_prereg.md: model internals only, no plant readings or action
                "internals_probe": "all internals + grounding",
                "observables_probe_r0": "plant readings + proposed change",
                "combined_probe_r0": "readings + change + all internals + grounding",
                "internals_probe_r0": "all internals + grounding"}
REJECT_MSG = "The previous proposal was rejected by a risk screen before validation."


def _verify(args):
    from src.cstr.episodes import verify

    snap, action = args
    return verify(snap, action)


def _key(a):
    return tuple(round(float(a[k]), 9) for k in ("T_sp", "L_sp", "Fin_sp"))


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--frozen_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--max_proposals", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--quantization", choices=["4bit"], default=None, help="must match the model's collection run")
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES,
                   help="subset of policies (supplementary runs pair with an earlier run by episode)")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


class Runner:
    def __init__(self, args):
        self.args = args
        fz = REPO_ROOT / args.frozen_dir
        man = json.loads((fz / "freeze_manifest.json").read_text())
        for f, h in man["files_sha256"].items():
            if sha256_file(fz / f) != h:
                raise SystemExit(f"frozen file changed since the freeze: {f}")
        rules = json.loads((fz / "rules.json").read_text())
        self.rules, self.probes = rules["rules"], joblib.load(fz / "probes.joblib")
        self.tkey = f"t_accept_{int(round(args.tolerance * 100)):02d}"
        dev = pd.read_csv(fz / "dev_routing.csv")
        r = dev[(dev.signal == PROBE_SIGNAL["combined_probe"]) & (np.isclose(dev.tolerance, args.tolerance))].iloc[0]
        self.p_acc, self.p_rej = r.accepted / r.dev_n, r.rejected / r.dev_n

        self.model, self.tok = load_hf_model_and_tokenizer(args.model, device_map=args.device, torch_dtype="bfloat16",
                                                           attn_implementation="sdpa", local_files_only=args.local_files_only,
                                                           quantization=args.quantization)
        self.cfg = InternalsConfig(max_new_tokens=args.max_new_tokens)
        self.cfg_g = InternalsConfig(max_new_tokens=0, collect_hidden=False)
        self.blocks = selected_layers(self.model.config.num_hidden_layers, self.cfg_g.relative_layers)
        self.gen_cache, self.ver_cache = {}, {}

    def set_prefix(self, snap):
        ids = S25.shared_prefix_ids(self.tok, build_messages_v3(snap, version=VERSION), self.cfg.chat_template_kwargs)
        self.cfg.prefix_cache = PrefixCache(self.model, ids)

    # ----------------------------------------------------------------- model calls (cached)
    def generate(self, snap, fb, prev):
        msgs = build_messages_v3(snap, fb, prev, version=VERSION)
        h = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()
        if h in self.gen_cache:
            return self.gen_cache[h]
        out = None
        for attempt in range(2):  # CUDA out-of-memory on the 8 GB card: free cached blocks and retry once
            try:
                out = run_candidate(self.model, self.tok, msgs, lambda rendered: region_char_spans(rendered, msgs), self.cfg)
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(f"CUDA out of memory (attempt {attempt + 1}); emptying cache and retrying", flush=True)
                torch.cuda.empty_cache()
        if out is None:  # both attempts failed: treated as an unparseable answer (the episode ends unresolved)
            out = {"feature_status": "oom", "answer_text": "", "generated_ids": [], "answer_token_count": 0,
                   "gen_latency_s": 0.0, "feat_latency_s": 0.0}
        hid = out.pop("hidden", {}) or {}
        vec = np.concatenate([hid[k].astype(np.float32) for k in sorted(hid) if k.startswith("hid_")]) if hid else None
        parsed = parse_action(out.get("answer_text", ""))
        cand = {"sha": h, "msgs": msgs, "out": out, "parsed": parsed, "hidden_vec": vec,
                "gen_s": float(out.get("gen_latency_s") or 0.0), "feat_s": float(out.get("feat_latency_s") or 0.0),
                "g31": None, "grd_s": None}
        self.gen_cache[h] = cand
        return cand

    @torch.no_grad()
    def grounding(self, cand):
        if cand["g31"] is not None:
            return cand["g31"]
        t0 = time.perf_counter()
        msgs, out, pc = cand["msgs"], cand["out"], self.cfg.prefix_cache
        rendered = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs)
        enc = self.tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        P, n_pre = len(enc["input_ids"]), len(pc.ids)
        ans = out["generated_ids"][: int(out["answer_token_count"])]
        rows = setpoint_token_rows(self.tok, ans)
        feats = candidate_features_v31(None, self.cfg_g.relative_layers)
        if rows is not None and enc["input_ids"][:n_pre] == pc.ids:
            masks = token_masks(enc["offset_mapping"], cstr_regions_v31(rendered, msgs[0]["content"], msgs[1]["content"]))
            ids = torch.tensor([enc["input_ids"]], device=self.model.device)
            cache = copy.deepcopy(pc.cache)
            for s0 in range(n_pre, P - 1, 1024):
                self.model(ids[:, s0: min(s0 + 1024, P - 1)], past_key_values=cache, use_cache=True, logits_to_keep=1)
            _, attn = decision_pass(self.model, cache, ids, torch.tensor(ans, device=self.model.device), self.cfg_g,
                                    attn_blocks=self.blocks)
            fg = grounding_v31(attn, self.blocks, self.cfg_g.relative_layers, masks, rows, P)
            feats = candidate_features_v31(fg, self.cfg_g.relative_layers)
            # release the cache copy and attention maps at once: on an 8 GB card the allocator otherwise keeps
            # them reserved and the next generation spills into shared system memory (10x slower)
            del cache, attn, ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cand["g31"], cand["grd_s"] = feats, time.perf_counter() - t0
        return feats

    def risk(self, label, cand, snap, rnd, prev_reason):
        r = self.rules[label]
        row = {**context_action(snap.state, cand["parsed"]["action"], rnd, prev_reason), **(cand["out"].get("features") or {})}
        if any(c.startswith("g31_") for c in r["scalar"]):
            row.update(self.grounding(cand))
        df = pd.DataFrame([row]).reindex(columns=r["scalar"])
        df["instance_id"] = "x"
        X = S12.build_matrix(df, r["scalar"], r["hidden"], {"x": cand["hidden_vec"]})
        pr = self.probes[label]
        return float(pr["platt"].predict_proba(logit(pr["pipeline"].predict_proba(X)[:, 1])[:, None])[:, 1][0])


def main():
    args = parse_args()
    ds = REPO_ROOT / args.dataset_dir
    set_kg_context((ds / "kg_context.ttl").read_bytes().decode("utf-8"))
    snaps = [s for f in sorted(ds.glob("snapshots_*.pkl")) for s in pickle.load(open(f, "rb"))]
    snaps = snaps[: args.limit] if args.limit else snaps
    run_dir = (REPO_ROOT / args.run_dir) if args.run_dir else make_run_dir(REPO_ROOT / "outputs/cstr_closed_loop", args.tag)
    done = set()
    if (run_dir / "episodes.jsonl").exists():
        done = {json.loads(l)["episode_id"] for l in open(run_dir / "episodes.jsonl", encoding="utf-8")}
    R = Runner(args)
    R.set_prefix(snaps[0])
    if not (run_dir / "run_manifest.json").exists():
        write_json(run_dir / "run_manifest.json", build_manifest(
            prereg={"file": "docs/cstr_closed_loop_prereg.md", "sha256": sha256_file(REPO_ROOT / "docs/cstr_closed_loop_prereg.md")},
            model={"hf_id": args.model, "resolved_revision": resolved_revision(R.model), "quantization": args.quantization},
            prompt=VERSION,
            frozen_dir=args.frozen_dir, tolerance=args.tolerance, policies=args.policies, probe_signal=PROBE_SIGNAL,
            random_rates={"accept": R.p_acc, "reject": R.p_rej}, max_proposals=args.max_proposals,
            dataset_dir=args.dataset_dir, n_episodes=len(snaps), seed=args.seed))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    t_start = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for ei, s in enumerate(snaps):
            e = s.spec.episode_id
            if e in done:
                continue
            futures = {}

            def verdict(action):
                k = (e, _key(action))
                if k not in futures:
                    futures[k] = pool.submit(_verify, (s, {kk: float(action[kk]) for kk in ("T_sp", "L_sp", "Fin_sp")}))
                return futures[k]

            from src.cstr.episodes import current_setpoints
            nochange = verdict(current_setpoints(s))
            state = {p: {"fb": None, "prev": None, "prev_reason": None, "seen": set(), "calls": 0, "proposals": 0,
                         "t_gen": 0.0, "t_feat": 0.0, "t_grd": 0.0, "t_ver": 0.0, "done": False, "outcome": None,
                         "executed": None, "exec_round": None, "unchecked_rejects": 0, "good_rejected": 0}
                     for p in args.policies}
            steps = []
            for rnd in range(args.max_proposals):
                active = [p for p in args.policies if not state[p]["done"]]
                if not active:
                    break
                cands = {}
                for p in active:  # generate first (GPU), verifications run meanwhile (CPU)
                    st = state[p]
                    c = R.generate(s, st["fb"], st["prev"])
                    cands[p] = c
                    if c["parsed"]["schema_valid"] == 1:
                        verdict(c["parsed"]["action"])
                for p in active:
                    st, c = state[p], cands[p]
                    st["proposals"] += 1
                    st["t_gen"] += c["gen_s"]
                    st["t_feat"] += c["feat_s"] if p.startswith(("combined_probe", "internals_probe")) else 0.0  # need internals
                    step = {"episode_id": e, "policy": p, "round": rnd, "prompt_sha": c["sha"],
                            "schema_valid": c["parsed"]["schema_valid"], "action": c["parsed"]["action"]}
                    if c["parsed"]["schema_valid"] != 1:  # CAR aborts on an unparseable answer
                        st.update(done=True, outcome="unresolved_format")
                        step["decision"] = "format_failure"
                        steps.append(step)
                        continue
                    a = c["parsed"]["action"]
                    res = verdict(a).result()
                    repeat = _key(a) in st["seen"]
                    st["seen"].add(_key(a))
                    if p == "always_validate":
                        decision = "validate"
                    elif p == "never_validate":
                        decision = "accept"
                    elif p in PROBE_SIGNAL:
                        lab = PROBE_SIGNAL[p]
                        risk = R.risk(lab, c, s, rnd, st["prev_reason"])
                        if any(col.startswith("g31_") for col in R.rules[lab]["scalar"]):  # grounding pass needed
                            st["t_grd"] += c["grd_s"] or 0.0
                        ta, tr = R.rules[lab][R.tkey], R.rules[lab]["t_reject"]
                        decision = "accept" if ta is not None and risk <= ta else (
                            "reject" if tr is not None and risk >= tr else "validate")
                        if p.endswith("_r0") and rnd > 0 and decision == "accept":
                            decision = "validate"  # first-proposal-only rule: retries are always validated
                        step["risk"] = risk
                    else:
                        u = np.random.default_rng([args.seed, ei, rnd]).random()
                        decision = "accept" if u < R.p_acc else ("reject" if u < R.p_acc + R.p_rej else "validate")
                    step.update(decision=decision, verifier_pass=res["verifier_pass"], repeat=repeat)
                    if decision == "validate" and not repeat:
                        st["calls"] += 1
                        st["t_ver"] += res["wall_s"]
                    if decision == "accept" or (decision == "validate" and res["verifier_pass"]):
                        st.update(done=True, executed=a, exec_round=rnd,
                                  outcome="recovered" if res["verifier_pass"] else f"executed_failure:{res['fail_reason']}")
                    elif decision == "reject":
                        st["unchecked_rejects"] += 1
                        st["good_rejected"] += int(bool(res["verifier_pass"]))
                        st.update(fb=REJECT_MSG, prev=a, prev_reason=None)
                    else:
                        st.update(fb=neutral_feedback(res), prev=a, prev_reason=res["fail_reason"])
                    steps.append(step)
            nc = nochange.result()
            ep_rows = []
            for p in args.policies:
                st = state[p]
                if not st["done"]:
                    st["outcome"] = "unresolved_fallback"
                ep_rows.append({"episode_id": e, "family": s.spec.family, "policy": p, "outcome": st["outcome"],
                                "recovered": st["outcome"] == "recovered",
                                "fallback_recovered": bool(nc["verifier_pass"]) if st["outcome"].startswith("unresolved") else None,
                                "nochange_pass": bool(nc["verifier_pass"]), "exec_round": st["exec_round"],
                                "proposals": st["proposals"], "validator_calls": st["calls"],
                                "unchecked_rejects": st["unchecked_rejects"], "good_rejected": st["good_rejected"],
                                "t_generation_s": st["t_gen"], "t_internals_s": st["t_feat"], "t_grounding_s": st["t_grd"],
                                "t_validator_s": st["t_ver"],
                                "t_total_s": st["t_gen"] + st["t_feat"] + st["t_grd"] + st["t_ver"]})
            with open(run_dir / "steps.jsonl", "a", encoding="utf-8") as f:
                for st_ in steps:
                    f.write(json.dumps(st_) + "\n")
            with open(run_dir / "episodes.jsonl", "a", encoding="utf-8") as f:
                for row in ep_rows:
                    f.write(json.dumps(row) + "\n")
            rec = {p: sum(1 for r in ep_rows if r["policy"] == p and r["recovered"]) for p in args.policies}
            print(f"[{ei + 1}/{len(snaps)}] {e} {s.spec.family}: recovered {rec} | cache {len(R.gen_cache)} gens | "
                  f"{(time.time() - t_start) / 60:.1f} min", flush=True)
            R.gen_cache.clear()  # prompts contain the snapshot, so generations are shared only within an episode
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({"run_dir": str(run_dir), "hours": round((time.time() - t_start) / 3600, 2)}))


if __name__ == "__main__":
    main()
