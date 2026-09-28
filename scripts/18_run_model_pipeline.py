"""
Run the full FSM pipeline for one model, in protocol order, with the policy freeze placed
before the certification inference:

    smoke (5) -> pilot inference -> probes (12) -> routing (13) -> time (14) -> agreement (15)
    -> FREEZE (16) -> certification inference -> certify (17)

Each step is a subprocess; the new output directory of each step is discovered by diffing
the output root, and the whole chain is logged to outputs/model_pipelines/<run>/pipeline.json.
Stops at the first failing step (later steps depend on earlier ones).

Example:
    python -m scripts.18_run_model_pipeline --model meta-llama/Llama-3.2-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.utils.manifest import REPO_ROOT, make_run_dir, write_json

PILOT_DATA = "data/v2/fsm_pilot_seed20260923"
CERT_DATA = "data/v2/fsm_cert_seed20260924"
ROUTING_SETTINGS = ["0.0:0.10", "0.05:0.05", "0.10:0.10", "0.20:0.20"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=96)
    return p.parse_args()


def new_dir(root: Path, before: set) -> Path:
    after = {d for d in root.iterdir() if d.is_dir()} if root.exists() else set()
    created = sorted(after - before, key=lambda d: d.stat().st_mtime)
    if not created:
        raise RuntimeError(f"No new output directory under {root}")
    return created[-1]


def main():
    args = parse_args()
    tag = args.model.split("/")[-1].lower().replace(".", "")
    log_dir = make_run_dir(REPO_ROOT / "outputs/model_pipelines", tag)
    log = {"model": args.model, "steps": [], "status": "running"}
    py = sys.executable

    def step(name: str, module: str, argv: list, out_root: str):
        root = REPO_ROOT / out_root
        before = {d for d in root.iterdir() if d.is_dir()} if root.exists() else set()
        t0 = time.time()
        with open(log_dir / f"{len(log['steps']):02d}_{name}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run([py, "-m", module, *argv], cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        entry = {"step": name, "returncode": rc, "wall_s": round(time.time() - t0, 1)}
        if rc == 0:
            entry["output_dir"] = str(new_dir(root, before).relative_to(REPO_ROOT))
        log["steps"].append(entry)
        write_json(log_dir / "pipeline.json", log)
        if rc != 0:
            log["status"] = f"failed at {name}"
            write_json(log_dir / "pipeline.json", log)
            raise SystemExit(f"{name} failed (rc={rc}); see {log_dir}")
        return entry["output_dir"]

    common = ["--model", args.model, "--max_new_tokens", str(args.max_new_tokens), "--local_files_only"]
    step("smoke", "scripts.11_run_fsm_inference_v2",
         ["--dataset_dir", PILOT_DATA, "--partitions", "train", "--limit", "5", "--tag", "smoke5",
          "--out_root", "outputs/fsm_inference_smoke", *common], "outputs/fsm_inference_smoke")
    pilot = step("pilot_inference", "scripts.11_run_fsm_inference_v2",
                 ["--dataset_dir", PILOT_DATA, "--tag", "pilot3000", *common], "outputs/fsm_inference")
    base = step("probes", "scripts.12_fit_fsm_baseline", ["--inference_dir", pilot, "--tag", f"{tag}_pilot"],
                "outputs/fsm_baseline")
    routing = step("routing", "scripts.13_routing_table",
                   ["--baseline_dir", base, "--tag", f"{tag}_pilot", "--settings", *ROUTING_SETTINGS],
                   "outputs/fsm_selective_verification")
    tdir = step("time_savings", "scripts.14_time_savings", ["--routing_dir", routing, "--tag", f"{tag}_pilot"],
                "outputs/fsm_time_savings")
    step("agreement", "scripts.15_agreement_routing", ["--baseline_dir", base, "--time_dir", tdir, "--tag", f"{tag}_pilot"],
         "outputs/fsm_agreement_routing")
    frozen = step("freeze", "scripts.16_freeze_policies", ["--baseline_dir", base, "--tag", f"{tag}_frozen_policies"],
                  "outputs/certification")
    cert_inf = step("cert_inference", "scripts.11_run_fsm_inference_v2",
                    ["--dataset_dir", CERT_DATA, "--partitions", "cert", "--tag", "cert3000", *common], "outputs/fsm_inference")
    step("certify", "scripts.17_certify", ["--frozen_dir", frozen, "--inference_dir", cert_inf, "--tag", f"{tag}_certify"],
         "outputs/certification")
    log["status"] = "complete"
    write_json(log_dir / "pipeline.json", log)
    print(f"Pipeline complete: {log_dir}")


if __name__ == "__main__":
    main()
