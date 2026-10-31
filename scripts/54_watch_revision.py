"""
Live status of the revision queue (docs/revision_prereg.md): every --every minutes, print one block with a
progress bar per step (parsed from the job logs), ETA, GPU load and the last queue-log lines. The same block is
written to outputs/revision_status.txt. Stops by itself once the queue log says ITEM 2B DONE.

Example (opens nothing; run it in a terminal you keep open):
    python -m scripts.54_watch_revision --every 10
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
QLOG = OUT / "revision_queue.log"
MODELS = [("smollm2-17b", "SmolLM2-1.7B"), ("qwen25-15b", "Qwen2.5-1.5B"), ("qwen25-3b", "Qwen2.5-3B"),
          ("llama-32-3b", "Llama-3.2-3B"), ("qwen25-7b", "Qwen2.5-7B")]
# (label, job key in the queue log); a step is done when "END   <key>" appears
STEPS = ([("build 400 fresh closed-loop episodes", "build_closedloop2"),
          ("item 2  fresh R0 test   SmolLM2-1.7B", "closedloop2_smollm2-17b"),
          ("item 2  fresh R0 test   Qwen2.5-7B", "closedloop2_qwen25-7b")]
         + [(f"item 1  cert2 {stage:<9} {name}", f"cert2_{stage}_{key}")
            for key, name in MODELS for stage in ("collect", "grounding", "eval")]
         + [("item 2b R0 + validate rejections Qwen2.5-7B", "closedloop2v_qwen25-7b"),
            ("item 2b R0 + validate rejections SmolLM2-1.7B", "closedloop2v_smollm2-17b")])
DONE = "ITEM 2B DONE"  # last line written by outputs/revision_queue_2b.ps1, which runs after the main queue


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="replace") if p.exists() else ""


def bar(i, n, width=28):
    f = 0 if not n else min(1.0, i / n)
    full = int(f * width)
    return "[" + "#" * full + "-" * (width - full) + f"] {100 * f:5.1f}%  {i}/{n}"


def job_progress(key: str):
    """(done, total, eta-string) parsed from a job log: closed-loop '[i/n] ... | m min' lines or the last tqdm bar."""
    text = read(OUT / f"queue_rev_{key}_resume.log") or read(OUT / f"queue_rev_{key}.log")
    if not text:
        return None
    cl = re.findall(r"^\[(\d+)/(\d+)\].*\|\s*([\d.]+) min", text, flags=re.M)
    if cl:
        pts = [(int(a), int(b), float(c)) for a, b, c in cl]
        i, n, m = pts[-1]
        seg = pts[-30:]
        rate = (seg[-1][2] - seg[0][2]) / max(1, seg[-1][0] - seg[0][0]) if len(seg) > 1 else None
        eta = (dt.datetime.now() + dt.timedelta(minutes=(n - i) * rate)).strftime("%a %H:%M") if rate else "?"
        return i, n, f"ETA {eta}"
    tq = re.findall(r"\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+)", text)
    if tq:
        i, n, _, left = tq[-1]
        return int(i), int(n), f"{left} left"
    return None


def started(log: str, key: str):
    """Start time of the latest attempt (a restart logs the job again as <key>_resume)."""
    m = re.findall(rf"^(\S+) START {re.escape(key)}(?:_resume)?$", log, flags=re.M)
    return dt.datetime.fromisoformat(m[-1]) if m else None


def gpu():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        u, used, tot = [x.strip() for x in q.split(",")]
        return f"GPU {u}% | {int(used) / 1024:.1f}/{int(tot) / 1024:.1f} GB"
    except Exception:
        return "GPU n/a"


def snapshot() -> str:
    log = read(QLOG)
    now = dt.datetime.now()
    lines = [f"=== revision queue status {now:%Y-%m-%d %H:%M} | {gpu()} ==="]
    for label, key in STEPS:
        if f"END   {key}" in log:
            ex = re.findall(rf"END   {re.escape(key)} \(exit (\d+)\)", log)
            mark = "done " if ex and ex[-1] == "0" else f"FAIL (exit {ex[-1]})" if ex else "done "
            lines.append(f"  [{mark}] {label}")
        elif (t0 := started(log, key)):
            p = job_progress(key)
            el = str(now - t0).split(".")[0]
            lines.append(f"  [ RUN ] {label}  {bar(p[0], p[1]) + '  ' + p[2] if p else 'running'}  (elapsed {el})")
        else:
            lines.append(f"  [ ... ] {label}")
    b = read(OUT / "build_cert2.log")
    cert2_built = (ROOT / "data" / "cstr" / "cert2_v4" / "dataset_manifest.json").exists()
    passes = re.findall(r"cert2: (\d+)/(\d+) usable this pass", b)
    lines.append(f"  [{'done ' if cert2_built else ' RUN '}] build 2000 cert2 episodes (parallel)"
                 + ("" if cert2_built else f"  {len(passes)} pass(es) finished" if passes else "  first pass running"))
    tail = [l for l in log.splitlines() if l.strip()][-3:]
    lines += ["  last queue lines:"] + [f"    {l}" for l in tail]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--every", type=float, default=10.0, help="minutes between status blocks")
    a = p.parse_args()
    while True:
        s = snapshot()
        print(s + "\n", flush=True)
        (OUT / "revision_status.txt").write_text(s + "\n", encoding="utf-8")
        if DONE in read(QLOG):
            print("Queue finished.", flush=True)
            break
        time.sleep(60 * a.every)


if __name__ == "__main__":
    main()
