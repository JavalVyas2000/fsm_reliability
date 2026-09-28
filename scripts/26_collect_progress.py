"""
Progress of a running (or finished) scripts/25 collection, from its saved records.

Example:
    python -m scripts.26_collect_progress                      # newest run
    python -m scripts.26_collect_progress --run_dir outputs/cstr_collect/<run>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from src.utils.manifest import REPO_ROOT


def bar(done: int, total: int, width: int = 40) -> str:
    frac = done / total if total else 0.0
    fill = int(round(width * frac))
    return f"[{'#' * fill}{'.' * (width - fill)}] {done}/{total} ({100 * frac:.1f}%)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None)
    args = p.parse_args()
    if args.run_dir:
        run = REPO_ROOT / args.run_dir
    else:
        runs = sorted((REPO_ROOT / "outputs/cstr_collect").glob("*/"), key=lambda d: d.stat().st_mtime)
        run = runs[-1]
    prog = json.loads((run / "progress.json").read_text()) if (run / "progress.json").exists() else {}
    recs = [json.loads(l) for l in open(run / "records.jsonl", encoding="utf-8")] if (run / "records.jsonl").exists() else []
    total = prog.get("first_proposals_total", 3000)
    eps = defaultdict(list)
    for r in recs:
        eps[r["graph_hash"]].append(r)
    r0 = [r for r in recs if r["round"] == 0 and r.get("verifier_pass") is not None]
    rounds = Counter(r["round"] for r in recs)
    solved = sum(any(x.get("verifier_pass") for x in rs) for rs in eps.values())
    print(f"run: {run.name}   (progress.json updated {prog.get('updated', '?')})")
    print("first proposals  " + bar(prog.get("first_proposals_done", len({r['graph_hash'] for r in recs if r['round'] == 0})), total))
    print(f"candidates saved: {len(recs)}   by round: {dict(sorted(rounds.items()))}")
    if r0:
        print(f"round-0 pass rate: {sum(bool(r['verifier_pass']) for r in r0)}/{len(r0)} = "
              f"{sum(bool(r['verifier_pass']) for r in r0) / len(r0):.1%}")
    print(f"episodes solved (any round): {solved}/{len(eps)}   repeats stopped: {sum(r.get('repeat_of_round') is not None for r in recs)}")
    fam = defaultdict(lambda: [0, 0])
    for r in r0:
        fam[r["family"]][0] += bool(r["verifier_pass"])
        fam[r["family"]][1] += 1
    if fam:
        print("round-0 pass by family: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(fam.items())))
    if prog:
        rate = prog.get("sec_per_proposal_recent") or 0
        left0 = total - prog.get("first_proposals_done", 0)
        print(f"queue: round-0 {prog.get('queue_round0')}  reprompt {prog.get('queue_reprompt')}  verifying {prog.get('verifying')}   "
              f"{rate:.1f} s/proposal   session {prog.get('hours_this_session')} h")
        if rate and left0:
            print(f"ETA all first proposals: ~{left0 * rate / 3600:.1f} h")


if __name__ == "__main__":
    main()
