"""
Interpretable account of routing decisions for FSM answers (grounding policy set).

For each routed answer: the decision and risk, the weakest-grounded step (where the model
looked among the adjacency lines while writing it, versus the line that decides the
step), and the probe's per-feature contributions to the risk score. The verifier's
verdict is shown separately as offline ground truth; it is never an input.

Inputs: a certification run of the grounding policy set (scripts/17), which points to the
frozen policies and to the augmented certification records with grounding.jsonl.

Outputs (in the certification run directory):
    explanations.jsonl   one structured explanation per routed answer
    explanations.md      readable cards for a selection of cases

Example:
    python -m scripts.29_explain_flags --cert_run outputs/certification/<grounding certify run>
"""
from __future__ import annotations

# torch must be imported before pandas on this Windows setup (WinError 1114 otherwise).
import torch  # noqa: F401,I001

import argparse
import importlib
import json

import joblib
import numpy as np
import pandas as pd

from src.utils.manifest import REPO_ROOT

baseline = importlib.import_module("scripts.12_fit_fsm_baseline")

READABLE = {
    "num_nodes": "graph size (nodes)", "num_edges": "number of edges", "edge_density": "edge density",
    "shortest_length": "shortest path length", "answer_length": "answer path length",
    "answer_is_two_node": "answer is a single hop", "answer_repeats_node": "answer revisits a node",
    "answer_endpoints_match": "answer starts/ends at the right states",
    "grd_min_share": "weakest step: attention share on the deciding line",
    "grd_mean_share": "average attention share on the deciding line",
    "grd_top1_frac": "share of steps where the deciding line got the most attention",
    "grd_first_share": "first step: attention share on the deciding line",
    "grd_min_share_hmax": "weakest step: best single head's share on the deciding line",
}


def readable(name: str) -> str:
    if name in READABLE:
        return READABLE[name]
    if name.startswith("grd_L"):
        layer = name[4:8]
        kind = "min share on deciding line" if "min_share" in name else "fraction of steps deciding line is top"
        return f"layer {layer[1:]}% depth: {kind}"
    return name


def contributions(probe, row: pd.Series):
    """Per-feature contribution to the logit of the linear probe (scalar-only probes)."""
    if probe["hidden"]:
        return None
    pipe = probe["pipeline"]
    x = row[probe["scalar"]].to_numpy(dtype=float)[None, :]
    z = pipe.named_steps["features"].transform(x)[0]
    coef = pipe.named_steps["lr"].coef_[0]
    return sorted(((probe["scalar"][j], float(coef[j] * z[j]), float(x[0, j])) for j in range(len(z))),
                  key=lambda t: -abs(t[1]))


def bar(v: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "█" * n + "·" * (width - n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cert_run", type=str, required=True)
    p.add_argument("--policy", type=str, default="G1_context_grounding")
    p.add_argument("--n_cards", type=int, default=4, help="cards per case type in the markdown")
    args = p.parse_args()

    cdir = REPO_ROOT / args.cert_run
    cfg = json.loads((cdir / "run_config.json").read_text())
    frozen = REPO_ROOT / cfg["frozen_dir"]
    aug = REPO_ROOT / cfg["inference_dir"]
    gdir = aug.parent
    policies = json.loads((frozen / "policies.json").read_text())
    probes = joblib.load(frozen / "probes.joblib")
    pol = next(x for x in policies["policies"] if x["id"] == args.policy)
    probe = probes[pol["allow"][0]]

    df = baseline.load_records(aug).set_index("instance_id")
    grd = {json.loads(l)["instance_id"]: json.loads(l) for l in open(gdir / "grounding.jsonl", encoding="utf-8")}
    dec = pd.read_csv(cdir / "cert_routing_decisions.csv")
    dec = dec[dec.policy == args.policy]
    inf_man = json.loads((aug / "run_manifest.json").read_text())
    ds_dir = REPO_ROOT / inf_man["data"]["dataset_dir"]
    rows = pd.concat([pd.read_csv(f) for f in ds_dir.glob("*.csv")]).set_index("instance_id")

    def logit(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))

    out = []
    for _, d in dec.iterrows():
        iid = d.instance_id
        r = df.loc[iid]
        X = baseline.build_matrix(df.loc[[iid]], probe["scalar"], probe["hidden"], {})
        risk = float(probe["platt"].predict_proba(logit(probe["pipeline"].predict_proba(X)[:, 1])[:, None])[0, 1])
        g = grd.get(iid, {})
        steps = g.get("steps", [])
        graph_lines = {int(l.split(":")[0]): l for l in rows.loc[iid, "graph_text"].split("\n")}
        weakest = None
        if steps:
            sh = [np.nanmean([s[f"{t}_share"] for t in ("L025", "L050", "L075", "L100")]) for s in steps]
            k = int(np.nanargmin(sh))
            s = steps[k]
            dist = {int(u): v for u, v in s.get("line_share_layer_mean", {}).items()}
            top_line = max(dist, key=dist.get) if dist else None
            weakest = {
                "step_index": k + 1, "u": s["u"], "v": s["v"], "share_on_deciding_line": float(sh[k]),
                "deciding_line": graph_lines.get(s["u"]), "most_attended_line": graph_lines.get(top_line) if top_line is not None else None,
                "most_attended_share": float(dist[top_line]) if top_line is not None else None,
                "step_valid_offline_truth": bool(s["valid"]), "attention_distribution": dist,
            }
        per_step = []
        for j, st in enumerate(steps):
            dist_j = {int(u): v for u, v in st.get("line_share_layer_mean", {}).items()}
            top_j = max(dist_j, key=dist_j.get) if dist_j else None
            per_step.append({"step": j + 1, "u": st["u"], "v": st["v"],
                             "share_on_deciding_line": float(np.nanmean([st[f"{t}_share"] for t in ("L025", "L050", "L075", "L100")])),
                             "deciding_line_is_most_attended": top_j == st["u"], "most_attended_node": top_j,
                             "legal_offline_truth": bool(st["valid"])})
        endpoints_ok = (e_ok := (isinstance(r.get("parsed_path"), list) and len(r["parsed_path"]) > 0
                                 and r["parsed_path"][0] == int(rows.loc[iid, "start"]) and r["parsed_path"][-1] == int(rows.loc[iid, "goal"])))
        contrib = contributions(probe, r)
        out.append({
            "instance_id": iid, "policy": args.policy, "route": d.route, "risk": risk,
            "answer": r["answer_text"], "start": int(rows.loc[iid, "start"]), "goal": int(rows.loc[iid, "goal"]),
            "weakest_step": weakest, "steps": per_step, "endpoints_match": bool(endpoints_ok),
            "top_contributions": [{"feature": f, "readable": readable(f), "contribution_to_logit": c, "value": v}
                                  for f, c, v in (contrib or [])[:6]],
            "verifier_verdict_offline": "invalid" if int(d.y_fail) == 1 else "valid",
        })
    with open(cdir / "explanations.jsonl", "w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e) + "\n")

    def card(e):
        w = e["weakest_step"]
        lines = [f"### {e['instance_id']} — route **{e['route']}**, predicted failure risk {e['risk']:.2f} "
                 f"(verifier, offline: **{e['verifier_verdict_offline']}**)",
                 f"Task: path from {e['start']} to {e['goal']}. Answer: `{e['answer']}`", ""]
        if w:
            lines += ["**Step-by-step grounding** (share of graph attention on the line that decides each step; legality is offline truth):"]
            for st in e["steps"]:
                lines.append(f"- step {st['step']}: {st['u']} → {st['v']} — {st['share_on_deciding_line']:.0%} on `{st['u']}:` line"
                             + ("" if st["deciding_line_is_most_attended"] else f" (looked most at line `{st['most_attended_node']}:`)")
                             + f" — {'legal' if st['legal_offline_truth'] else '**ILLEGAL**'}")
            if not e["endpoints_match"]:
                lines.append(f"- the answer does not start at {e['start']} and end at {e['goal']} (visible without the verifier)")
            lines += [""]
            lines += [f"Weakest-grounded step {w['step_index']}: {w['u']} → {w['v']}. While writing "
                      f"`{w['v']}` the model put **{w['share_on_deciding_line']:.0%}** of its graph attention on the line that "
                      f"decides this step, `{w['deciding_line']}`"
                      + (f"; it looked most at `{w['most_attended_line']}` ({w['most_attended_share']:.0%})." if w["most_attended_line"] and w["most_attended_line"] != w["deciding_line"] else ".")
                      + f" (Offline truth: this step is {'legal' if w['step_valid_offline_truth'] else 'illegal'}.)", "",
                      "Attention over graph lines for this step:", "```"]
            shown = sorted(w["attention_distribution"].items(), key=lambda t: -t[1])[:6]
            if w["u"] not in [u for u, _ in shown] and w["u"] in w["attention_distribution"]:
                shown.append((w["u"], w["attention_distribution"][w["u"]]))
            for u, v in shown:
                mark = "  <- deciding line" if u == w["u"] else ""
                lines.append(f"{str(u).rjust(3)}: {bar(v)} {v:5.1%}{mark}")
            lines += ["```"]
        if e["top_contributions"]:
            lines += ["**Why the probe scored this risk** (contribution to the risk logit; + raises risk):"]
            for c in e["top_contributions"][:4]:
                lines.append(f"- {c['readable']}: {c['contribution_to_logit']:+.2f} (value {c['value']:.3g})")
        return "\n".join(lines + [""])

    E = pd.DataFrame(out)
    groups = [
        ("Failures caught (DISALLOW, verifier: invalid)", (E.route == "DISALLOW") & (E.verifier_verdict_offline == "invalid")),
        ("Correctly trusted (ALLOW, verifier: valid)", (E.route == "ALLOW") & (E.verifier_verdict_offline == "valid")),
        ("Failures let through (ALLOW, verifier: invalid)", (E.route == "ALLOW") & (E.verifier_verdict_offline == "invalid")),
        ("Valid answers rejected (DISALLOW, verifier: valid)", (E.route == "DISALLOW") & (E.verifier_verdict_offline == "valid")),
    ]
    md = [f"# Interpretable routing decisions — {args.policy} ({json.loads((frozen / 'freeze_manifest.json').read_text()).get('baseline_dir')})", "",
          "Verifier verdicts are shown as offline truth only; they are never used by the policy.", ""]
    for title, mask in groups:
        sub = [out[i] for i in np.flatnonzero(mask.to_numpy())]
        md += [f"## {title} — {len(sub)} cases", ""]
        for e in sub[: args.n_cards]:
            md.append(card(e))
    (cdir / "explanations.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"explanations": str(cdir / "explanations.md"), "n": len(out),
                      "counts": {t: int(m.sum()) for t, m in groups}}))


if __name__ == "__main__":
    main()
