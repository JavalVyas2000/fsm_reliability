"""
CSTR episode specifications (fresh start): nominal plant, fault onset and fault severity.

Every episode draws, independently and uniformly within the frozen ranges file:
  - a nominal plant (UA, Tin_base, Tc_base, CA_in_base, CD_in_base, Fc_max, L_sp, Fin_sp; T_sp fixed at 310 K);
  - a fault onset time;
  - a fault family (balanced: an equal number per family in every split) and its severity.

`severity_level` puts each family's severity parameter on a common 0-1 scale within its range
(0 = mildest, 1 = most severe). Direction: a lower pump_degrade_factor / outlet_block_factor
(remaining outlet-flow fraction) or stuck_opening (cooling valve opening) is more severe; a
higher fouling_max (fraction of UA lost) is more severe. fouling_tau is the fouling time constant.

No simulation is run here: whether an episode triggers the monitor is decided later.

Example:
    python -m scripts.33_cstr_episode_specs --sizes train=3000 val=1000 test=1000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.cstr.specs import draw_episode
from src.utils.manifest import REPO_ROOT, build_manifest, sha256_file, write_json

PLANT_UNITS = {"Tin_base": "K", "Tc_base": "K"}
COLUMNS = {
    "episode_id": "Unique episode identifier",
    "split": "train / val / test",
    "family": "Fault family",
    "severity_param": "Name of the family's severity parameter",
    "severity_value": "Value of the severity parameter",
    "severity_level": "Severity on a 0-1 scale within the family's range (0 mildest, 1 most severe)",
    "fouling_tau_s": "Fouling time constant in s (fouling only; blank otherwise)",
    "onset_s": "Fault onset time in s after simulation start",
    "noise_seed": "Simulator noise seed (unique per episode)",
    "T_sp": "Temperature setpoint, K (fixed)",
    "UA": "Heat-transfer coefficient x area (simulator units)",
    "Tin_base": "Feed inlet temperature, K",
    "Tc_base": "Coolant inlet temperature, K",
    "CA_in_base": "Inlet concentration of A (simulator units)",
    "CD_in_base": "Inlet concentration of D (simulator units)",
    "Fc_max": "Coolant supply capacity (simulator units)",
    "L_sp": "Nominal level setpoint (simulator units)",
    "Fin_sp": "Nominal feed-flow setpoint (simulator units)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ranges", type=str, default="configs/cstr_severity_ranges_v3.json")
    p.add_argument("--sizes", nargs="+", default=["train=3000", "val=1000", "test=1000"])
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--out", type=str, default="data/cstr/cstr_episode_specs.xlsx")
    return p.parse_args()


def sample_split(rng, ranges, split, n, seeds):
    fams = list(ranges["families"])
    if n % len(fams):
        raise ValueError(f"{split}: size {n} is not divisible by the {len(fams)} families")
    fam_order = rng.permutation(np.repeat(fams, n // len(fams)))
    rows = []
    for i, fam in enumerate(fam_order):
        ep = draw_episode(rng, ranges, fam)
        rows.append({"episode_id": f"cstr_{split}_{i:05d}", "split": split, **ep, "noise_seed": int(next(seeds))})
    return pd.DataFrame(rows, columns=list(COLUMNS))


def main():
    args = parse_args()
    ranges_path = REPO_ROOT / args.ranges
    ranges = json.loads(ranges_path.read_text())
    sizes = {k: int(v) for k, v in (s.split("=") for s in args.sizes)}
    rng = np.random.default_rng(args.seed)
    seeds = iter(rng.choice(2**31 - 1, size=sum(sizes.values()), replace=False))
    splits = {s: sample_split(rng, ranges, s, n, seeds) for s, n in sizes.items()}
    allr = pd.concat(splits.values(), ignore_index=True)

    rng_rows = [{"quantity": f"plant.{k}", "low": a, "high": b, "unit": PLANT_UNITS.get(k, "")}
                for k, (a, b) in ranges["plant"].items()]
    rng_rows.append({"quantity": "onset_s", "low": ranges["onset_s"][0], "high": ranges["onset_s"][1], "unit": "s"})
    for fam, d in ranges["families"].items():
        for k, (a, b) in d["params"].items():
            rng_rows.append({"quantity": f"{fam}.{k}", "low": a, "high": b, "unit": "s" if k.endswith("tau") else ""})
    summary = (allr.groupby(["split", "family"]).agg(n=("episode_id", "size"),
                                                     severity_value_min=("severity_value", "min"),
                                                     severity_value_max=("severity_value", "max"),
                                                     onset_s_mean=("onset_s", "mean"))
               .reset_index())
    readme = pd.DataFrame({
        "item": ["generator", "ranges file", "seed", "sizes", "sampling", "not yet known"] + list(COLUMNS),
        "description": [
            "scripts/33_cstr_episode_specs.py", f"{args.ranges} ({ranges['version']})", str(args.seed),
            ", ".join(f"{k}={v}" for k, v in sizes.items()),
            "independent uniform draws within the ranges; families balanced within each split",
            "whether each episode triggers the monitor (decided when the episodes are simulated)",
        ] + list(COLUMNS.values()),
    })

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for s, df in splits.items():
            df.to_excel(xw, sheet_name=s, index=False)
        summary.to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(rng_rows).to_excel(xw, sheet_name="ranges", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 90)

    write_json(out.with_suffix(".manifest.json"), build_manifest(
        dataset={"type": "cstr_episode_specs", "seed": args.seed, "sizes": sizes, "ranges_file": args.ranges,
                 "ranges_sha256": sha256_file(ranges_path), "output": args.out}))
    print(f"Wrote {out} ({len(allr)} episodes: {sizes})")


if __name__ == "__main__":
    main()
