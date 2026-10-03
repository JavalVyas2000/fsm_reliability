"""
Summarise the shift analysis (scripts/45; docs/cstr_shift_prereg.md) into paper tables and a figure.

Per model and shift (feed, cooling, family-* pooled over the four held-out families, control):
AUROC, calls avoided, correct / failing actions let through, failing actions stopped, good actions
rejected, and the paired difference in calls skipped vs observables. Figure: AUROC drop under shift
(shift minus control) per signal and model.

Example:
    python -m scripts.46_cstr_shift_summary --run_dir outputs/cstr_shift_v4/<run>
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.manifest import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "paper_outputs"
NAMES = {"plant readings + proposed change": "Observables", "plant readings only": "Plant readings only",
         "readings + change + grounding": "Obs. + grounding", "readings + change + all internals": "Obs. + all internals",
         "readings + change + all internals + grounding": "Obs. + internals + grounding",
         "region-based attention grounding": "Region grounding", "attention by prompt region": "Attention (regions)",
         "hidden states": "Hidden states", "all internals": "All internals", "all internals + grounding": "All internals + grounding",
         "token confidence": "Token confidence"}
ORDER = list(NAMES.values())
FIG_SIGNALS = [("Observables", "#2a78d6"), ("Obs. + internals + grounding", "#eb6834"), ("All internals", "#1baf7a"),
               ("Token confidence", "#eda100")]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    return p.parse_args()


def pooled(R):
    """Family shifts pooled over the four held-out families (counts summed, AUROC averaged)."""
    fam = R[R["shift"].str.startswith("family-")]
    agg = fam.groupby(["model", "signal", "tolerance"]).agg(
        n_target=("n_target", "sum"), auroc=("auroc", "mean"), accepted=("accepted", "sum"),
        accepted_failures=("accepted_failures", "sum"), rejected=("rejected", "sum"), rejected_good=("rejected_good", "sum"),
        skipped_n=("skipped_n", "sum"), good_total=("good_total", "sum")).reset_index()
    agg["shift"] = "family (pooled)"
    agg["skipped_pct"] = 100 * agg.skipped_n / agg.n_target
    return pd.concat([R, agg], ignore_index=True)


def main():
    args = parse_args()
    d = REPO_ROOT / args.run_dir
    R = pd.read_csv(d / "results.csv")
    R["skipped_n"] = (R.skipped_pct / 100 * R.n_target).round().astype(int)
    R["good_total"] = (R.n_target * (1 - R.target_failure_rate)).round().astype(int)
    R = pooled(R)
    R["signal_name"] = R.signal.map(NAMES)
    R["accepted_correct"] = R.accepted - R.accepted_failures
    R["rejected_correct"] = R.rejected - R.rejected_good
    D = pd.read_csv(d / "deltas.csv")
    shifts = ["control", "feed", "cooling", "family (pooled)"]
    models = list(dict.fromkeys(R.model))
    R.to_csv(OUT / "tables" / "shift_all.csv", index=False)

    md = ["# CSTR under plant / fault shift (`docs/cstr_shift_prereg.md`)", "",
          "Probes and thresholds fitted on the source part only; evaluated once on the target. "
          "family (pooled) = the four held-out-family shifts, counts summed, AUROC averaged.", ""]
    for x in (0.10, 0.05):
        md += [f"## X = {x:.0%}", ""]
        for m in models:
            md += [f"### {m}", "", "| shift | signal | AUROC | calls avoided | correct let through | **failing let through** | "
                   "failing stopped | good rejected |", "|---|---|---|---|---|---|---|---|"]
            for sh in shifts:
                sub = R[(R.model == m) & (R["shift"] == sh) & np.isclose(R.tolerance, x)].set_index("signal_name").reindex(ORDER)
                for sig, r in sub.iterrows():
                    if pd.isna(r.n_target):
                        continue
                    md.append(f"| {sh} | {sig} | {r.auroc:.3f} | {int(r.skipped_n)} / {int(r.n_target)} ({r.skipped_pct:.0f}%) | "
                              f"{int(r.accepted_correct)} | **{int(r.accepted_failures)}** | {int(r.rejected_correct)} | "
                              f"{int(r.rejected_good)} / {int(r.good_total)} |")
            md.append("")
    (OUT / "tables" / "shift.md").write_text("\n".join(md), encoding="utf-8")

    # AUROC change under shift relative to the control split
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True, sharex=True)
    for ax, sh in zip(axes, ["feed", "cooling", "family (pooled)"]):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for k, (sig, col) in enumerate(FIG_SIGNALS):
            for i, m in enumerate(models):
                a = R[(R.model == m) & (R["shift"] == sh) & (R.signal_name == sig) & np.isclose(R.tolerance, 0.10)].auroc
                c = R[(R.model == m) & (R["shift"] == "control") & (R.signal_name == sig) & np.isclose(R.tolerance, 0.10)].auroc
                if a.empty or c.empty or pd.isna(a.iloc[0]):
                    continue
                y = len(models) - 1 - i + (1.5 - k) * 0.18
                ax.plot([0, a.iloc[0] - c.iloc[0]], [y, y], color=col, linewidth=2, solid_capstyle="round")
                ax.plot(a.iloc[0] - c.iloc[0], y, "o", color=col, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=sig if i == 0 else None)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(f"shift: {sh}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("AUROC on target − AUROC in control", fontsize=8, color=INK2)
        ax.set_yticks(range(len(models))[::-1])
        ax.set_yticklabels(models, fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"shift_auroc_change.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    key = R[np.isclose(R.tolerance, 0.10) & R.signal_name.isin([s for s, _ in FIG_SIGNALS] + ["Obs. + all internals", "Hidden states"])]
    print(key.pivot_table(index=["model", "signal_name"], columns="shift", values="auroc").round(3).to_string())
    print()
    print(key.pivot_table(index=["model", "signal_name"], columns="shift", values="accepted_failures", aggfunc="sum").to_string())
    print()
    dd = D[np.isclose(D.tolerance, 0.10) & D.signal.isin(["readings + change + all internals + grounding", "readings + change + all internals"])]
    print(dd.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
