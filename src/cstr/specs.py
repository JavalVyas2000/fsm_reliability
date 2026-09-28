"""
Episode-spec sampling for the fresh CSTR datasets (scripts 33 and 35).

One episode = fault family + severity + onset + nominal plant + noise seed, drawn uniformly
within a frozen ranges file. `severity_level` maps each family's severity parameter to 0-1
within its range (0 mildest, 1 most severe): a lower pump_degrade_factor / outlet_block_factor
(remaining outlet-flow fraction) or stuck_opening (cooling valve opening) is more severe, a higher
fouling_max (fraction of UA lost) is more severe.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

SEVERITY = {  # family -> (severity parameter, higher value is more severe)
    "fouling": ("fouling_max", True),
    "pump_degrade": ("pump_degrade_factor", False),
    "cool_stuck_closed": ("stuck_opening", False),
    "outlet_block": ("outlet_block_factor", False),
}
PLANT_KEYS = ("CA_in_base", "CD_in_base", "Fc_max", "Fin_sp", "L_sp", "Tc_base", "Tin_base", "UA")


def draw_episode(rng: np.random.Generator, ranges: Dict[str, Any], family: str) -> Dict[str, Any]:
    """Severity, onset and plant for one episode of `family` (draw order is part of the dataset definition)."""
    fr = ranges["families"][family]["params"]
    name, higher_worse = SEVERITY[family]
    lo, hi = fr[name]
    v = float(rng.uniform(lo, hi))
    level = (v - lo) / (hi - lo) if higher_worse else (hi - v) / (hi - lo)
    tau = float(rng.uniform(*fr["fouling_tau"])) if family == "fouling" else np.nan
    onset = float(rng.uniform(*ranges["onset_s"]))
    plant = {k: float(rng.uniform(a, b)) for k, (a, b) in sorted(ranges["plant"].items())}
    return {"family": family, "severity_param": name, "severity_value": round(v, 6), "severity_level": round(level, 4),
            "fouling_tau_s": round(tau, 1), "onset_s": round(onset, 1), "T_sp": ranges["fixed"]["T_sp"],
            **{k: round(x, 6) for k, x in plant.items()}}


def row_to_spec(row: Dict[str, Any]):
    """Spreadsheet row -> EpisodeSpec (the simulator input)."""
    from src.cstr.episodes import EpisodeSpec

    fam = row["family"]
    params = {SEVERITY[fam][0]: float(row["severity_value"])}
    if fam == "fouling":
        params["fouling_tau"] = float(row["fouling_tau_s"])
    plant = {k: float(row[k]) for k in PLANT_KEYS}
    return EpisodeSpec(str(row["episode_id"]), fam, params, float(row["onset_s"]), int(row["noise_seed"]), plant=plant)
