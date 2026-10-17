"""Generate the fixed normalized Sobol bank used by the broad pilot.

This utility generates numbers only.  It never imports or runs the MuJoCo
model, so the simulator version used by this helper cannot affect dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "data" / "sobol_point_bank_power7.json"
WAVES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0)
POWER = 7
BASE_SEED = 20260831


def main():
    bank = {}
    for wave_index, wave in enumerate(WAVES):
        sampler = qmc.Sobol(d=3, scramble=True, seed=BASE_SEED + wave_index)
        bank[f"{wave:g}"] = sampler.random_base2(POWER).tolist()
    payload = {
        "generator": "scipy.stats.qmc.Sobol(scramble=True)",
        "dimension": 3,
        "power": POWER,
        "points_per_wave": 2**POWER,
        "base_seed": BASE_SEED,
        "waves": list(WAVES),
        "normalized_points": bank,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
