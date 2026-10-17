"""Prepare deterministic, coupled three-amplitude refinement task banks."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
WAVES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0)
LOWER = (0.0, 0.0, 0.0)
UPPER = (36.0, 90.0, 60.0)
STAGE_CONFIG = {
    1: {"radii_deg": (4.0, 10.0, 6.0), "seed_count": 3},
    2: {"radii_deg": (2.0, 5.0, 3.0), "seed_count": 2},
    3: {"radii_deg": (0.75, 2.0, 1.25), "seed_count": 1},
}


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def amplitudes(row):
    return (
        float(row["body_yaw_amplitude_deg"]),
        float(row["leg_yaw_amplitude_deg"]),
        float(row["leg_pitch_amplitude_deg"]),
    )


def false_count(mapping):
    return sum(not bool(value) for value in mapping.values())


def rank_key(row):
    if row.get("feasible_v2_pilot"):
        tier = 0
        failures = 0
    elif row.get("provisional_base_feasible"):
        tier = 1
        failures = false_count(row.get("naturality_gates", {}))
    else:
        tier = 2
        failures = false_count(row.get("base_gates_v2", {}))
    return (tier, failures, -float(row.get("forward_displacement_m", -math.inf)))


def distant(point, selected, radii):
    return all(
        math.sqrt(sum(((a - b) / radius) ** 2 for a, b, radius in zip(point, amplitudes(seed), radii)))
        >= 1.0
        for seed in selected
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    config = STAGE_CONFIG[args.stage]
    rows = load_jsonl(DATA / "broad_sobol_pilot_certified_trials.jsonl")
    for prior_stage in range(1, args.stage):
        for path in sorted(DATA.glob(f"adaptive_refinement_trials_s{prior_stage}_*.jsonl")):
            rows.extend(load_jsonl(path))
    if not rows:
        raise RuntimeError("no completed broad/refinement rows available")

    existing = {
        (float(row["wave"]),) + tuple(round(value, 8) for value in amplitudes(row))
        for row in rows
    }
    tasks = []
    seed_audit = {}
    offsets = list(itertools.product((-1.0, 0.0, 1.0), repeat=3))
    offsets.remove((0.0, 0.0, 0.0))
    for wave in WAVES:
        group = [row for row in rows if math.isclose(float(row["wave"]), wave)]
        ranked = sorted(group, key=rank_key)
        selected = []
        for row in ranked:
            point = amplitudes(row)
            if distant(point, selected, config["radii_deg"]):
                selected.append(row)
            if len(selected) >= config["seed_count"]:
                break
        if not selected:
            raise RuntimeError(f"wave {wave:g} has no seed rows")
        seed_audit[f"{wave:g}"] = [
            {
                "candidate_id": row["candidate_id"],
                "amplitudes_deg": amplitudes(row),
                "feasible": bool(row.get("feasible_v2_pilot")),
                "forward_displacement_m": row.get("forward_displacement_m"),
            }
            for row in selected
        ]
        for seed_index, seed in enumerate(selected):
            center = amplitudes(seed)
            for offset_index, offset in enumerate(offsets):
                point = tuple(
                    min(high, max(low, value + sign * radius))
                    for value, sign, radius, low, high in zip(
                        center, offset, config["radii_deg"], LOWER, UPPER
                    )
                )
                key = (wave,) + tuple(round(value, 8) for value in point)
                if key in existing:
                    continue
                existing.add(key)
                tasks.append(
                    {
                        "candidate_id": f"w{wave:g}_refine_s{args.stage}_{seed_index:02d}_{offset_index:02d}",
                        "source": f"adaptive_refinement_stage_{args.stage}",
                        "wave": wave,
                        "body_yaw_amplitude_deg": point[0],
                        "leg_yaw_amplitude_deg": point[1],
                        "leg_pitch_amplitude_deg": point[2],
                        "duration_s": 6.0,
                        "eval_start_s": 1.5,
                    }
                )
    payload = {
        "stage": args.stage,
        "scope": "Centipede Model only; Robot Step 18/19 untouched",
        "selection": "rank hard-gate-feasible displacement first, then nearest-to-feasible diagnostics",
        "radii_deg": config["radii_deg"],
        "seed_count_per_wave": config["seed_count"],
        "coupled_offsets": "full 3x3x3 neighborhood excluding the already-evaluated center",
        "tasks": tasks,
        "seeds": seed_audit,
        "continuous_global_claim": False,
    }
    output = DATA / f"adaptive_refinement_stage{args.stage}_tasks.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "tasks": len(tasks), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
