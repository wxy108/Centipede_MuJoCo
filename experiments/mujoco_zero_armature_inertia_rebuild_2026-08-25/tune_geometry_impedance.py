"""Sweep natural-frequency multipliers for the geometry-derived PD seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import run_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "impedance_tuning"
FREQUENCY_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
FAMILY_GRID = (
    (4.0, 32.0),
    (4.0, 64.0),
    (4.0, 128.0),
    (8.0, 32.0),
    (8.0, 64.0),
    (8.0, 128.0),
)


def two_band_impedance(body_multiplier: float, leg_multiplier: float):
    result = {}
    for family, (kp, kv) in pipeline.GEOMETRY_IMPEDANCE_SEED.items():
        multiplier = body_multiplier if family.startswith("body") or family == "head_pitch" else leg_multiplier
        result[family] = (kp * multiplier * multiplier, kv * multiplier)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for multiplier in (() if args.grid_only else FREQUENCY_MULTIPLIERS):
        # f_n scales by s: Kp scales by s^2 and Kv by s, preserving zeta.
        impedance = {
            family: (kp * multiplier * multiplier, kv * multiplier)
            for family, (kp, kv) in pipeline.GEOMETRY_IMPEDANCE_SEED.items()
        }
        result = pipeline.run_trial(
            pipeline.TrialConfig(wave=3, q=0.0, seed=0, duration_s=4.0, eval_start_s=1.25),
            model_path=pipeline.MODEL_PATH,
            armature_override_kg_m2=0.0,
            impedance=impedance,
        )
        row = {
            "frequency_multiplier": multiplier,
            "kp_scale": multiplier * multiplier,
            "kv_scale": multiplier,
            **result,
        }
        rows.append(row)
        print(
            f"s={multiplier:g} finite={result['finite']} "
            f"x={result['absolute_forward_displacement_m']:.6g} "
            f"root_gz_min={result['minimum_root_upright_cos']:.4g} "
            f"chassis_gz_min={result['minimum_chassis_upright_cos']:.4g} "
            f"tort_max={result['maximum_spine_tortuosity']:.4g} "
            f"qvel={result['peak_abs_active_joint_speed_rad_s']:.4g} "
            f"warnings={result['warning_counts']}",
            flush=True,
        )

    if rows:
        with (OUT / "frequency_multiplier_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / "frequency_multiplier_sweep.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    family_rows = []
    for body_multiplier, leg_multiplier in FAMILY_GRID:
        result = pipeline.run_trial(
            pipeline.TrialConfig(wave=3, q=0.0, seed=0, duration_s=4.0, eval_start_s=1.25),
            model_path=pipeline.MODEL_PATH,
            armature_override_kg_m2=0.0,
            impedance=two_band_impedance(body_multiplier, leg_multiplier),
        )
        row = {
            "body_frequency_multiplier": body_multiplier,
            "leg_frequency_multiplier": leg_multiplier,
            **result,
        }
        family_rows.append(row)
        print(
            f"body={body_multiplier:g} leg={leg_multiplier:g} finite={result['finite']} "
            f"x={result['absolute_forward_displacement_m']:.6g} "
            f"root_gz_min={result['minimum_root_upright_cos']:.4g} "
            f"chassis_gz_min={result['minimum_chassis_upright_cos']:.4g} "
            f"tort_max={result['maximum_spine_tortuosity']:.4g} "
            f"qvel={result['peak_abs_active_joint_speed_rad_s']:.4g} "
            f"warnings={result['warning_counts']}",
            flush=True,
        )
    with (OUT / "body_leg_frequency_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    (OUT / "body_leg_frequency_grid.json").write_text(
        json.dumps(family_rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
