"""Validation gates for the geometry-inertia, zero-armature Model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "validation"


def main() -> None:
    cases = [
        pipeline.TrialConfig(wave=3, q=0.0, seed=0, duration_s=8.0, eval_start_s=2.0),
        *[
            pipeline.TrialConfig(wave=3, q=0.3, seed=seed, duration_s=8.0, eval_start_s=2.0)
            for seed in range(3)
        ],
    ]
    rows = []
    for case in cases:
        row = pipeline.run_trial(
            case,
            model_path=pipeline.MODEL_PATH,
            armature_override_kg_m2=0.0,
            impedance=pipeline.IMPEDANCE,
        )
        rows.append(row)
        print(
            f"q={case.q:g} seed={case.seed} finite={row['finite']} "
            f"forward={row['absolute_forward_displacement_m']:.5f} "
            f"min|gz|={row['minimum_root_upright_cos']:.3f} "
            f"tort={row['maximum_spine_tortuosity']:.3f} "
            f"speed={row['peak_abs_active_joint_speed_rad_s']:.2f}",
            flush=True,
        )

    gates = {
        "all_cases_finite": all(bool(row["finite"]) for row in rows),
        "all_cases_warning_free": all(row["warning_counts"] == "{}" for row in rows),
        "all_active_armatures_exactly_zero": all(
            float(row["active_armature_min_kg_m2"]) == 0.0
            and float(row["active_armature_max_kg_m2"]) == 0.0
            for row in rows
        ),
        "robot_projected_gravity_tip_gate_clear": all(
            float(row["minimum_root_upright_cos"]) >= pipeline.ISAAC_TIP_COS
            and float(row["root_tip_fraction"]) == 0.0
            for row in rows
        ),
        "no_chassis_segment_tip_samples": all(
            float(row["mean_chassis_tip_fraction"]) == 0.0 for row in rows
        ),
        "no_spine_coil_tortuosity_le_1_25": all(
            float(row["maximum_spine_tortuosity"]) <= 1.25 for row in rows
        ),
        "no_ground_penetrating_chassis": all(float(row["min_chassis_z_m"]) > 0.0 for row in rows),
        "within_robot_velocity_limit_100_rad_s": all(
            float(row["peak_abs_active_joint_speed_rad_s"]) <= 100.0 for row in rows
        ),
        "headward_forward_motion_all_cases": all(
            float(row["absolute_forward_displacement_m"]) > 0.0 for row in rows
        ),
    }
    build = json.loads((ROOT / "outputs" / "geometry_inertia_build_audit.json").read_text(encoding="utf-8"))
    gates["mass_matrix_positive_definite_full_rank"] = bool(
        build["mass_matrix_at_reset"]["positive_definite"]
        and build["mass_matrix_at_reset"]["rank"] == build["mass_matrix_at_reset"]["nv"]
    )
    gates["total_mass_exactly_preserved"] = float(build["mass_error_kg"]) == 0.0

    payload = {
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "Centipede Model only; Robot is read-only reference",
        "model": str(pipeline.MODEL_PATH),
        "armature_kg_m2": 0.0,
        "impedance": pipeline.IMPEDANCE,
        "gates": gates,
        "aggregate": {
            "baseline_forward_m": float(rows[0]["absolute_forward_displacement_m"]),
            "random_forward_mean_m": float(
                np.mean([float(row["absolute_forward_displacement_m"]) for row in rows[1:]])
            ),
            "minimum_root_upright_cos": min(float(row["minimum_root_upright_cos"]) for row in rows),
            "maximum_spine_tortuosity": max(float(row["maximum_spine_tortuosity"]) for row in rows),
            "maximum_active_speed_rad_s": max(
                float(row["peak_abs_active_joint_speed_rad_s"]) for row in rows
            ),
            "maximum_active_acceleration_rad_s2": max(
                float(row["peak_abs_active_joint_acceleration_rad_s2"]) for row in rows
            ),
        },
        "cases": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zero_armature_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (OUT / "zero_armature_validation_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": payload["status"], "gates": gates, "aggregate": payload["aggregate"]}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
