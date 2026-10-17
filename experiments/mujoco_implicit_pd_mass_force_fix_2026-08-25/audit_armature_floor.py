"""Re-test the MuJoCo armature floor under the repaired implicit-PD runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "armature_audit"
DEFAULT_LEVELS = (0.0, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5)


def mass_matrix_audit(armature: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(pipeline.MODEL_PATH))
    index = pipeline.ModelIndex(model)
    joint_type_by_dof = model.jnt_type[model.dof_jntid]
    internal = joint_type_by_dof != int(mujoco.mjtJoint.mjJNT_FREE)
    model.dof_armature[internal] = armature
    model.dof_armature[~internal] = 0.0
    pipeline.configure_native_implicit_pd(model, index, armature)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    knees = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knees]] = math.pi / 6.0
    # Build M without calling mj_forward: exact-zero armature can make the
    # Hessian factorization fail, but the unfactorized mass matrix itself is
    # still the evidence we need to diagnose rank and conditioning.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_makeM(model, data)

    dense = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, dense)
    eigenvalues = np.linalg.eigvalsh(dense)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    tolerance = maximum * model.nv * np.finfo(float).eps
    numerical_rank = int(np.count_nonzero(eigenvalues > tolerance))
    return {
        "armature_kg_m2": armature,
        "nv": int(model.nv),
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "rank_tolerance": float(tolerance),
        "numerical_rank": numerical_rank,
        "full_rank": bool(numerical_rank == model.nv and minimum > 0.0),
        "condition_number": float(maximum / minimum) if minimum > 0.0 else float("inf"),
    }


def run_case(case: tuple[float, float, int]) -> dict:
    armature, q, seed = case
    try:
        result = pipeline.run_trial(
            pipeline.TrialConfig(
                wave=3,
                q=q,
                seed=seed,
                duration_s=8.0,
                eval_start_s=2.0,
            ),
            armature_override_kg_m2=armature,
        )
        return {
            "status": "ok",
            "armature_kg_m2": armature,
            "q": q,
            "seed": seed,
            **result,
        }
    except Exception as exc:
        return {
            "status": "exception",
            "armature_kg_m2": armature,
            "q": q,
            "seed": seed,
            "finite": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(levels: tuple[float, ...], cases: list[dict], matrices: list[dict]) -> list[dict]:
    matrix_by_level = {float(row["armature_kg_m2"]): row for row in matrices}
    rows = []
    for level in levels:
        selected = [row for row in cases if float(row["armature_kg_m2"]) == level]
        successful = [row for row in selected if row["status"] == "ok"]
        warnings = [row for row in successful if row.get("warning_counts", "{}") != "{}"]
        baseline = next((row for row in successful if float(row["q"]) == 0.0), None)
        random_rows = [row for row in successful if float(row["q"]) > 0.0]
        matrix = matrix_by_level[level]
        rows.append(
            {
                "armature_kg_m2": level,
                "mass_matrix_full_rank_at_reset": matrix["full_rank"],
                "mass_matrix_minimum_eigenvalue": matrix["minimum_eigenvalue"],
                "mass_matrix_condition_number": matrix["condition_number"],
                "completed_cases": len(successful),
                "finite_cases": sum(bool(row.get("finite", False)) for row in successful),
                "warning_cases": len(warnings),
                "numerically_clean_all_cases": bool(
                    len(successful) == len(selected)
                    and all(bool(row.get("finite", False)) for row in successful)
                    and not warnings
                ),
                "baseline_displacement_m": (
                    float(baseline["absolute_forward_displacement_m"])
                    if baseline is not None else float("nan")
                ),
                "random_displacement_mean_m": (
                    float(np.mean([row["absolute_forward_displacement_m"] for row in random_rows]))
                    if random_rows else float("nan")
                ),
                "peak_active_speed_rad_s": (
                    max(float(row["peak_abs_active_joint_speed_rad_s"]) for row in successful)
                    if successful else float("nan")
                ),
                "peak_active_acceleration_rad_s2": (
                    max(
                        float(row["peak_abs_active_joint_acceleration_rad_s2"])
                        for row in successful
                    )
                    if successful else float("nan")
                ),
            }
        )
    return rows


def model_structure() -> dict:
    model = mujoco.MjModel.from_xml_path(str(pipeline.MODEL_PATH))
    joint_counts = np.bincount(model.jnt_bodyid, minlength=model.nbody)
    multi_joint_bodies = np.flatnonzero(joint_counts > 1)
    positive_mass = model.body_mass[model.body_mass > 0.0]
    positive_inertia = model.body_inertia[model.body_inertia > 0.0]
    return {
        "total_mass_kg": float(np.sum(model.body_mass)),
        "body_count": int(model.nbody),
        "joint_count": int(model.njnt),
        "dof_count": int(model.nv),
        "zero_mass_body_count": int(np.count_nonzero(model.body_mass == 0.0)),
        "minimum_positive_body_mass_kg": float(np.min(positive_mass)),
        "minimum_positive_body_inertia_kg_m2": float(np.min(positive_inertia)),
        "bodies_with_multiple_joints": [
            {
                "body_id": int(body_id),
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)),
                "joint_count": int(joint_counts[body_id]),
                "mass_kg": float(model.body_mass[body_id]),
            }
            for body_id in multi_joint_bodies
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--levels",
        default=",".join(f"{value:g}" for value in DEFAULT_LEVELS),
    )
    args = parser.parse_args()
    levels = tuple(float(value) for value in args.levels.split(","))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices = [mass_matrix_audit(level) for level in levels]
    cases = [
        (level, q, seed)
        for level in levels
        for q, seeds in ((0.0, (0,)), (0.3, (0, 1, 2)))
        for seed in seeds
    ]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_case, case): case for case in cases}
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{count:02d}/{len(cases):02d}] arm={result['armature_kg_m2']:.0e} "
                f"q={result['q']:.1f} seed={result['seed']} status={result['status']} "
                f"finite={result.get('finite', False)}",
                flush=True,
            )

    results.sort(key=lambda row: (float(row["armature_kg_m2"]), float(row["q"]), int(row["seed"])))
    summary = summarize(levels, results, matrices)
    report = {
        "scope": "Centipede Model only; Robot Step 18/19 excluded",
        "controller_architecture": "MuJoCo native affine PD + implicitfast",
        "model": model_structure(),
        "mass_matrix_at_reset": matrices,
        "summary": summary,
        "cases": results,
    }
    (OUTPUT_DIR / "armature_floor_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "armature_floor_summary.csv", summary)
    write_csv(OUTPUT_DIR / "armature_floor_cases.csv", results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
