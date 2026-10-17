"""Audit collision-restoration impulses for the strong independent channel."""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import numpy as np

import run_pipeline as base
from run_severity_sweep import SeverityCase, build_strong_events


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "strong_independent_erasure" / "recovery_audit"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(case: SeverityCase) -> list[dict]:
    model = mujoco.MjModel.from_xml_path(str(base.MODEL_PATH))
    data = mujoco.MjData(model)
    index = base.ModelIndex(model)
    controller = base.make_farms_controller(model, case.wave)
    events, _ = build_strong_events(case)
    manager = base.ErasureManager(model, index, events)

    mujoco.mj_resetData(model, data)
    knee_positions = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knee_positions]] = np.pi / 6.0
    mujoco.mj_forward(model, data)

    previous_active = np.zeros(base.N_LEG_PAIRS, dtype=bool)
    previous_qvel = data.qvel.copy()
    rows: list[dict] = []
    fromto = np.zeros(6, dtype=float)
    floor_id = base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    try:
        while data.time < case.duration_s - 0.5 * model.opt.timestep:
            t = float(data.time)
            active = manager.update(t).copy()
            started = np.flatnonzero(~previous_active & active)
            restored = np.flatnonzero(previous_active & ~active)
            minimum_restore_distance = float("nan")
            if len(restored):
                distances: list[float] = []
                for segment in restored:
                    for geom_id in index.segment_collision_geoms[int(segment)]:
                        distances.append(
                            float(
                                mujoco.mj_geomDistance(
                                    model,
                                    data,
                                    floor_id,
                                    int(geom_id),
                                    1.0,
                                    fromto,
                                )
                            )
                        )
                minimum_restore_distance = min(distances)

            controller.step(model, data, t)
            mujoco.mj_step(model, data)
            delta = data.qvel - previous_qvel
            if t >= case.eval_start_s:
                transition = (
                    "restore"
                    if len(restored)
                    else ("erase_start" if len(started) else "ordinary")
                )
                rows.append(
                    {
                        "wave": case.wave,
                        "noise_level_nu": case.nu,
                        "seed": case.seed,
                        "time_s": t,
                        "transition": transition,
                        "segments": ";".join(
                            map(str, restored if len(restored) else started)
                        ),
                        "minimum_geom_floor_distance_m": minimum_restore_distance,
                        "root_linear_delta_mps": float(np.linalg.norm(delta[:3])),
                        "root_angular_delta_radps": float(np.linalg.norm(delta[3:6])),
                        "max_active_joint_speed_delta_radps": float(
                            np.max(np.abs(delta[index.dof_addresses]))
                        ),
                    }
                )
            previous_active = active
            previous_qvel = data.qvel.copy()
    finally:
        manager.restore()
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for transition in ("ordinary", "erase_start", "restore"):
        subset = [row for row in rows if row["transition"] == transition]
        for metric in (
            "root_linear_delta_mps",
            "root_angular_delta_radps",
            "max_active_joint_speed_delta_radps",
        ):
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            output.append(
                {
                    "transition": transition,
                    "metric": metric,
                    "n": len(values),
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(np.max(values)),
                }
            )

    distances = np.asarray(
        [
            float(row["minimum_geom_floor_distance_m"])
            for row in rows
            if row["transition"] == "restore"
        ],
        dtype=float,
    )
    output.extend(
        [
            {
                "transition": "restore",
                "metric": "minimum_geom_floor_distance_m",
                "n": len(distances),
                "median": float(np.median(distances)),
                "p95": float(np.quantile(distances, 0.95)),
                "max": float(np.max(distances)),
            },
            {
                "transition": "restore",
                "metric": "fraction_restored_while_overlapping_floor",
                "n": len(distances),
                "median": float(np.mean(distances < 0.0)),
                "p95": float("nan"),
                "max": float("nan"),
            },
        ]
    )
    return output


def main() -> None:
    cases = [
        SeverityCase(wave=1, nu=0.5, seed=seed, duration_s=8.0)
        for seed in (0, 1)
    ] + [
        SeverityCase(wave=3, nu=0.5, seed=seed, duration_s=8.0)
        for seed in (0, 1)
    ]
    rows = [sample for case in cases for sample in run(case)]
    summary = summarize(rows)
    write_csv(OUT / "recovery_transition_samples.csv", rows)
    write_csv(OUT / "recovery_transition_summary.csv", summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
