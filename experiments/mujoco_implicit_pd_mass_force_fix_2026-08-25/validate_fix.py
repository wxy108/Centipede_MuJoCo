"""Numerical gate for the isolated MuJoCo implicit-PD repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "outputs" / "validation" / "implicit_pd_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run 6 s zero-noise and random-contact smoke trials",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(pipeline.MODEL_PATH))
    index = pipeline.ModelIndex(model)
    pipeline.configure_native_implicit_pd(model, index)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    knee_positions = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knee_positions]] = math.pi / 6.0
    # Deterministic nonzero errors ensure the check is not a zero-force pass.
    data.qpos[index.qpos_addresses[index.body_positions[4]]] = 0.07
    data.qvel[index.dof_addresses[index.leg_positions[3, 0, 0]]] = -0.12
    data.time = 1.2
    mujoco.mj_forward(model, data)

    controller = pipeline.LockedTravelingWaveController(model, index, wave=3)
    errors = []
    peak_torque = 0.0
    peak_ctrl = 0.0
    for _ in range(2):
        expected_torque, _ = controller.step(data)
        mujoco.mj_forward(model, data)
        actual_torque = data.actuator_force[index.active_actuator_ids]
        errors.append(float(np.max(np.abs(actual_torque - expected_torque))))
        peak_torque = max(peak_torque, float(np.max(np.abs(expected_torque))))
        peak_ctrl = max(
            peak_ctrl,
            float(np.max(np.abs(data.ctrl[index.active_actuator_ids]))),
        )
        data.time += float(model.opt.timestep)

    force_identity_error = max(errors)
    if force_identity_error > 1.0e-12:
        raise RuntimeError(
            f"Native actuator force does not match the requested PD law: {force_identity_error:g}"
        )
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise RuntimeError("Runtime integrator is not implicitfast")
    if np.allclose(
        data.ctrl[index.active_actuator_ids],
        data.actuator_force[index.active_actuator_ids],
    ):
        raise RuntimeError("data.ctrl still has torque semantics")

    short_trial = pipeline.run_trial(
        pipeline.TrialConfig(
            wave=3,
            q=0.0,
            seed=0,
            duration_s=2.25,
            eval_start_s=1.25,
        )
    )
    if not bool(short_trial["finite"]):
        raise RuntimeError("Short implicit-PD dynamics trial was not finite")

    smoke_trials = []
    if args.full:
        for q, seed in ((0.0, 0), (0.3, 2)):
            trial = pipeline.run_trial(
                pipeline.TrialConfig(
                    wave=3,
                    q=q,
                    seed=seed,
                    duration_s=6.0,
                    eval_start_s=2.0,
                )
            )
            if not bool(trial["finite"]):
                raise RuntimeError(f"Full smoke trial q={q:g} was not finite")
            smoke_trials.append(trial)

    result = {
        "status": "pass",
        "scope": "Centipede Model only; Centipede Robot Step 18/19 excluded",
        "model_path": str(pipeline.MODEL_PATH),
        "model_sha256": sha256(pipeline.MODEL_PATH),
        "pipeline_sha256": sha256(Path(pipeline.__file__)),
        "mujoco_version": mujoco.__version__,
        "integrator": "implicitfast",
        "timestep_s": float(model.opt.timestep),
        "active_joint_count": int(len(index.active_joint_ids)),
        "active_armature_min_kg_m2": float(
            np.min(model.dof_armature[index.dof_addresses])
        ),
        "active_armature_max_kg_m2": float(
            np.max(model.dof_armature[index.dof_addresses])
        ),
        "active_passive_damping_max": float(
            np.max(np.abs(model.dof_damping[index.dof_addresses]))
        ),
        "active_passive_stiffness_max": float(
            np.max(np.abs(model.jnt_stiffness[index.active_joint_ids]))
        ),
        "native_force_identity_max_abs_error_nm": force_identity_error,
        "probe_peak_abs_torque_nm": peak_torque,
        "probe_peak_abs_composite_reference": peak_ctrl,
        "short_trial": short_trial,
        "full_smoke_trials": smoke_trials,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
