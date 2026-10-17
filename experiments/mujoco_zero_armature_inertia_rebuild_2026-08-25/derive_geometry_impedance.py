"""Derive auditable family-level PD seeds from the rebuilt mass matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "geometry_impedance_seed.json"

TARGET_FREQUENCY_HZ = {
    "body_yaw": 8.0,
    "head_pitch": 8.0,
    "body_pitch": 8.0,
    "body_roll": 8.0,
    "legbase_yaw": 10.0,
    "legbase_pitch": 10.0,
    "legbase_roll": 10.0,
    "knee": 12.0,
}
DAMPING_RATIO = 0.8


def dense_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_crb(model, data)
    mujoco.mj_makeM(model, data)
    dense = np.empty((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, dense)
    return dense


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(pipeline.MODEL_PATH))
    index = pipeline.ModelIndex(model)
    data = mujoco.MjData(model)
    knee = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knee]] = math.pi / 6.0
    mass = dense_mass_matrix(model, data)
    inverse = np.linalg.inv(mass)

    family_positions = {
        "body_yaw": index.body_positions,
        "head_pitch": index.pitch_positions[:1],
        "body_pitch": index.pitch_positions[1:],
        "body_roll": index.roll_positions,
        "legbase_yaw": index.leg_positions[:, :, 0].reshape(-1),
        "legbase_pitch": index.leg_positions[:, :, 1].reshape(-1),
        "legbase_roll": index.leg_positions[:, :, 2].reshape(-1),
        "knee": index.leg_positions[:, :, 3].reshape(-1),
    }

    families = {}
    impedance = {}
    for family, positions in family_positions.items():
        dofs = index.dof_addresses[np.asarray(positions, dtype=int)]
        locked_inertia = np.diag(mass)[dofs]
        free_inertia = 1.0 / np.diag(inverse)[dofs]
        representative = float(np.median(free_inertia))
        omega = 2.0 * math.pi * TARGET_FREQUENCY_HZ[family]
        kp = representative * omega * omega
        kv = 2.0 * DAMPING_RATIO * representative * omega
        impedance[family] = [kp, kv]
        families[family] = {
            "joint_count": int(len(dofs)),
            "locked_joint_inertia_kg_m2": statistics(locked_inertia),
            "free_joint_effective_inertia_kg_m2": statistics(free_inertia),
            "target_frequency_hz": TARGET_FREQUENCY_HZ[family],
            "damping_ratio": DAMPING_RATIO,
            "seed_kp_nm_per_rad": kp,
            "seed_kv_nm_s_per_rad": kv,
        }

    payload = {
        "scope": "Centipede Model only",
        "model": str(pipeline.MODEL_PATH),
        "armature_kg_m2": float(np.max(model.dof_armature)),
        "derivation": (
            "I_eff=1/diag(inv(M)); Kp=I_eff*(2*pi*f_n)^2; "
            "Kv=2*zeta*I_eff*(2*pi*f_n); family median at reset knee=pi/6"
        ),
        "families": families,
        "impedance": impedance,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
