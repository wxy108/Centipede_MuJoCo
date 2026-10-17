"""Small read-only diagnostic for contact-frame normal handling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage1_debug_import", ROOT / "run_stage1.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import stage-1 pipeline")
stage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage
spec.loader.exec_module(stage)

model_path = ROOT / "models" / "centipede_flat_scaled_dt0p5ms_q2pm20p0deg.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
index = stage.base.ModelIndex(model, impedance=stage.selected_impedance())
stage.configure_native_implicit_pd(model, index)
data = mujoco.MjData(model)
controller = stage.ctrl.FloatWaveController(model, index, 1.5)
mujoco.mj_resetData(model, data)
q3 = index.leg_positions[:, :, 3].reshape(-1)
data.qpos[index.qpos_addresses[q3]] = stage.ctrl.KNEE_DC_RAD
mujoco.mj_forward(model, data)
ground = stage.ground_geom_id(model)
feet = {
    stage.base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{segment}_{side}_collision")
    for segment in range(stage.base.N_LEG_PAIRS)
    for side in range(2)
}
previous: dict[int, np.ndarray] = {}
large = []
samples = []
while data.time < 4.5:
    controller.step(data)
    mujoco.mj_step(model, data)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 == ground and geom2 in feet:
            foot = geom2
        elif geom2 == ground and geom1 in feet:
            foot = geom1
        else:
            continue
        normal = np.asarray(contact.frame[:3], dtype=float).copy()
        raw = normal.copy()
        if normal[2] < 0.0:
            normal = -normal
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        if len(samples) < 20:
            samples.append(
                {
                    "time": float(data.time),
                    "geom1": geom1,
                    "geom2": geom2,
                    "raw": raw.tolist(),
                    "normal": normal.tolist(),
                }
            )
        if foot in previous:
            angle = float(
                np.degrees(np.arccos(np.clip(np.dot(normal, previous[foot]), -1.0, 1.0)))
            )
            if angle > 1.0 and len(large) < 20:
                large.append(
                    {
                        "time": float(data.time),
                        "foot": int(foot),
                        "angle_deg": angle,
                        "previous": previous[foot].tolist(),
                        "normal": normal.tolist(),
                    }
                )
        previous[foot] = normal

print({"samples": samples, "large_changes": large})
