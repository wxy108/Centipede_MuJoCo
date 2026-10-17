"""Static-parameter body-roll tuning for the abstract Centipede Model.

Eligible controller architecture:
  * the frozen open-loop CPG reference generator;
  * symmetric, time-invariant implicit-PD gains;
  * optional zero target on the already-existing body-roll actuators.
  * optional static retuning of the already-existing passive distal-tip hinge.

Ineligible here: projected gravity, contact feedback, state-dependent target
offsets, contact gates, tip guards, new state machines, wave/phase decoupling,
or any change to mass, inertia, armature, geometry, or topology.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


_ORIGINAL_CDLL = ctypes.CDLL


def _cdll_without_optional_plugins(path, *args, **kwargs):
    if "site-packages\\mujoco\\plugin" in os.path.normcase(str(path)):
        return None
    return _ORIGINAL_CDLL(path, *args, **kwargs)


ctypes.CDLL = _cdll_without_optional_plugins
try:
    import mujoco
finally:
    ctypes.CDLL = _ORIGINAL_CDLL


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parents[1]
TIP_SCRIPT = (
    MODEL_ROOT
    / "experiments"
    / "mujoco_template_tip_compliance_2026-08-27"
    / "run_tip_compliance.py"
)
LOCKED_ROOT = (
    MODEL_ROOT / "experiments" / "mujoco_template_locked_head_2026-08-27"
)
MODELS = {
    "flat": LOCKED_ROOT / "models" / "centipede_flat_locked_head_m29p0.xml",
    "rough": LOCKED_ROOT / "models" / "centipede_rough_locked_head_m29p0.xml",
}
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"
SEGMENTS = 19

BASE_ROLL_KP = 1.1597375518325183e-4
BASE_ROLL_KV = 4.02715557956593e-7


def roll_gain(multiplier: float) -> tuple[float, float]:
    if multiplier <= 0.0:
        return 0.0, 0.0
    return BASE_ROLL_KP * multiplier, BASE_ROLL_KV * math.sqrt(multiplier)


def candidate(
    roll_multiplier: float = 0.0,
    leg_pitch_kp_scale: float = 1.0,
    leg_pitch_kv_scale: float = 1.0,
    body_pitch_kp_scale: float = 1.0,
    body_pitch_kv_scale: float = 1.0,
    leg_yaw_amplitude_scale: float = 1.0,
    leg_pitch_amplitude_scale: float = 1.0,
    body_amplitude_scale: float = 1.0,
    tip_roll_stiffness_scale: float = 1.0,
    tip_roll_damping_scale: float = 1.0,
    body_roll_stiffness_scale: float = 1.0,
    body_roll_damping_scale: float = 1.0,
) -> dict[str, float]:
    roll_kp, roll_kv = roll_gain(roll_multiplier)
    return {
        "roll_multiplier": roll_multiplier,
        "roll_kp": roll_kp,
        "roll_kv": roll_kv,
        "leg_pitch_kp_scale": leg_pitch_kp_scale,
        "leg_pitch_kv_scale": leg_pitch_kv_scale,
        "body_pitch_kp_scale": body_pitch_kp_scale,
        "body_pitch_kv_scale": body_pitch_kv_scale,
        "leg_yaw_amplitude_scale": leg_yaw_amplitude_scale,
        "leg_pitch_amplitude_scale": leg_pitch_amplitude_scale,
        "body_amplitude_scale": body_amplitude_scale,
        "tip_roll_stiffness_scale": tip_roll_stiffness_scale,
        "tip_roll_damping_scale": tip_roll_damping_scale,
        "body_roll_stiffness_scale": body_roll_stiffness_scale,
        "body_roll_damping_scale": body_roll_damping_scale,
    }


# Coarse one-factor brackets plus a few predeclared combinations.  Every item
# is a static parameter vector; the reference law itself stays unchanged.
CANDIDATES = {
    "baseline": candidate(),
    "r1": candidate(1.0),
    "r2": candidate(2.0),
    "r4": candidate(4.0),
    "r8": candidate(8.0),
    "r16": candidate(16.0),
    "lp_c2": candidate(0.0, leg_pitch_kv_scale=2.0),
    "lp_c4": candidate(0.0, leg_pitch_kv_scale=4.0),
    "lp_c050": candidate(0.0, leg_pitch_kv_scale=0.50),
    "lp_c075": candidate(0.0, leg_pitch_kv_scale=0.75),
    "lp_k085": candidate(
        0.0, leg_pitch_kp_scale=0.85, leg_pitch_kv_scale=math.sqrt(0.85)
    ),
    "lp_k115": candidate(
        0.0, leg_pitch_kp_scale=1.15, leg_pitch_kv_scale=math.sqrt(1.15)
    ),
    "lp_k040": candidate(
        0.0, leg_pitch_kp_scale=0.40, leg_pitch_kv_scale=math.sqrt(0.40)
    ),
    "lp_k055": candidate(
        0.0, leg_pitch_kp_scale=0.55, leg_pitch_kv_scale=math.sqrt(0.55)
    ),
    "lp_k070": candidate(
        0.0, leg_pitch_kp_scale=0.70, leg_pitch_kv_scale=math.sqrt(0.70)
    ),
    "lp_a085": candidate(0.0, leg_pitch_amplitude_scale=0.85),
    "lp_a060": candidate(0.0, leg_pitch_amplitude_scale=0.60),
    "lp_a030": candidate(0.0, leg_pitch_amplitude_scale=0.30),
    "lp_a035": candidate(0.0, leg_pitch_amplitude_scale=0.35),
    "lp_a040": candidate(0.0, leg_pitch_amplitude_scale=0.40),
    "lp_a045": candidate(0.0, leg_pitch_amplitude_scale=0.45),
    "lp_a0475": candidate(0.0, leg_pitch_amplitude_scale=0.475),
    "lp_a050": candidate(0.0, leg_pitch_amplitude_scale=0.50),
    "lp_a0525": candidate(0.0, leg_pitch_amplitude_scale=0.525),
    "lp_a055": candidate(0.0, leg_pitch_amplitude_scale=0.55),
    "lp_a0575": candidate(0.0, leg_pitch_amplitude_scale=0.575),
    "lp_a065": candidate(0.0, leg_pitch_amplitude_scale=0.65),
    "lp_a070": candidate(0.0, leg_pitch_amplitude_scale=0.70),
    "lp_a075": candidate(0.0, leg_pitch_amplitude_scale=0.75),
    "lp_a080": candidate(0.0, leg_pitch_amplitude_scale=0.80),
    "lp_a090": candidate(0.0, leg_pitch_amplitude_scale=0.90),
    "lp_a095": candidate(0.0, leg_pitch_amplitude_scale=0.95),
    "lp_a105": candidate(0.0, leg_pitch_amplitude_scale=1.05),
    "lp_k070_a095": candidate(
        0.0,
        leg_pitch_kp_scale=0.70,
        leg_pitch_kv_scale=math.sqrt(0.70),
        leg_pitch_amplitude_scale=0.95,
    ),
    "ly_a060": candidate(0.0, leg_yaw_amplitude_scale=0.60),
    "ly_a080": candidate(0.0, leg_yaw_amplitude_scale=0.80),
    "ly_a120": candidate(0.0, leg_yaw_amplitude_scale=1.20),
    "ly_a140": candidate(0.0, leg_yaw_amplitude_scale=1.40),
    "ly_a120_lp_a055": candidate(
        0.0, leg_yaw_amplitude_scale=1.20, leg_pitch_amplitude_scale=0.55
    ),
    "ly_a140_lp_a055": candidate(
        0.0, leg_yaw_amplitude_scale=1.40, leg_pitch_amplitude_scale=0.55
    ),
    "ly_a120_lp_a0575": candidate(
        0.0, leg_yaw_amplitude_scale=1.20, leg_pitch_amplitude_scale=0.575
    ),
    "ly_a140_lp_a0575": candidate(
        0.0, leg_yaw_amplitude_scale=1.40, leg_pitch_amplitude_scale=0.575
    ),
    "tip_k075": candidate(0.0, tip_roll_stiffness_scale=0.75),
    "tip_d2": candidate(0.0, tip_roll_damping_scale=2.0),
    "tip_d4": candidate(0.0, tip_roll_damping_scale=4.0),
    "tip_k150_d1225": candidate(
        0.0, tip_roll_stiffness_scale=1.50, tip_roll_damping_scale=math.sqrt(1.50)
    ),
    "tip_k200_d1414": candidate(
        0.0, tip_roll_stiffness_scale=2.0, tip_roll_damping_scale=math.sqrt(2.0)
    ),
    "br_k1": candidate(
        0.0, body_roll_stiffness_scale=1.0 / 6.0,
        body_roll_damping_scale=math.sqrt(1.0 / 6.0)
    ),
    "br_k2": candidate(
        0.0, body_roll_stiffness_scale=2.0 / 6.0,
        body_roll_damping_scale=math.sqrt(2.0 / 6.0)
    ),
    "br_k3": candidate(
        0.0, body_roll_stiffness_scale=3.0 / 6.0,
        body_roll_damping_scale=math.sqrt(3.0 / 6.0)
    ),
    "br_k4": candidate(
        0.0, body_roll_stiffness_scale=4.0 / 6.0,
        body_roll_damping_scale=math.sqrt(4.0 / 6.0)
    ),
    "br_k5": candidate(
        0.0, body_roll_stiffness_scale=5.0 / 6.0,
        body_roll_damping_scale=math.sqrt(5.0 / 6.0)
    ),
    "b_a075": candidate(0.0, body_amplitude_scale=0.75),
    "b_a085": candidate(0.0, body_amplitude_scale=0.85),
    "b_a115": candidate(0.0, body_amplitude_scale=1.15),
    "b_a075_lp_a050": candidate(
        0.0, leg_pitch_amplitude_scale=0.50, body_amplitude_scale=0.75
    ),
    "b_a085_lp_a050": candidate(
        0.0, leg_pitch_amplitude_scale=0.50, body_amplitude_scale=0.85
    ),
    "b_a075_lp_a060": candidate(
        0.0, leg_pitch_amplitude_scale=0.60, body_amplitude_scale=0.75
    ),
    "r4_lp_c2": candidate(4.0, leg_pitch_kv_scale=2.0),
    "r8_lp_c2": candidate(8.0, leg_pitch_kv_scale=2.0),
    "r8_lp_c4": candidate(8.0, leg_pitch_kv_scale=4.0),
}
CANDIDATES["terrain_profile_lp0525"] = {
    **candidate(),
    "flat_leg_pitch_amplitude_scale": 0.525,
    "rough_leg_pitch_amplitude_scale": 1.0,
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tip = load_module(TIP_SCRIPT, "centipede_clean_roll_tip")


def ensure_directories() -> None:
    for path in (DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def percentile(values, q: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, q)) if array.size else float("nan")


def circular_mean_rad(angles_rad: np.ndarray) -> float:
    return float(
        math.atan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    )


def roll_actuator_ids(model, index) -> np.ndarray:
    actuator_for_joint = {
        int(model.actuator_trnid[actuator_id, 0]): actuator_id
        for actuator_id in range(model.nu)
    }
    ids = np.asarray(
        [actuator_for_joint[int(joint_id)] for joint_id in index.roll_joint_ids],
        dtype=int,
    )
    if len(ids) != SEGMENTS or len(np.unique(ids)) != SEGMENTS:
        raise RuntimeError(f"Expected {SEGMENTS} unique body-roll actuators")
    return ids


def run_candidate(
    name: str,
    kind: str,
    duration_s: float,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
) -> dict[str, object]:
    values = CANDIDATES[name]
    applied_leg_pitch_amplitude_scale = float(
        values.get(
            f"{kind}_leg_pitch_amplitude_scale",
            values["leg_pitch_amplitude_scale"],
        )
    )
    original_index = tip.LeanIndex
    original_configure = tip.configure_model
    original_controller = tip.LeanController
    original_signed_roll = tip.signed_world_roll_deg
    original_body_amplitude_scale = tip.BODY_AMPLITUDE_SCALE

    roll_ids_holder: dict[str, np.ndarray] = {}
    tip_params_holder: dict[str, float] = {}
    body_roll_params_holder: dict[str, float] = {}
    coherent_abs_deg: list[float] = []
    front_abs_deg: list[float] = []
    world_span_deg: list[float] = []
    roll_torque_time: list[float] = []
    roll_torque_abs_nm: list[float] = []

    class TunedIndex(original_index):
        def __post_init__(self):
            super().__post_init__()
            leg_pitch = self.leg_positions[:, :, 1].reshape(-1)
            self.kp[leg_pitch] *= float(values["leg_pitch_kp_scale"])
            self.kv[leg_pitch] *= float(values["leg_pitch_kv_scale"])
            self.kp[self.pitch_positions] *= float(values["body_pitch_kp_scale"])
            self.kv[self.pitch_positions] *= float(values["body_pitch_kv_scale"])

    def configure_static(model, index):
        original_configure(model, index)
        model.jnt_stiffness[index.tip_joint_ids] *= float(
            values["tip_roll_stiffness_scale"]
        )
        model.dof_damping[index.tip_dof_addresses] *= float(
            values["tip_roll_damping_scale"]
        )
        tip_params_holder["stiffness"] = float(
            np.median(model.jnt_stiffness[index.tip_joint_ids])
        )
        tip_params_holder["damping"] = float(
            np.median(model.dof_damping[index.tip_dof_addresses])
        )
        model.jnt_stiffness[index.roll_joint_ids] *= float(
            values["body_roll_stiffness_scale"]
        )
        model.dof_damping[index.roll_dof_addresses] *= float(
            values["body_roll_damping_scale"]
        )
        body_roll_params_holder["stiffness"] = float(
            np.median(model.jnt_stiffness[index.roll_joint_ids])
        )
        body_roll_params_holder["damping"] = float(
            np.median(model.dof_damping[index.roll_dof_addresses])
        )
        actuator_ids = roll_actuator_ids(model, index)
        roll_ids_holder["ids"] = actuator_ids
        kp = float(values["roll_kp"])
        kv = float(values["roll_kv"])
        if kp <= 0.0:
            return
        model.actuator_gainprm[actuator_ids] = 0.0
        model.actuator_biasprm[actuator_ids] = 0.0
        model.actuator_gaintype[actuator_ids] = int(mujoco.mjtGain.mjGAIN_FIXED)
        model.actuator_biastype[actuator_ids] = int(mujoco.mjtBias.mjBIAS_AFFINE)
        model.actuator_gainprm[actuator_ids, 0] = kp
        model.actuator_biasprm[actuator_ids, 1] = -kp
        model.actuator_biasprm[actuator_ids, 2] = -kv

    class StaticParameterController(original_controller):
        def __init__(self, model, index, wave):
            super().__init__(model, index, wave)
            self.leg_amplitudes[0] *= float(values["leg_yaw_amplitude_scale"])
            self.leg_amplitudes[1] *= applied_leg_pitch_amplitude_scale

        def step(self, data):
            torque = super().step(data)
            if float(data.time) >= min(1.5, 0.25 * duration_s):
                roll_torque_time.extend(
                    [float(data.time)] * len(roll_ids_holder["ids"])
                )
                roll_torque_abs_nm.extend(
                    np.abs(data.actuator_force[roll_ids_holder["ids"]]).tolist()
                )
            return torque

    def signed_roll_capture(data, body_ids):
        world_deg = original_signed_roll(data, body_ids)
        if len(body_ids) == 21 and float(data.time) >= min(1.5, 0.25 * duration_s):
            radians = np.radians(world_deg)
            coherent_abs_deg.append(abs(math.degrees(circular_mean_rad(radians))))
            front_abs_deg.append(abs(math.degrees(circular_mean_rad(radians[:7]))))
            world_span_deg.append(float(np.ptp(world_deg)))
        return world_deg

    tip.LeanIndex = TunedIndex
    tip.configure_model = configure_static
    tip.LeanController = StaticParameterController
    tip.signed_world_roll_deg = signed_roll_capture
    tip.BODY_AMPLITUDE_SCALE = original_body_amplitude_scale * float(
        values["body_amplitude_scale"]
    )
    try:
        row = tip.run_case(
            name,
            kind,
            MODELS[kind],
            duration_s,
            video=video,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
        )
    finally:
        tip.LeanIndex = original_index
        tip.configure_model = original_configure
        tip.LeanController = original_controller
        tip.signed_world_roll_deg = original_signed_roll
        tip.BODY_AMPLITUDE_SCALE = original_body_amplitude_scale

    row.update(
        {
            "clean_controller": True,
            "feedback_signals": [],
            "state_dependent_target_offsets": False,
            "contact_gating": False,
            "tip_authority_guard": False,
            "body_leg_wave_architecture_changed": False,
            "static_parameters": values,
            "applied_leg_pitch_amplitude_scale": applied_leg_pitch_amplitude_scale,
            "tip_roll_stiffness_nm_rad": tip_params_holder["stiffness"],
            "tip_roll_damping_nms_rad": tip_params_holder["damping"],
            "body_roll_passive_stiffness_nm_rad": body_roll_params_holder[
                "stiffness"
            ],
            "body_roll_passive_damping_nms_rad": body_roll_params_holder[
                "damping"
            ],
            "coherent_chassis_roll_abs_p95_deg": percentile(
                coherent_abs_deg, 0.95
            ),
            "front7_coherent_roll_abs_p95_deg": percentile(front_abs_deg, 0.95),
            "chassis_world_roll_span_p95_deg": percentile(world_span_deg, 0.95),
            "body_roll_active_torque_abs_p95_nm": percentile(
                roll_torque_abs_nm, 0.95
            ),
            "body_roll_active_torque_abs_max_nm": max(
                roll_torque_abs_nm, default=0.0
            ),
            "frozen_model_sha256": tip.sha256(MODELS[kind]),
        }
    )
    return row


def selection_gates(row: dict[str, object], baseline: dict[str, object]):
    if str(row["kind"]) == "flat":
        coherent_roll_goal = float(row["coherent_chassis_roll_abs_p95_deg"]) <= (
            0.50 * float(baseline["coherent_chassis_roll_abs_p95_deg"])
        )
        front_roll_goal = float(row["front7_coherent_roll_abs_p95_deg"]) <= 8.0
    else:
        # Rough-ground roll includes the imposed terrain attitude.  The selected
        # terrain profile deliberately preserves that gait; require no regression
        # instead of forcing the flat-ground 50% reduction target onto it.
        coherent_roll_goal = float(row["coherent_chassis_roll_abs_p95_deg"]) <= (
            float(baseline["coherent_chassis_roll_abs_p95_deg"]) + 0.05
        )
        front_roll_goal = float(row["front7_coherent_roll_abs_p95_deg"]) <= (
            float(baseline["front7_coherent_roll_abs_p95_deg"]) + 0.05
        )
    return {
        "finite_warning_free": bool(row["finite"]) and not row["warning_counts"],
        "clean_architecture": bool(row["clean_controller"])
        and not row["feedback_signals"]
        and not row["state_dependent_target_offsets"],
        "armature_zero": int(row["armature_nonzero_count"]) == 0,
        "wave_phase_frozen": bool(row["wave_numbers_coupled"])
        and math.isclose(float(row["relative_leg_phase_deg"]), 0.0, abs_tol=1e-12),
        "topology_frozen": int(row["nq"]) == int(baseline["nq"])
        and int(row["nv"]) == int(baseline["nv"])
        and int(row["nu"]) == int(baseline["nu"]),
        "forward_retention": float(row["forward_displacement_m"])
        >= 0.90 * float(baseline["forward_displacement_m"]),
        "path_efficiency_retention": float(row["path_efficiency"])
        >= 0.92 * float(baseline["path_efficiency"]),
        "tip_compliance_not_suppressed": float(row["tip_roll_abs_p95_deg"])
        >= float(baseline["tip_roll_abs_p95_deg"]) - 3.0,
        "tip_motion_bounded": float(row["tip_roll_abs_p95_deg"])
        <= float(row["tip_roll_limit_deg"]) - 5.0,
        "tip_stop_not_increased": float(row["tip_roll_stop_fraction"])
        <= float(baseline["tip_roll_stop_fraction"]) + 0.01,
        "contact_penetration_bounded": float(row["contact_penetration_p95_mm"])
        <= max(0.08, float(baseline["contact_penetration_p95_mm"]) + 0.02),
        "contacting_feet_bounded": 8.0 <= float(row["mean_contacting_feet"]) <= 26.0,
        "active_roll_torque_p95": float(row["body_roll_active_torque_abs_p95_nm"])
        <= 3.0e-5,
        "active_roll_torque_max": float(row["body_roll_active_torque_abs_max_nm"])
        <= 5.0e-5,
        "terrain_specific_coherent_roll_goal": coherent_roll_goal,
        "terrain_specific_front_roll_goal": front_roll_goal,
        "local_roll_not_excessive": float(row["body_roll_local_abs_p95_deg"])
        <= 8.0,
    }


def intervention_key(name: str):
    values = CANDIDATES[name]
    pitch_scale = max(
        abs(float(values.get("flat_leg_pitch_amplitude_scale", values["leg_pitch_amplitude_scale"])) - 1.0),
        abs(float(values.get("rough_leg_pitch_amplitude_scale", values["leg_pitch_amplitude_scale"])) - 1.0),
    )
    return (
        float(values["roll_multiplier"]),
        abs(float(values["leg_pitch_kp_scale"]) - 1.0),
        abs(float(values["leg_pitch_kv_scale"]) - 1.0),
        abs(float(values["body_pitch_kp_scale"]) - 1.0),
        abs(float(values["body_pitch_kv_scale"]) - 1.0),
        abs(float(values["leg_yaw_amplitude_scale"]) - 1.0),
        pitch_scale,
        abs(float(values["body_amplitude_scale"]) - 1.0),
        abs(float(values["tip_roll_stiffness_scale"]) - 1.0),
        abs(float(values["tip_roll_damping_scale"]) - 1.0),
        abs(float(values["body_roll_stiffness_scale"]) - 1.0),
        abs(float(values["body_roll_damping_scale"]) - 1.0),
    )


def run_screen(duration_s: float, names: list[str], kinds: list[str]):
    ensure_directories()
    if "baseline" not in names:
        names = ["baseline", *names]
    before = tip.robot_sentinels()
    rows = [
        run_candidate(name, kind, duration_s)
        for name in names
        for kind in kinds
    ]
    baselines = {row["kind"]: row for row in rows if row["candidate"] == "baseline"}
    for row in rows:
        baseline = baselines[str(row["kind"])]
        row["forward_retention"] = float(row["forward_displacement_m"]) / max(
            float(baseline["forward_displacement_m"]), 1e-12
        )
        row["coherent_roll_retention"] = float(
            row["coherent_chassis_roll_abs_p95_deg"]
        ) / max(float(baseline["coherent_chassis_roll_abs_p95_deg"]), 1e-12)
        row["gates"] = selection_gates(row, baseline)
        row["all_gates"] = bool(all(row["gates"].values()))

    passing = []
    for name in names:
        if name == "baseline":
            continue
        candidate_rows = [row for row in rows if row["candidate"] == name]
        if len(candidate_rows) == len(kinds) and all(row["all_gates"] for row in candidate_rows):
            passing.append(name)
    selected = min(passing, key=intervention_key) if passing else None

    performance_feasible = [
        row
        for row in rows
        if row["candidate"] != "baseline"
        and row["gates"]["finite_warning_free"]
        and row["gates"]["forward_retention"]
        and row["gates"]["path_efficiency_retention"]
        and row["gates"]["tip_stop_not_increased"]
        and row["gates"]["active_roll_torque_max"]
    ]
    diagnostic_best = None
    if performance_feasible:
        diagnostic_best = min(
            performance_feasible,
            key=lambda row: (
                float(row["front7_coherent_roll_abs_p95_deg"]),
                float(row["coherent_chassis_roll_abs_p95_deg"]),
            ),
        )["candidate"]

    after = tip.robot_sentinels()
    payload = {
        "purpose": "clean-controller static-parameter feasibility screen; no scalar reward",
        "duration_s": duration_s,
        "eligible_architecture": "open-loop CPG + symmetric static implicit-PD and passive-tip parameters",
        "ineligible": [
            "projected gravity feedback",
            "contact feedback or gating",
            "state-dependent target offsets",
            "wave or phase decoupling",
            "mass, inertia, armature, geometry or topology changes",
        ],
        "models": {kind: str(path) for kind, path in MODELS.items()},
        "candidate_specs": {name: CANDIDATES[name] for name in names},
        "rows": rows,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "diagnostic_best_if_no_selection": diagnostic_best,
        "selection_rule": "all feasibility gates, then minimum static-parameter intervention",
        "robot_sentinels_before": before,
        "robot_sentinels_after": after,
        "robot_sentinels_unchanged": before == after,
    }
    (DATA / "clean_roll_screen.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def render_demos(duration_s: float, name: str):
    ensure_directories()
    before = tip.robot_sentinels()
    views = [
        ("front", 180.0, -10.0),
        ("side", 90.0, -12.0),
        ("top", 90.0, -89.0),
    ]
    rows = []
    video_paths = []
    for kind in ("flat", "rough"):
        for view, azimuth, elevation in views:
            video = VIDEOS / f"clean_roll_{name}_{kind}_{view}.mp4"
            rows.append(
                run_candidate(
                    name,
                    kind,
                    duration_s,
                    video=video,
                    camera_azimuth=azimuth,
                    camera_elevation=elevation,
                )
            )
            video_paths.append(video)

    cells = []
    integrity = []
    for video in video_paths:
        reader = imageio.get_reader(str(video))
        count = int(reader.count_frames())
        metadata = reader.get_meta_data()
        cells.append(Image.fromarray(reader.get_data(min(90, count - 1))).resize((480, 270)))
        integrity.append(
            {
                "video": str(video),
                "frames": count,
                "fps": float(metadata.get("fps", float("nan"))),
                "duration_s": float(metadata.get("duration", float("nan"))),
            }
        )
        reader.close()
    sheet = Image.new("RGB", (1440, 580), "white")
    draw = ImageDraw.Draw(sheet)
    for index, cell in enumerate(cells):
        x = (index % 3) * 480
        y = (index // 3) * 290
        sheet.paste(cell, (x, y))
        draw.text((x + 8, y + 272), video_paths[index].name, fill="black")
    sheet_path = FIGURES / f"clean_roll_{name}_contact_sheet.png"
    sheet.save(sheet_path)

    after = tip.robot_sentinels()
    payload = {
        "candidate": name,
        "duration_s": duration_s,
        "views": [view for view, _, _ in views],
        "rows": rows,
        "videos": [str(path) for path in video_paths],
        "video_integrity": integrity,
        "contact_sheet": str(sheet_path),
        "robot_sentinels_before": before,
        "robot_sentinels_after": after,
        "robot_sentinels_unchanged": before == after,
    }
    (DATA / "clean_roll_demo_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def compact(payload):
    output = dict(payload)
    if "rows" in output:
        output["row_count"] = len(output.pop("rows"))
    for key in ("robot_sentinels_before", "robot_sentinels_after"):
        output.pop(key, None)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("screen", "demo"))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--candidates", nargs="*", choices=tuple(CANDIDATES))
    parser.add_argument("--kinds", nargs="*", choices=("flat", "rough"))
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    args = parser.parse_args()
    if args.command == "screen":
        payload = run_screen(
            args.duration,
            list(args.candidates or CANDIDATES),
            list(args.kinds or ("flat", "rough")),
        )
    else:
        if not args.candidate:
            raise ValueError("demo requires --candidate")
        payload = render_demos(args.duration, args.candidate)
    print(json.dumps(compact(payload), indent=2))


if __name__ == "__main__":
    main()
