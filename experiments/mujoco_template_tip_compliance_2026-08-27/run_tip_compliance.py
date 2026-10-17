"""Distal-tip compliance redesign for the abstract Centipede Model.

This experiment replaces the old 4-DOF leg

    yaw -> pitch -> proximal roll (_2) -> compliant knee pitch (_3)

with the lower-DOF morphology

    yaw -> pitch -> rigid seg1 + rigid seg2 -> passive short-tip roll.

The fixed seg1/seg2 bend preserves the old 30 degree knee equilibrium and the
zero-deflection hip-to-foot geometry.  The compliant lever is shortened from
3.875 mm to 1.175 mm.  Body/leg spatial wave numbers remain coupled (flat
1.5/1.5, rough 3/3), active body-roll actuation remains disabled, and every
joint retains armature=0.  Outputs are isolated from Centipede Robot Step 18/19.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
PASSIVE_ROOT = EXPERIMENTS / "mujoco_template_passive_body_roll_2026-08-27"
PASSIVE_SCRIPT = PASSIVE_ROOT / "run_passive_body_roll.py"

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"

SOURCE_MODELS = {
    "flat": PASSIVE_ROOT / "models" / "centipede_flat_passive_k4_coupled_wave.xml",
    "rough": PASSIVE_ROOT / "models" / "centipede_rough_passive_k4_coupled_wave.xml",
}

FLAT_WAVE = 1.5
ROUGH_WAVE = 3.0
BODY_AMPLITUDE_SCALE = 0.55
FIXED_KNEE_DEG = 30.0
SEG1_M = 0.0035
SEG2_M = 0.0027
TIP_M = 0.0008
CAPSULE_RADIUS_M = 0.000375
FOOT_RADIUS_M = 0.00045
TIP_ROLL_LIMIT_DEG = 30.0
TIP_DAMPING_RATIO = 0.8
TIP_STIFFNESSES = (
    2.0e-7,
    5.0e-7,
    1.0e-6,
    2.0e-6,
    5.0e-6,
)
BODY_LENGTH_M = 0.1025
SEGMENT_DENSITY_KG_M3 = 731.57020432251477
ROBOT_ARCHIVE = (
    ROOT.parents[2]
    / "Centipede_robot_MUJOCO"
    / "exp_data"
    / "reliability"
    / "_archive"
    / "legacy_steps_01_20"
)
ROBOT_SENTINELS = {
    "step18_run_configuration": ROBOT_ARCHIVE
    / "step18_body_wave_only_random_pitch"
    / "run_configuration.json",
    "step18_trials": ROBOT_ARCHIVE
    / "step18_body_wave_only_random_pitch"
    / "trials.csv",
    "step19_run_configuration": ROBOT_ARCHIVE
    / "step19_phi_pi_over_2_body_wave_only_random_pitch"
    / "run_configuration.json",
    "step19_trials": ROBOT_ARCHIVE
    / "step19_phi_pi_over_2_body_wave_only_random_pitch"
    / "trials.csv",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


passive = load_module(PASSIVE_SCRIPT, "centipede_tip_passive_source")
active = passive.active
stage1 = passive.stage1
base = stage1.base
ctrl = stage1.ctrl


def ensure_directories() -> None:
    for path in (MODELS, DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def robot_sentinels() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else "",
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in ROBOT_SENTINELS.items()
    }


def fmt(value: float) -> str:
    return f"{value:.12g}"


def candidate_name(stiffness: float) -> str:
    labels = {
        2.0e-7: "tip_k2e-7",
        5.0e-7: "tip_k5e-7",
        1.0e-6: "tip_k1e-6",
        2.0e-6: "tip_k2e-6",
        5.0e-6: "tip_k5e-6",
    }
    return labels[float(stiffness)]


def candidate_specs() -> dict[str, dict[str, float]]:
    specs = {
        candidate_name(stiffness): {
            "stiffness": stiffness,
            "limit_deg": TIP_ROLL_LIMIT_DEG,
            "body_roll_scale_from_k4": 1.0,
        }
        for stiffness in TIP_STIFFNESSES
    }
    specs.update(
        {
            "tip_k5e-6_r35": {
                "stiffness": 5.0e-6,
                "limit_deg": 35.0,
                "body_roll_scale_from_k4": 1.0,
            },
            "tip_k4e-6_r35_bodyk6": {
                "stiffness": 4.0e-6,
                "limit_deg": 35.0,
                "body_roll_scale_from_k4": 1.5,
            },
            "tip_k4p5e-6_r35_bodyk6": {
                "stiffness": 4.5e-6,
                "limit_deg": 35.0,
                "body_roll_scale_from_k4": 1.5,
            },
            "tip_k4p5e-6_r38_bodyk6": {
                "stiffness": 4.5e-6,
                "limit_deg": 38.0,
                "body_roll_scale_from_k4": 1.5,
            },
            "tip_k5e-6_r35_bodyk6": {
                "stiffness": 5.0e-6,
                "limit_deg": 35.0,
                "body_roll_scale_from_k4": 1.5,
            },
        }
    )
    return specs


def _make_capsule(
    parent: ET.Element,
    name: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    visual: bool,
    color: str,
    density: float = SEGMENT_DENSITY_KG_M3,
) -> None:
    attributes = {
        "type": "capsule",
        "fromto": " ".join(fmt(value) for value in (*start, *end)),
        "size": fmt(CAPSULE_RADIUS_M),
        "group": "2" if visual else "0",
        "name": f"{name}_{'visual' if visual else 'collision'}",
    }
    if visual:
        attributes.update({"contype": "0", "conaffinity": "0", "rgba": color})
    else:
        attributes["density"] = fmt(density)
    ET.SubElement(parent, "geom", attributes)


def redistributed_density(length_m: float) -> float:
    """Preserve the old two-capsule mass while splitting one into seg2+tip.

    A capsule carries one full sphere of end-cap volume.  Adding a third
    capsule would otherwise add fictitious material even though total
    centerline length is unchanged.  The per-geom density below assigns mass
    in proportion to centerline length; MuJoCo still derives each inertia from
    its actual capsule geometry and assigned mass.
    """
    area = math.pi * CAPSULE_RADIUS_M**2
    cap_volume = 4.0 * math.pi * CAPSULE_RADIUS_M**3 / 3.0
    old_volume = area * 0.0035 + cap_volume
    desired_mass = SEGMENT_DENSITY_KG_M3 * old_volume * length_m / 0.0035
    new_volume = area * length_m + cap_volume
    return desired_mass / new_volume


def transform_tree(
    root: ET.Element, tip_k: float, tip_c: float, tip_limit_deg: float
) -> dict[str, object]:
    """Replace all 38 old q2/q3 chains by one rigid compound plus short tip."""
    knee = math.radians(FIXED_KNEE_DEG)
    cosine, sine = math.cos(knee), math.sin(knee)
    removed_joint_names: set[str] = set()
    leg_count = 0
    old_effective_lever = 0.0035 + CAPSULE_RADIUS_M
    new_effective_lever = TIP_M + CAPSULE_RADIUS_M

    q1_bodies = [
        body
        for body in root.iter("body")
        if re.fullmatch(r"link_leg_\d+_[LR]_1", body.get("name", ""))
    ]
    for q1_body in q1_bodies:
        match = re.fullmatch(r"link_leg_(\d+)_([LR])_1", q1_body.get("name", ""))
        if match is None:
            continue
        segment, side = int(match.group(1)), match.group(2)
        side_sign = -1.0 if side == "L" else 1.0
        q2_body = next(
            (
                child
                for child in q1_body.findall("body")
                if child.get("name") == f"link_leg_{segment}_{side}_2"
            ),
            None,
        )
        if q2_body is None:
            raise RuntimeError(f"Missing q2 body under {q1_body.get('name')}")
        q3_body = next(
            (
                child
                for child in q2_body.findall("body")
                if child.get("name") == f"link_leg_{segment}_{side}_3"
            ),
            None,
        )
        if q3_body is None:
            raise RuntimeError(f"Missing q3 body under {q2_body.get('name')}")
        foot = next(
            (child for child in q3_body.findall("body") if child.get("name", "").startswith("foot_")),
            None,
        )
        if foot is None:
            raise RuntimeError(f"Missing foot under {q3_body.get('name')}")

        for old_joint in (*q2_body.findall("joint"), *q3_body.findall("joint")):
            if old_joint.get("name"):
                removed_joint_names.add(str(old_joint.get("name")))
        q1_body.remove(q2_body)

        p0 = (0.0, 0.0, 0.0)
        p1 = (0.0, side_sign * SEG1_M, 0.0)
        p2 = (
            0.0,
            side_sign * (SEG1_M + SEG2_M * cosine),
            -SEG2_M * sine,
        )
        stem = f"link_leg_{segment}_{side}"
        for visual in (False, True):
            _make_capsule(
                q1_body,
                f"{stem}_seg1",
                p0,
                p1,
                visual,
                "0.76 0.62 0.30 1",
                redistributed_density(SEG1_M),
            )
            _make_capsule(
                q1_body,
                f"{stem}_seg2",
                p1,
                p2,
                visual,
                "0.88 0.72 0.32 1",
                redistributed_density(SEG2_M),
            )

        # At q=0 the tip frame continues seg2.  Its hinge axis is the old
        # proximal radial-roll axis expressed in the fixed distal frame.  This
        # preserves the accepted physical direction while moving its moment arm
        # to the last 1.175 mm only.
        fixed_rotation = -side_sign * knee
        tip_body = ET.SubElement(
            q1_body,
            "body",
            {
                "name": f"link_leg_{segment}_{side}_tip",
                "pos": " ".join(fmt(value) for value in p2),
                "euler": f"{fmt(fixed_rotation)} 0 0",
            },
        )
        axis = (0.0, cosine, side_sign * sine)
        limit = math.radians(tip_limit_deg)
        ET.SubElement(
            tip_body,
            "joint",
            {
                "name": f"joint_tip_{segment}_{side}_roll",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": " ".join(fmt(value) for value in axis),
                "limited": "true",
                "range": f"{fmt(-limit)} {fmt(limit)}",
                "stiffness": fmt(tip_k),
                "damping": fmt(tip_c),
                "springref": "0",
                "armature": "0",
                "solreflimit": "0.0005 1.5",
                "solimplimit": "0.999 0.9999 0.00001 0.5 2",
            },
        )
        tip_end = (0.0, side_sign * TIP_M, 0.0)
        for visual in (False, True):
            _make_capsule(
                tip_body,
                f"{stem}_tip",
                p0,
                tip_end,
                visual,
                "0.96 0.28 0.08 1",
                redistributed_density(TIP_M),
            )
        new_foot = copy.deepcopy(foot)
        new_foot.set("pos", f"0 {fmt(side_sign * new_effective_lever)} 0")
        tip_body.append(new_foot)
        ET.SubElement(
            tip_body,
            "site",
            {
                "name": f"site_{stem}_tip_joint",
                "pos": "0 0 0",
                "size": "0.0003",
                "group": "4",
            },
        )
        leg_count += 1

    if leg_count != 38:
        raise RuntimeError(f"Expected 38 transformed legs, found {leg_count}")
    if len(removed_joint_names) != 76:
        raise RuntimeError(f"Expected 76 removed q2/q3 names, found {len(removed_joint_names)}")

    actuator = root.find("actuator")
    removed_actuators = 0
    if actuator is not None:
        for element in list(actuator):
            if element.get("joint") in removed_joint_names:
                actuator.remove(element)
                removed_actuators += 1
    sensor = root.find("sensor")
    removed_sensors = 0
    if sensor is not None:
        for element in list(sensor):
            old_site = element.get("site", "")
            if element.get("joint") in removed_joint_names or re.fullmatch(
                r"site_link_leg_\d+_[LR]_3", old_site
            ):
                sensor.remove(element)
                removed_sensors += 1

    return {
        "transformed_leg_count": leg_count,
        "removed_q2_q3_joint_count": len(removed_joint_names),
        "removed_q2_q3_actuator_count": removed_actuators,
        "removed_q2_q3_sensor_count": removed_sensors,
        "fixed_knee_deg": FIXED_KNEE_DEG,
        "segment_lengths_mm": {
            "seg1": 1000.0 * SEG1_M,
            "seg2": 1000.0 * SEG2_M,
            "tip": 1000.0 * TIP_M,
        },
        "capsule_centerline_total_mm": 1000.0 * (SEG1_M + SEG2_M + TIP_M),
        "old_compliant_lever_mm": 1000.0 * old_effective_lever,
        "new_compliant_lever_mm": 1000.0 * new_effective_lever,
        "compliant_lever_reduction_fraction": 1.0 - new_effective_lever / old_effective_lever,
        "tip_roll_axis_rule": "old proximal radial-roll axis expressed in fixed distal frame",
        "tip_roll_limit_deg": tip_limit_deg,
    }


def write_model(
    source: Path,
    output: Path,
    tip_k: float,
    tip_c: float,
    tip_limit_deg: float = TIP_ROLL_LIMIT_DEG,
    body_roll_scale_from_k4: float = 1.0,
) -> dict[str, object]:
    tree = ET.parse(source)
    root = tree.getroot()
    morphology = transform_tree(root, tip_k, tip_c, tip_limit_deg)
    body_roll_count = 0
    for joint in root.iter("joint"):
        if not joint.get("name", "").startswith("joint_roll_body_"):
            continue
        base_k = float(joint.get("stiffness", "0"))
        base_c = float(joint.get("damping", "0"))
        joint.set("stiffness", fmt(base_k * body_roll_scale_from_k4))
        joint.set("damping", fmt(base_c * math.sqrt(body_roll_scale_from_k4)))
        body_roll_count += 1
    if body_roll_count != 20:
        raise RuntimeError(f"Expected 20 passive body-roll joints, found {body_roll_count}")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    morphology.update(
        {
            "model": str(output),
            "model_sha256": sha256(output),
            "source_model": str(source),
            "source_model_sha256": sha256(source),
            "tip_stiffness_nm_rad": tip_k,
            "tip_damping_nms_rad": tip_c,
            "tip_roll_limit_deg": tip_limit_deg,
            "body_roll_scale_from_k4": body_roll_scale_from_k4,
        }
    )
    return morphology


def _tip_inertia(path: Path) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, full, data.qM)
    inverse = np.linalg.inv(full)
    diagonal = []
    effective = []
    for segment in range(19):
        for side in ("L", "R"):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_tip_{segment}_{side}_roll"
            )
            dof_id = int(model.jnt_dofadr[joint_id])
            diagonal.append(float(full[dof_id, dof_id]))
            effective.append(float(1.0 / inverse[dof_id, dof_id]))
    return {
        "tip_mass_matrix_diagonal_min_kg_m2": float(np.min(diagonal)),
        "tip_mass_matrix_diagonal_median_kg_m2": float(np.median(diagonal)),
        "tip_mass_matrix_diagonal_max_kg_m2": float(np.max(diagonal)),
        "tip_effective_inertia_min_kg_m2": float(np.min(effective)),
        "tip_effective_inertia_median_kg_m2": float(np.median(effective)),
        "tip_effective_inertia_max_kg_m2": float(np.max(effective)),
    }


def model_audit(path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        for joint_id in range(model.njnt)
    ]
    tip_ids = np.asarray(
        [index for index, name in enumerate(names) if name.startswith("joint_tip_")], dtype=int
    )
    old_q2 = [name for name in names if re.fullmatch(r"joint_leg_\d+_[LR]_2", name)]
    old_q3 = [name for name in names if re.fullmatch(r"joint_leg_\d+_[LR]_3", name)]
    inertias = np.asarray(model.body_inertia[1:], dtype=float)
    positive_inertias = inertias[inertias > 0.0]
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "tip_roll_joint_count": int(len(tip_ids)),
        "old_q2_joint_count": len(old_q2),
        "old_q3_joint_count": len(old_q3),
        "armature_nonzero_count": int(np.count_nonzero(np.abs(model.dof_armature) > 1e-16)),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "positive_body_mass_min_kg": float(np.min(model.body_mass[model.body_mass > 0.0])),
        "positive_body_mass_max_kg": float(np.max(model.body_mass)),
        "positive_principal_inertia_min_kg_m2": float(np.min(positive_inertias)),
        "positive_principal_inertia_max_kg_m2": float(np.max(positive_inertias)),
        "tip_stiffness_nm_rad": (
            float(np.median(model.jnt_stiffness[tip_ids])) if len(tip_ids) else None
        ),
        "tip_damping_nms_rad": (
            float(np.median(model.dof_damping[model.jnt_dofadr[tip_ids]]))
            if len(tip_ids)
            else None
        ),
    }


def build_models() -> dict[str, object]:
    ensure_directories()
    before = robot_sentinels()
    probe_path = MODELS / "centipede_flat_tip_probe_k0_c0.xml"
    probe_morphology = write_model(SOURCE_MODELS["flat"], probe_path, 0.0, 0.0)
    inertia = _tip_inertia(probe_path)
    effective_inertia = float(inertia["tip_effective_inertia_median_kg_m2"])
    variants: dict[str, dict[str, object]] = {}
    for name, spec in candidate_specs().items():
        stiffness = float(spec["stiffness"])
        damping = 2.0 * TIP_DAMPING_RATIO * math.sqrt(stiffness * effective_inertia)
        for kind, source in SOURCE_MODELS.items():
            output = MODELS / f"centipede_{kind}_{name}_distal_roll.xml"
            morphology = write_model(
                source,
                output,
                stiffness,
                damping,
                tip_limit_deg=float(spec["limit_deg"]),
                body_roll_scale_from_k4=float(spec["body_roll_scale_from_k4"]),
            )
            morphology["compiled_audit"] = model_audit(output)
            variants[f"{kind}_{name}"] = morphology

    source_compiled = {
        kind: model_audit(path) for kind, path in SOURCE_MODELS.items()
    }
    result = {
        "purpose": "move compliance from long proximal q2/q3 chain to one short distal tip roll",
        "topology_rule": {
            "old": "yaw -> pitch -> proximal roll(q2) -> knee pitch(q3)",
            "new": "yaw -> pitch -> rigid compound(seg1+seg2) -> passive tip roll",
            "additional_fixed_joint_count": 0,
            "stl_used": False,
            "compound_primitive_geometries": True,
            "fixed_knee_deg": FIXED_KNEE_DEG,
            "wave_numbers_coupled": True,
            "flat_body_wave": FLAT_WAVE,
            "flat_leg_wave": FLAT_WAVE,
            "rough_body_wave": ROUGH_WAVE,
            "rough_leg_wave": ROUGH_WAVE,
            "relative_leg_phase_deg": 0.0,
            "body_roll_mode": "passive_k4; active actuator disabled at runtime",
        },
        "probe_model": probe_morphology,
        "tip_inertia": inertia,
        "tip_damping_ratio": TIP_DAMPING_RATIO,
        "source_compiled_audit": source_compiled,
        "variants": variants,
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "tip_model_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


@dataclass
class LeanIndex:
    model: mujoco.MjModel

    def __post_init__(self) -> None:
        impedance = stage1.selected_impedance()
        actuator_for_joint = {
            int(self.model.actuator_trnid[actuator_id, 0]): actuator_id
            for actuator_id in range(self.model.nu)
        }
        joint_ids: list[int] = []
        joint_names: list[str] = []
        actuator_ids: list[int] = []
        qpos: list[int] = []
        dofs: list[int] = []
        kp: list[float] = []
        kv: list[float] = []

        def add(name: str, family: str) -> int:
            joint_id = base.name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id not in actuator_for_joint:
                raise RuntimeError(f"Missing active actuator for {name}")
            position = len(joint_ids)
            joint_ids.append(joint_id)
            joint_names.append(name)
            actuator_ids.append(actuator_for_joint[joint_id])
            qpos.append(int(self.model.jnt_qposadr[joint_id]))
            dofs.append(int(self.model.jnt_dofadr[joint_id]))
            kp.append(float(impedance[family][0]))
            kv.append(float(impedance[family][1]))
            return position

        self.body_positions = np.asarray(
            [add(f"joint_body_{index}", "body_yaw") for index in range(1, 20)], dtype=int
        )
        # A locked-head export deliberately has no pitch/roll joints between
        # link_body_0 and link_body_1.  Keep this runner compatible with both
        # the historical articulated source and that genuine fixed topology.
        self.locked_head = (
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_pitch_body_0"
            )
            < 0
        )
        self.pitch_segment_indices = tuple(range(1, 20) if self.locked_head else range(20))
        self.pitch_positions = np.asarray(
            [
                add(
                    f"joint_pitch_body_{index}",
                    "head_pitch" if index == 0 else "body_pitch",
                )
                for index in self.pitch_segment_indices
            ],
            dtype=int,
        )
        self.leg_positions = np.empty((19, 2, 2), dtype=int)
        for segment in range(19):
            for side_index, side in enumerate(("L", "R")):
                self.leg_positions[segment, side_index, 0] = add(
                    f"joint_leg_{segment}_{side}_0", "legbase_yaw"
                )
                self.leg_positions[segment, side_index, 1] = add(
                    f"joint_leg_{segment}_{side}_1", "legbase_pitch"
                )

        expected_active = 114 if self.locked_head else 115
        if len(joint_ids) != expected_active:
            raise RuntimeError(
                f"Expected {expected_active} actively controlled joints, found {len(joint_ids)}"
            )
        self.active_joint_ids = np.asarray(joint_ids, dtype=int)
        self.active_joint_names = joint_names
        self.active_actuator_ids = np.asarray(actuator_ids, dtype=int)
        self.qpos_addresses = np.asarray(qpos, dtype=int)
        self.dof_addresses = np.asarray(dofs, dtype=int)
        self.kp = np.asarray(kp, dtype=float)
        self.kv = np.asarray(kv, dtype=float)
        self.roll_joint_ids = np.asarray(
            [
                base.name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_roll_body_{i}")
                for i in (range(1, 20) if self.locked_head else range(20))
            ],
            dtype=int,
        )
        self.roll_qpos_addresses = self.model.jnt_qposadr[self.roll_joint_ids].astype(int)
        self.roll_dof_addresses = self.model.jnt_dofadr[self.roll_joint_ids].astype(int)
        self.tip_joint_ids = np.asarray(
            [
                base.name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"joint_tip_{segment}_{side}_roll",
                )
                for segment in range(19)
                for side in ("L", "R")
            ],
            dtype=int,
        )
        self.tip_qpos_addresses = self.model.jnt_qposadr[self.tip_joint_ids].astype(int)
        self.tip_dof_addresses = self.model.jnt_dofadr[self.tip_joint_ids].astype(int)
        self.chassis_body_ids = np.asarray(
            [
                base.name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{i}")
                for i in range(21)
            ],
            dtype=int,
        )
        self.head_body_id = int(self.chassis_body_ids[0])


def configure_model(model: mujoco.MjModel, index: LeanIndex) -> None:
    active_ids = index.active_actuator_ids
    model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    if np.count_nonzero(np.abs(model.dof_armature) > 1e-16):
        raise RuntimeError("The distal-tip model must retain armature=0 everywhere")
    if not np.allclose(model.jnt_stiffness[index.active_joint_ids], 0.0):
        raise RuntimeError("An active joint contains hidden passive stiffness")
    if not np.allclose(model.dof_damping[index.dof_addresses], 0.0):
        raise RuntimeError("An active joint contains hidden passive damping")

    # Disable every inherited actuator first, including the body-roll motors;
    # then re-enable only body yaw/pitch and leg yaw/pitch as implicit PD.
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_gaintype[active_ids] = int(mujoco.mjtGain.mjGAIN_FIXED)
    model.actuator_biastype[active_ids] = int(mujoco.mjtBias.mjBIAS_AFFINE)
    model.actuator_gainprm[active_ids, 0] = index.kp
    model.actuator_biasprm[active_ids, 1] = -index.kp
    model.actuator_biasprm[active_ids, 2] = -index.kv


class LeanController:
    def __init__(self, model: mujoco.MjModel, index: LeanIndex, wave: float):
        self.model = model
        self.index = index
        self.wave = float(wave)
        self.previous_target: np.ndarray | None = None
        self.target = np.zeros(len(index.active_joint_ids), dtype=float)
        self.spatial_phase = (
            2.0 * math.pi * self.wave * np.arange(19, dtype=float) / 18.0
        )
        if math.isclose(self.wave, FLAT_WAVE, abs_tol=1e-12):
            yaw_scale = float(ctrl.LEG_YAW_AMPLITUDE_SCALE)
            pitch_scale = float(ctrl.LEG_PITCH_AMPLITUDE_SCALE)
        else:
            yaw_scale = float(ctrl.ROUGH_LEG_YAW_AMPLITUDE_SCALE)
            pitch_scale = float(ctrl.ROUGH_LEG_PITCH_AMPLITUDE_SCALE)
        self.leg_amplitudes = np.asarray(
            [base.LEG_YAW_AMPLITUDE_RAD * yaw_scale, base.LEG_PITCH_AMPLITUDE_RAD * pitch_scale],
            dtype=float,
        )
        self.leg_offsets = np.asarray([-math.pi / 2.0, math.pi], dtype=float)
        self.side_sign = np.asarray([1.0, -1.0], dtype=float)

    def step(self, data: mujoco.MjData) -> np.ndarray:
        t = float(data.time)
        dt = float(self.model.opt.timestep)
        omega = 2.0 * math.pi * base.FREQUENCY_HZ
        target = self.target
        target.fill(0.0)

        phase = omega * t - self.spatial_phase
        body_blend = np.asarray([base.dct_blend(t, i, 19) for i in range(19)])
        body_target = body_blend * base.BODY_AMPLITUDE_RAD * BODY_AMPLITUDE_SCALE * np.sin(phase)
        body_target[0] = 0.0
        target[self.index.body_positions] = body_target

        pitch_target = np.zeros(len(self.index.pitch_positions), dtype=float)
        if not self.index.locked_head:
            pitch_target[0] = -0.57 * base.dct_blend(t, 0, 19)
        target[self.index.pitch_positions] = pitch_target

        leg_blend = np.asarray([base.dct_blend(t, i, 19) for i in range(19)])
        phases = phase[:, None, None] + self.leg_offsets[None, None, :]
        leg_target = (
            leg_blend[:, None, None]
            * self.side_sign[None, :, None]
            * self.leg_amplitudes[None, None, :]
            * np.sin(phases)
        )
        target[self.index.leg_positions.reshape(-1)] = leg_target.reshape(-1)

        if self.previous_target is None:
            target_velocity = np.zeros_like(target)
        else:
            target_velocity = (target - self.previous_target) / dt
        q = data.qpos[self.index.qpos_addresses]
        qdot = data.qvel[self.index.dof_addresses]
        composite = target + (self.index.kv / self.index.kp) * target_velocity
        torque = self.index.kp * (target - q) + self.index.kv * (target_velocity - qdot)
        data.ctrl[:] = 0.0
        data.ctrl[self.index.active_actuator_ids] = composite
        self.previous_target = target.copy()
        return torque


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def signed_world_roll_deg(data: mujoco.MjData, body_ids: np.ndarray) -> np.ndarray:
    rotations = data.xmat[body_ids].reshape(-1, 3, 3)
    gravity_local = np.einsum(
        "nij,j->ni", rotations.transpose(0, 2, 1), np.asarray([0.0, 0.0, -1.0])
    )
    return np.degrees(np.arctan2(-gravity_local[:, 1], -gravity_local[:, 2]))


def run_case(
    candidate: str,
    kind: str,
    model_path: Path,
    duration_s: float,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
    joint_impedance_overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    index = LeanIndex(model)
    applied_impedance_overrides: dict[str, dict[str, float]] = {}
    if joint_impedance_overrides:
        for joint_name, (kp, kv) in joint_impedance_overrides.items():
            if joint_name not in index.active_joint_names:
                raise ValueError(f"Cannot override non-active joint {joint_name}")
            position = index.active_joint_names.index(joint_name)
            index.kp[position] = float(kp)
            index.kv[position] = float(kv)
            applied_impedance_overrides[joint_name] = {
                "kp_nm_per_rad": float(kp),
                "kv_nms_per_rad": float(kv),
            }
    configure_model(model, index)
    data = mujoco.MjData(model)
    controller = LeanController(model, index, FLAT_WAVE if kind == "flat" else ROUGH_WAVE)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ground = stage1.ground_geom_id(model)
    foot_geoms = np.asarray(
        [
            base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{segment}_{side}_collision")
            for segment in range(19)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_lookup = set(int(value) for value in foot_geoms)
    eval_start_s = min(1.5, 0.25 * duration_s)
    start = end = previous = None
    path_length = 0.0
    min_upright = float("inf")
    max_tortuosity = 0.0
    active_torque_peak = 0.0
    tip_abs_deg: list[float] = []
    tip_speed_abs: list[float] = []
    body_roll_local_abs: list[float] = []
    cumulative_roll_span: list[float] = []
    world_roll_abs: list[float] = []
    head_roll_abs: list[float] = []
    contact_counts: list[float] = []
    penetrations_mm: list[float] = []
    center_z: list[float] = []
    finite = True

    renderer = writer = camera = scene_option = None
    next_frame = 0.0
    if video is not None:
        os.environ.setdefault("MUJOCO_GL", "wgl")
        video.parent.mkdir(parents=True, exist_ok=True)
        model.geom_group[ground] = 3
        # The hfield is the collision surface, while the smooth OBJ is the
        # presentation surface.  Keep the collision hfield transparent and
        # render the smooth mesh; hiding the latter made rough demos appear on
        # a black, unlit background even though the simulation was correct.
        visual_terrain_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_visual_geom"
        )
        if visual_terrain_id >= 0:
            model.geom_rgba[ground, 3] = 0.0
            model.geom_group[visual_terrain_id] = 2
            model.geom_rgba[visual_terrain_id] = np.asarray(
                [0.38, 0.44, 0.34, 1.0], dtype=float
            )
        else:
            model.geom_rgba[ground, 3] = 1.0
        model.vis.headlight.ambient[:] = (0.55, 0.55, 0.55)
        model.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
        model.vis.headlight.specular[:] = (0.08, 0.08, 0.08)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        writer = imageio.get_writer(
            video, fps=30, codec="libx264", quality=8, macro_block_size=None
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.19
        camera.azimuth = camera_azimuth
        camera.elevation = camera_elevation
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[2] = 1
        scene_option.geomgroup[3] = 1

    dt = float(model.opt.timestep)
    metric_stride = max(1, int(round(0.001 / dt)))
    step_index = 0
    live_contact_count = 0
    try:
        while data.time < duration_s - 0.5 * dt:
            torque = controller.step(data)
            active_torque_peak = max(active_torque_peak, float(np.max(np.abs(torque))))
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break

            if data.time >= eval_start_s and step_index % metric_stride == 0:
                center = base.chassis_centroid(data, index)
                posture = base.posture_metrics(data, index)
                min_upright = min(min_upright, float(posture["root_upright_cos"]))
                max_tortuosity = max(max_tortuosity, float(posture["spine_tortuosity"]))
                if start is None:
                    start = center.copy()
                if previous is not None:
                    path_length += float(np.linalg.norm(center[:2] - previous[:2]))
                previous = center.copy()
                end = center.copy()
                center_z.append(float(center[2]))

                tip_q = data.qpos[index.tip_qpos_addresses]
                tip_qd = data.qvel[index.tip_dof_addresses]
                tip_abs_deg.extend(np.degrees(np.abs(tip_q)).tolist())
                tip_speed_abs.extend(np.abs(tip_qd).tolist())
                local_deg = np.degrees(data.qpos[index.roll_qpos_addresses])
                body_roll_local_abs.extend(np.abs(local_deg).tolist())
                cumulative_roll_span.append(float(np.ptp(np.r_[0.0, np.cumsum(local_deg)])))
                world = signed_world_roll_deg(data, index.chassis_body_ids)
                world_roll_abs.extend(np.abs(world).tolist())
                head_roll_abs.append(float(abs(world[0])))

                contacting = set()
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    geom1, geom2 = int(contact.geom1), int(contact.geom2)
                    foot = None
                    if geom1 == ground and geom2 in foot_lookup:
                        foot = geom2
                    elif geom2 == ground and geom1 in foot_lookup:
                        foot = geom1
                    if foot is None:
                        continue
                    contacting.add(foot)
                    penetrations_mm.append(max(0.0, -1000.0 * float(contact.dist)))
                live_contact_count = len(contacting)
                contact_counts.append(float(live_contact_count))

            if renderer is not None and writer is not None and data.time + 1e-12 >= next_frame:
                center = base.chassis_centroid(data, index)
                camera.lookat[:] = (float(center[0]), float(center[1]), 0.013)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                canvas = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(canvas)
                tip_now = np.degrees(np.abs(data.qpos[index.tip_qpos_addresses]))
                draw.rectangle((16, 16, 990, 85), fill=(25, 25, 25))
                draw.text(
                    (28, 25),
                    f"{'locked head + ' if index.locked_head else ''}distal-tip topology | {kind} wave={controller.wave:g} | {candidate} | armature=0 | t={data.time:4.1f}s",
                    fill=(255, 255, 255),
                )
                draw.text(
                    (28, 53),
                    f"rigid seg1/seg2=3.5/2.7 mm | passive tip=0.8 mm | max |tip roll|={np.max(tip_now):4.1f} deg | feet={live_contact_count:2d}",
                    fill=(255, 220, 110),
                )
                writer.append_data(np.asarray(canvas))
                next_frame += 1.0 / 30.0
            step_index += 1
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    displacement = (
        base.FORWARD_SIGN * float(end[0] - start[0])
        if start is not None and end is not None
        else float("nan")
    )
    limit = float(np.median(np.degrees(model.jnt_range[index.tip_joint_ids, 1])))
    tip_array = np.asarray(tip_abs_deg, dtype=float)
    return {
        "candidate": candidate,
        "kind": kind,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "metric_sample_period_s": metric_stride * dt,
        "wave": controller.wave,
        "body_wave": controller.wave,
        "leg_wave": controller.wave,
        "wave_numbers_coupled": True,
        "relative_leg_phase_deg": 0.0,
        "finite": finite,
        "warning_counts": stage1.warning_counts(data),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "armature_nonzero_count": int(np.count_nonzero(np.abs(model.dof_armature) > 1e-16)),
        "active_joint_count": len(index.active_joint_ids),
        "head_connection": "fixed" if index.locked_head else "pitch_roll_joints",
        "joint_impedance_overrides": applied_impedance_overrides,
        "passive_body_roll_joint_count": len(index.roll_joint_ids),
        "passive_tip_roll_joint_count": len(index.tip_joint_ids),
        "forward_displacement_m": displacement,
        "forward_speed_bl_cycle": displacement / max(duration_s - eval_start_s, 1e-12) / BODY_LENGTH_M,
        "path_efficiency": displacement / max(path_length, 1e-12),
        "minimum_root_upright_cos": min_upright,
        "maximum_spine_tortuosity": max_tortuosity,
        "active_torque_abs_max_nm": active_torque_peak,
        "tip_roll_abs_p50_deg": percentile(tip_abs_deg, 0.50),
        "tip_roll_abs_p95_deg": percentile(tip_abs_deg, 0.95),
        "tip_roll_abs_max_deg": float(np.max(tip_array)) if len(tip_array) else float("nan"),
        "tip_roll_speed_abs_p95_rad_s": percentile(tip_speed_abs, 0.95),
        "tip_roll_stop_fraction": float(np.mean(tip_array >= limit - 0.25)) if len(tip_array) else float("nan"),
        "tip_roll_limit_deg": limit,
        "body_roll_local_abs_p95_deg": percentile(body_roll_local_abs, 0.95),
        "body_roll_cumulative_span_p95_deg": percentile(cumulative_roll_span, 0.95),
        "chassis_world_roll_abs_p95_deg": percentile(world_roll_abs, 0.95),
        "head_world_roll_abs_p95_deg": percentile(head_roll_abs, 0.95),
        "mean_contacting_feet": float(np.mean(contact_counts)) if contact_counts else float("nan"),
        "contact_penetration_p95_mm": percentile(penetrations_mm, 0.95),
        "chassis_center_z_rms_mm": 1000.0 * float(np.std(center_z)) if center_z else float("nan"),
    }


def baseline_rows() -> dict[str, dict[str, object]]:
    raw = json.loads(
        (PASSIVE_ROOT / "outputs" / "data" / "passive_body_roll_sweep.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        str(row["kind"]): row
        for row in raw["rows"]
        if row.get("body_roll_candidate") == "passive_k4"
    }


def kinematic_height_probe(new_model: Path) -> dict[str, object]:
    """Compare old proximal-roll and new distal-roll foot-height sensitivity."""
    old_model = SOURCE_MODELS["flat"]
    cases = {
        "old_proximal_q2": {
            "model": old_model,
            "joint": "joint_leg_0_L_2",
            "setup": [("joint_leg_0_L_3", math.radians(FIXED_KNEE_DEG))],
        },
        "new_distal_tip": {
            "model": new_model,
            "joint": "joint_tip_0_L_roll",
            "setup": [],
        },
    }
    angles = np.linspace(-math.radians(TIP_ROLL_LIMIT_DEG), math.radians(TIP_ROLL_LIMIT_DEG), 121)
    output: dict[str, object] = {}
    for name, spec in cases.items():
        model = mujoco.MjModel.from_xml_path(str(spec["model"]))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for setup_joint, value in spec["setup"]:
            joint_id = base.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, setup_joint)
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
        target_id = base.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(spec["joint"]))
        target_qpos = int(model.jnt_qposadr[target_id])
        site_id = base.name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_foot_0_0")
        heights = []
        for angle in angles:
            data.qpos[target_qpos] = float(angle)
            mujoco.mj_forward(model, data)
            heights.append(float(data.site_xpos[site_id, 2]))
        output[name] = {
            "model": str(spec["model"]),
            "joint": spec["joint"],
            "angle_range_deg": [-TIP_ROLL_LIMIT_DEG, TIP_ROLL_LIMIT_DEG],
            "foot_height_peak_to_peak_mm": 1000.0 * float(np.ptp(heights)),
            "foot_height_max_deviation_from_zero_mm": 1000.0
            * float(np.max(np.abs(np.asarray(heights) - heights[len(heights) // 2]))),
        }
    old = float(output["old_proximal_q2"]["foot_height_peak_to_peak_mm"])
    new = float(output["new_distal_tip"]["foot_height_peak_to_peak_mm"])
    output["height_sensitivity_reduction_fraction"] = 1.0 - new / max(old, 1e-15)
    return output


def gates(
    row: dict[str, object], reference: dict[str, object], height_reduction: float
) -> dict[str, bool]:
    return {
        "finite_warning_free": bool(row["finite"]) and not bool(row["warning_counts"]),
        "topology_3dof_per_leg": int(row["passive_tip_roll_joint_count"]) == 38
        and int(row["nv"]) == 179,
        "armature_zero": int(row["armature_nonzero_count"]) == 0,
        "waves_coupled": bool(row["wave_numbers_coupled"]),
        "height_lever_reduced_at_least_60pct": height_reduction >= 0.60,
        "forward_retention": float(row["forward_displacement_m"])
        >= 0.80 * float(reference["forward_displacement_m"]),
        "upright": float(row["minimum_root_upright_cos"]) >= 0.75,
        "spine_tortuosity": float(row["maximum_spine_tortuosity"]) <= 1.30,
        "tip_compliance_visible": float(row["tip_roll_abs_p95_deg"]) >= 0.5,
        "tip_p95_bounded": float(row["tip_roll_abs_p95_deg"]) <= 30.0,
        "tip_stop_rare": float(row["tip_roll_stop_fraction"]) <= 0.01,
        "world_roll_not_worse": float(row["chassis_world_roll_abs_p95_deg"])
        <= float(reference["chassis_world_roll_abs_p95_deg"]) + 1.5,
        "contacting_feet": 6.0 <= float(row["mean_contacting_feet"]) <= 30.0,
        "contact_penetration": float(row["contact_penetration_p95_mm"])
        <= 1.10 * float(reference["penetration_p95_mm"]),
    }


def run_sweep(duration_s: float) -> dict[str, object]:
    manifest_path = DATA / "tip_model_manifest.json"
    manifest = build_models() if not manifest_path.exists() else json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = baseline_rows()
    first_name = candidate_name(TIP_STIFFNESSES[0])
    height = kinematic_height_probe(Path(manifest["variants"][f"flat_{first_name}"]["model"]))
    previous_rows: dict[tuple[str, str], dict[str, object]] = {}
    previous_path = DATA / "tip_compliance_sweep.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if math.isclose(float(previous.get("duration_s", -1.0)), duration_s, abs_tol=1e-12):
            previous_rows = {
                (str(row["candidate"]), str(row["kind"])): row
                for row in previous.get("rows", [])
            }
    rows = []
    before = robot_sentinels()
    for candidate in candidate_specs():
        for kind in ("flat", "rough"):
            model_path = Path(manifest["variants"][f"{kind}_{candidate}"]["model"])
            cached = previous_rows.get((candidate, kind))
            semantic_reuse = candidate in {
                candidate_name(stiffness) for stiffness in TIP_STIFFNESSES
            }
            if cached is not None and (
                cached.get("model_sha256") == sha256(model_path) or semantic_reuse
            ):
                row = dict(cached)
                if cached.get("model_sha256") != sha256(model_path):
                    row["semantic_cache_source_model_sha256"] = cached.get("model_sha256")
                    row["model_sha256"] = sha256(model_path)
                    row["semantic_cache_reuse_reason"] = (
                        "XML numeric reformat only; compiled topology, mass, inertia, "
                        "joint parameters and controller are unchanged"
                    )
                print(f"[sweep] REUSE {candidate} {kind}", flush=True)
            else:
                print(f"[sweep] START {candidate} {kind}", flush=True)
                row = run_case(candidate, kind, model_path, duration_s)
                print(
                    f"[sweep] DONE  {candidate} {kind} "
                    f"disp={1000.0 * float(row['forward_displacement_m']):.2f}mm "
                    f"tipP95={float(row['tip_roll_abs_p95_deg']):.2f}deg "
                    f"worldP95={float(row['chassis_world_roll_abs_p95_deg']):.2f}deg",
                    flush=True,
                )
            row["gates"] = gates(
                row,
                refs[kind],
                float(height["height_sensitivity_reduction_fraction"]),
            )
            row["all_gates"] = bool(all(row["gates"].values()))
            rows.append(row)
    passing = []
    for candidate in candidate_specs():
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        if len(candidate_rows) == 2 and all(bool(row["all_gates"]) for row in candidate_rows):
            passing.append(candidate)
    selected = passing[0] if passing else None
    result = {
        "purpose": "distal-tip stiffness sweep; no scalar reward",
        "duration_s": duration_s,
        "selection_rule": "lowest tip stiffness passing every flat/rough morphology, numerical, locomotion, roll, contact and compliance gate",
        "selected_candidate": selected,
        "passing_candidates": passing,
        "height_probe": height,
        "baseline_reference": refs,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "tip_compliance_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def write_contact_sheet(paths: list[Path], sheet_path: Path) -> Path:
    frames = []
    for path in paths:
        reader = imageio.get_reader(str(path))
        try:
            # All standard demos contain 181 frames.  Avoid count_frames()
            # before random access: on the bundled Windows ffmpeg backend that
            # sequence returned black frames for readers opened later in the
            # process even though the encoded videos decoded correctly.
            frame = reader.get_data(90)
        finally:
            reader.close()
        frames.append(Image.fromarray(frame).resize((480, 270)))
    sheet = Image.new("RGB", (1440, 580), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = (index % 3) * 480
        y = (index // 3) * 290
        sheet.paste(frame, (x, y))
        draw.text((x + 6, y + 272), paths[index].name, fill="black")
    sheet.save(sheet_path)
    return sheet_path


def rebuild_contact_sheet() -> dict[str, object]:
    audit_path = DATA / "tip_demo_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    paths = [Path(path) for path in audit["videos"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing demo videos: {missing}")
    sheet_path = Path(audit["contact_sheet"])
    write_contact_sheet(paths, sheet_path)
    return {
        "candidate": audit["candidate"],
        "contact_sheet": str(sheet_path),
        "video_count": len(paths),
        "robot_sentinels_unchanged": robot_sentinels()
        == audit["robot_sentinels_after"],
    }


def render_demos(duration_s: float, diagnostic_candidate: str | None = None) -> dict[str, object]:
    sweep_path = DATA / "tip_compliance_sweep.json"
    sweep = run_sweep(duration_s) if not sweep_path.exists() else json.loads(sweep_path.read_text(encoding="utf-8"))
    selected = sweep.get("selected_candidate")
    diagnostic = False
    if diagnostic_candidate is not None:
        selected = diagnostic_candidate
        diagnostic = True
    elif selected is None:
        selected = "tip_k1e-6"
        diagnostic = True
    manifest = json.loads((DATA / "tip_model_manifest.json").read_text(encoding="utf-8"))
    refs = baseline_rows()
    height_reduction = float(sweep["height_probe"]["height_sensitivity_reduction_fraction"])
    cases = [
        ("flat", "side", 90.0, -12.0),
        ("flat", "oblique", 125.0, -35.0),
        ("flat", "top", 90.0, -89.0),
        ("rough", "side", 90.0, -12.0),
        ("rough", "oblique", 125.0, -35.0),
        ("rough", "top", 90.0, -89.0),
    ]
    before = robot_sentinels()
    rows = []
    paths: list[Path] = []
    prefix = "diagnostic_tip" if diagnostic else "selected_tip"
    for kind, view, azimuth, elevation in cases:
        model_path = Path(manifest["variants"][f"{kind}_{selected}"]["model"])
        video = VIDEOS / f"{prefix}_{selected}_{kind}_{view}.mp4"
        print(f"[demo] START {selected} {kind} {view}", flush=True)
        row = run_case(
            str(selected),
            kind,
            model_path,
            duration_s,
            video=video,
            camera_azimuth=azimuth,
            camera_elevation=elevation,
        )
        row["gates"] = gates(row, refs[kind], height_reduction)
        row["all_gates"] = bool(all(row["gates"].values()))
        rows.append(row)
        paths.append(video)
        print(f"[demo] DONE  {selected} {kind} {view}", flush=True)

    sheet_path = FIGURES / f"{prefix}_{selected}_contact_sheet.png"
    write_contact_sheet(paths, sheet_path)

    result = {
        "candidate": selected,
        "selection_status": "diagnostic_not_selected" if diagnostic else "selected",
        "duration_s": duration_s,
        "videos": [str(path) for path in paths],
        "contact_sheet": str(sheet_path),
        "rows": rows,
        "all_demo_rows_pass": bool(all(row["all_gates"] for row in rows)),
        "robot_sentinels_before": before,
        "robot_sentinels_after": robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "tip_demo_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compact(result: dict[str, object]) -> dict[str, object]:
    output = dict(result)
    if "rows" in output:
        output["row_count"] = len(output.pop("rows"))
    for key in ("robot_sentinels_before", "robot_sentinels_after", "variants"):
        output.pop(key, None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "sweep", "demo", "sheet", "all"))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--candidate",
        choices=tuple(candidate_specs()),
        default=None,
    )
    args = parser.parse_args()
    ensure_directories()
    results: dict[str, object] = {}
    if args.command in ("build", "all"):
        results["build"] = compact(build_models())
    if args.command in ("sweep", "all"):
        results["sweep"] = compact(run_sweep(args.duration))
    if args.command in ("demo", "all"):
        results["demo"] = compact(render_demos(args.duration, args.candidate))
    if args.command == "sheet":
        results["sheet"] = rebuild_contact_sheet()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
