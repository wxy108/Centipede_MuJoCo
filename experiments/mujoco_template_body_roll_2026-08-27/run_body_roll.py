"""Evidence-first body-roll audit for the abstract Centipede Model.

The accepted q2-compliance plant is frozen.  This isolated experiment varies
only the explicit body-roll controller gains and body-roll anatomical range,
measures local/cumulative/world-frame roll, and never writes to Centipede
Robot Step 18/19.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
Q2_ROOT = EXPERIMENTS / "mujoco_template_q2_compliance_2026-08-26"
Q2_SCRIPT = Q2_ROOT / "run_q2_compliance.py"
Q2_MANIFEST = Q2_ROOT / "outputs" / "data" / "passive_model_manifest.json"

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"

Q2_CANDIDATE = "bounded_k20_c18"
Q2_LIMIT_DEG = 30.0
BODY_SCALE = 0.55
HOLD_FIRST_BODY_YAW = True
ROUGH_WAVE = 3.0

# Kp multiplier; Kv follows sqrt(multiplier), retaining the baseline damping
# ratio for the same effective inertia.  Limits are explicit MJCF ranges.
CANDIDATES = {
    "g1_r30": {"kp_multiplier": 1.0, "roll_limit_deg": 30.0, "pg_gain": 0.0},
    "g2_r30": {"kp_multiplier": 2.0, "roll_limit_deg": 30.0, "pg_gain": 0.0},
    "g4_r30": {"kp_multiplier": 4.0, "roll_limit_deg": 30.0, "pg_gain": 0.0},
    "g1_r8": {"kp_multiplier": 1.0, "roll_limit_deg": 8.0, "pg_gain": 0.0},
    "g2_r8": {"kp_multiplier": 2.0, "roll_limit_deg": 8.0, "pg_gain": 0.0},
    "g4_r8": {"kp_multiplier": 4.0, "roll_limit_deg": 8.0, "pg_gain": 0.0},
    "g8_r6": {"kp_multiplier": 8.0, "roll_limit_deg": 6.0, "pg_gain": 0.0},
    # Low-bandwidth projected-gravity feedback is applied with opposite signs
    # in the left/right joint coordinates.  The joint axes and lateral link
    # directions both reverse across sides, so opposite q offsets (not equal
    # offsets) are the physical differential leg-height command.
    "pgp025_g1_r8": {"kp_multiplier": 1.0, "roll_limit_deg": 8.0, "pg_gain": 0.25},
    "pgn025_g1_r8": {"kp_multiplier": 1.0, "roll_limit_deg": 8.0, "pg_gain": -0.25},
    "pgp050_g1_r8": {"kp_multiplier": 1.0, "roll_limit_deg": 8.0, "pg_gain": 0.50},
    "pgn050_g1_r8": {"kp_multiplier": 1.0, "roll_limit_deg": 8.0, "pg_gain": -0.50},
    "pgn100_g1_r30": {
        "kp_multiplier": 1.0,
        "roll_limit_deg": 30.0,
        "pg_gain": -1.0,
        "pg_clip_rad": 0.24,
    },
    "pgn150_g1_r30": {
        "kp_multiplier": 1.0,
        "roll_limit_deg": 30.0,
        "pg_gain": -1.5,
        "pg_clip_rad": 0.30,
    },
    "pgn100_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": -1.0,
        "pg_clip_rad": 0.24,
    },
    "pgn150_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": -1.5,
        "pg_clip_rad": 0.30,
    },
    "sgn010_g1_r12": {
        "kp_multiplier": 1.0,
        "roll_limit_deg": 12.0,
        "pg_gain": -0.10,
        "pg_scope": "segmental",
    },
    "sgn025_g1_r12": {
        "kp_multiplier": 1.0,
        "roll_limit_deg": 12.0,
        "pg_gain": -0.25,
        "pg_scope": "segmental",
    },
    "sgn050_g1_r12": {
        "kp_multiplier": 1.0,
        "roll_limit_deg": 12.0,
        "pg_gain": -0.50,
        "pg_scope": "segmental",
    },
    "sgn025_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": -0.25,
        "pg_scope": "segmental",
    },
    "c_hn050_sn025_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -0.50,
        "pg_segment_gain": -0.25,
        "pg_clip_rad": 0.24,
    },
    "c_hn100_sn025_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.25,
        "pg_clip_rad": 0.30,
    },
    "c_hn050_sn050_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -0.50,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.24,
    },
    "c_hn100_sn050_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.30,
    },
    "c_hn100_sn050_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.30,
    },
    "c_hn150_sn100_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
    },
    "c_hn150_sn100_g4_r8": {
        "kp_multiplier": 4.0,
        "roll_limit_deg": 8.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
    },
    "c_hn200_sn150_g4_r8": {
        "kp_multiplier": 4.0,
        "roll_limit_deg": 8.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -2.00,
        "pg_segment_gain": -1.50,
        "pg_clip_rad": 0.40,
    },
    "lp075_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "leg_pitch_amplitude_factor": 0.75,
    },
    "lp050_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "leg_pitch_amplitude_factor": 0.50,
    },
    "lp075_c_hn150_sn100_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "leg_pitch_amplitude_factor": 0.75,
    },
    "lp050_c_hn150_sn100_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "leg_pitch_amplitude_factor": 0.50,
    },
    "lw3_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 3.0,
    },
    "lw4p5_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 4.5,
    },
    "lw3_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 3.0,
    },
    "lw4p5_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 4.5,
    },
    "lw2_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 2.0,
    },
    "lw2p25_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 2.25,
    },
    "lw2p5_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 2.5,
    },
    "lw2p75_g2_r12": {
        "kp_multiplier": 2.0,
        "roll_limit_deg": 12.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 2.75,
    },
    "lw3_phm90_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": -90.0,
    },
    "lw3_phm45_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": -45.0,
    },
    "lw3_php45_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
    },
    "lw3_php90_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 90.0,
    },
    "lw3_php45_c_hn100_sn050_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.30,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
    },
    "lw3_php45_c_hn150_sn100_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
    },
    "lw3_php90_c_hn100_sn050_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.30,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 90.0,
    },
    "lw3_php90_c_hn150_sn100_g3_r10": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 90.0,
    },
    "lw3_php45_c1_g3_r10_q2g28": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.30,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.0,
    },
    "lw3_php45_c1_g3_r10_q2g28p5": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.00,
        "pg_segment_gain": -0.50,
        "pg_clip_rad": 0.30,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.5,
    },
    "lw3_php45_c2_g3_r10_q2g28": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.0,
    },
    "lw3_php45_c2_g3_r10_q2g28p5": {
        "kp_multiplier": 3.0,
        "roll_limit_deg": 10.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.5,
    },
    "lw3_php45_c2_g4_r8_q2g28p5": {
        "kp_multiplier": 4.0,
        "roll_limit_deg": 8.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.5,
    },
    "lw3_php45_c3_g4_r8_q2g28p5": {
        "kp_multiplier": 4.0,
        "roll_limit_deg": 8.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -2.00,
        "pg_segment_gain": -1.50,
        "pg_clip_rad": 0.40,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.5,
    },
    "lw3_php45_c2_g5_r8_q2g28p5": {
        "kp_multiplier": 5.0,
        "roll_limit_deg": 8.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -1.50,
        "pg_segment_gain": -1.00,
        "pg_clip_rad": 0.35,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.5,
    },
    "lw3_php45_c3_g5_r8_q2g28p5": {
        "kp_multiplier": 5.0,
        "roll_limit_deg": 8.0,
        "pg_gain": 0.0,
        "pg_scope": "combined",
        "pg_head_gain": -2.00,
        "pg_segment_gain": -1.50,
        "pg_clip_rad": 0.40,
        "flat_leg_wave": 3.0,
        "flat_leg_phase_shift_deg": 45.0,
        "q2_guard_limit_deg": 28.5,
    },
}

PG_FILTER_TAU_S = 0.12
PG_TARGET_CLIP_RAD = 0.12

BODY_ROLL_STOP_SOLREF = "0.0005 1.5"
BODY_ROLL_STOP_SOLIMP = "0.999 0.9999 0.00001 0.5 2"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q2 = load_module(Q2_SCRIPT, "centipede_body_roll_q2_source")
stage1 = q2.stage1


def ensure_directories() -> None:
    for path in (MODELS, DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def source_models() -> dict[str, Path]:
    manifest = json.loads(Q2_MANIFEST.read_text(encoding="utf-8"))
    return {
        kind: Path(manifest["variants"][f"{kind}_{Q2_CANDIDATE}"]["model"])
        for kind in ("flat", "rough")
    }


def impedance_for(candidate: str) -> dict[str, tuple[float, float]]:
    impedance = stage1.selected_impedance()
    multiplier = float(CANDIDATES[candidate]["kp_multiplier"])
    kp, kv = impedance["body_roll"]
    impedance["body_roll"] = (kp * multiplier, kv * math.sqrt(multiplier))
    return impedance


def build_models() -> dict[str, object]:
    sources = source_models()
    variants: dict[str, dict[str, object]] = {}
    for candidate, spec in CANDIDATES.items():
        limit_deg = float(spec["roll_limit_deg"])
        limit_rad = math.radians(limit_deg)
        for kind, source in sources.items():
            tree = ET.parse(source)
            root = tree.getroot()
            count = 0
            for joint in root.iter("joint"):
                name = joint.get("name", "")
                if (
                    name.startswith("joint_leg_")
                    and name.endswith("_2")
                    and "q2_guard_limit_deg" in spec
                ):
                    q2_guard_rad = math.radians(float(spec["q2_guard_limit_deg"]))
                    joint.set("limited", "true")
                    joint.set("range", f"{-q2_guard_rad:.12g} {q2_guard_rad:.12g}")
                if not name.startswith("joint_roll_body_"):
                    continue
                joint.set("limited", "true")
                joint.set("range", f"{-limit_rad:.12g} {limit_rad:.12g}")
                joint.set("solreflimit", BODY_ROLL_STOP_SOLREF)
                joint.set("solimplimit", BODY_ROLL_STOP_SOLIMP)
                # Body-roll remains an active zero-target coordinate.  Do not
                # add a second hidden passive spring/damper path.
                joint.set("stiffness", "0")
                joint.set("damping", "0")
                joint.set("armature", "0")
                count += 1
            if count != 20:
                raise RuntimeError(f"Expected 20 body-roll joints in {source}, found {count}")
            output = MODELS / f"centipede_{kind}_bodyroll_{candidate}.xml"
            tree.write(output, encoding="utf-8", xml_declaration=True)
            impedance = impedance_for(candidate)
            variants[f"{kind}_{candidate}"] = {
                "kind": kind,
                "candidate": candidate,
                "model": str(output),
                "model_sha256": stage1.sha256(output),
                "source_model": str(source),
                "source_model_sha256": stage1.sha256(source),
                "body_roll_joint_count": count,
                "body_roll_limit_deg": limit_deg,
                "body_roll_kp_nm_rad": impedance["body_roll"][0],
                "body_roll_kv_nms_rad": impedance["body_roll"][1],
                "body_roll_kp_multiplier": spec["kp_multiplier"],
                "projected_gravity_leg_pitch_gain": spec.get("pg_gain", 0.0),
                "projected_gravity_feedback_scope": spec.get("pg_scope", "head"),
                "projected_gravity_head_gain": spec.get(
                    "pg_head_gain",
                    spec.get("pg_gain", 0.0)
                    if spec.get("pg_scope", "head") == "head"
                    else 0.0,
                ),
                "projected_gravity_segment_gain": spec.get(
                    "pg_segment_gain",
                    spec.get("pg_gain", 0.0)
                    if spec.get("pg_scope", "head") == "segmental"
                    else 0.0,
                ),
                "projected_gravity_filter_tau_s": PG_FILTER_TAU_S,
                "projected_gravity_target_clip_rad": spec.get(
                    "pg_clip_rad", PG_TARGET_CLIP_RAD
                ),
                "flat_leg_pitch_amplitude_factor": spec.get(
                    "leg_pitch_amplitude_factor", 1.0
                ),
                "flat_leg_spatial_wave": spec.get("flat_leg_wave", 1.5),
                "flat_leg_phase_shift_deg": spec.get(
                    "flat_leg_phase_shift_deg", 0.0
                ),
                "q2_guard_limit_deg": spec.get("q2_guard_limit_deg", Q2_LIMIT_DEG),
                "body_roll_stop_solref": BODY_ROLL_STOP_SOLREF,
                "body_roll_stop_solimp": BODY_ROLL_STOP_SOLIMP,
            }
    result = {
        "purpose": "body-roll-only variants of the accepted bounded_k20_c18 q2 plant",
        "q2_candidate_frozen": Q2_CANDIDATE,
        "body_amplitude_scale_frozen": BODY_SCALE,
        "hold_first_body_yaw_frozen": HOLD_FIRST_BODY_YAW,
        "variants": variants,
        "robot_sentinels_before": stage1.robot_sentinels(),
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "body_roll_model_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else float("nan")


def run_with_roll_capture(
    candidate: str,
    kind: str,
    model_path: Path,
    duration_s: float,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
    camera_distance: float = 0.19,
) -> dict[str, object]:
    impedance = impedance_for(candidate)
    original_selected_impedance = stage1.selected_impedance
    original_posture_metrics = stage1.base.posture_metrics
    original_controller = stage1.Stage1FloatWaveController

    local_abs_deg: list[float] = []
    frame_local_rms_deg: list[float] = []
    frame_local_max_deg: list[float] = []
    cumulative_span_deg: list[float] = []
    tail_relative_twist_abs_deg: list[float] = []
    world_abs_deg: list[float] = []
    frame_world_span_deg: list[float] = []
    head_world_roll_abs_deg: list[float] = []
    body_roll_torque_abs_nm: list[float] = []
    feedback_correction_abs_rad: list[float] = []

    def selected_impedance_override() -> dict[str, tuple[float, float]]:
        return dict(impedance)

    pg_gain = float(CANDIDATES[candidate].get("pg_gain", 0.0))
    pg_clip_rad = float(CANDIDATES[candidate].get("pg_clip_rad", PG_TARGET_CLIP_RAD))
    pg_scope = str(CANDIDATES[candidate].get("pg_scope", "head"))
    pg_head_gain = float(
        CANDIDATES[candidate].get(
            "pg_head_gain", pg_gain if pg_scope == "head" else 0.0
        )
    )
    pg_segment_gain = float(
        CANDIDATES[candidate].get(
            "pg_segment_gain", pg_gain if pg_scope == "segmental" else 0.0
        )
    )
    leg_pitch_amplitude_factor = float(
        CANDIDATES[candidate].get("leg_pitch_amplitude_factor", 1.0)
    )
    flat_leg_wave = float(CANDIDATES[candidate].get("flat_leg_wave", 1.5))
    flat_leg_phase_shift_rad = math.radians(
        float(CANDIDATES[candidate].get("flat_leg_phase_shift_deg", 0.0))
    )

    class RollStabilizedController(original_controller):
        def __init__(self, model, index, wave, body_amplitude_scale, hold_first_body_yaw):
            super().__init__(
                model, index, wave, body_amplitude_scale, hold_first_body_yaw
            )
            self.pg_head_filtered_y = 0.0
            self.pg_segment_filtered_y = np.zeros(
                stage1.base.N_LEG_PAIRS, dtype=float
            )
            if math.isclose(float(wave), 1.5, rel_tol=0.0, abs_tol=1e-12):
                self.leg_amplitudes[1] *= leg_pitch_amplitude_factor

        def step(self, data):
            previous = None if self.previous_target is None else self.previous_target.copy()
            torque, qdot = super().step(data)
            affected_positions: list[int] = []
            if not math.isclose(
                flat_leg_wave, 1.5, rel_tol=0.0, abs_tol=1e-12
            ) and math.isclose(float(self.wave), 1.5, rel_tol=0.0, abs_tol=1e-12):
                t = float(data.time)
                omega = 2.0 * math.pi * stage1.base.FREQUENCY_HZ
                spatial_phase = (
                    2.0
                    * math.pi
                    * flat_leg_wave
                    * np.arange(stage1.base.N_LEG_PAIRS, dtype=float)
                    / (stage1.base.N_BODY_YAW - 1)
                )
                leg_phase = omega * t - spatial_phase + flat_leg_phase_shift_rad
                leg_blend = np.asarray(
                    [
                        stage1.base.dct_blend(t, i, stage1.base.N_LEG_PAIRS)
                        for i in range(stage1.base.N_LEG_PAIRS)
                    ],
                    dtype=float,
                )
                for dof in (0, 1):
                    positions = self.index.leg_positions[:, :, dof]
                    phase = leg_phase[:, None] + self.leg_offsets[dof]
                    targets = (
                        leg_blend[:, None]
                        * self.side_sign[None, :]
                        * self.leg_amplitudes[dof]
                        * np.sin(phase)
                        + self.leg_dc[dof]
                    )
                    self.target[positions] = targets
                    affected_positions.extend(positions.reshape(-1).tolist())

            feedback_enabled = not math.isclose(
                pg_head_gain, 0.0, abs_tol=1e-15
            ) or not math.isclose(pg_segment_gain, 0.0, abs_tol=1e-15)
            if not affected_positions and not feedback_enabled:
                return torque, qdot

            dt = float(self.model.opt.timestep)
            if feedback_enabled:
                alpha = dt / (PG_FILTER_TAU_S + dt)
                corrections = np.zeros(stage1.base.N_LEG_PAIRS, dtype=float)
                if not math.isclose(pg_head_gain, 0.0, abs_tol=1e-15):
                    gravity = stage1.base.projected_gravity_body(data, self.index.head_body_id)
                    self.pg_head_filtered_y += alpha * (
                        float(gravity[1]) - self.pg_head_filtered_y
                    )
                    corrections += pg_head_gain * self.pg_head_filtered_y
                if not math.isclose(pg_segment_gain, 0.0, abs_tol=1e-15):
                    body_ids = self.index.chassis_body_ids[1 : 1 + stage1.base.N_LEG_PAIRS]
                    rotations = data.xmat[body_ids].reshape(-1, 3, 3)
                    gravity_local = np.einsum(
                        "nij,j->ni",
                        rotations.transpose(0, 2, 1),
                        np.asarray([0.0, 0.0, -1.0]),
                    )
                    measured_y = gravity_local[:, 1]
                    self.pg_segment_filtered_y += alpha * (
                        measured_y - self.pg_segment_filtered_y
                    )
                    corrections += pg_segment_gain * self.pg_segment_filtered_y
                corrections = np.clip(corrections, -pg_clip_rad, pg_clip_rad)
                feedback_correction_abs_rad.extend(np.abs(corrections).tolist())
                pitch_matrix = self.index.leg_positions[:, :, 1]
                self.target[pitch_matrix] += corrections[:, None] * self.side_sign[None, :]
                affected_positions.extend(pitch_matrix.reshape(-1).tolist())

            controlled_positions = np.asarray(
                sorted(set(affected_positions)), dtype=int
            )
            if previous is None:
                target_velocity = np.zeros(len(controlled_positions), dtype=float)
            else:
                target_velocity = (
                    self.target[controlled_positions] - previous[controlled_positions]
                ) / dt
            q = data.qpos[self.index.qpos_addresses[controlled_positions]]
            qvel = data.qvel[self.index.dof_addresses[controlled_positions]]
            kp = self.index.kp[controlled_positions]
            kv = self.index.kv[controlled_positions]
            composite_reference = self.target[controlled_positions] + (kv / kp) * target_velocity
            actuator_ids = self.index.active_actuator_ids[controlled_positions]
            data.ctrl[actuator_ids] = composite_reference
            torque[controlled_positions] = kp * (self.target[controlled_positions] - q) + kv * (
                target_velocity - qvel
            )
            self.previous_target = self.target.copy()
            return torque, qdot

    def posture_capture(data, index):
        result = original_posture_metrics(data, index)
        q_local = np.asarray(
            data.qpos[index.qpos_addresses[index.roll_positions]], dtype=float
        )
        local_deg = np.degrees(q_local)
        local_abs_deg.extend(np.abs(local_deg).tolist())
        frame_local_rms_deg.append(float(np.sqrt(np.mean(np.square(local_deg)))))
        frame_local_max_deg.append(float(np.max(np.abs(local_deg))))

        cumulative = np.concatenate(([0.0], np.cumsum(local_deg)))
        cumulative_span_deg.append(float(np.ptp(cumulative)))
        tail_relative_twist_abs_deg.append(float(abs(cumulative[-1])))

        rotations = data.xmat[index.chassis_body_ids].reshape(-1, 3, 3)
        gravity_local = np.einsum(
            "nij,j->ni", rotations.transpose(0, 2, 1), np.asarray([0.0, 0.0, -1.0])
        )
        signed_world_roll = np.degrees(
            np.arctan2(-gravity_local[:, 1], -gravity_local[:, 2])
        )
        world_abs_deg.extend(np.abs(signed_world_roll).tolist())
        frame_world_span_deg.append(float(np.ptp(signed_world_roll)))
        head_world_roll_abs_deg.append(float(abs(signed_world_roll[0])))

        actuator_ids = index.active_actuator_ids[index.roll_positions]
        body_roll_torque_abs_nm.extend(
            np.abs(np.asarray(data.actuator_force[actuator_ids], dtype=float)).tolist()
        )
        return result

    stage1.selected_impedance = selected_impedance_override
    stage1.base.posture_metrics = posture_capture
    stage1.Stage1FloatWaveController = RollStabilizedController
    try:
        row = stage1.run_diagnostic(
            f"{kind} body-roll {candidate}",
            model_path,
            1.5 if kind == "flat" else ROUGH_WAVE,
            Q2_LIMIT_DEG,
            duration_s=duration_s,
            video=video,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
            q2_mode="passive",
            q2_passive_name=Q2_CANDIDATE,
            body_amplitude_scale=BODY_SCALE,
            hold_first_body_yaw=HOLD_FIRST_BODY_YAW,
        )
    finally:
        stage1.selected_impedance = original_selected_impedance
        stage1.base.posture_metrics = original_posture_metrics
        stage1.Stage1FloatWaveController = original_controller

    limit_deg = float(CANDIDATES[candidate]["roll_limit_deg"])
    local_array = np.asarray(local_abs_deg, dtype=float)
    row.update(
        {
            "kind": kind,
            "body_roll_candidate": candidate,
            "body_roll_kp_multiplier": CANDIDATES[candidate]["kp_multiplier"],
            "body_roll_kp_nm_rad": impedance["body_roll"][0],
            "body_roll_kv_nms_rad": impedance["body_roll"][1],
            "body_roll_limit_deg": limit_deg,
            "q2_nominal_limit_deg": Q2_LIMIT_DEG,
            "q2_guard_limit_deg": float(
                CANDIDATES[candidate].get("q2_guard_limit_deg", Q2_LIMIT_DEG)
            ),
            "projected_gravity_leg_pitch_gain": pg_gain,
            "projected_gravity_feedback_scope": pg_scope,
            "projected_gravity_head_gain": pg_head_gain,
            "projected_gravity_segment_gain": pg_segment_gain,
            "flat_leg_pitch_amplitude_factor": leg_pitch_amplitude_factor,
            "flat_leg_pitch_amplitude_rad": float(
                stage1.base.LEG_PITCH_AMPLITUDE_RAD
                * stage1.ctrl.LEG_PITCH_AMPLITUDE_SCALE
                * leg_pitch_amplitude_factor
            ),
            "flat_leg_spatial_wave": flat_leg_wave,
            "flat_leg_phase_shift_deg": math.degrees(flat_leg_phase_shift_rad),
            "projected_gravity_filter_tau_s": PG_FILTER_TAU_S,
            "projected_gravity_target_clip_rad": pg_clip_rad,
            "projected_gravity_correction_abs_p95_deg": math.degrees(
                percentile(feedback_correction_abs_rad, 0.95)
            )
            if feedback_correction_abs_rad
            else 0.0,
            "projected_gravity_correction_abs_max_deg": math.degrees(
                percentile(feedback_correction_abs_rad, 1.0)
            )
            if feedback_correction_abs_rad
            else 0.0,
            "body_roll_local_abs_rms_deg": float(np.sqrt(np.mean(np.square(local_array)))),
            "body_roll_local_abs_p95_deg": percentile(local_abs_deg, 0.95),
            "body_roll_local_abs_max_deg": percentile(local_abs_deg, 1.0),
            "body_roll_frame_local_rms_p95_deg": percentile(frame_local_rms_deg, 0.95),
            "body_roll_frame_local_max_p95_deg": percentile(frame_local_max_deg, 0.95),
            "body_roll_cumulative_span_p95_deg": percentile(cumulative_span_deg, 0.95),
            "body_roll_cumulative_span_max_deg": percentile(cumulative_span_deg, 1.0),
            "body_roll_tail_relative_twist_p95_deg": percentile(
                tail_relative_twist_abs_deg, 0.95
            ),
            "body_roll_tail_relative_twist_max_deg": percentile(
                tail_relative_twist_abs_deg, 1.0
            ),
            "chassis_world_roll_abs_p95_deg": percentile(world_abs_deg, 0.95),
            "chassis_world_roll_abs_max_deg": percentile(world_abs_deg, 1.0),
            "chassis_world_roll_span_p95_deg": percentile(frame_world_span_deg, 0.95),
            "chassis_world_roll_span_max_deg": percentile(frame_world_span_deg, 1.0),
            "head_world_roll_abs_p95_deg": percentile(head_world_roll_abs_deg, 0.95),
            "head_world_roll_abs_max_deg": percentile(head_world_roll_abs_deg, 1.0),
            "body_roll_torque_abs_p95_nm": percentile(body_roll_torque_abs_nm, 0.95),
            "body_roll_torque_abs_max_nm": percentile(body_roll_torque_abs_nm, 1.0),
            "body_roll_limit_contact_fraction": float(np.mean(local_array > limit_deg)),
            "body_roll_limit_penetration_max_deg": max(
                0.0, percentile(local_abs_deg, 1.0) - limit_deg
            ),
        }
    )
    # Preserve the accepted q2 bounded-stop gates as an independent feasibility block.
    q2.add_bounded_gates(row, q2.baseline_by_kind()[kind])
    row["q2_all_bounded_gates"] = row["all_bounded_gates"]
    row["q2_bounded_gates"] = row["bounded_gates"]
    return row


def run_probe(duration_s: float) -> dict[str, object]:
    manifest = build_models()
    before = stage1.robot_sentinels()
    rows = []
    for kind in ("flat", "rough"):
        path = Path(manifest["variants"][f"{kind}_g1_r30"]["model"])
        rows.append(run_with_roll_capture("g1_r30", kind, path, duration_s))
    result = {
        "purpose": "measure the accepted q2 model's previously unreported body-roll motion",
        "duration_s": duration_s,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "body_roll_baseline_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_screen(
    duration_s: float, candidates: list[str], kinds: list[str]
) -> dict[str, object]:
    """Run a targeted screen before committing to the full rough panel."""

    manifest = build_models()
    before = stage1.robot_sentinels()
    rows = []
    for candidate in candidates:
        for kind in kinds:
            model_path = Path(
                manifest["variants"][f"{kind}_{candidate}"]["model"]
            )
            rows.append(
                run_with_roll_capture(candidate, kind, model_path, duration_s)
            )
    result = {
        "purpose": "targeted body-roll causal screen; no selection is made",
        "duration_s": duration_s,
        "candidates": candidates,
        "kinds": kinds,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    tag = "_".join(kinds)
    candidate_tag = f"{candidates[0]}_to_{candidates[-1]}"
    (DATA / f"body_roll_screen_{tag}_{candidate_tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def selection_gates(
    row: dict[str, object], baseline: dict[str, object]
) -> dict[str, bool]:
    return {
        "finite_warning_free": bool(row["finite"]) and not row["warning_counts"],
        "accepted_q2_plant": bool(row["q2_all_bounded_gates"]),
        "forward_displacement_retention": float(row["forward_displacement_m"])
        >= 0.90 * float(baseline["forward_displacement_m"]),
        "path_efficiency_retention": float(row["path_efficiency"])
        >= 0.95 * float(baseline["path_efficiency"]),
        "upright": float(row["minimum_root_upright_cos"]) >= 0.75,
        "spine_tortuosity": float(row["maximum_spine_tortuosity"]) <= 1.30,
        # Template-model observability brackets.  They are engineering gates
        # anchored by the >60% measured reduction from the accepted baseline,
        # not claimed species-identification confidence intervals.
        "local_roll_p95": float(row["body_roll_local_abs_p95_deg"]) <= 3.0,
        "local_roll_max": float(row["body_roll_local_abs_max_deg"]) <= 8.0,
        "cumulative_span_p95": float(row["body_roll_cumulative_span_p95_deg"]) <= 15.0,
        "world_roll_p95": float(row["chassis_world_roll_abs_p95_deg"]) <= 10.0,
        "head_roll_p95": float(row["head_world_roll_abs_p95_deg"]) <= 7.0,
        "active_speed": float(row["peak_active_joint_speed_rad_s"]) <= 100.0,
        "contacting_feet": 8.0 <= float(row["mean_contacting_feet"]) <= 26.0,
        "body_roll_torque": float(row["body_roll_torque_abs_max_nm"]) <= 4.0e-5,
        "body_roll_stop_penetration": float(row["body_roll_limit_penetration_max_deg"])
        <= 0.5,
        "body_roll_stop_contact_fraction": float(row["body_roll_limit_contact_fraction"])
        <= 0.01,
    }


def finalize_selection() -> dict[str, object]:
    """Consolidate the predeclared final 4x/5x bracket without rerunning stale screens."""

    selected = "lw3_php45_c3_g4_r8_q2g28p5"
    upper = "lw3_php45_c3_g5_r8_q2g28p5"
    baseline_payload = json.loads(
        (DATA / "body_roll_baseline_probe.json").read_text(encoding="utf-8")
    )
    flat_payload = json.loads(
        (
            DATA
            / "body_roll_screen_flat_lw3_php45_c2_g4_r8_q2g28p5_to_lw3_php45_c3_g5_r8_q2g28p5.json"
        ).read_text(encoding="utf-8")
    )
    rough_payload = json.loads(
        (
            DATA
            / "body_roll_screen_rough_lw3_php45_c3_g4_r8_q2g28p5_to_lw3_php45_c3_g5_r8_q2g28p5.json"
        ).read_text(encoding="utf-8")
    )
    baseline = {row["kind"]: row for row in baseline_payload["rows"]}
    rows = list(baseline_payload["rows"])
    rows.extend(
        row
        for payload in (flat_payload, rough_payload)
        for row in payload["rows"]
        if row["body_roll_candidate"] in {selected, upper}
    )
    for row in rows:
        kind = str(row["kind"])
        row["forward_displacement_retention"] = float(row["forward_displacement_m"]) / max(
            float(baseline[kind]["forward_displacement_m"]), 1e-12
        )
        row["path_efficiency_retention"] = float(row["path_efficiency"]) / max(
            float(baseline[kind]["path_efficiency"]), 1e-12
        )
        row["body_roll_gates"] = selection_gates(row, baseline[kind])
        row["all_body_roll_gates"] = bool(all(row["body_roll_gates"].values()))

    by_candidate = {}
    for candidate in (selected, upper):
        pair = [row for row in rows if row["body_roll_candidate"] == candidate]
        by_candidate[candidate] = {
            **CANDIDATES[candidate],
            "all_conditions_pass": bool(all(row["all_body_roll_gates"] for row in pair)),
            "minimum_displacement_retention": min(
                float(row["forward_displacement_retention"]) for row in pair
            ),
            "minimum_path_efficiency_retention": min(
                float(row["path_efficiency_retention"]) for row in pair
            ),
            "maximum_local_roll_p95_deg": max(
                float(row["body_roll_local_abs_p95_deg"]) for row in pair
            ),
            "maximum_cumulative_span_p95_deg": max(
                float(row["body_roll_cumulative_span_p95_deg"]) for row in pair
            ),
            "maximum_world_roll_p95_deg": max(
                float(row["chassis_world_roll_abs_p95_deg"]) for row in pair
            ),
            "maximum_head_roll_p95_deg": max(
                float(row["head_world_roll_abs_p95_deg"]) for row in pair
            ),
            "maximum_body_roll_torque_nm": max(
                float(row["body_roll_torque_abs_max_nm"]) for row in pair
            ),
        }
    if not by_candidate[selected]["all_conditions_pass"]:
        raise RuntimeError("Predeclared 4x final candidate does not pass final gates")
    if not by_candidate[upper]["all_conditions_pass"]:
        raise RuntimeError("5x upper bracket unexpectedly fails final gates")

    before = stage1.robot_sentinels()
    result = {
        "purpose": "final body-roll feasibility panel; no scalar reward",
        "duration_s": 6.0,
        "q2_candidate_frozen": Q2_CANDIDATE,
        "baseline_candidate": "g1_r30",
        "rows": rows,
        "by_candidate": by_candidate,
        "selected_candidate": selected,
        "upper_bracket_candidate": upper,
        "selection_status": (
            "4x is the lowest tested gain passing flat/rough roll, q2, contact, posture "
            "and performance-retention gates; 5x adds marginal roll reduction"
        ),
        "selection_rule": "all gates first; choose the lower sufficient body-roll gain",
        "gate_semantics": {
            "local_roll_p95_deg": 3.0,
            "local_roll_max_deg": 8.0,
            "cumulative_span_p95_deg": 15.0,
            "world_roll_p95_deg": 10.0,
            "head_roll_p95_deg": 7.0,
            "peak_active_joint_speed_rad_s": 100.0,
            "mean_contacting_feet_range": [8.0, 26.0],
            "body_roll_torque_max_nm": 4.0e-5,
            "engineering_template_brackets_not_species_confidence_intervals": True,
        },
        "q2_guard_semantics": {
            "solver_contact_start_deg": 28.5,
            "nominal_anatomical_envelope_deg": 30.0,
            "interior_stiffness_nm_rad": 2.0e-5,
            "interior_damping_nms_rad": 1.8e-7,
            "interpretation": "explicit early guard for rare re-entry impulse; not hidden stiffness",
        },
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "body_roll_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_sweep(duration_s: float) -> dict[str, object]:
    manifest = build_models()
    before = stage1.robot_sentinels()
    rows = []
    for candidate in CANDIDATES:
        cache = DATA / f"candidate_{candidate}.json"
        cached_rows = None
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if math.isclose(float(payload.get("duration_s", -1)), duration_s) and len(
                payload.get("rows", [])
            ) == 2:
                cached_rows = payload["rows"]
        if cached_rows is None:
            candidate_rows = []
            candidate_before = stage1.robot_sentinels()
            for kind in ("flat", "rough"):
                path = Path(manifest["variants"][f"{kind}_{candidate}"]["model"])
                candidate_rows.append(
                    run_with_roll_capture(candidate, kind, path, duration_s)
                )
            payload = {
                "candidate": candidate,
                "duration_s": duration_s,
                "rows": candidate_rows,
                "robot_sentinels_before": candidate_before,
                "robot_sentinels_after": stage1.robot_sentinels(),
            }
            payload["robot_sentinels_unchanged"] = (
                payload["robot_sentinels_before"] == payload["robot_sentinels_after"]
            )
            cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.extend(candidate_rows)
        else:
            rows.extend(cached_rows)

    baseline = {
        row["kind"]: row for row in rows if row["body_roll_candidate"] == "g1_r30"
    }
    by_candidate = {}
    for row in rows:
        kind = str(row["kind"])
        row["forward_displacement_retention"] = float(row["forward_displacement_m"]) / max(
            float(baseline[kind]["forward_displacement_m"]), 1e-12
        )
        row["path_efficiency_retention"] = float(row["path_efficiency"]) / max(
            float(baseline[kind]["path_efficiency"]), 1e-12
        )
        row["body_roll_gates"] = selection_gates(row, baseline[kind])
        row["all_body_roll_gates"] = bool(all(row["body_roll_gates"].values()))

    for candidate in CANDIDATES:
        pair = [row for row in rows if row["body_roll_candidate"] == candidate]
        by_candidate[candidate] = {
            **CANDIDATES[candidate],
            "all_conditions_pass": bool(all(row["all_body_roll_gates"] for row in pair)),
            "minimum_displacement_retention": min(
                float(row["forward_displacement_retention"]) for row in pair
            ),
            "minimum_path_efficiency_retention": min(
                float(row["path_efficiency_retention"]) for row in pair
            ),
            "maximum_local_roll_p95_deg": max(
                float(row["body_roll_local_abs_p95_deg"]) for row in pair
            ),
            "maximum_local_roll_deg": max(
                float(row["body_roll_local_abs_max_deg"]) for row in pair
            ),
            "maximum_cumulative_span_p95_deg": max(
                float(row["body_roll_cumulative_span_p95_deg"]) for row in pair
            ),
            "maximum_world_roll_p95_deg": max(
                float(row["chassis_world_roll_abs_p95_deg"]) for row in pair
            ),
            "maximum_head_roll_p95_deg": max(
                float(row["head_world_roll_abs_p95_deg"]) for row in pair
            ),
            "maximum_body_roll_torque_nm": max(
                float(row["body_roll_torque_abs_max_nm"]) for row in pair
            ),
        }

    passing = [name for name, value in by_candidate.items() if value["all_conditions_pass"]]
    if passing:
        # Minimal-intervention hierarchy: lowest active gain first; for a tie,
        # prefer the wider anatomical range so selection cannot be a hard-stop artifact.
        selected = min(
            passing,
            key=lambda name: (
                float(CANDIDATES[name]["kp_multiplier"]),
                -float(CANDIDATES[name]["roll_limit_deg"]),
            ),
        )
        status = "passes body-roll, q2, posture, contact and performance-retention gates"
    else:
        selected = min(
            CANDIDATES,
            key=lambda name: -sum(
                sum(int(value) for value in row["body_roll_gates"].values())
                for row in rows
                if row["body_roll_candidate"] == name
            ),
        )
        status = "diagnostic fallback; no candidate passes every body-roll gate"

    result = {
        "purpose": "body-roll landscape; feasibility vector only, no scalar reward",
        "duration_s": duration_s,
        "q2_candidate_frozen": Q2_CANDIDATE,
        "body_amplitude_scale_frozen": BODY_SCALE,
        "hold_first_body_yaw_frozen": HOLD_FIRST_BODY_YAW,
        "baseline_candidate": "g1_r30",
        "rows": rows,
        "by_candidate": by_candidate,
        "selected_candidate": selected,
        "selection_status": status,
        "selection_rule": "all gates first; then lowest gain multiplier; then widest range",
        "gate_semantics": {
            "local_roll_p95_deg": 3.0,
            "local_roll_max_deg": 8.0,
            "cumulative_span_p95_deg": 15.0,
            "world_roll_p95_deg": 5.0,
            "head_roll_p95_deg": 3.0,
            "engineering_template_brackets_not_species_confidence_intervals": True,
        },
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "body_roll_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def render_demos(duration_s: float) -> dict[str, object]:
    sweep_path = DATA / "body_roll_sweep.json"
    if not sweep_path.exists():
        run_sweep(duration_s)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    selected = str(sweep["selected_candidate"])
    manifest = json.loads(
        (DATA / "body_roll_model_manifest.json").read_text(encoding="utf-8")
    )
    before = stage1.robot_sentinels()
    cases = [
        ("flat", "side", 90.0, -12.0),
        ("flat", "oblique", 125.0, -35.0),
        ("flat", "top", 90.0, -89.0),
        ("rough", "side", 90.0, -12.0),
        ("rough", "oblique", 125.0, -35.0),
        ("rough", "top", 90.0, -89.0),
    ]
    rows = []
    video_paths = []
    for kind, view, azimuth, elevation in cases:
        model_path = Path(manifest["variants"][f"{kind}_{selected}"]["model"])
        video = VIDEOS / f"bodyroll_{selected}_{kind}_{view}.mp4"
        rows.append(
            run_with_roll_capture(
                selected,
                kind,
                model_path,
                duration_s,
                video=video,
                camera_azimuth=azimuth,
                camera_elevation=elevation,
            )
        )
        video_paths.append(video)

    frames = []
    for video in video_paths:
        reader = imageio.get_reader(str(video))
        frames.append(Image.fromarray(reader.get_data(150)).resize((480, 270)))
        reader.close()
    sheet = Image.new("RGB", (1440, 580), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = (index % 3) * 480
        y = (index // 3) * 290
        sheet.paste(frame, (x, y))
        draw.text((x + 6, y + 272), video_paths[index].name, fill="black")
    sheet_path = FIGURES / "body_roll_demo_contact_sheet.png"
    sheet.save(sheet_path)

    baseline = {
        row["kind"]: row
        for row in sweep["rows"]
        if row["body_roll_candidate"] == "g1_r30"
    }
    for row in rows:
        kind = str(row["kind"])
        row["body_roll_gates"] = selection_gates(row, baseline[kind])
        row["all_body_roll_gates"] = bool(all(row["body_roll_gates"].values()))

    result = {
        "selected_candidate": selected,
        "selection_status": sweep["selection_status"],
        "duration_s": duration_s,
        "rows": rows,
        "all_demo_rows_pass": bool(all(row["all_body_roll_gates"] for row in rows)),
        "contact_sheet": str(sheet_path),
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "demo_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compact(result: dict[str, object]) -> dict[str, object]:
    output = dict(result)
    if "rows" in output:
        output["row_count"] = len(output.pop("rows"))
    for key in ("robot_sentinels_before", "robot_sentinels_after"):
        output.pop(key, None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("probe", "screen", "finalize", "sweep", "demo", "all")
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--candidates", nargs="*", choices=tuple(CANDIDATES))
    parser.add_argument("--kinds", nargs="*", choices=("flat", "rough"))
    args = parser.parse_args()
    ensure_directories()
    output = {}
    if args.command == "probe":
        output["probe"] = run_probe(args.duration)
    if args.command == "screen":
        output["screen"] = run_screen(
            args.duration,
            list(args.candidates or CANDIDATES),
            list(args.kinds or ["flat"]),
        )
    if args.command == "finalize":
        output["finalize"] = finalize_selection()
    if args.command in {"sweep", "all"}:
        output["sweep"] = run_sweep(args.duration)
    if args.command in {"demo", "all"}:
        output["demo"] = render_demos(args.duration)
    print(json.dumps({key: compact(value) for key, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
