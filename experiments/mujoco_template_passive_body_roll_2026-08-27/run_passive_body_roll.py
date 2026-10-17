"""Coupled-wave passive body-roll study for the Centipede Model.

The body and leg spatial wave numbers are a frozen coordination constraint:
flat uses 1.5/1.5 and rough uses 3.0/3.0, with no relative phase override.
Body-roll active actuators are disabled.  Only explicit XML spring/damper
coefficients at the 20 body-roll joints are varied.  The accepted q2 plant,
mass, inertia and armature=0 model are otherwise unchanged.
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
ACTIVE_ROOT = EXPERIMENTS / "mujoco_template_body_roll_2026-08-27"
ACTIVE_SCRIPT = ACTIVE_ROOT / "run_body_roll.py"

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"

FLAT_WAVE = 1.5
ROUGH_WAVE = 3.0
RELATIVE_LEG_PHASE_DEG = 0.0
ROLL_LIMIT_DEG = 30.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


active = load_module(ACTIVE_SCRIPT, "centipede_passive_roll_active_source")
stage1 = active.stage1

BASE_ROLL_K = float(stage1.selected_impedance()["body_roll"][0])
BASE_ROLL_C = float(stage1.selected_impedance()["body_roll"][1])

PASSIVE_CANDIDATES = {
    name: {
        "stiffness_multiplier": multiplier,
        "stiffness_nm_rad": BASE_ROLL_K * multiplier,
        "damping_nms_rad": BASE_ROLL_C * math.sqrt(multiplier),
    }
    for name, multiplier in (
        ("passive_k1", 1.0),
        ("passive_k2", 2.0),
        ("passive_k3", 3.0),
        ("passive_k4", 4.0),
        ("passive_k6", 6.0),
    )
}


def ensure_directories() -> None:
    for path in (MODELS, DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def register_active_runner_candidates() -> None:
    for name in PASSIVE_CANDIDATES:
        active.CANDIDATES[name] = {
            "kp_multiplier": 1.0,
            "roll_limit_deg": ROLL_LIMIT_DEG,
            "flat_leg_wave": FLAT_WAVE,
            "flat_leg_phase_shift_deg": RELATIVE_LEG_PHASE_DEG,
            "pg_gain": 0.0,
        }


def build_models() -> dict[str, object]:
    ensure_directories()
    register_active_runner_candidates()
    sources = active.source_models()
    before = stage1.robot_sentinels()
    variants: dict[str, dict[str, object]] = {}
    for candidate, spec in PASSIVE_CANDIDATES.items():
        for kind, source in sources.items():
            tree = ET.parse(source)
            root = tree.getroot()
            count = 0
            for joint in root.iter("joint"):
                name = joint.get("name", "")
                if not name.startswith("joint_roll_body_"):
                    continue
                joint.set("limited", "true")
                limit = math.radians(ROLL_LIMIT_DEG)
                joint.set("range", f"{-limit:.12g} {limit:.12g}")
                joint.set("stiffness", f"{spec['stiffness_nm_rad']:.15g}")
                joint.set("damping", f"{spec['damping_nms_rad']:.15g}")
                joint.set("springref", "0")
                joint.set("armature", "0")
                count += 1
            if count != 20:
                raise RuntimeError(f"Expected 20 body-roll joints in {source}, found {count}")
            output = MODELS / f"centipede_{kind}_{candidate}_coupled_wave.xml"
            tree.write(output, encoding="utf-8", xml_declaration=True)
            variants[f"{kind}_{candidate}"] = {
                "kind": kind,
                "candidate": candidate,
                "model": str(output),
                "model_sha256": stage1.sha256(output),
                "source_model": str(source),
                "source_model_sha256": stage1.sha256(source),
                "body_roll_joint_count": count,
                "body_roll_passive_stiffness_nm_rad": spec["stiffness_nm_rad"],
                "body_roll_passive_damping_nms_rad": spec["damping_nms_rad"],
                "body_roll_active_actuator_enabled": False,
                "body_roll_limit_deg": ROLL_LIMIT_DEG,
                "flat_body_wave": FLAT_WAVE,
                "flat_leg_wave": FLAT_WAVE,
                "rough_body_wave": ROUGH_WAVE,
                "rough_leg_wave": ROUGH_WAVE,
                "relative_leg_phase_deg": RELATIVE_LEG_PHASE_DEG,
                "q2_candidate_frozen": active.Q2_CANDIDATE,
                "q2_limit_deg": active.Q2_LIMIT_DEG,
            }
    result = {
        "purpose": "coupled body/leg wave with explicit passive body-roll springs",
        "coordination_rule": {
            "flat_body_wave": FLAT_WAVE,
            "flat_leg_wave": FLAT_WAVE,
            "rough_body_wave": ROUGH_WAVE,
            "rough_leg_wave": ROUGH_WAVE,
            "relative_leg_phase_deg": RELATIVE_LEG_PHASE_DEG,
            "frozen": True,
        },
        "body_roll_active_actuator_enabled": False,
        "variants": variants,
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "passive_body_roll_model_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_case(
    candidate: str,
    kind: str,
    model_path: Path,
    duration_s: float,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -12.0,
    knee_multiplier: float = 1.0,
) -> dict[str, object]:
    register_active_runner_candidates()
    spec = PASSIVE_CANDIDATES[candidate]
    passive_torque_abs: list[float] = []

    original_configure = stage1.configure_native_implicit_pd
    original_posture = stage1.base.posture_metrics
    original_impedance_for = active.impedance_for

    def impedance_with_knee_multiplier(name):
        impedance = original_impedance_for(name)
        kp, kv = impedance["knee"]
        impedance["knee"] = (
            kp * knee_multiplier,
            kv * math.sqrt(knee_multiplier),
        )
        return impedance

    def configure_passive_roll(model, index, q2_positions=None, q2_mode="active_pd"):
        roll_positions = np.asarray(index.roll_positions, dtype=int)
        roll_joint_ids = np.asarray(index.active_joint_ids, dtype=int)[roll_positions]
        roll_dof_ids = np.asarray(index.dof_addresses, dtype=int)[roll_positions]
        saved_k = model.jnt_stiffness[roll_joint_ids].copy()
        saved_c = model.dof_damping[roll_dof_ids].copy()
        model.jnt_stiffness[roll_joint_ids] = 0.0
        model.dof_damping[roll_dof_ids] = 0.0
        original_configure(
            model, index, q2_positions=q2_positions, q2_mode=q2_mode
        )
        model.jnt_stiffness[roll_joint_ids] = saved_k
        model.dof_damping[roll_dof_ids] = saved_c
        roll_actuators = np.asarray(index.active_actuator_ids, dtype=int)[roll_positions]
        model.actuator_gainprm[roll_actuators] = 0.0
        model.actuator_biasprm[roll_actuators] = 0.0

    def posture_with_passive_torque(data, index):
        result = original_posture(data, index)
        positions = np.asarray(index.roll_positions, dtype=int)
        joint_ids = np.asarray(index.active_joint_ids, dtype=int)[positions]
        qpos_ids = np.asarray(index.qpos_addresses, dtype=int)[positions]
        dof_ids = np.asarray(index.dof_addresses, dtype=int)[positions]
        q = np.asarray(data.qpos[qpos_ids], dtype=float)
        qref = np.asarray(data.model.qpos_spring[qpos_ids], dtype=float)
        qvel = np.asarray(data.qvel[dof_ids], dtype=float)
        torque = (
            -np.asarray(data.model.jnt_stiffness[joint_ids], dtype=float) * (q - qref)
            - np.asarray(data.model.dof_damping[dof_ids], dtype=float) * qvel
        )
        passive_torque_abs.extend(np.abs(torque).tolist())
        return result

    stage1.configure_native_implicit_pd = configure_passive_roll
    stage1.base.posture_metrics = posture_with_passive_torque
    active.impedance_for = impedance_with_knee_multiplier
    try:
        row = active.run_with_roll_capture(
            candidate,
            kind,
            model_path,
            duration_s,
            video=video,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
        )
    finally:
        stage1.configure_native_implicit_pd = original_configure
        stage1.base.posture_metrics = original_posture
        active.impedance_for = original_impedance_for

    active_p95 = float(row["body_roll_torque_abs_p95_nm"])
    active_max = float(row["body_roll_torque_abs_max_nm"])
    passive_array = np.asarray(passive_torque_abs, dtype=float)
    passive_p95 = float(np.quantile(passive_array, 0.95))
    passive_max = float(np.max(passive_array))
    expected_wave = FLAT_WAVE if kind == "flat" else ROUGH_WAVE
    row.update(
        {
            "experiment_class": "passive_body_roll_coupled_wave",
            "body_wave": expected_wave,
            "leg_wave": expected_wave,
            "wave_numbers_coupled": True,
            "relative_leg_phase_deg": RELATIVE_LEG_PHASE_DEG,
            "flat_leg_spatial_wave": FLAT_WAVE,
            "flat_leg_phase_shift_deg": RELATIVE_LEG_PHASE_DEG,
            "projected_gravity_feedback_scope": "disabled",
            "projected_gravity_head_gain": 0.0,
            "projected_gravity_segment_gain": 0.0,
            "body_roll_active_actuator_enabled": False,
            "body_roll_active_kp_nm_rad": 0.0,
            "body_roll_active_kv_nms_rad": 0.0,
            "body_roll_active_torque_abs_p95_nm": active_p95,
            "body_roll_active_torque_abs_max_nm": active_max,
            "body_roll_passive_stiffness_multiplier": spec["stiffness_multiplier"],
            "body_roll_passive_stiffness_nm_rad": spec["stiffness_nm_rad"],
            "body_roll_passive_damping_nms_rad": spec["damping_nms_rad"],
            "body_roll_passive_torque_abs_p95_nm": passive_p95,
            "body_roll_passive_torque_abs_max_nm": passive_max,
            "body_roll_torque_abs_p95_nm": passive_p95,
            "body_roll_torque_abs_max_nm": passive_max,
            "q2_guard_limit_deg": active.Q2_LIMIT_DEG,
            "knee_kp_multiplier": knee_multiplier,
            "knee_kp_nm_rad": stage1.selected_impedance()["knee"][0]
            * knee_multiplier,
            "knee_kv_nms_rad": stage1.selected_impedance()["knee"][1]
            * math.sqrt(knee_multiplier),
        }
    )
    if active_p95 > 1.0e-15 or active_max > 1.0e-15:
        raise RuntimeError("Body-roll active actuator was not fully disabled")
    return row


def run_flat_knee_support_screen(duration_s: float) -> dict[str, object]:
    manifest_path = DATA / "passive_body_roll_model_manifest.json"
    if not manifest_path.exists():
        build_models()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = Path(manifest["variants"]["flat_passive_k4"]["model"])
    reference = reference_rows()["flat"]
    before = stage1.robot_sentinels()
    rows = []
    for multiplier in (1.0, 1.5, 2.0, 3.0, 4.0):
        row = run_case(
            "passive_k4",
            "flat",
            model,
            duration_s,
            knee_multiplier=multiplier,
        )
        row["knee_support_candidate"] = f"knee_k{multiplier:g}"
        row["passive_body_roll_gates"] = gates(row, reference)
        row["all_passive_body_roll_gates"] = bool(
            all(row["passive_body_roll_gates"].values())
        )
        rows.append(row)
    result = {
        "purpose": "flat-only causal screen: can restored knee support make passive body-roll stiffness stabilize the global roll mode?",
        "duration_s": duration_s,
        "body_roll_candidate_frozen": "passive_k4",
        "wave_numbers_coupled": True,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "flat_knee_support_screen.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def reference_rows() -> dict[str, dict[str, object]]:
    raw = json.loads(
        (
            active.Q2_ROOT
            / "outputs"
            / "data"
            / "candidate_bounded_k20_c18.json"
        ).read_text(encoding="utf-8")
    )
    return {str(row["kind"]): row for row in raw["rows"]}


def gates(row: dict[str, object], reference: dict[str, object]) -> dict[str, bool]:
    kind = str(row["kind"])
    q3_p95_limit = float(reference["q3_deviation_p95_deg"]) * 1.15
    q3_max_limit = float(reference["q3_deviation_max_deg"]) * 1.30
    return {
        "finite_warning_free": bool(row["finite"]) and not bool(row["warning_counts"]),
        "wave_numbers_coupled": bool(row["wave_numbers_coupled"]),
        "relative_leg_phase_zero": abs(float(row["relative_leg_phase_deg"])) < 1.0e-12,
        "active_roll_actuator_zero": float(row["body_roll_active_torque_abs_max_nm"]) <= 1.0e-15,
        "accepted_q2_plant": bool(row["q2_all_bounded_gates"]),
        "forward_displacement_retention": float(row["forward_displacement_m"]) / float(reference["forward_displacement_m"]) >= 0.90,
        "path_efficiency_retention": float(row["path_efficiency"]) / float(reference["path_efficiency"]) >= 0.95,
        "upright": float(row["minimum_root_upright_cos"]) >= 0.75,
        "spine_tortuosity": float(row["maximum_spine_tortuosity"]) <= 1.30,
        "knee_support_p95": float(row["q3_deviation_p95_deg"]) <= q3_p95_limit,
        "knee_support_max": float(row["q3_deviation_max_deg"]) <= q3_max_limit,
        "local_roll_p95": float(row["body_roll_local_abs_p95_deg"]) <= 5.0,
        "local_roll_max": float(row["body_roll_local_abs_max_deg"]) <= 8.0,
        "cumulative_roll_span_p95": float(row["body_roll_cumulative_span_p95_deg"]) <= 22.0,
        "world_roll_p95": float(row["chassis_world_roll_abs_p95_deg"]) <= 14.0,
        "head_roll_p95": float(row["head_world_roll_abs_p95_deg"]) <= 12.0,
        "contacting_feet": 8.0 <= float(row["mean_contacting_feet"]) <= 26.0,
        "passive_torque_bounded": float(row["body_roll_passive_torque_abs_max_nm"]) <= 6.0e-5,
        "body_roll_stop_unused": float(row["body_roll_limit_contact_fraction"]) == 0.0,
    }


def run_sweep(duration_s: float) -> dict[str, object]:
    manifest = build_models()
    refs = reference_rows()
    before = stage1.robot_sentinels()
    rows: list[dict[str, object]] = []
    for candidate in PASSIVE_CANDIDATES:
        for kind in ("flat", "rough"):
            model = Path(manifest["variants"][f"{kind}_{candidate}"]["model"])
            row = run_case(candidate, kind, model, duration_s)
            row["passive_body_roll_gates"] = gates(row, refs[kind])
            row["all_passive_body_roll_gates"] = bool(
                all(row["passive_body_roll_gates"].values())
            )
            rows.append(row)

    passing = []
    for candidate in PASSIVE_CANDIDATES:
        candidate_rows = [r for r in rows if r["body_roll_candidate"] == candidate]
        if len(candidate_rows) == 2 and all(
            bool(r["all_passive_body_roll_gates"]) for r in candidate_rows
        ):
            passing.append(candidate)
    selected = passing[0] if passing else None
    result = {
        "purpose": "passive body-roll stiffness brackets with body/leg waves locked equal",
        "duration_s": duration_s,
        "selection_rule": "lowest passive stiffness passing every flat/rough feasibility gate; no scalar reward",
        "selected_candidate": selected,
        "passing_candidates": passing,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "passive_body_roll_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def render_demos(
    duration_s: float, diagnostic_candidate: str | None = None
) -> dict[str, object]:
    sweep_path = DATA / "passive_body_roll_sweep.json"
    if not sweep_path.exists():
        run_sweep(duration_s)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    selected = diagnostic_candidate or sweep.get("selected_candidate")
    if not selected:
        raise RuntimeError("No passive candidate passes all gates; no final Demo may be labeled selected")
    if selected not in PASSIVE_CANDIDATES:
        raise ValueError(f"Unknown passive candidate {selected}")
    diagnostic_only = diagnostic_candidate is not None
    manifest = json.loads(
        (DATA / "passive_body_roll_model_manifest.json").read_text(encoding="utf-8")
    )
    refs = reference_rows()
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
    paths = []
    for kind, view, azimuth, elevation in cases:
        model = Path(manifest["variants"][f"{kind}_{selected}"]["model"])
        prefix = "diagnostic_passiveroll" if diagnostic_only else "passiveroll"
        video = VIDEOS / f"{prefix}_{selected}_{kind}_{view}.mp4"
        row = run_case(
            selected,
            kind,
            model,
            duration_s,
            video=video,
            camera_azimuth=azimuth,
            camera_elevation=elevation,
        )
        row["passive_body_roll_gates"] = gates(row, refs[kind])
        row["all_passive_body_roll_gates"] = bool(
            all(row["passive_body_roll_gates"].values())
        )
        rows.append(row)
        paths.append(video)

    frames = []
    for video in paths:
        reader = imageio.get_reader(str(video))
        frames.append(Image.fromarray(reader.get_data(150)).resize((480, 270)))
        reader.close()
    sheet = Image.new("RGB", (1440, 580), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = (index % 3) * 480
        y = (index // 3) * 290
        sheet.paste(frame, (x, y))
        draw.text((x + 6, y + 272), paths[index].name, fill="black")
    sheet_name = (
        "diagnostic_passive_body_roll_demo_contact_sheet.png"
        if diagnostic_only
        else "passive_body_roll_demo_contact_sheet.png"
    )
    sheet_path = FIGURES / sheet_name
    sheet.save(sheet_path)

    result = {
        "selected_candidate": selected,
        "selection_status": (
            "diagnostic_not_selected" if diagnostic_only else "selected"
        ),
        "duration_s": duration_s,
        "rows": rows,
        "all_demo_rows_pass": bool(
            all(row["all_passive_body_roll_gates"] for row in rows)
        ),
        "contact_sheet": str(sheet_path),
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    audit_name = "diagnostic_demo_audit.json" if diagnostic_only else "demo_audit.json"
    (DATA / audit_name).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
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
        "command",
        choices=("build", "sweep", "knee-screen", "demo", "diagnostic-demo", "all"),
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--candidate", choices=tuple(PASSIVE_CANDIDATES), default="passive_k4"
    )
    args = parser.parse_args()
    ensure_directories()
    result: dict[str, object] = {}
    if args.command == "build":
        result["build"] = build_models()
    if args.command in {"sweep", "all"}:
        result["sweep"] = run_sweep(args.duration)
    if args.command == "knee-screen":
        result["knee_screen"] = run_flat_knee_support_screen(args.duration)
    if args.command in {"demo", "all"}:
        result["demo"] = render_demos(args.duration)
    if args.command == "diagnostic-demo":
        result["diagnostic_demo"] = render_demos(
            args.duration, diagnostic_candidate=args.candidate
        )
    print(json.dumps({key: compact(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
