"""Model-only q2 compliance sweep on the accepted Stage-1 controller.

This experiment is strictly isolated from Centipede Robot Step 18/19.  It
reuses the Stage-1 geometry, contact calibration and diagnostic definitions,
then varies only the explicit passive q2 stiffness/damping pair.  Selection is
feasibility-first and retains a metric vector; no scalar reward is computed.
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
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
STAGE1_ROOT = EXPERIMENTS / "mujoco_template_fidelity_stage1_2026-08-26"
STAGE1_SCRIPT = STAGE1_ROOT / "run_stage1.py"
STAGE1_BASELINE = STAGE1_ROOT / "outputs" / "data" / "head_first_body_hold_probe.json"

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"
TERRAINS = OUTPUTS / "terrains"
RIGS = OUTPUTS / "single_foot_rigs"

BODY_SCALE = 0.55
HOLD_FIRST_BODY_YAW = True
PROFILE_NAME = "stiff_dt0p25ms"
Q2_LIMIT_DEG = 30.0
ROUGH_WAVE = 3.0

Q2_CANDIDATES = {
    # Stiffness [N m/rad], damping [N m s/rad].  The accepted Stage-1
    # reference is k30/c18; it is read from the baseline JSON and is not
    # rerun as an optimization candidate.
    "soft_k20_c18": (2.0e-5, 1.8e-7),
    "bounded_k20_c18": (2.0e-5, 1.8e-7),
    "boundary_k25_c18": (2.5e-5, 1.8e-7),
    "visco_k22p5_c27": (2.25e-5, 2.7e-7),
    "soft_k15_c18": (1.5e-5, 1.8e-7),
    "bounded_k15_c18": (1.5e-5, 1.8e-7),
    "soft_k10_c18": (1.0e-5, 1.8e-7),
    "visco_k10_c27": (1.0e-5, 2.7e-7),
}

TIGHT_STOP_CANDIDATES = {"bounded_k20_c18", "bounded_k15_c18"}
TIGHT_STOP_SOLREF = "0.0005 1.5"
TIGHT_STOP_SOLIMP = "0.999 0.9999 0.00001 0.5 2"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage1 = load_module(STAGE1_SCRIPT, "centipede_q2_compliance_stage1")


def configure_isolated_stage1() -> None:
    """Redirect every Stage-1 output global into this experiment."""

    stage1.ROOT = ROOT
    stage1.MODELS = MODELS
    stage1.OUTPUTS = OUTPUTS
    stage1.DATA = DATA
    stage1.FIGURES = FIGURES
    stage1.VIDEOS = VIDEOS
    stage1.TERRAINS = TERRAINS
    stage1.RIGS = RIGS
    stage1.PASSIVE_Q2 = dict(Q2_CANDIDATES)
    stage1.PASSIVE_SWEEP_PATH = DATA / "q2_compliance_sweep.json"
    stage1.ensure_directories()


def baseline_by_kind() -> dict[str, dict[str, object]]:
    raw = json.loads(STAGE1_BASELINE.read_text(encoding="utf-8"))
    if not raw.get("all_rows_pass", False):
        raise RuntimeError("Stage-1 reference is not an accepted baseline")
    rows = {str(row["kind"]): row for row in raw["rows"]}
    if set(rows) != {"flat", "rough"}:
        raise RuntimeError("Stage-1 baseline lacks flat/rough rows")
    return rows


def performance_gates(
    row: dict[str, object], baseline: dict[str, object]
) -> dict[str, bool]:
    gates = stage1.diagnostic_gates(row)
    displacement_retention = float(row["forward_displacement_m"]) / max(
        float(baseline["forward_displacement_m"]), 1e-12
    )
    path_efficiency_retention = float(row["path_efficiency"]) / max(
        float(baseline["path_efficiency"]), 1e-12
    )
    gates["forward_displacement_retention"] = displacement_retention >= 0.90
    gates["path_efficiency_retention"] = path_efficiency_retention >= 0.95
    return gates


def decorate_row(
    row: dict[str, object], kind: str, baseline: dict[str, object]
) -> dict[str, object]:
    row["kind"] = kind
    candidate = str(row["q2_passive_name"])
    row["q2_stop_profile"] = (
        "tight_anatomical_stop" if candidate in TIGHT_STOP_CANDIDATES else "stage1_stop"
    )
    row["baseline_forward_displacement_m"] = float(baseline["forward_displacement_m"])
    row["forward_displacement_retention"] = float(row["forward_displacement_m"]) / max(
        float(baseline["forward_displacement_m"]), 1e-12
    )
    row["baseline_path_efficiency"] = float(baseline["path_efficiency"])
    row["path_efficiency_retention"] = float(row["path_efficiency"]) / max(
        float(baseline["path_efficiency"]), 1e-12
    )
    row["gates"] = performance_gates(row, baseline)
    row["gate_pass_count"] = int(sum(row["gates"].values()))
    row["all_gates"] = bool(all(row["gates"].values()))
    add_bounded_gates(row, baseline)
    return row


def add_bounded_gates(
    row: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    """Add gates appropriate to an explicit anatomical-stop model.

    The strict Stage-1 gates remain in ``gates`` and are never rewritten.
    Bounded candidates instead report the solver penetration past the nominal
    stop and stop-contact occupancy explicitly.
    """

    gates = performance_gates(row, baseline)
    gates.pop("q2_within_stop", None)
    gates.pop("q2_stop_occupancy", None)
    candidate = str(row["q2_passive_name"])
    gates["explicit_tight_stop_model"] = candidate in TIGHT_STOP_CANDIDATES
    gates["q2_stop_penetration_max"] = (
        max(0.0, float(row["q2_abs_max_deg"]) - Q2_LIMIT_DEG) <= 0.5
    )
    gates["q2_stop_contact_fraction"] = (
        float(row["q2_above_nominal_limit_fraction"]) <= 0.01
    )
    row["bounded_stop_penetration_max_deg"] = max(
        0.0, float(row["q2_abs_max_deg"]) - Q2_LIMIT_DEG
    )
    row["bounded_gates"] = gates
    row["all_bounded_gates"] = bool(all(gates.values()))
    return row


def build_models(profile) -> dict[str, object]:
    manifest = stage1.build_passive_all(profile, q2_limit_deg=Q2_LIMIT_DEG)
    for candidate in TIGHT_STOP_CANDIDATES:
        for kind in ("flat", "rough"):
            key = f"{kind}_{candidate}"
            path = Path(manifest["variants"][key]["model"])
            tree = ET.parse(path)
            root = tree.getroot()
            count = 0
            for joint in root.iter("joint"):
                name = joint.get("name", "")
                if name.startswith("joint_leg_") and name.endswith("_2"):
                    joint.set("solreflimit", TIGHT_STOP_SOLREF)
                    joint.set("solimplimit", TIGHT_STOP_SOLIMP)
                    count += 1
            if count != 38:
                raise RuntimeError(f"Expected 38 q2 stops in {path}, found {count}")
            tree.write(path, encoding="utf-8", xml_declaration=True)
            manifest["variants"][key]["q2_stop_profile"] = "tight_anatomical_stop"
            manifest["variants"][key]["q2_stop_solref"] = TIGHT_STOP_SOLREF
            manifest["variants"][key]["q2_stop_solimp"] = TIGHT_STOP_SOLIMP
            manifest["variants"][key]["model_sha256"] = stage1.sha256(path)
    manifest["tight_stop_candidates"] = sorted(TIGHT_STOP_CANDIDATES)
    manifest["tight_stop_solref"] = TIGHT_STOP_SOLREF
    manifest["tight_stop_solimp"] = TIGHT_STOP_SOLIMP
    (DATA / "passive_model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_sweep(profile, duration_s: float) -> dict[str, object]:
    manifest = build_models(profile)
    baseline = baseline_by_kind()
    before = stage1.robot_sentinels()
    rows = []
    for candidate in Q2_CANDIDATES:
        candidate_path = DATA / f"candidate_{candidate}.json"
        cached = None
        if candidate_path.exists():
            candidate_raw = json.loads(candidate_path.read_text(encoding="utf-8"))
            if (
                math.isclose(
                    float(candidate_raw.get("duration_s", -1.0)),
                    duration_s,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and candidate_raw.get("profile") == profile.name
                and candidate_raw.get("candidate") == candidate
                and len(candidate_raw.get("rows", [])) == 2
            ):
                cached = candidate_raw["rows"]
        if cached is not None:
            rows.extend(cached)
            continue

        candidate_rows = []
        candidate_before = stage1.robot_sentinels()
        for kind, wave in (("flat", 1.5), ("rough", ROUGH_WAVE)):
            model_path = Path(manifest["variants"][f"{kind}_{candidate}"]["model"])
            row = stage1.run_diagnostic(
                f"{kind} q2 compliance {candidate}",
                model_path,
                wave,
                Q2_LIMIT_DEG,
                duration_s=duration_s,
                q2_mode="passive",
                q2_passive_name=candidate,
                body_amplitude_scale=BODY_SCALE,
                hold_first_body_yaw=HOLD_FIRST_BODY_YAW,
            )
            candidate_rows.append(decorate_row(row, kind, baseline[kind]))
        candidate_result = {
            "candidate": candidate,
            "profile": profile.name,
            "duration_s": duration_s,
            "stiffness_nm_rad": Q2_CANDIDATES[candidate][0],
            "damping_nms_rad": Q2_CANDIDATES[candidate][1],
            "rows": candidate_rows,
            "robot_sentinels_before": candidate_before,
            "robot_sentinels_after": stage1.robot_sentinels(),
        }
        candidate_result["robot_sentinels_unchanged"] = (
            candidate_result["robot_sentinels_before"]
            == candidate_result["robot_sentinels_after"]
        )
        candidate_path.write_text(json.dumps(candidate_result, indent=2), encoding="utf-8")
        rows.extend(candidate_rows)

    # Cached candidate files predate or may outlive selection-semantics
    # revisions.  Recompute bounded gates from raw metrics every aggregation.
    for row in rows:
        add_bounded_gates(row, baseline[str(row["kind"])])

    by_candidate = {}
    for name, (stiffness, damping) in Q2_CANDIDATES.items():
        pair = [row for row in rows if row["q2_passive_name"] == name]
        by_candidate[name] = {
            "stiffness_nm_rad": stiffness,
            "damping_nms_rad": damping,
            "q2_stop_profile": (
                "tight_anatomical_stop" if name in TIGHT_STOP_CANDIDATES else "stage1_stop"
            ),
            "stiffness_reduction_from_stage1": 1.0 - stiffness / 3.0e-5,
            "all_conditions_pass": bool(all(row["all_gates"] for row in pair)),
            "all_bounded_conditions_pass": bool(
                all(row["all_bounded_gates"] for row in pair)
            ),
            "minimum_displacement_retention": min(
                float(row["forward_displacement_retention"]) for row in pair
            ),
            "minimum_path_efficiency_retention": min(
                float(row["path_efficiency_retention"]) for row in pair
            ),
            "maximum_spine_tortuosity": max(
                float(row["maximum_spine_tortuosity"]) for row in pair
            ),
            "minimum_root_upright_cos": min(
                float(row["minimum_root_upright_cos"]) for row in pair
            ),
            "maximum_q2_p95_deg": max(float(row["q2_abs_p95_deg"]) for row in pair),
            "maximum_q2_deg": max(float(row["q2_abs_max_deg"]) for row in pair),
            "maximum_q2_speed_rad_s": max(
                float(row["q2_speed_max_rad_s"]) for row in pair
            ),
            "maximum_penetration_p95_mm": max(
                float(row["penetration_p95_mm"]) for row in pair
            ),
            "maximum_normal_jump_p95_deg": max(
                float(row["contact_normal_jump_nonzero_p95_deg"]) for row in pair
            ),
        }

    strict_passing = [
        name for name in Q2_CANDIDATES if by_candidate[name]["all_conditions_pass"]
    ]
    bounded_passing = [
        name
        for name in Q2_CANDIDATES
        if by_candidate[name]["all_bounded_conditions_pass"]
    ]
    if strict_passing:
        # Primary feasibility rule: lowest static stiffness.  When stiffness
        # ties, prefer lower damping so the result is genuinely more compliant
        # rather than merely more dissipative.
        selected = min(
            strict_passing,
            key=lambda name: (Q2_CANDIDATES[name][0], Q2_CANDIDATES[name][1]),
        )
        status = "passes all plant/contact/posture and performance-retention gates"
        selection_semantics = "strict_stage1"
    elif bounded_passing:
        selected = min(
            bounded_passing,
            key=lambda name: (Q2_CANDIDATES[name][0], Q2_CANDIDATES[name][1]),
        )
        status = (
            "passes explicit bounded-compliance gates; strict Stage-1 stop-occupancy gate "
            "remains false and is reported separately"
        )
        selection_semantics = "explicit_bounded_stop"
    else:
        selected = min(
            Q2_CANDIDATES,
            key=lambda name: (
                -sum(
                    int(row["gate_pass_count"])
                    for row in rows
                    if row["q2_passive_name"] == name
                ),
                Q2_CANDIDATES[name][0],
            ),
        )
        status = "diagnostic only; no softer q2 candidate passed every gate"
        selection_semantics = "diagnostic_fallback"

    result = {
        "purpose": (
            "q2 compliance landscape with Stage-1 contact, body wave and head hold frozen; "
            "no scalar reward"
        ),
        "profile": profile.name,
        "duration_s": duration_s,
        "body_amplitude_scale": BODY_SCALE,
        "hold_first_body_yaw": HOLD_FIRST_BODY_YAW,
        "stage1_reference_stiffness_nm_rad": 3.0e-5,
        "stage1_reference_damping_nms_rad": 1.8e-7,
        "performance_retention_gates": {
            "forward_displacement": 0.90,
            "path_efficiency": 0.95,
        },
        "rows": rows,
        "by_candidate": by_candidate,
        "selected_candidate": selected,
        "selection_status": status,
        "selection_semantics": selection_semantics,
        "selection_rule": (
            "feasibility first; among fully passing candidates choose lowest stiffness, "
            "then lowest damping"
        ),
        "tight_stop_definition": {
            "candidates": sorted(TIGHT_STOP_CANDIDATES),
            "solref": TIGHT_STOP_SOLREF,
            "solimp": TIGHT_STOP_SOLIMP,
            "interpretation": (
                "soft passive spring inside nominal ROM; explicit high-impedance anatomical stop "
                "only outside +/-30 deg"
            ),
            "bounded_acceptance": {
                "maximum_solver_penetration_deg": 0.5,
                "maximum_stop_contact_fraction": 0.01,
                "strict_stage1_gates_preserved_separately": True,
            },
        },
        "stage1_baseline": str(STAGE1_BASELINE),
        "stage1_baseline_rows": baseline,
        "robot_sentinels_before": before,
        "robot_sentinels_after": stage1.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "q2_compliance_sweep.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def render_demos(profile, duration_s: float) -> dict[str, object]:
    sweep_path = DATA / "q2_compliance_sweep.json"
    if not sweep_path.exists():
        run_sweep(profile, duration_s)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    selected = str(sweep["selected_candidate"])
    manifest = json.loads((DATA / "passive_model_manifest.json").read_text(encoding="utf-8"))
    baseline = baseline_by_kind()
    before = stage1.robot_sentinels()
    paths = {
        "flat": Path(manifest["variants"][f"flat_{selected}"]["model"]),
        "rough": Path(manifest["variants"][f"rough_{selected}"]["model"]),
    }
    cases = [
        ("flat side", "flat", 1.5, 90.0, -12.0, 0.19),
        ("flat oblique", "flat", 1.5, 125.0, -35.0, 0.19),
        ("flat top", "flat", 1.5, 90.0, -89.0, 0.19),
        ("rough side", "rough", ROUGH_WAVE, 90.0, -12.0, 0.19),
        ("rough oblique", "rough", ROUGH_WAVE, 125.0, -35.0, 0.19),
        ("rough top", "rough", ROUGH_WAVE, 90.0, -89.0, 0.19),
    ]
    rows = []
    video_paths = []
    for label, kind, wave, azimuth, elevation, distance in cases:
        view = label.split()[-1]
        video = VIDEOS / f"q2compliance_{selected}_{kind}_{view}.mp4"
        row = stage1.run_diagnostic(
            label,
            paths[kind],
            wave,
            Q2_LIMIT_DEG,
            duration_s=duration_s,
            video=video,
            camera_azimuth=azimuth,
            camera_elevation=elevation,
            camera_distance=distance,
            q2_mode="passive",
            q2_passive_name=selected,
            body_amplitude_scale=BODY_SCALE,
            hold_first_body_yaw=HOLD_FIRST_BODY_YAW,
        )
        rows.append(decorate_row(row, kind, baseline[kind]))
        video_paths.append(video)

    frames = []
    labels = []
    for video in video_paths:
        reader = imageio.get_reader(str(video))
        frames.append(Image.fromarray(reader.get_data(150)).resize((480, 270)))
        labels.append(video.name)
        reader.close()
    sheet = Image.new("RGB", (1440, 580), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = (index % 3) * 480
        y = (index // 3) * 290
        sheet.paste(frame, (x, y))
        draw.text((x + 6, y + 272), labels[index], fill="black")
    sheet_path = FIGURES / "q2_compliance_demo_contact_sheet.png"
    sheet.save(sheet_path)

    result = {
        "selected_candidate": selected,
        "selection_status": sweep["selection_status"],
        "selection_semantics": sweep["selection_semantics"],
        "profile": profile.name,
        "duration_s": duration_s,
        "body_amplitude_scale": BODY_SCALE,
        "hold_first_body_yaw": HOLD_FIRST_BODY_YAW,
        "rows": rows,
        "all_demo_rows_pass": bool(
            all(
                row["all_bounded_gates"]
                if sweep["selection_semantics"] == "explicit_bounded_stop"
                else row["all_gates"]
                for row in rows
            )
        ),
        "all_demo_rows_strict_stage1_pass": bool(
            all(row["all_gates"] for row in rows)
        ),
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
        output["row_count"] = len(output["rows"])
        del output["rows"]
    for key in ("robot_sentinels_before", "robot_sentinels_after", "stage1_baseline_rows"):
        output.pop(key, None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("sweep", "demo", "all"))
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    configure_isolated_stage1()
    profile = stage1.CONTACT_PROFILES[PROFILE_NAME]
    output = {}
    if args.command in {"sweep", "all"}:
        output["sweep"] = run_sweep(profile, args.duration)
    if args.command in {"demo", "all"}:
        output["demo"] = render_demos(profile, args.duration)
    print(json.dumps({key: compact(value) for key, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
