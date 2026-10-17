"""Targeted support sweep for joint_pitch_body_1 in the abstract Centipede Model.

Only the effective implicit-PD impedance of the link1-link2 pitch connection is
changed.  Head geometry, the fixed -29 degree head/link1 topology, all other
body/leg impedances, distal-tip compliance, mass, inertia and armature remain
unchanged.  Centipede Robot Step 18/19 is read only through hash sentinels.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
LOCKED_ROOT = ROOT.parent / "mujoco_template_locked_head_2026-08-27"
sys.path.insert(0, str(LOCKED_ROOT))
import run_head_contact_diagnostic as diagnostic  # noqa: E402
import run_locked_head as locked  # noqa: E402


MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
VIDEOS = OUTPUTS / "videos"
REPORT = ROOT / "FIRST_PITCH_SUPPORT_REPORT.md"

BASE_MODELS = {
    "flat": LOCKED_ROOT / "models" / "centipede_flat_locked_head_m29p0.xml",
    "rough": LOCKED_ROOT / "models" / "centipede_rough_locked_head_m29p0.xml",
}

BASE_KP, BASE_KD = locked.tip.stage1.selected_impedance()["body_pitch"]
CANDIDATES = {
    "baseline": {"kp_scale": 1.0, "kd_scale": 1.0},
    "k0p75_zeta": {"kp_scale": 0.75, "kd_scale": math.sqrt(0.75)},
    "k0p75_d1": {"kp_scale": 0.75, "kd_scale": 1.0},
    "k1p5_zeta": {"kp_scale": 1.5, "kd_scale": math.sqrt(1.5)},
    "k2_zeta": {"kp_scale": 2.0, "kd_scale": math.sqrt(2.0)},
    "k1p5_d1p5": {"kp_scale": 1.5, "kd_scale": 1.5},
}


def ensure_directories() -> None:
    for path in (MODELS, DATA, FIGURES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def candidate_path(candidate: str) -> Path:
    return DATA / f"candidate_{candidate}.json"


def evaluate_candidate(candidate: str, duration_s: float) -> dict[str, object]:
    ensure_directories()
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate}")
    spec = CANDIDATES[candidate]
    before = locked.tip.robot_sentinels()
    rows = []
    for kind in ("flat", "rough"):
        print(f"[first-pitch] START {candidate} {kind}", flush=True)
        row = diagnostic.run_case(
            "locked_m29",
            kind,
            BASE_MODELS[kind],
            duration_s,
            first_pitch_kp_scale=float(spec["kp_scale"]),
            first_pitch_kd_scale=float(spec["kd_scale"]),
        )
        row["candidate"] = candidate
        rows.append(row)
        candidate_path(candidate).write_text(
            json.dumps(
                {
                    "candidate": candidate,
                    "spec": spec,
                    "duration_s": duration_s,
                    "rows": rows,
                    "robot_sentinels_before": before,
                    "robot_sentinels_current": locked.tip.robot_sentinels(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        q95 = row["first_spine_pitch"]["joint_angle_abs_deg"]["p95"]
        contact = 100.0 * row["link1_ground_contact_after_1p5s"]["contact_time_fraction"]
        print(
            f"[first-pitch] DONE  {candidate} {kind} "
            f"q1P95={q95:.2f}deg link1={contact:.2f}% "
            f"disp={1000.0 * row['forward_displacement_m']:.2f}mm",
            flush=True,
        )
    result = {
        "candidate": candidate,
        "spec": spec,
        "effective_kp_nm_per_rad": BASE_KP * float(spec["kp_scale"]),
        "effective_kd_nms_per_rad": BASE_KD * float(spec["kd_scale"]),
        "duration_s": duration_s,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": locked.tip.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    candidate_path(candidate).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def row_by_kind(result: dict[str, object], kind: str) -> dict[str, object]:
    return next(row for row in result["rows"] if row["kind"] == kind)


def finite_or_zero(value: float) -> float:
    return 0.0 if not math.isfinite(float(value)) else float(value)


def gates(
    row: dict[str, object], baseline: dict[str, object]
) -> dict[str, bool]:
    q95 = float(row["first_spine_pitch"]["joint_angle_abs_deg"]["p95"])
    q_speed = float(row["first_spine_pitch"]["speed_abs_rad_s"]["p95"])
    q_speed_base = float(baseline["first_spine_pitch"]["speed_abs_rad_s"]["p95"])
    q_accel = float(row["first_spine_pitch"]["acceleration_abs_rad_s2"]["p95"])
    q_accel_base = float(
        baseline["first_spine_pitch"]["acceleration_abs_rad_s2"]["p95"]
    )
    contact = float(row["link1_ground_contact_after_1p5s"]["contact_time_fraction"])
    contact_base = float(
        baseline["link1_ground_contact_after_1p5s"]["contact_time_fraction"]
    )
    head_contact = float(row["head_ground_contact_after_1p5s"]["contact_time_fraction"])
    head_contact_base = float(
        baseline["head_ground_contact_after_1p5s"]["contact_time_fraction"]
    )
    force = finite_or_zero(
        row["link1_ground_contact_after_1p5s"]["normal_force_mn"]["p95"]
    )
    force_base = finite_or_zero(
        baseline["link1_ground_contact_after_1p5s"]["normal_force_mn"]["p95"]
    )
    q1_torque_max = float(
        row["first_spine_pitch"]["controller_torque_abs_micronewton_m"]["max"]
    )
    return {
        "finite_warning_free": bool(row["finite"]) and not bool(row["warning_counts"]),
        "mass_and_armature_unchanged": int(row["armature_nonzero_count"]) == 0
        and math.isclose(
            float(row["total_mass_kg"]), float(baseline["total_mass_kg"]), abs_tol=1e-15
        ),
        "q1_p95_bounded_at_10deg": q95 <= 10.0,
        "q1_speed_p95_not_over_120pct": q_speed <= 1.20 * q_speed_base,
        "q1_acceleration_p95_not_over_120pct": q_accel <= 1.20 * q_accel_base,
        "link1_contact_not_worse": contact <= contact_base + 1e-12,
        "head_contact_not_worse": head_contact <= head_contact_base + 1e-12,
        "front_contact_materially_improved": contact <= 0.90 * contact_base
        or head_contact <= 0.90 * head_contact_base,
        "link1_force_p95_not_over_115pct": force <= 1.15 * max(force_base, 1e-12),
        "q1_torque_max_at_most_50unm": q1_torque_max <= 50.0,
        "forward_retention_at_least_90pct": float(row["forward_displacement_m"])
        >= 0.90 * float(baseline["forward_displacement_m"]),
        "upright": float(row["minimum_root_upright_cos"]) >= 0.75,
        "world_roll_not_worse_by_1p5deg": float(row["chassis_world_roll_abs_p95_deg"])
        <= float(baseline["chassis_world_roll_abs_p95_deg"]) + 1.5,
        "tip_stop_rare": float(row["tip_roll"]["stop_fraction"]) <= 0.01,
        "foot_penetration_not_worse_by_10pct": float(row["foot_contact_penetration_p95_mm"])
        <= 1.10 * float(baseline["foot_contact_penetration_p95_mm"]),
    }


def annotate_model(source: Path, output: Path, candidate: str, spec: dict[str, float]) -> Path:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", f"centipede_{candidate}_{source.stem}")
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    for name, value in (
        ("first_pitch_kp_nm_per_rad", BASE_KP * float(spec["kp_scale"])),
        ("first_pitch_kd_nms_per_rad", BASE_KD * float(spec["kd_scale"])),
        ("first_pitch_kp_scale", float(spec["kp_scale"])),
        ("first_pitch_kd_scale", float(spec["kd_scale"])),
    ):
        ET.SubElement(custom, "numeric", {"name": name, "data": f"{value:.12g}"})
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def select_candidate() -> dict[str, object]:
    results = {
        name: json.loads(candidate_path(name).read_text(encoding="utf-8"))
        for name in CANDIDATES
        if candidate_path(name).exists()
    }
    missing = [name for name in CANDIDATES if name not in results]
    if missing:
        raise FileNotFoundError(f"Missing candidate results: {missing}")
    baseline = results["baseline"]
    assessed = []
    passing = []
    for candidate, result in results.items():
        candidate_rows = []
        for kind in ("flat", "rough"):
            row = row_by_kind(result, kind)
            if candidate == "baseline":
                row_gates = {"reference_only": True}
                all_gates = False
            else:
                row_gates = gates(row, row_by_kind(baseline, kind))
                all_gates = bool(all(row_gates.values()))
            candidate_rows.append({"kind": kind, "gates": row_gates, "all_gates": all_gates})
        passes = candidate != "baseline" and all(item["all_gates"] for item in candidate_rows)
        if passes:
            passing.append(candidate)
        assessed.append(
            {
                "candidate": candidate,
                "spec": result["spec"],
                "rows": candidate_rows,
                "passes_both_terrains": passes,
            }
        )
    selected = (
        min(
            passing,
            key=lambda name: (
                abs(float(CANDIDATES[name]["kp_scale"]) - 1.0),
                abs(float(CANDIDATES[name]["kd_scale"]) - 1.0),
            ),
        )
        if passing
        else None
    )
    models = {}
    if selected is not None:
        for kind in ("flat", "rough"):
            output = MODELS / f"centipede_{kind}_locked_head_m29_{selected}.xml"
            annotate_model(BASE_MODELS[kind], output, selected, CANDIDATES[selected])
            models[kind] = str(output)
    result = {
        "purpose": "joint_pitch_body_1 support sweep; no scalar reward",
        "selection_rule": "smallest Kp departure from baseline, then smallest Kd departure, among candidates passing every flat/rough front-contact, bounded-q1, force, locomotion, posture, tip and numerical gate",
        "candidates": CANDIDATES,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "assessed": assessed,
        "selected_models": models,
        "robot_sentinels_unchanged": all(
            bool(result["robot_sentinels_unchanged"]) for result in results.values()
        ),
    }
    (DATA / "first_pitch_support_selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def demo_gates(row: dict[str, object], reference: dict[str, object]) -> dict[str, bool]:
    return {
        "finite_warning_free": bool(row["finite"]) and not bool(row["warning_counts"]),
        "fixed_head": row["head_connection"] == "fixed",
        "armature_zero": int(row["armature_nonzero_count"]) == 0,
        "forward_retention": float(row["forward_displacement_m"])
        >= 0.90 * float(reference["forward_displacement_m"]),
        "upright": float(row["minimum_root_upright_cos"]) >= 0.75,
        "world_roll_not_worse": float(row["chassis_world_roll_abs_p95_deg"])
        <= float(reference["chassis_world_roll_abs_p95_deg"]) + 1.5,
        "tip_stop_rare": float(row["tip_roll_stop_fraction"]) <= 0.01,
        "penetration_not_worse": float(row["contact_penetration_p95_mm"])
        <= 1.10 * float(reference["foot_contact_penetration_p95_mm"]),
    }


def render_demos(duration_s: float, diagnostic_candidate: str | None = None) -> dict[str, object]:
    selection = json.loads(
        (DATA / "first_pitch_support_selection.json").read_text(encoding="utf-8")
    )
    selected = selection["selected_candidate"]
    diagnostic = diagnostic_candidate is not None
    if diagnostic:
        selected = diagnostic_candidate
    if selected is None:
        raise RuntimeError("No candidate passed; provide --candidate for a diagnostic demo")
    spec = CANDIDATES[selected]
    kp = BASE_KP * float(spec["kp_scale"])
    kd = BASE_KD * float(spec["kd_scale"])
    overrides = {"joint_pitch_body_1": (kp, kd)}
    model_paths = {}
    for kind in ("flat", "rough"):
        if diagnostic:
            output = MODELS / f"centipede_{kind}_locked_head_m29_diagnostic_{selected}.xml"
            annotate_model(BASE_MODELS[kind], output, selected, spec)
            model_paths[kind] = output
        else:
            model_paths[kind] = Path(selection["selected_models"][kind])
    baseline = json.loads(candidate_path("baseline").read_text(encoding="utf-8"))
    cases = [
        ("flat", "side", 90.0, -12.0),
        ("flat", "oblique", 125.0, -35.0),
        ("flat", "top", 90.0, -89.0),
        ("rough", "side", 90.0, -12.0),
        ("rough", "oblique", 125.0, -35.0),
        ("rough", "top", 90.0, -89.0),
    ]
    before = locked.tip.robot_sentinels()
    rows = []
    paths = []
    for kind, view, azimuth, elevation in cases:
        model_path = model_paths[kind]
        prefix = "diagnostic_first_pitch" if diagnostic else "selected_first_pitch"
        video = VIDEOS / f"{prefix}_{selected}_{kind}_{view}.mp4"
        print(f"[first-pitch demo] START {selected} {kind} {view}", flush=True)
        row = locked.tip.run_case(
            f"q1 support {selected}",
            kind,
            model_path,
            duration_s,
            video=video,
            camera_azimuth=azimuth,
            camera_elevation=elevation,
            joint_impedance_overrides=overrides,
        )
        row["view"] = view
        row["gates"] = demo_gates(row, row_by_kind(baseline, kind))
        row["all_gates"] = bool(all(row["gates"].values()))
        rows.append(row)
        paths.append(video)
        print(f"[first-pitch demo] DONE  {selected} {kind} {view}", flush=True)
    prefix = "diagnostic_first_pitch" if diagnostic else "selected_first_pitch"
    sheet = FIGURES / f"{prefix}_{selected}_contact_sheet.png"
    locked.tip.write_contact_sheet(paths, sheet)
    result = {
        "selected_candidate": selected,
        "selection_status": "diagnostic_not_selected" if diagnostic else "selected",
        "duration_s": duration_s,
        "videos": [str(path) for path in paths],
        "contact_sheet": str(sheet),
        "rows": rows,
        "all_demo_rows_pass": bool(all(row["all_gates"] for row in rows)),
        "robot_sentinels_before": before,
        "robot_sentinels_after": locked.tip.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    (DATA / "first_pitch_support_demo_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def write_report() -> Path:
    selection = json.loads(
        (DATA / "first_pitch_support_selection.json").read_text(encoding="utf-8")
    )
    selected = selection["selected_candidate"]
    baseline = json.loads(candidate_path("baseline").read_text(encoding="utf-8"))
    lines = [
        "# First-pitch support audit",
        "",
        "This experiment changes only `joint_pitch_body_1` effective Kp/Kd in the biological/template Centipede Model.",
        "",
        f"Selected candidate: **{selected if selected is not None else 'none; baseline retained'}**.",
        "",
        "| Candidate | Kp/Kd scale | Flat q1 P95 | Flat link1 contact | Rough q1 P95 | Rough link1 contact | Rough displacement | Rough tip-stop | Both-terrain gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    assessed = {item["candidate"]: item for item in selection["assessed"]}
    for candidate in CANDIDATES:
        result = json.loads(candidate_path(candidate).read_text(encoding="utf-8"))
        flat = row_by_kind(result, "flat")
        rough = row_by_kind(result, "rough")
        spec = CANDIDATES[candidate]
        status = (
            "reference"
            if candidate == "baseline"
            else str(assessed[candidate]["passes_both_terrains"])
        )
        lines.append(
            f"| {candidate} | {spec['kp_scale']:.3g}/{spec['kd_scale']:.3g} | "
            f"{flat['first_spine_pitch']['joint_angle_abs_deg']['p95']:.2f} deg | "
            f"{100.0 * flat['link1_ground_contact_after_1p5s']['contact_time_fraction']:.2f}% | "
            f"{rough['first_spine_pitch']['joint_angle_abs_deg']['p95']:.2f} deg | "
            f"{100.0 * rough['link1_ground_contact_after_1p5s']['contact_time_fraction']:.2f}% | "
            f"{1000.0 * rough['forward_displacement_m']:.2f} mm | "
            f"{100.0 * rough['tip_roll']['stop_fraction']:.3f}% | "
            f"{status} |"
        )
    lines.extend(
        [
            "",
            f"Passing candidates: {selection['passing_candidates']}.",
            f"Robot sentinels unchanged: {selection['robot_sentinels_unchanged']}.",
            "",
            "Selection uses explicit gates and no scalar reward.  Since no candidate passes both terrains, the production baseline is unchanged; k1p5_zeta demos are diagnostic only.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def compact(result: dict[str, object]) -> dict[str, object]:
    output = dict(result)
    if "rows" in output:
        output["row_count"] = len(output.pop("rows"))
    for key in ("robot_sentinels_before", "robot_sentinels_after"):
        output.pop(key, None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "select", "demo", "report"))
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()
    ensure_directories()
    if args.command == "evaluate":
        if args.candidate is None:
            raise ValueError("--candidate is required for evaluate")
        result = compact(evaluate_candidate(args.candidate, args.duration))
    elif args.command == "select":
        result = select_candidate()
    elif args.command == "demo":
        result = compact(render_demos(args.duration, args.candidate))
    else:
        result = {"report": str(write_report())}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
