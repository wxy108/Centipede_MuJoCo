"""Quantify whether removing distal tip compliance reduces body twitch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE_DIR = PROJECT / "experiments" / "mujoco_nearground_pd_search_2026-09-01" / "outputs" / "common"
DATA = ROOT / "outputs" / "data"
TOPOLOGY = ROOT / "outputs" / "topology_audit.json"
MASS_KG = json.loads(TOPOLOGY.read_text())["rigid_total_mass_kg"]
WEIGHT_N = MASS_KG * 9.81

PAIRS = {
    "2.5": (
        "legyaw_refine_k0p29_w2p5_t6_dt0p00025",
        "rigid_tip_legyaw_refine_k0p29_w2p5_t6_dt0p00025",
    ),
    "3": (
        "basin15_yk2_pd4_rk1_w3_t6_dt0p00025",
        "rigid_tip_basin15_yk2_pd4_rk1_w3_t6_dt0p00025",
    ),
}
HALF_PAIRS = {
    "2.5": (
        "legyaw_refine_k0p29_w2p5_t6_dt0p000125",
        "rigid_tip_legyaw_refine_k0p29_w2p5_t6_dt0p000125",
    ),
    "3": (
        "basin15_yk2_pd4_rk1_w3_t6_dt0p000125",
        "rigid_tip_basin15_yk2_pd4_rk1_w3_t6_dt0p000125",
    ),
}


def load_pair(directory: Path, stem: str):
    row = json.loads((directory / f"{stem}.json").read_text())
    with np.load(directory / f"{stem}.npz") as archive:
        trace = {key: archive[key] for key in archive.files}
    return row, trace


def percentile(values, q):
    values = np.asarray(values)
    return float(np.quantile(values, q)) if values.size else float("nan")


def max_window_integral(values, dt, width=0.01):
    n = max(1, int(round(width / dt)))
    cs = np.r_[0.0, np.cumsum(values)]
    return float(np.max(cs[n:] - cs[:-n], initial=0.0) * dt)


def twitch_metrics(row, trace):
    t = np.asarray(trace["t"], dtype=float)
    good = t >= 1.5
    sampled_dt = float(np.median(np.diff(t)))
    bv = np.asarray(trace["bv"], dtype=float)[good]
    acceleration = np.diff(bv, axis=0) / sampled_dt
    jerk = np.diff(acceleration, axis=0) / sampled_dt
    accel_abs = np.abs(acceleration)
    jerk_abs = np.abs(jerk)
    physics_dt = float(row["dt"])
    start = int(round(1.5 / physics_dt))
    force = np.asarray(trace["force"], dtype=float)[start:]
    feet = force[:, :38].sum(axis=1) / WEIGHT_N
    chassis = force[:, 38:59].sum(axis=1) / WEIGHT_N
    head_link1 = force[:, 38:40].sum(axis=1) / WEIGHT_N
    return {
        "body_joint_accel_p99_rad_s2": percentile(accel_abs, 0.99),
        "body_joint_accel_peak_rad_s2": float(accel_abs.max()),
        "body_joint_jerk_p99_rad_s3": percentile(jerk_abs, 0.99),
        "foot_force_step_change_p99_bw": percentile(np.abs(np.diff(feet)), 0.99),
        "chassis_force_peak_bw_recomputed": float(chassis.max()),
        "chassis_force_p99_bw": percentile(chassis, 0.99),
        "chassis_contact_fraction_recomputed": float(np.mean(chassis > 1e-7)),
        "head_link1_force_peak_bw": float(head_link1.max()),
        "head_link1_force_p99_bw": percentile(head_link1, 0.99),
        "head_link1_contact_fraction": float(np.mean(head_link1 > 1e-7)),
        "head_link1_impulse_10ms_bw_s": max_window_integral(head_link1, physics_dt),
    }


def common_metrics(row):
    m = row["metrics"]
    return {
        key: m[key]
        for key in (
            "displacement_mm",
            "path_efficiency",
            "coherent_roll_p95_deg",
            "root_roll_abs_peak_deg",
            "body_yaw_speed_peak",
            "body_pitch_speed_peak",
            "body_roll_speed_peak",
            "body_deltaomega_10ms_peak",
            "body_pitch_delta10ms_deg",
            "body_roll_delta10ms_deg",
            "ground_force_peak_bw",
            "chassis_force_peak_bw",
            "chassis_contact_fraction",
            "loaded_foot_slip_p95_mm_s",
            "leg_yaw_tracking_rms_deg",
            "leg_pitch_tracking_rms_deg",
            "swing_foot_height_p95_mm",
            "work_per_distance_n",
        )
    }


def ratio_change(rigid, compliant):
    return 100.0 * (rigid / compliant - 1.0) if compliant else float("nan")


results = {}
support_votes = []
for wave, (source_stem, rigid_stem) in PAIRS.items():
    source_row, source_trace = load_pair(SOURCE_DIR, source_stem)
    rigid_row, rigid_trace = load_pair(DATA, rigid_stem)
    source_common, rigid_common = common_metrics(source_row), common_metrics(rigid_row)
    source_twitch, rigid_twitch = twitch_metrics(source_row, source_trace), twitch_metrics(rigid_row, rigid_trace)
    primary = (
        "body_deltaomega_10ms_peak",
        "body_pitch_speed_peak",
        "body_roll_speed_peak",
        "chassis_force_peak_bw",
    )
    primary_changes = {
        key: ratio_change(rigid_common[key], source_common[key]) for key in primary
    }
    twitch_changes = {
        key: ratio_change(rigid_twitch[key], source_twitch[key])
        for key in (
            "body_joint_accel_p99_rad_s2",
            "body_joint_jerk_p99_rad_s3",
            "head_link1_force_peak_bw",
            "head_link1_impulse_10ms_bw_s",
        )
    }
    support = sum(change < -5.0 for change in (*primary_changes.values(), *twitch_changes.values()))
    against = sum(change > 5.0 for change in (*primary_changes.values(), *twitch_changes.values()))
    support_votes.append((support, against))
    results[wave] = {
        "controller_and_pd_identical": True,
        "amplitudes_identical": True,
        "compliant_tip": {
            "id": source_stem,
            "common_metrics": source_common,
            "twitch_metrics": source_twitch,
        },
        "rigid_tip": {
            "id": rigid_stem,
            "common_metrics": rigid_common,
            "twitch_metrics": rigid_twitch,
            "all_applicable_absolute_gates_pass": rigid_row["feasible_applicable_gates"],
            "failed_applicable_gates": rigid_row["failed_applicable_gates"],
            "not_applicable_original_tip_gates": rigid_row["not_applicable_original_gates"],
        },
        "primary_percent_changes_rigid_vs_compliant": primary_changes,
        "twitch_percent_changes_rigid_vs_compliant": twitch_changes,
        "evidence_votes_support_tip_hypothesis": support,
        "evidence_votes_against_tip_hypothesis": against,
    }

half_results = {}
for wave, (source_stem, rigid_stem) in HALF_PAIRS.items():
    source_row, source_trace = load_pair(SOURCE_DIR, source_stem)
    rigid_row, rigid_trace = load_pair(DATA, rigid_stem)
    source_common, rigid_common = common_metrics(source_row), common_metrics(rigid_row)
    source_twitch, rigid_twitch = twitch_metrics(source_row, source_trace), twitch_metrics(rigid_row, rigid_trace)
    primary = ("body_deltaomega_10ms_peak", "body_pitch_speed_peak", "body_roll_speed_peak", "chassis_force_peak_bw")
    primary_changes = {key: ratio_change(rigid_common[key], source_common[key]) for key in primary}
    twitch_changes = {
        key: ratio_change(rigid_twitch[key], source_twitch[key])
        for key in ("body_joint_accel_p99_rad_s2", "body_joint_jerk_p99_rad_s3", "head_link1_force_peak_bw", "head_link1_impulse_10ms_bw_s")
    }
    support = sum(change < -5.0 for change in (*primary_changes.values(), *twitch_changes.values()))
    against = sum(change > 5.0 for change in (*primary_changes.values(), *twitch_changes.values()))
    support_votes.append((support, against))
    half_results[wave] = {
        "dt_s": 0.000125,
        "compliant_tip": {"id": source_stem, "common_metrics": source_common, "twitch_metrics": source_twitch},
        "rigid_tip": {
            "id": rigid_stem,
            "common_metrics": rigid_common,
            "twitch_metrics": rigid_twitch,
            "all_applicable_absolute_gates_pass": rigid_row["feasible_applicable_gates"],
            "failed_applicable_gates": rigid_row["failed_applicable_gates"],
        },
        "primary_percent_changes_rigid_vs_compliant": primary_changes,
        "twitch_percent_changes_rigid_vs_compliant": twitch_changes,
        "evidence_votes_support_tip_hypothesis": support,
        "evidence_votes_against_tip_hypothesis": against,
    }

total_support = sum(item[0] for item in support_votes)
total_against = sum(item[1] for item in support_votes)
if total_support >= 10 and total_against <= 3:
    conclusion = "supports_tip_as_major_twitch_source"
elif total_against >= total_support:
    conclusion = "does_not_support_tip_as_major_twitch_source"
else:
    conclusion = "mixed_evidence_tip_is_contributor_not_sole_source"

output = {
    "scope": "paired single-variable tip-compliance ablation",
    "topology_audit": json.loads(TOPOLOGY.read_text()),
    "comparison": results,
    "half_timestep_comparison": half_results,
    "hypothesis_conclusion": conclusion,
    "support_votes": total_support,
    "against_votes": total_against,
    "causal_limit": "two paired wave conditions; sufficient for this model/controller ablation, not a biological proof",
}
(ROOT / "outputs" / "comparison.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


report = [
    "# Rigid-tip ablation",
    "",
    "## 设计",
    "",
    "只移除 38 个 distal tip-roll 关节；tip、foot、seg1、seg2 的几何和质量仍保留为同一个刚性运动链。没有恢复 knee。足端标称位置、腿长、质量、惯量、接触和控制参数均不变。",
    "",
    "## 成对结果",
    "",
    "| wave | topology | D mm | Δω 10ms rad/s | pitch/roll speed rad/s | chassis force BW | accel P99 rad/s² | head+link1 force BW | slip P95 mm/s | gates |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for wave in ("2.5", "3"):
    item = results[wave]
    for label, key in (("compliant tip", "compliant_tip"), ("rigid tip", "rigid_tip")):
        c = item[key]["common_metrics"]
        t = item[key]["twitch_metrics"]
        gates = "reference pass" if key == "compliant_tip" else (
            "applicable pass" if item[key]["all_applicable_absolute_gates_pass"] else ", ".join(item[key]["failed_applicable_gates"])
        )
        report.append(
            f"| {wave} | {label} | {fmt(c['displacement_mm'])} | {fmt(c['body_deltaomega_10ms_peak'])} | "
            f"{fmt(c['body_pitch_speed_peak'])}/{fmt(c['body_roll_speed_peak'])} | {fmt(c['chassis_force_peak_bw'])} | "
            f"{fmt(t['body_joint_accel_p99_rad_s2'])} | {fmt(t['head_link1_force_peak_bw'])} | "
            f"{fmt(c['loaded_foot_slip_p95_mm_s'])} | {gates} |"
        )
report += [
    "",
    "## 半时间步复验（0.125 ms）",
    "",
    "| wave | topology | D mm | Δω 10ms rad/s | accel P99 rad/s² | failed gates |",
    "|---:|---|---:|---:|---:|---|",
]
for wave in ("2.5", "3"):
    item = half_results[wave]
    for label, key in (("compliant tip", "compliant_tip"), ("rigid tip", "rigid_tip")):
        c = item[key]["common_metrics"]
        t = item[key]["twitch_metrics"]
        failed = "reference pass" if key == "compliant_tip" else (
            "none" if not item[key]["failed_applicable_gates"] else ", ".join(item[key]["failed_applicable_gates"])
        )
        report.append(
            f"| {wave} | {label} | {fmt(c['displacement_mm'])} | {fmt(c['body_deltaomega_10ms_peak'])} | "
            f"{fmt(t['body_joint_accel_p99_rad_s2'])} | {failed} |"
        )
report += [
    "",
    "## 判定",
    "",
    f"机器判定：`{conclusion}`。支持票 {total_support}，反对票 {total_against}。支持/反对票只来自预先列出的 body jump、pitch/roll speed、chassis force、加速度、jerk 与 head/link1 contact 指标，避免只凭视频印象。",
    "",
    "原 tip-angle/tip-speed 四个门槛在 rigid topology 中没有对应自由度，因此明确标为 N/A，没有伪造为 pass。",
]
(ROOT / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print(json.dumps({"conclusion": conclusion, "support": total_support, "against": total_against}, indent=2))
