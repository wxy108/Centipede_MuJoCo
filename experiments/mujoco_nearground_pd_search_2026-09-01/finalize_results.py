"""Create the selected-profile manifest and an evidence-first Chinese report."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OLD = ROOT.parent / "mujoco_pd_multimetric_stability_2026-09-01" / "outputs"


def load(path: Path):
    return json.loads(path.read_text())


def compact_metrics(row):
    m = row["metrics"]
    return {
        "displacement_mm": m["displacement_mm"],
        "path_efficiency": m["path_efficiency"],
        "coherent_roll_p95_deg": m["coherent_roll_p95_deg"],
        "root_roll_abs_peak_deg": m["root_roll_abs_peak_deg"],
        "body_pitch_speed_peak_rad_s": m["body_pitch_speed_peak"],
        "body_roll_speed_peak_rad_s": m["body_roll_speed_peak"],
        "body_deltaomega_10ms_peak_rad_s": m["body_deltaomega_10ms_peak"],
        "ground_force_peak_bw": m["ground_force_peak_bw"],
        "chassis_force_peak_bw": m["chassis_force_peak_bw"],
        "tip_speed_peak_rad_s": m["tip_speed_peak"],
        "leg_yaw_tracking_rms_deg": m["leg_yaw_tracking_rms_deg"],
        "leg_pitch_tracking_rms_deg": m["leg_pitch_tracking_rms_deg"],
        "loaded_foot_slip_p95_mm_s": m["loaded_foot_slip_p95_mm_s"],
        "work_per_distance_n": m["work_per_distance_n"],
        "swing_foot_center_height_p95_mm": m["swing_foot_height_p95_mm"],
        "all_gates_pass": row["feasible"],
        "failed_gates": row["failed"],
    }


def f(value, digits=3):
    return f"{value:.{digits}f}"


assessment = {row["id"]: row for row in load(OUT / "assessment.json")}
ledger = load(OUT / "final_selection.json")
contract = load(OUT / "contract_checks.json")
pose = load(OUT / "pose_checks.json")
pruning = load(OUT / "pruning_regression_checks.json")

profiles = {
    "2.5": {
        "nominal": "legyaw_refine_k0p29_w2p5_t6_dt0p00025",
        "long": "legyaw_refine_k0p29_w2p5_t12_dt0p00025",
        "half_dt": "legyaw_refine_k0p29_w2p5_t6_dt0p000125",
        "baseline": "baseline_w2p5_t6_dt0p00025",
        "audit": "legyaw_refine_k0p29_w2p5_t6_dt0p00025_trace_audit.json",
        "videos": {
            "three_views": "pd_w2p5_winner_three_views.mp4",
            "local": "pd_w2p5_winner_local.mp4",
        },
    },
    "3": {
        "nominal": "basin15_yk2_pd4_rk1_w3_t6_dt0p00025",
        "long": "basin15_yk2_pd4_rk1_w3_t12_dt0p00025",
        "half_dt": "basin15_yk2_pd4_rk1_w3_t6_dt0p000125",
        "baseline": "baseline_w3_t6_dt0p00025",
        "audit": "basin15_yk2_pd4_rk1_w3_t6_dt0p00025_trace_audit.json",
        "videos": {
            "three_views": "pd_w3_winner_three_views.mp4",
            "local": "pd_w3_winner_local.mp4",
        },
    },
}

selected = {}
for wave, spec in profiles.items():
    rows = {key: assessment[spec[key]] for key in ("nominal", "long", "half_dt", "baseline")}
    assert all(rows[key]["feasible"] for key in ("nominal", "long", "half_dt"))
    audit = load(OUT / "common" / spec["audit"])
    selected[wave] = {
        "status": "validated_best_tested_in_registered_finite_search",
        "config_name": rows["nominal"]["config"]["name"],
        "multipliers_relative_to_baseline": {
            key: value
            for key, value in rows["nominal"]["config"].items()
            if key != "name"
        },
        "absolute_physical_gains": audit["physical_gains"],
        "realized_motion_audit": {
            key: value
            for key, value in audit.items()
            if key
            in (
                "leg_pitch_realized_half_excursion_range_deg",
                "passive_roll_angle_abs_sampled_peak_deg",
                "passive_roll_torque_abs_sampled_peak_nm",
                "foot_sphere_bottom_swing_height_p95_mm",
            )
        },
        "nominal_6s": compact_metrics(rows["nominal"]),
        "matching_baseline_6s": compact_metrics(rows["baseline"]),
        "long_horizon_12s": compact_metrics(rows["long"]),
        "half_timestep_6s": compact_metrics(rows["half_dt"]),
        "validation_ids": [spec["nominal"], spec["long"], spec["half_dt"]],
        "videos": {
            key: str(OUT / "videos" / name) for key, name in spec["videos"].items()
        },
    }

new_init = assessment[profiles["2.5"]["baseline"]]
old_init = load(OLD / "baseline_w2p5_t6_dt0p00025.json")
unresolved = {}
for wave, best in ledger["best_feasible"].items():
    if best is None:
        diagnostic = ledger["diagnostic_candidates"][wave][0]
        unresolved[wave] = {
            "best_diagnostic_id": diagnostic["id"],
            "failed_gates": diagnostic["failed"],
        }

manifest = {
    "scope": "Centipede Model only; Centipede Robot and Robot Step 18/19 untouched",
    "status": "two_wave_specific_profiles_validated; universal_PD_not_found",
    "objective": "maximize forward displacement only among all-gates-feasible trials",
    "amplitudes_frozen_deg": ledger["fixed_amplitudes_deg"],
    "body_and_leg_wave_number_coupled": True,
    "controller_law_unchanged": True,
    "model_xml_unchanged": True,
    "model_sha256": contract["model_sha256"],
    "mass_g": contract["mass_g"],
    "armature_zero": True,
    "near_ground_initialization": {
        **contract["initialization"],
        "old_air_drop_startup_force_peak_bw": old_init["startup"]["ground_force_peak_bw"],
        "new_startup_force_peak_bw": new_init["startup"]["ground_force_peak_bw"],
        "new_rest_t0p9_to_1p0": new_init["rest_before_gait"],
    },
    "search_coverage": {
        "planned_nominal": ledger["planned_nominal"],
        "completed_nominal": ledger["completed_nominal"],
        "full_nominal": ledger["full_nominal"],
        "proven_infeasible_nominal": ledger["proven_infeasible_nominal"],
        "feasible_nominal": ledger["feasible_nominal"],
        "wave_numbers": ledger["wave_numbers"],
        "continuous_global_proof": False,
        "claim": "best tested within the preregistered finite multistage search, not a proof over a continuous domain",
    },
    "selected_profiles": selected,
    "unresolved_wave_numbers": unresolved,
    "verification": {
        "frozen_physics_candidates_checked": contract["checked_candidates"],
        "all_frozen_fields_equal": all(row["frozen_fields_equal"] for row in contract["candidates"]),
        "pose_reconstruction_checks": len(pose),
        "pose_max_error_m": max(
            max(row["max_body_error_m"], row["max_foot_error_m"]) for row in pose
        ),
        "pruning_replays_and_original_regression": pruning,
    },
    "default_controller_replaced": False,
    "reason_default_not_replaced": "the valid result is a wave-specific map, not one universal PD profile",
}
(OUT / "selected_pd_profiles.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)


def row_metrics(wave, key):
    return selected[wave][key]


lines = [
    "# 近地初始化与固定幅值 PD 搜索结果",
    "",
    "## 结论",
    "",
    "近地初始化已经稳定落地：只变换自由根位姿，所有相对关节零位和初速度保持不变，最低碰撞几何离地 25 µm；0.9–1.0 s 静置窗内总法向力峰值为 1.000001 BW。原 XML 空中落体初始化的启动峰值为 "
    + f(old_init["startup"]["ground_force_peak_bw"], 2)
    + " BW，新初始化完整启动窗峰值降为 "
    + f(new_init["startup"]["ground_force_peak_bw"], 2)
    + " BW。",
    "",
    "在固定三项幅值、body/leg wave number 保持一致、控制律不变的条件下，登记的 1629 个名义点全部完成。只有 wave=2.5 与 wave=3 找到并验证了可行 PD；其余波数没有通过全部门槛的点。这里的“最优”严格指登记的有限多阶段搜索中的 best-tested，不是连续参数域的数学全局最优证明。",
    "",
    "固定幅值：body yaw 16.9183°，leg yaw 49.3775°，leg pitch 22.9183°。",
    "",
    "## 选定 PD",
    "",
    "| wave | body yaw Kp/Kd | body pitch Kp/Kd | leg yaw Kp/Kd | leg pitch Kp/Kd | passive body roll k/c |",
    "|---:|---:|---:|---:|---:|---:|",
]
for wave in ("2.5", "3"):
    g = selected[wave]["absolute_physical_gains"]
    a = g["active"]
    lines.append(
        f"| {wave} | {a['body_yaw']['kp_nm_rad']:.6g} / {a['body_yaw']['kd_nms_rad']:.6g} | "
        f"{a['body_pitch']['kp_nm_rad']:.6g} / {a['body_pitch']['kd_nms_rad']:.6g} | "
        f"{a['leg_yaw']['kp_nm_rad']:.6g} / {a['leg_yaw']['kd_nms_rad']:.6g} | "
        f"{a['leg_pitch']['kp_nm_rad']:.6g} / {a['leg_pitch']['kd_nms_rad']:.6g} | "
        f"{g['passive_body_roll']['k_nm_rad']:.6g} / {g['passive_body_roll']['c_nms_rad']:.6g} |"
    )

lines += [
    "",
    "单位：Kp、被动 k 为 N·m/rad；Kd、被动 c 为 N·m·s/rad。tip compliance 未改变。",
    "",
    "## 名义 6 秒对照",
    "",
    "| wave | 方案 | 位移 mm | coherent roll P95 ° | body pitch/roll 峰值 rad/s | tip 速度峰值 rad/s | 峰值地面力 BW | yaw 跟踪 RMS ° | slip P95 mm/s |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]
for wave in ("2.5", "3"):
    for label, key in (("原基线", "matching_baseline_6s"), ("选定 PD", "nominal_6s")):
        m = row_metrics(wave, key)
        lines.append(
            f"| {wave} | {label} | {f(m['displacement_mm'])} | {f(m['coherent_roll_p95_deg'])} | "
            f"{f(m['body_pitch_speed_peak_rad_s'])} / {f(m['body_roll_speed_peak_rad_s'])} | "
            f"{f(m['tip_speed_peak_rad_s'])} | {f(m['ground_force_peak_bw'])} | "
            f"{f(m['leg_yaw_tracking_rms_deg'])} | {f(m['loaded_foot_slip_p95_mm_s'])} |"
        )

lines += [
    "",
    "## 鲁棒性复验",
    "",
    "| wave | 6 s / 0.25 ms | 12 s / 0.25 ms | 6 s / 0.125 ms |",
    "|---:|---:|---:|---:|",
]
for wave in ("2.5", "3"):
    lines.append(
        f"| {wave} | {f(row_metrics(wave, 'nominal_6s')['displacement_mm'])} mm，全部通过 | "
        f"{f(row_metrics(wave, 'long_horizon_12s')['displacement_mm'])} mm，全部通过 | "
        f"{f(row_metrics(wave, 'half_timestep_6s')['displacement_mm'])} mm，全部通过 |"
    )

lines += [
    "",
    "“全部通过”包括原硬门槛、leg-yaw 跟踪补充门槛，以及相同 wave/duration/dt 基线的位移、功耗、slip、左右力不平衡和冲量非退化门槛。",
    "",
    "## 搜索边界与未解决项",
    "",
    f"- 已完成 {ledger['completed_nominal']}/{ledger['planned_nominal']} 个登记名义点；{ledger['proven_infeasible_nominal']} 个由不可逆逐步硬门槛证据提前判为不可行，{ledger['full_nominal']} 个完成全量评估。",
    "- wave=2 的高位移诊断仍只因既有 foot-center swing-height 门槛失败；本轮没有更改门槛，也没有把它升级为 winner。",
    "- wave=0.5、1、1.5、2、3.5、4、4.5、5、6、7、8、9 尚无全门槛可行点，因此当前结果必须作为 wave-specific PD map，而不能写成一套 universal PD。",
    "- 默认控制器没有自动替换；模型 XML、质量、惯量、几何、接触、armature、tip compliance 和幅值均未改变。",
    "",
    "## 可复核文件",
    "",
    "- `outputs/selected_pd_profiles.json`：完整选定参数、绝对物理增益、验证 ID 与指标。",
    "- `outputs/final_selection.json`：全搜索账本与每个 wave 的诊断结果。",
    "- `PROTOCOL.md`：预注册门槛、搜索域与修订记录。",
    "- `outputs/pose_checks.json`：录像轨迹重建误差检查。",
    "- `outputs/contract_checks.json`：冻结物理字段与模型哈希检查。",
]
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(
    json.dumps(
        {
            "selected": list(selected),
            "all_selected_validated": True,
            "unresolved": list(unresolved),
            "manifest": str(OUT / "selected_pd_profiles.json"),
            "report": str(ROOT / "RESULTS.md"),
        },
        indent=2,
    )
)
