"""Write the evidence ledger and human-readable conclusion."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = OUT / "data"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


experiment = load(ROOT / "optimize.py", "compliant_phase_finalize_source")
np = experiment.np


def row(stem: str) -> dict:
    return json.loads((DATA / f"{stem}.json").read_text())


PROFILES = {
    "2.5": {
        "nominal": row("sobol_027_w2p5_t6_dt0p00025"),
        "long": row("recovery_sobol_027_w2p5_t12_dt0p00025"),
        "half": row("recovery_sobol_027_w2p5_t6_dt0p000125"),
    },
    "3": {
        "nominal": row("target_02_w3_t6_dt0p00025"),
        "long": row("selected_w3_w3_t12_dt0p00025"),
        "half": row("selected_w3_w3_t6_dt0p000125"),
    },
}


def absolute_parameters(config: dict) -> dict:
    evaluator = experiment.opt.FastEvaluator()
    model, index = evaluator.model, evaluator.index
    experiment.pilot.apply_impedance(model, index, config)

    def active(positions):
        positions = np.asarray(positions).reshape(-1)
        return {
            "kp_nm_rad": float(np.unique(index.kp[positions])[0]),
            "kd_nms_rad": float(np.unique(index.kv[positions])[0]),
        }

    return {
        "active": {
            "body_yaw": active(index.body_positions),
            "body_pitch": active(index.pitch_positions),
            "leg_yaw": active(index.leg_positions[:, :, 0]),
            "leg_pitch": active(index.leg_positions[:, :, 1]),
        },
        "passive_body_roll": {
            "k_nm_rad": float(np.unique(model.jnt_stiffness[index.roll_joint_ids])[0]),
            "c_nms_rad": float(np.unique(model.dof_damping[index.roll_dof_addresses])[0]),
        },
        "passive_tip_roll_unchanged": {
            "k_nm_rad": float(np.unique(model.jnt_stiffness[index.tip_joint_ids])[0]),
            "c_nms_rad": float(np.unique(model.dof_damping[index.tip_dof_addresses])[0]),
        },
    }


full_rows = []
for path in DATA.glob("*.json"):
    try:
        candidate = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue
    if candidate.get("metrics"):
        full_rows.append(candidate)

assessment = {
    "scope": "latest compliant-tip Centipede Model only",
    "status": "phase-corrected nominal profiles found; robustness not certified; default not replaced",
    "rigid_tip_is_ablation_only": True,
    "robot_project_read_only": True,
    "robot_step18_step19_untouched": True,
    "controller_structure": "open-loop three-amplitude CPG; no feedback, clipping or smoothing added",
    "amplitudes_fixed_deg": experiment.fixed_case(2.5),
    "body_and_leg_wave_number_coupled": True,
    "tip_topology": "original passive compliant roll tip retained",
    "tip_parameters_optimized": False,
    "model_sha256": experiment.opt.tip.sha256(experiment.opt.MODEL),
    "search_evidence": {
        "full_evaluations_available": len(full_rows),
        "continuous_global_optimum_proven": False,
        "decision_rule": "maximize displacement only among candidates passing every registered gate",
    },
    "phase_definition": {
        "body": "theta_body[j] = omega*t - j*(2*pi*wave/18)",
        "leg": "theta_leg[i] = theta_body[i] + f*(2*pi*wave/18)",
        "robot_geometry_midpoint_f": 0.5,
        "applied_to": ["leg_yaw", "leg_pitch"],
    },
    "profiles": {},
    "decision": {
        "replace_previous_validated_profile": False,
        "reason": "each nonzero-phase nominal profile failed at least one registered long-horizon or half-timestep robustness gate",
        "next_step": "carry the phase-collocation correction as a nominal diagnostic, but begin the next study from the previously certified compliant-tip shared-phase profiles unless robustness gates are explicitly revisited",
    },
    "videos": {
        "wave2.5_three_views": str(OUT / "videos" / "compliant_tip_phase_w2p5_three_views.mp4"),
        "wave2.5_local": str(OUT / "videos" / "compliant_tip_phase_w2p5_local.mp4"),
        "wave3_three_views": str(OUT / "videos" / "compliant_tip_phase_w3p0_three_views.mp4"),
        "wave3_local": str(OUT / "videos" / "compliant_tip_phase_w3p0_local.mp4"),
    },
}

for wave, records in PROFILES.items():
    nominal = records["nominal"]
    assessment["profiles"][wave] = {
        "phase_fraction": nominal["phase_collocation"]["leg_phase_fraction"],
        "phase_offset_deg": nominal["phase_collocation"]["leg_phase_offset_deg"],
        "multipliers_relative_to_baseline": nominal["config"],
        "absolute_physical_parameters": absolute_parameters(nominal["config"]),
        "nominal_6s": {
            "id": nominal["id"],
            "displacement_mm": nominal["metrics"]["displacement_mm"],
            "all_registered_gates_pass": nominal["all_registered_gates_pass"],
            "failed": nominal["failed_registered_gates"],
        },
        "long_12s": {
            "id": records["long"]["id"],
            "displacement_mm": records["long"]["metrics"]["displacement_mm"],
            "all_registered_gates_pass": records["long"]["all_registered_gates_pass"],
            "failed": records["long"]["failed_registered_gates"],
        },
        "half_timestep_6s": {
            "id": records["half"]["id"],
            "displacement_mm": records["half"]["metrics"]["displacement_mm"],
            "all_registered_gates_pass": records["half"]["all_registered_gates_pass"],
            "failed": records["half"]["failed_registered_gates"],
        },
    }

(OUT / "final_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")


def f(value):
    return f"{float(value):.4g}"


lines = [
    "# Compliant-tip phase + PD optimization",
    "",
    "结论：柔性 tip 始终保留且其被动 k/c 未改变。body-yaw 与 leg cluster 的空间采样已拆开，body/leg wave number 仍完全一致，控制器没有加入反馈、滤波、限幅或特例稳定器。两组非零相位方案都在 6 s nominal 下通过全部登记门槛，但没有同时通过 12 s 与半时间步复核，因此不覆盖上一版已认证 profile。",
    "",
    "## Phase definition",
    "",
    "`theta_leg[i] = theta_body[i] + f*(2*pi*wave/18)`；几何中点为 `f=0.5`。本轮 nominal winners 分别为 wave=2.5: `f=0.4853` (24.27 deg)，wave=3: `f=0.35` (21 deg)。",
    "",
    "## Nominal profiles and validation",
    "",
    "| wave | phase offset | 6 s nominal | 12 s | 6 s half-dt | decision |",
    "|---:|---:|---:|---:|---:|---|",
]
for wave, records in PROFILES.items():
    n, long, half = records["nominal"], records["long"], records["half"]
    lines.append(
        f"| {wave} | {f(n['phase_collocation']['leg_phase_offset_deg'])} deg | "
        f"{f(n['metrics']['displacement_mm'])} mm, PASS | "
        f"{f(long['metrics']['displacement_mm'])} mm, "
        f"{'PASS' if long['all_registered_gates_pass'] else 'FAIL: ' + ', '.join(long['failed_registered_gates'])} | "
        f"{f(half['metrics']['displacement_mm'])} mm, "
        f"{'PASS' if half['all_registered_gates_pass'] else 'FAIL: ' + ', '.join(half['failed_registered_gates'])} | nominal diagnostic only |"
    )

lines.extend([
    "",
    "## Exact impedance values",
    "",
    "| wave | body yaw Kp/Kd | body pitch Kp/Kd | leg yaw Kp/Kd | leg pitch Kp/Kd | passive body roll k/c | tip k/c |",
    "|---:|---|---|---|---|---|---|",
])
for wave, profile in assessment["profiles"].items():
    p = profile["absolute_physical_parameters"]
    a, r, tip = p["active"], p["passive_body_roll"], p["passive_tip_roll_unchanged"]
    lines.append(
        f"| {wave} | {a['body_yaw']['kp_nm_rad']:.6g}/{a['body_yaw']['kd_nms_rad']:.6g} | "
        f"{a['body_pitch']['kp_nm_rad']:.6g}/{a['body_pitch']['kd_nms_rad']:.6g} | "
        f"{a['leg_yaw']['kp_nm_rad']:.6g}/{a['leg_yaw']['kd_nms_rad']:.6g} | "
        f"{a['leg_pitch']['kp_nm_rad']:.6g}/{a['leg_pitch']['kd_nms_rad']:.6g} | "
        f"{r['k_nm_rad']:.6g}/{r['c_nms_rad']:.6g} | "
        f"{tip['k_nm_rad']:.6g}/{tip['c_nms_rad']:.6g} |"
    )

lines.extend([
    "",
    "## Decision",
    "",
    "- 默认仍是上一版柔性-tip、共享空间相位但已通过 nominal/12 s/half-dt 的两个 wave-specific PD profiles。",
    "- 本轮证明：几何相位拆分不是只改一行 phase 就能免费获得的修正；它会改变足端冲击和左右载荷，需要新的 robustness 标定。",
    "- rigid-tip 搜索已明确中止并标记为不得使用；Robot 目录与 Step 18/19 未写入。",
    "- Demo 使用 1 kHz 记录的真实物理姿态、无平滑、contact erasure OFF；画面明确标注新方案仅 nominal pass。",
])
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"assessment": str(OUT / "final_assessment.json"), "report": str(ROOT / "RESULTS.md")}, indent=2))
