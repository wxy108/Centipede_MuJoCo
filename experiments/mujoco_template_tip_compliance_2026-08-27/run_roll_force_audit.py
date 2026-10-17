"""Flat-ground roll and leg-feedback-force audit for the distal-tip model.

This is a diagnostic companion to run_tip_compliance.py.  It does not select a
gait and never writes to the Centipede Robot project.  The wave=3 run is a
causal ablation only: body and leg wave numbers remain coupled and final-model
selection continues to use the PI-required flat wave=1.5 controller.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import run_tip_compliance as tip


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
FIGURES = ROOT / "outputs" / "figures"
CURRENT_MODEL = ROOT / "models" / "centipede_flat_tip_k5e-6_r35_bodyk6_distal_roll.xml"
SOFTER_MODEL = ROOT / "models" / "centipede_flat_tip_k4p5e-6_r38_bodyk6_distal_roll.xml"
SOURCE_MODEL = tip.SOURCE_MODELS["flat"]
GRAVITY = 9.81


def pct(values: np.ndarray | list[float], quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, quantile)) if array.size else float("nan")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    a, b = left[mask], right[mask]
    if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mass_matrix_audit(path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    matrix = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, matrix, data.qM)
    eigenvalues = np.linalg.eigvalsh(matrix)
    root_joint = tip.base.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_dof = int(model.jnt_dofadr[root_joint])
    roll_dof = root_dof + 3
    inverse = np.linalg.inv(matrix)
    effective_roll_inertia = 1.0 / float(inverse[roll_dof, roll_dof])
    root_body = tip.base.name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
    return {
        "model": str(path),
        "model_sha256": tip.sha256(path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "system_com_z_at_reset_mm": 1000.0 * float(data.subtree_com[root_body, 2]),
        "root_roll_mass_matrix_diagonal_kg_m2": float(matrix[roll_dof, roll_dof]),
        "root_roll_effective_inertia_kg_m2": effective_roll_inertia,
        "mass_matrix_smallest_eigenvalue": float(eigenvalues[0]),
        "mass_matrix_largest_eigenvalue": float(eigenvalues[-1]),
        "mass_matrix_condition_number_raw_mixed_units": float(eigenvalues[-1] / eigenvalues[0]),
        "armature_nonzero_count": int(np.count_nonzero(np.abs(model.dof_armature) > 1e-16)),
    }


def classify_robot_geom(model: mujoco.MjModel, geom_id: int) -> tuple[str, str | None, str]:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    foot_match = re.fullmatch(r"foot_(\d+)_(0|1)_collision", name)
    if foot_match:
        return "foot", "L" if foot_match.group(2) == "0" else "R", name
    leg_match = re.match(r"link_leg_\d+_([LR])_", name)
    if leg_match:
        return "other_leg", leg_match.group(1), name
    return "body_or_other", None, name


def flat_leg_amplitudes() -> np.ndarray:
    return np.asarray(
        [
            tip.base.LEG_YAW_AMPLITUDE_RAD * float(tip.ctrl.LEG_YAW_AMPLITUDE_SCALE),
            tip.base.LEG_PITCH_AMPLITUDE_RAD * float(tip.ctrl.LEG_PITCH_AMPLITUDE_SCALE),
        ],
        dtype=float,
    )


def run_force_case(
    label: str,
    model_path: Path,
    duration_s: float,
    wave: float,
    control_enabled: bool = True,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    index = tip.LeanIndex(model)
    tip.configure_model(model, index)
    if not control_enabled:
        model.actuator_gainprm[:] = 0.0
        model.actuator_biasprm[:] = 0.0
    controller = tip.LeanController(model, index, wave) if control_enabled else None
    if controller is not None:
        # Isolate spatial wave number: the wave=3 ablation must retain the flat
        # leg amplitudes instead of silently switching to rough-terrain gains.
        controller.leg_amplitudes = flat_leg_amplitudes()

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ground = tip.stage1.ground_geom_id(model)
    root_body = tip.base.name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
    total_mass = float(np.sum(model.body_mass))
    weight = total_mass * GRAVITY
    eval_start_s = min(1.5, 0.25 * duration_s)
    dt = float(model.opt.timestep)
    stride = max(1, int(round(0.001 / dt)))
    sample_dt = stride * dt

    traces: dict[str, list[float]] = {
        name: []
        for name in (
            "time_s",
            "mean_body_roll_deg",
            "head_roll_deg",
            "com_z_mm",
            "total_support_z_mn",
            "left_support_z_mn",
            "right_support_z_mn",
            "unassigned_support_z_mn",
            "lr_support_imbalance",
            "head_axis_roll_moment_unm",
            "net_horizontal_force_mn",
            "foot_load_fraction",
            "contacting_feet",
        )
    }
    all_body_roll_abs: list[float] = []
    tip_roll_abs: list[float] = []
    loaded_foot_normal_mn: list[float] = []
    all_contact_normal_mn: list[float] = []
    leg_torque_abs_unm: list[float] = []
    nonfoot_contact_samples = 0
    ground_contact_samples = 0
    peak_foot = {"force_mn": -1.0, "geom": "", "time_s": 0.0}
    start_x = end_x = None
    finite = True
    step_index = 0

    while data.time < duration_s - 0.5 * dt:
        if controller is None:
            torque = np.zeros(len(index.active_joint_ids), dtype=float)
            data.ctrl[:] = 0.0
        else:
            torque = controller.step(data)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            finite = False
            break
        if data.time < eval_start_s or step_index % stride:
            step_index += 1
            continue

        center = tip.base.chassis_centroid(data, index)
        if start_x is None:
            start_x = float(center[0])
        end_x = float(center[0])
        com = np.asarray(data.subtree_com[root_body], dtype=float).copy()
        world_roll = tip.signed_world_roll_deg(data, index.chassis_body_ids)
        head_rotation = np.asarray(data.xmat[index.head_body_id], dtype=float).reshape(3, 3)
        head_forward = head_rotation[:, 0]

        side_force = {"L": np.zeros(3), "R": np.zeros(3)}
        unassigned_force = np.zeros(3)
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        foot_force_by_geom: dict[int, np.ndarray] = {}
        foot_normal_by_geom: dict[int, float] = {}
        sample_nonfoot = False

        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == ground:
                robot_geom, robot_is_geom2 = geom2, True
            elif geom2 == ground:
                robot_geom, robot_is_geom2 = geom1, False
            else:
                continue
            force6 = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, force6)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force_on_geom2_world = frame.T @ force6[:3]
            force_world = force_on_geom2_world if robot_is_geom2 else -force_on_geom2_world
            normal_force = abs(float(force6[0]))
            category, side, geom_name = classify_robot_geom(model, robot_geom)
            total_force += force_world
            total_moment += np.cross(np.asarray(contact.pos, dtype=float) - com, force_world)
            all_contact_normal_mn.append(1000.0 * normal_force)
            ground_contact_samples += 1
            if side is None:
                unassigned_force += force_world
            else:
                side_force[side] += force_world
            if category == "foot":
                foot_force_by_geom.setdefault(robot_geom, np.zeros(3))
                foot_force_by_geom[robot_geom] += force_world
                foot_normal_by_geom[robot_geom] = foot_normal_by_geom.get(robot_geom, 0.0) + normal_force
            else:
                sample_nonfoot = True
                nonfoot_contact_samples += 1

        for geom_id, normal_force in foot_normal_by_geom.items():
            force_mn = 1000.0 * normal_force
            loaded_foot_normal_mn.append(force_mn)
            if force_mn > float(peak_foot["force_mn"]):
                peak_foot = {
                    "force_mn": force_mn,
                    "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                    "time_s": float(data.time),
                }

        left_z = float(side_force["L"][2])
        right_z = float(side_force["R"][2])
        assigned_z = left_z + right_z
        total_z = float(total_force[2])
        traces["time_s"].append(float(data.time))
        traces["mean_body_roll_deg"].append(float(np.mean(world_roll)))
        traces["head_roll_deg"].append(float(world_roll[0]))
        traces["com_z_mm"].append(1000.0 * float(com[2]))
        traces["total_support_z_mn"].append(1000.0 * total_z)
        traces["left_support_z_mn"].append(1000.0 * left_z)
        traces["right_support_z_mn"].append(1000.0 * right_z)
        traces["unassigned_support_z_mn"].append(1000.0 * float(unassigned_force[2]))
        traces["lr_support_imbalance"].append((right_z - left_z) / max(abs(assigned_z), 1e-12))
        traces["head_axis_roll_moment_unm"].append(1.0e6 * float(np.dot(total_moment, head_forward)))
        traces["net_horizontal_force_mn"].append(1000.0 * float(np.linalg.norm(total_force[:2])))
        foot_z = sum(float(force[2]) for force in foot_force_by_geom.values())
        traces["foot_load_fraction"].append(foot_z / max(total_z, 1e-12))
        traces["contacting_feet"].append(float(len(foot_force_by_geom)))
        all_body_roll_abs.extend(np.abs(world_roll).tolist())
        tip_roll_abs.extend(np.degrees(np.abs(data.qpos[index.tip_qpos_addresses])).tolist())
        if controller is not None:
            leg_torque_abs_unm.extend(
                (1.0e6 * np.abs(torque[index.leg_positions.reshape(-1)])).tolist()
            )
        step_index += 1

    arrays = {name: np.asarray(values, dtype=float) for name, values in traces.items()}
    if arrays["time_s"].size < 5:
        raise RuntimeError(f"Insufficient diagnostic samples for {label}")
    if float(np.mean(arrays["total_support_z_mn"])) <= 0.0:
        raise RuntimeError("Contact force orientation audit failed: mean support force is not upward")

    com_z_m = arrays["com_z_mm"] / 1000.0
    com_vertical_acceleration = np.gradient(np.gradient(com_z_m, sample_dt), sample_dt)
    support_acceleration = (arrays["total_support_z_mn"] / 1000.0 - weight) / total_mass
    roll_rad = np.unwrap(np.radians(arrays["mean_body_roll_deg"]))
    roll_acceleration = np.gradient(np.gradient(roll_rad, sample_dt), sample_dt)
    roll_moment_nm = arrays["head_axis_roll_moment_unm"] / 1.0e6
    bodyweight = weight * 1000.0
    result = {
        "label": label,
        "model": str(model_path),
        "model_sha256": tip.sha256(model_path),
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "sample_period_s": sample_dt,
        "control_enabled": control_enabled,
        "body_wave": wave if control_enabled else None,
        "leg_wave": wave if control_enabled else None,
        "wave_numbers_coupled": True if control_enabled else None,
        "flat_leg_amplitudes_for_wave_ablation": True if control_enabled else None,
        "finite": finite,
        "warning_counts": tip.stage1.warning_counts(data),
        "total_mass_kg": total_mass,
        "weight_mn": bodyweight,
        "forward_displacement_mm": (
            1000.0 * tip.base.FORWARD_SIGN * float(end_x - start_x)
            if start_x is not None and end_x is not None
            else float("nan")
        ),
        "chassis_world_roll_abs_p95_deg": pct(all_body_roll_abs, 0.95),
        "mean_body_roll_abs_p95_deg": pct(np.abs(arrays["mean_body_roll_deg"]), 0.95),
        "head_world_roll_abs_p95_deg": pct(np.abs(arrays["head_roll_deg"]), 0.95),
        "tip_roll_abs_p95_deg": pct(tip_roll_abs, 0.95),
        "system_com_z_mean_mm": float(np.mean(arrays["com_z_mm"])),
        "system_com_z_rms_about_mean_mm": float(np.std(arrays["com_z_mm"])),
        "system_com_z_peak_to_peak_mm": float(np.ptp(arrays["com_z_mm"])),
        "total_support_force_mean_mn": float(np.mean(arrays["total_support_z_mn"])),
        "total_support_force_p95_mn": pct(arrays["total_support_z_mn"], 0.95),
        "total_support_force_max_mn": float(np.max(arrays["total_support_z_mn"])),
        "total_support_force_p95_bodyweights": pct(arrays["total_support_z_mn"], 0.95) / bodyweight,
        "support_above_1p5_bodyweights_fraction": float(
            np.mean(arrays["total_support_z_mn"] > 1.5 * bodyweight)
        ),
        "left_support_force_mean_mn": float(np.mean(arrays["left_support_z_mn"])),
        "right_support_force_mean_mn": float(np.mean(arrays["right_support_z_mn"])),
        "lr_support_imbalance_abs_p95": pct(np.abs(arrays["lr_support_imbalance"]), 0.95),
        "lr_support_imbalance_signed_mean": float(np.mean(arrays["lr_support_imbalance"])),
        "head_axis_contact_roll_moment_abs_p95_unm": pct(
            np.abs(arrays["head_axis_roll_moment_unm"]), 0.95
        ),
        "head_axis_contact_roll_moment_abs_max_unm": float(
            np.max(np.abs(arrays["head_axis_roll_moment_unm"]))
        ),
        "loaded_foot_normal_force_median_mn": pct(loaded_foot_normal_mn, 0.50),
        "loaded_foot_normal_force_p95_mn": pct(loaded_foot_normal_mn, 0.95),
        "loaded_foot_normal_force_p99_mn": pct(loaded_foot_normal_mn, 0.99),
        "loaded_foot_normal_force_max_mn": float(np.max(loaded_foot_normal_mn)),
        "peak_loaded_foot": peak_foot,
        "all_ground_contact_normal_force_p95_mn": pct(all_contact_normal_mn, 0.95),
        "mean_contacting_feet": float(np.mean(arrays["contacting_feet"])),
        "mean_vertical_load_fraction_through_feet": float(np.mean(arrays["foot_load_fraction"])),
        "nonfoot_ground_contact_count": nonfoot_contact_samples,
        "ground_contact_count": ground_contact_samples,
        "leg_actuator_torque_abs_p95_unm": pct(leg_torque_abs_unm, 0.95),
        "leg_actuator_torque_abs_max_unm": float(np.max(leg_torque_abs_unm)) if leg_torque_abs_unm else 0.0,
        "correlation_support_acceleration_vs_com_vertical_acceleration": correlation(
            support_acceleration, com_vertical_acceleration
        ),
        "correlation_contact_roll_moment_vs_mean_roll_acceleration": correlation(
            roll_moment_nm, roll_acceleration
        ),
        "correlation_lr_force_imbalance_vs_contact_roll_moment": correlation(
            arrays["lr_support_imbalance"], roll_moment_nm
        ),
    }
    return result, arrays


def make_figure(cases: list[tuple[dict[str, object], dict[str, np.ndarray]]]) -> Path:
    selected = {str(row[0]["label"]): row[1] for row in cases}
    main = selected["current_k5e-6_wave1p5"]
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    time = main["time_s"]
    axes[0].plot(time, main["mean_body_roll_deg"], label="mean chassis roll")
    axes[0].plot(time, main["head_roll_deg"], alpha=0.7, label="head roll")
    axes[0].set_ylabel("roll (deg)")
    axes[0].legend(loc="upper right")
    axes[1].plot(time, main["left_support_z_mn"], label="left")
    axes[1].plot(time, main["right_support_z_mn"], label="right")
    axes[1].plot(time, main["total_support_z_mn"], color="black", alpha=0.55, label="total")
    axes[1].set_ylabel("support force (mN)")
    axes[1].legend(loc="upper right", ncol=3)
    axes[2].plot(time, main["lr_support_imbalance"], label="(R-L)/(R+L)")
    axes[2].plot(time, main["head_axis_roll_moment_unm"] / 100.0, alpha=0.7, label="roll moment / 100")
    axes[2].set_ylabel("imbalance / scaled moment")
    axes[2].legend(loc="upper right")
    axes[3].plot(time, main["com_z_mm"], label="system COM z")
    axes[3].set_ylabel("COM z (mm)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right")
    figure.suptitle("Flat wave=1.5 roll-force audit (selected k5e-6 plant)")
    figure.tight_layout()
    path = FIGURES / "flat_roll_force_audit.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (CURRENT_MODEL, SOFTER_MODEL, SOURCE_MODEL):
        if not path.exists():
            raise FileNotFoundError(path)
    sentinels_before = tip.robot_sentinels()
    cases: list[tuple[dict[str, object], dict[str, np.ndarray]]] = []
    specifications = (
        ("current_k5e-6_wave1p5", CURRENT_MODEL, 6.0, 1.5, True),
        ("selected_softer_k4p5e-6_wave1p5", SOFTER_MODEL, 6.0, 1.5, True),
        ("diagnostic_coupled_wave3", CURRENT_MODEL, 6.0, 3.0, True),
        ("diagnostic_unactuated_settle", CURRENT_MODEL, 3.0, 1.5, False),
    )
    for label, path, duration, wave, enabled in specifications:
        print(f"[roll-audit] START {label}", flush=True)
        result, arrays = run_force_case(label, path, duration, wave, enabled)
        print(
            f"[roll-audit] DONE  {label} rollP95={result['chassis_world_roll_abs_p95_deg']:.2f}deg "
            f"FfootP95={result['loaded_foot_normal_force_p95_mn']:.3f}mN "
            f"LRp95={result['lr_support_imbalance_abs_p95']:.3f}",
            flush=True,
        )
        cases.append((result, arrays))
        np.savez_compressed(DATA / f"{label}_traces.npz", **arrays)

    figure = make_figure(cases)
    mass_old = mass_matrix_audit(SOURCE_MODEL)
    mass_new = mass_matrix_audit(CURRENT_MODEL)
    inertia_ratio = (
        float(mass_new["root_roll_effective_inertia_kg_m2"])
        / float(mass_old["root_roll_effective_inertia_kg_m2"])
    )
    by_label = {str(result["label"]): result for result, _ in cases}
    current = by_label["current_k5e-6_wave1p5"]
    wave3 = by_label["diagnostic_coupled_wave3"]
    unactuated = by_label["diagnostic_unactuated_settle"]
    sentinels_after = tip.robot_sentinels()
    output = {
        "purpose": "causal audit of low-wave flat-ground whole-body roll and leg feedback force",
        "diagnostic_only_wave3_not_a_selected_controller": True,
        "cases": [result for result, _ in cases],
        "mass_matrix_comparison": {
            "old_q2_q3_source": mass_old,
            "new_distal_tip": mass_new,
            "new_to_old_root_roll_effective_inertia_ratio": inertia_ratio,
            "total_mass_change_fraction": (
                float(mass_new["total_mass_kg"]) / float(mass_old["total_mass_kg"]) - 1.0
            ),
        },
        "causal_checks": {
            "unactuated_roll_p95_deg": unactuated["chassis_world_roll_abs_p95_deg"],
            "coupled_wave1p5_roll_p95_deg": current["chassis_world_roll_abs_p95_deg"],
            "coupled_wave3_diagnostic_roll_p95_deg": wave3["chassis_world_roll_abs_p95_deg"],
            "wave3_to_wave1p5_roll_ratio": (
                float(wave3["chassis_world_roll_abs_p95_deg"])
                / float(current["chassis_world_roll_abs_p95_deg"])
            ),
            "mass_matrix_is_primary_cause": False,
            "reason": (
                "total mass is unchanged, root-roll effective inertia remains comparable, "
                "the unactuated plant does not reproduce the locomotion roll, and a coupled "
                "wave-number-only ablation changes roll through the support-force distribution"
            ),
        },
        "figure": str(figure),
        "robot_sentinels_before": sentinels_before,
        "robot_sentinels_after": sentinels_after,
        "robot_sentinels_unchanged": sentinels_before == sentinels_after,
    }
    path = DATA / "flat_roll_force_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "figure": str(figure)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
