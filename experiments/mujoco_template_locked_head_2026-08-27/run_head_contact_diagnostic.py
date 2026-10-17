"""Replay the selected models to separate head-ground impacts from head-joint motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_locked_head as locked


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "data" / "head_contact_tip_diagnostic.json"
MODELS = {
    "articulated": {
        "flat": locked.TIP_ROOT / "models" / "centipede_flat_tip_k4p5e-6_r38_bodyk6_distal_roll.xml",
        "rough": locked.TIP_ROOT / "models" / "centipede_rough_tip_k4p5e-6_r38_bodyk6_distal_roll.xml",
    },
    "locked_m29": {
        "flat": ROOT / "models" / "centipede_flat_locked_head_m29p0.xml",
        "rough": ROOT / "models" / "centipede_rough_locked_head_m29p0.xml",
    },
}


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    values = values * scale
    return {
        "count": int(len(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def signed_stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] * scale
    if not len(values):
        return {
            "count": 0,
            "min": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def contact_summary(
    times: np.ndarray,
    flags: np.ndarray,
    forces_n: np.ndarray,
    penetrations_m: np.ndarray,
    start_s: float,
    dt: float,
) -> dict[str, object]:
    mask = times >= start_s
    selected = flags[mask]
    indices = np.flatnonzero(mask & flags)
    episodes = []
    if len(indices):
        split_points = np.flatnonzero(np.diff(indices) > 1) + 1
        for group in np.split(indices, split_points):
            episodes.append(
                {
                    "start_s": float(times[group[0]]),
                    "end_s": float(times[group[-1]] + dt),
                    "duration_ms": float(len(group) * dt * 1000.0),
                    "normal_force_max_mn": float(np.max(forces_n[group]) * 1000.0),
                    "penetration_max_mm": float(np.max(penetrations_m[group]) * 1000.0),
                }
            )
    return {
        "evaluation_start_s": start_s,
        "contact_step_count": int(np.count_nonzero(selected)),
        "contact_time_fraction": float(np.mean(selected)) if len(selected) else float("nan"),
        "episode_count": len(episodes),
        "total_contact_duration_ms": float(np.count_nonzero(selected) * dt * 1000.0),
        "normal_force_mn": stats(forces_n[mask & flags], 1000.0),
        "penetration_mm": stats(penetrations_m[mask & flags], 1000.0),
        "episodes": episodes,
    }


def run_case(
    topology: str,
    kind: str,
    model_path: Path,
    duration_s: float,
    first_pitch_kp_scale: float = 1.0,
    first_pitch_kd_scale: float = 1.0,
) -> dict[str, object]:
    tip = locked.tip
    model = mujoco.MjModel.from_xml_path(str(model_path))
    index = tip.LeanIndex(model)
    first_pitch_active_position = index.active_joint_names.index("joint_pitch_body_1")
    baseline_first_pitch_kp = float(index.kp[first_pitch_active_position])
    baseline_first_pitch_kd = float(index.kv[first_pitch_active_position])
    index.kp[first_pitch_active_position] *= float(first_pitch_kp_scale)
    index.kv[first_pitch_active_position] *= float(first_pitch_kd_scale)
    tip.configure_model(model, index)
    controller = tip.LeanController(
        model, index, tip.FLAT_WAVE if kind == "flat" else tip.ROUGH_WAVE
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ground = tip.stage1.ground_geom_id(model)
    head_geom = tip.base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link_0_collision")
    link1_geom = tip.base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link_1_collision")
    foot_geoms = np.asarray(
        [
            tip.base.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{segment}_{side}_collision")
            for segment in range(19)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_lookup = {int(geom): offset for offset, geom in enumerate(foot_geoms)}

    head_pitch_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "joint_pitch_body_0"
    )
    head_pitch_qpos = (
        int(model.jnt_qposadr[head_pitch_id]) if head_pitch_id >= 0 else None
    )
    head_pitch_dof = (
        int(model.jnt_dofadr[head_pitch_id]) if head_pitch_id >= 0 else None
    )
    first_pitch_id = tip.base.name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "joint_pitch_body_1"
    )
    first_pitch_qpos = int(model.jnt_qposadr[first_pitch_id])
    first_pitch_dof = int(model.jnt_dofadr[first_pitch_id])

    times = []
    head_flags = []
    link1_flags = []
    head_forces = []
    link1_forces = []
    head_penetrations = []
    link1_penetrations = []
    pitch_q = []
    pitch_qd = []
    pitch_qacc = []
    first_pitch_q = []
    first_pitch_qd = []
    first_pitch_qacc = []
    first_pitch_torque = []
    root_angular_speed = []
    tip_angles = []
    loaded_tip_angles = []
    instantaneous_tip_max = []
    contacting_feet = []
    foot_penetrations_m = []
    world_roll_abs_deg = []
    local_roll_abs_deg = []
    chassis_center_z = []
    start_center = None
    end_center = None
    previous_center = None
    path_length = 0.0
    minimum_upright = float("inf")
    maximum_spine_tortuosity = 0.0
    active_torque_abs_max = 0.0
    dt = float(model.opt.timestep)
    metric_stride = max(1, int(round(0.001 / dt)))
    step_index = 0

    while data.time < duration_s - 0.5 * dt:
        commanded_torque = controller.step(data)
        active_torque_abs_max = max(
            active_torque_abs_max, float(np.max(np.abs(commanded_torque)))
        )
        mujoco.mj_step(model, data)
        head_contact = False
        link1_contact = False
        head_force = 0.0
        link1_force = 0.0
        head_penetration = 0.0
        link1_penetration = 0.0
        loaded_feet: set[int] = set()
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if ground not in (geom1, geom2):
                continue
            other = geom2 if geom1 == ground else geom1
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, force)
            normal_force = abs(float(force[0]))
            penetration = max(0.0, -float(contact.dist))
            if other == head_geom:
                head_contact = True
                head_force += normal_force
                head_penetration = max(head_penetration, penetration)
            if other == link1_geom:
                link1_contact = True
                link1_force += normal_force
                link1_penetration = max(link1_penetration, penetration)
            foot_offset = foot_lookup.get(other)
            if foot_offset is not None:
                loaded_feet.add(foot_offset)
                if data.time >= 1.5:
                    foot_penetrations_m.append(penetration)

        tip_abs = np.abs(data.qpos[index.tip_qpos_addresses])
        tip_angles.append(tip_abs.copy())
        instantaneous_tip_max.append(float(np.max(tip_abs)))
        if loaded_feet and data.time >= 1.5:
            loaded_tip_angles.extend(tip_abs[sorted(loaded_feet)].tolist())
        if data.time >= 1.5:
            contacting_feet.append(len(loaded_feet))
        if data.time >= 1.5 and step_index % metric_stride == 0:
            center = tip.base.chassis_centroid(data, index)
            posture = tip.base.posture_metrics(data, index)
            if start_center is None:
                start_center = center.copy()
            if previous_center is not None:
                path_length += float(np.linalg.norm(center[:2] - previous_center[:2]))
            previous_center = center.copy()
            end_center = center.copy()
            minimum_upright = min(
                minimum_upright, float(posture["root_upright_cos"])
            )
            maximum_spine_tortuosity = max(
                maximum_spine_tortuosity, float(posture["spine_tortuosity"])
            )
            world_roll_abs_deg.extend(
                np.abs(tip.signed_world_roll_deg(data, index.chassis_body_ids)).tolist()
            )
            local_roll_abs_deg.extend(
                np.abs(np.degrees(data.qpos[index.roll_qpos_addresses])).tolist()
            )
            chassis_center_z.append(float(center[2]))
        times.append(float(data.time))
        head_flags.append(head_contact)
        link1_flags.append(link1_contact)
        head_forces.append(head_force)
        link1_forces.append(link1_force)
        head_penetrations.append(head_penetration)
        link1_penetrations.append(link1_penetration)
        root_angular_speed.append(float(np.linalg.norm(data.qvel[3:6])))
        first_pitch_q.append(float(data.qpos[first_pitch_qpos]))
        first_pitch_qd.append(float(data.qvel[first_pitch_dof]))
        first_pitch_qacc.append(float(data.qacc[first_pitch_dof]))
        first_pitch_torque.append(
            float(commanded_torque[first_pitch_active_position])
        )
        if head_pitch_qpos is not None and head_pitch_dof is not None:
            pitch_q.append(float(data.qpos[head_pitch_qpos]))
            pitch_qd.append(float(data.qvel[head_pitch_dof]))
            pitch_qacc.append(float(data.qacc[head_pitch_dof]))
        step_index += 1

    times_array = np.asarray(times)
    head_array = np.asarray(head_flags, dtype=bool)
    link1_array = np.asarray(link1_flags, dtype=bool)
    head_force_array = np.asarray(head_forces)
    link1_force_array = np.asarray(link1_forces)
    head_pen_array = np.asarray(head_penetrations)
    link1_pen_array = np.asarray(link1_penetrations)
    tip_array = np.asarray(tip_angles)
    post = times_array >= 1.5
    tip_post = tip_array[post]
    tip_limit_rad = float(np.median(model.jnt_range[index.tip_joint_ids, 1]))
    tip_p95_rad = float(np.quantile(tip_post.reshape(-1), 0.95))
    lever_m = 0.001175
    tip_result = {
        "joint_count": int(len(index.tip_joint_ids)),
        "limit_deg": math.degrees(tip_limit_rad),
        "all_tip_samples_abs_deg": stats(tip_post.reshape(-1), 180.0 / math.pi),
        "loaded_tip_samples_abs_deg": stats(
            np.asarray(loaded_tip_angles),
            180.0 / math.pi,
        ),
        "instantaneous_max_across_38_abs_deg": stats(
            np.asarray(instantaneous_tip_max)[post], 180.0 / math.pi
        ),
        "per_tip_time_p95_abs_deg": stats(
            np.quantile(tip_post, 0.95, axis=0), 180.0 / math.pi
        ),
        "stop_fraction": float(np.mean(tip_post >= tip_limit_rad - math.radians(0.25))),
        "p95_tip_endpoint_chord_estimate_mm": float(
            2000.0 * lever_m * math.sin(0.5 * tip_p95_rad)
        ),
        "lever_joint_to_foot_center_mm": lever_m * 1000.0,
    }

    pitch_result: dict[str, object]
    if pitch_q:
        q = np.asarray(pitch_q)[post]
        qd = np.abs(np.asarray(pitch_qd)[post])
        qacc = np.abs(np.asarray(pitch_qacc)[post])
        contacts = head_array[post]
        link1_contacts = link1_array[post]
        radius = max(1, int(round(0.005 / dt)))
        near_contact = np.convolve(
            contacts.astype(int), np.ones(2 * radius + 1, dtype=int), mode="same"
        ) > 0
        near_link1_contact = np.convolve(
            link1_contacts.astype(int), np.ones(2 * radius + 1, dtype=int), mode="same"
        ) > 0
        high_acceleration = qacc >= np.quantile(qacc, 0.99)
        pitch_result = {
            "present": True,
            "angle_deg": stats(q, 180.0 / math.pi),
            "speed_abs_rad_s": stats(qd),
            "acceleration_abs_rad_s2": stats(qacc),
            "speed_abs_during_head_contact_rad_s": stats(qd[contacts]),
            "acceleration_abs_during_head_contact_rad_s2": stats(qacc[contacts]),
            "speed_abs_during_link1_contact_rad_s": stats(qd[link1_contacts]),
            "acceleration_abs_during_link1_contact_rad_s2": stats(qacc[link1_contacts]),
            "top_1pct_acceleration_within_5ms_of_head_contact_fraction": float(
                np.mean(near_contact[high_acceleration])
            )
            if np.any(high_acceleration)
            else float("nan"),
            "top_1pct_acceleration_within_5ms_of_link1_contact_fraction": float(
                np.mean(near_link1_contact[high_acceleration])
            )
            if np.any(high_acceleration)
            else float("nan"),
        }
    else:
        pitch_result = {
            "present": False,
            "relative_head_link1_pitch_is_structurally_constant": True,
        }

    root_speed = np.asarray(root_angular_speed)[post]
    first_q = np.asarray(first_pitch_q)[post]
    first_qd = np.asarray(first_pitch_qd)[post]
    first_qacc = np.asarray(first_pitch_qacc)[post]
    first_torque = np.asarray(first_pitch_torque)[post]
    first_contact = link1_array[post]
    angle_scale = 180.0 / math.pi
    first_pitch_result = {
        "joint": "joint_pitch_body_1",
        "connection": "link1_to_link2",
        "target_deg": 0.0,
        "baseline_kp_nm_per_rad": baseline_first_pitch_kp,
        "baseline_kd_nms_per_rad": baseline_first_pitch_kd,
        "kp_scale": float(first_pitch_kp_scale),
        "kd_scale": float(first_pitch_kd_scale),
        "effective_kp_nm_per_rad": float(index.kp[first_pitch_active_position]),
        "effective_kd_nms_per_rad": float(index.kv[first_pitch_active_position]),
        "joint_angle_signed_deg": signed_stats(first_q, angle_scale),
        "front_link1_relative_to_link2_pitch_signed_deg": signed_stats(
            -first_q, angle_scale
        ),
        "joint_angle_abs_deg": stats(np.abs(first_q), angle_scale),
        "joint_angle_signed_during_link1_contact_deg": signed_stats(
            first_q[first_contact], angle_scale
        ),
        "joint_angle_signed_without_link1_contact_deg": signed_stats(
            first_q[~first_contact], angle_scale
        ),
        "contact_minus_noncontact_mean_joint_angle_deg": float(
            (np.mean(first_q[first_contact]) - np.mean(first_q[~first_contact]))
            * angle_scale
        )
        if np.any(first_contact) and np.any(~first_contact)
        else float("nan"),
        "speed_abs_rad_s": stats(np.abs(first_qd)),
        "acceleration_abs_rad_s2": stats(np.abs(first_qacc)),
        "controller_torque_abs_micronewton_m": stats(
            np.abs(first_torque), 1.0e6
        ),
    }
    displacement = (
        tip.base.FORWARD_SIGN * float(end_center[0] - start_center[0])
        if start_center is not None and end_center is not None
        else float("nan")
    )
    return {
        "topology": topology,
        "kind": kind,
        "model": str(model_path),
        "duration_s": duration_s,
        "timestep_s": dt,
        "warning_counts": locked.tip.stage1.warning_counts(data),
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "armature_nonzero_count": int(
            np.count_nonzero(np.abs(model.dof_armature) > 1e-16)
        ),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "forward_displacement_m": displacement,
        "path_efficiency": displacement / max(path_length, 1e-12),
        "minimum_root_upright_cos": minimum_upright,
        "maximum_spine_tortuosity": maximum_spine_tortuosity,
        "chassis_world_roll_abs_p95_deg": float(
            np.quantile(world_roll_abs_deg, 0.95)
        ),
        "body_roll_local_abs_p95_deg": float(
            np.quantile(local_roll_abs_deg, 0.95)
        ),
        "chassis_center_z_rms_mm": float(np.std(chassis_center_z) * 1000.0),
        "mean_contacting_feet": float(np.mean(contacting_feet)),
        "foot_contact_penetration_p95_mm": float(
            np.quantile(foot_penetrations_m, 0.95) * 1000.0
        ),
        "active_torque_abs_max_nm": active_torque_abs_max,
        "head_ground_contact_all_time": contact_summary(
            times_array, head_array, head_force_array, head_pen_array, 0.0, dt
        ),
        "head_ground_contact_after_1p5s": contact_summary(
            times_array, head_array, head_force_array, head_pen_array, 1.5, dt
        ),
        "link1_ground_contact_after_1p5s": contact_summary(
            times_array, link1_array, link1_force_array, link1_pen_array, 1.5, dt
        ),
        "root_angular_speed_abs_rad_s_after_1p5s": stats(root_speed),
        "root_angular_speed_during_head_contact_rad_s": stats(root_speed[head_array[post]]),
        "root_angular_speed_during_link1_contact_rad_s": stats(root_speed[link1_array[post]]),
        "head_pitch_joint": pitch_result,
        "first_spine_pitch": first_pitch_result,
        "tip_roll": tip_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--topologies", default="articulated,locked_m29")
    parser.add_argument("--terrains", default="flat,rough")
    args = parser.parse_args()
    topologies = [value.strip() for value in args.topologies.split(",") if value.strip()]
    terrains = [value.strip() for value in args.terrains.split(",") if value.strip()]
    if any(value not in MODELS for value in topologies):
        raise ValueError(f"Unknown topology in {topologies}")
    if any(value not in ("flat", "rough") for value in terrains):
        raise ValueError(f"Unknown terrain in {terrains}")
    output = OUTPUT
    if topologies != ["articulated", "locked_m29"] or terrains != ["flat", "rough"]:
        suffix = "_".join(topologies + terrains)
        output = OUTPUT.with_name(f"head_contact_tip_diagnostic_{suffix}.json")
    before = locked.tip.robot_sentinels()
    rows = []
    for topology in topologies:
        for kind in terrains:
            print(f"[head-contact] START {topology} {kind}", flush=True)
            row = run_case(topology, kind, MODELS[topology][kind], args.duration)
            rows.append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "duration_s": args.duration,
                        "rows": rows,
                        "robot_sentinels_before": before,
                        "robot_sentinels_current": locked.tip.robot_sentinels(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[head-contact] DONE  {topology} {kind} "
                f"headEpisodes={row['head_ground_contact_after_1p5s']['episode_count']} "
                f"tipP95={row['tip_roll']['all_tip_samples_abs_deg']['p95']:.2f}deg",
                flush=True,
            )
    result = {
        "purpose": "separate ground-contact impulses from articulated head-joint motion",
        "duration_s": args.duration,
        "evaluation_start_s": 1.5,
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": locked.tip.robot_sentinels(),
    }
    result["robot_sentinels_unchanged"] = (
        result["robot_sentinels_before"] == result["robot_sentinels_after"]
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "row_count": len(rows), "robot_sentinels_unchanged": result["robot_sentinels_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
