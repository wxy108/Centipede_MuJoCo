"""Flat-ground spawn and upright-stability gate for the detailed FARMS model.

This gate intentionally runs before any erasure sweep.  It distinguishes
first-frame spawn penetration from a genuine in-motion buckle and records a
synchronized oblique/side/front/top video for visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import run_pipeline as experiment


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "stability_gate"
FARMS_CONTROLLER_DIR = experiment.CONTROLLER_DIR
FARMS_CONFIG = experiment.CONTROLLER_CONFIG
DEFAULT_OFFSETS_M = (0.0, 0.0035, 0.005, 0.010, 0.015, 0.025)
ROOT_TILT_LIMIT_DEG = 45.0
SEGMENT_TILT_LIMIT_DEG = 75.0
SPINE_JOINT_LIMIT_DEG = 35.0
INITIAL_PENETRATION_LIMIT_M = 1.0e-4
DYNAMIC_PENETRATION_LIMIT_M = 3.5e-3


def reset_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    index: experiment.ModelIndex,
    spawn_offset_m: float,
) -> None:
    mujoco.mj_resetData(model, data)
    # MuJoCo resolves link_body_0's baked 28.3 mm position into qpos0.  Add to
    # that absolute default; replacing qpos[2] would incorrectly lower it.
    data.qpos[2] += float(spawn_offset_m)
    knees = index.leg_positions[:, :, 3].reshape(-1)
    data.qpos[index.qpos_addresses[knees]] = math.pi / 6.0
    mujoco.mj_forward(model, data)


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float | int]:
    floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    floor_distances: list[float] = []
    self_distances: list[float] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        target = (
            floor_distances
            if floor_id in (int(contact.geom1), int(contact.geom2))
            else self_distances
        )
        target.append(float(contact.dist))
    return {
        "floor_contact_count": len(floor_distances),
        "self_contact_count": len(self_distances),
        "minimum_floor_distance_m": min(floor_distances, default=float("nan")),
        "minimum_self_distance_m": min(self_distances, default=float("nan")),
    }


def scan_offsets(
    offsets_m: tuple[float, ...], model_path: Path
) -> list[dict[str, float | int | bool]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    index = experiment.ModelIndex(model)
    rows: list[dict[str, float | int | bool]] = []
    for offset_m in offsets_m:
        data = mujoco.MjData(model)
        reset_pose(model, data, index, offset_m)
        contacts = contact_summary(model, data)
        centroid = experiment.chassis_centroid(data, index)
        floor_distance = float(contacts["minimum_floor_distance_m"])
        rows.append(
            {
                "model": str(model_path),
                "spawn_offset_m": offset_m,
                "absolute_root_height_m": float(data.xpos[index.head_body_id, 2]),
                "chassis_centroid_height_m": float(centroid[2]),
                **contacts,
                "initial_penetration_free": bool(
                    (math.isnan(floor_distance) or floor_distance >= -INITIAL_PENETRATION_LIMIT_M)
                    and int(contacts["self_contact_count"]) == 0
                ),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "spawn_clearance_scan.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def euler_roll_pitch(rotation: np.ndarray) -> tuple[float, float]:
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[2, 1]), float(rotation[2, 2])),
    )
    return roll, pitch


def tilt_deg(rotation: np.ndarray) -> float:
    up_dot = float(np.clip(rotation[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(up_dot))


def camera_views(center: np.ndarray) -> list[tuple[str, mujoco.MjvCamera]]:
    settings = (
        ("oblique", 135.0, -35.0, 0.18),
        ("side", 90.0, -22.0, 0.18),
        ("front", 0.0, -18.0, 0.13),
        ("top", 90.0, -90.0, 0.18),
    )
    views: list[tuple[str, mujoco.MjvCamera]] = []
    for label, azimuth, elevation, distance in settings:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (float(center[0]), float(center[1]), 0.015)
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        views.append((label, camera))
    return views


def render_multiview(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    center: np.ndarray,
    t: float,
    wave: int,
) -> np.ndarray:
    panels: list[Image.Image] = []
    for label, camera in camera_views(center):
        renderer.update_scene(data, camera=camera)
        panel = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(panel)
        draw.rectangle((8, 8, 105, 34), fill=(24, 24, 24))
        draw.text((15, 14), label, fill=(255, 255, 255))
        panels.append(panel)
    canvas = Image.new("RGB", (1280, 720), (0, 0, 0))
    for panel, xy in zip(panels, ((0, 0), (640, 0), (0, 360), (640, 360))):
        canvas.paste(panel, xy)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((12, 675, 360, 708), fill=(24, 24, 24))
    draw.text((22, 685), f"flat, no erasure | w={wave} | t={t:5.2f} s", fill=(255, 255, 255))
    return np.asarray(canvas)


def run_gate(
    wave: int,
    duration_s: float,
    spawn_offset_m: float,
    model_path: Path,
    controller_kind: str,
    render_fps: int = 20,
    render_video: bool = True,
) -> dict:
    os.environ.setdefault("MUJOCO_GL", "wgl")
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    index = experiment.ModelIndex(model)
    if controller_kind == "isaac_locked":
        controller = experiment.LockedTravelingWaveController(model, index, wave)
    elif controller_kind == "local_farms":
        controller = experiment.make_farms_controller(model, wave)
    else:
        raise ValueError(f"Unknown controller {controller_kind}")
    reset_pose(model, data, index, spawn_offset_m)

    initial_contacts = contact_summary(model, data)
    initial_center = experiment.chassis_centroid(data, index).copy()
    rows: list[dict[str, float | int]] = []
    finite = True
    next_sample_t = 0.0
    next_frame_t = 0.0
    output_stem = (
        f"{model_path.stem}_{controller_kind}_w{wave}_"
        f"offset_{spawn_offset_m * 1000:.1f}mm_T{duration_s:g}s"
    )
    video_path = OUT / f"{output_stem}_multiview.mp4"
    renderer = None
    writer = None
    if render_video:
        renderer = mujoco.Renderer(model, height=360, width=640)
        writer = imageio.get_writer(
            video_path, fps=render_fps, codec="libx264", quality=8, macro_block_size=None
        )
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None

    try:
        while data.time < duration_s - 0.5 * model.opt.timestep:
            if controller_kind == "isaac_locked":
                controller.step(data)
            else:
                controller.step(model, data, float(data.time))
            before = float(data.time)
            mujoco.mj_step(model, data)
            if (
                float(data.time) <= before
                or not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                finite = False
                break

            if data.time + 1e-12 >= next_sample_t:
                rotations = data.xmat[index.chassis_body_ids].reshape(-1, 3, 3)
                root_rotation = rotations[0]
                root_roll, root_pitch = euler_roll_pitch(root_rotation)
                segment_tilts = np.asarray([tilt_deg(rotation) for rotation in rotations])
                contacts = contact_summary(model, data)
                center = experiment.chassis_centroid(data, index)
                nonhead_pitch = data.qpos[index.qpos_addresses[index.pitch_positions[1:]]]
                spine_roll = data.qpos[index.qpos_addresses[index.roll_positions]]
                rows.append(
                    {
                        "time_s": float(data.time),
                        "centroid_x_m": float(center[0]),
                        "centroid_y_m": float(center[1]),
                        "centroid_z_m": float(center[2]),
                        "minimum_chassis_z_m": float(np.min(data.xpos[index.chassis_body_ids, 2])),
                        "root_roll_deg": math.degrees(root_roll),
                        "root_pitch_deg": math.degrees(root_pitch),
                        "root_tilt_deg": tilt_deg(root_rotation),
                        "max_segment_tilt_deg": float(np.max(segment_tilts)),
                        "max_body_pitch_joint_deg": float(np.max(np.abs(np.degrees(nonhead_pitch)))),
                        "max_body_roll_joint_deg": float(np.max(np.abs(np.degrees(spine_roll)))),
                        **contacts,
                    }
                )
                next_sample_t += 0.01

            if (
                renderer is not None
                and writer is not None
                and data.time + 1e-12 >= next_frame_t
            ):
                center = experiment.chassis_centroid(data, index)
                frame = render_multiview(renderer, data, center, float(data.time), wave)
                writer.append_data(frame)
                if first_frame is None:
                    first_frame = frame.copy()
                last_frame = frame.copy()
                next_frame_t += 1.0 / render_fps
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    trace_path = OUT / f"{output_stem}_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    if first_frame is not None:
        imageio.imwrite(OUT / f"{output_stem}_first.png", first_frame)
    if last_frame is not None:
        imageio.imwrite(OUT / f"{output_stem}_last.png", last_frame)

    values = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    initial_floor_distance = float(initial_contacts["minimum_floor_distance_m"])
    initial_clear = bool(
        (
            math.isnan(initial_floor_distance)
            or initial_floor_distance >= -INITIAL_PENETRATION_LIMIT_M
        )
        and int(initial_contacts["self_contact_count"]) == 0
    )
    result = {
        "model": str(model_path),
        "controller": controller_kind,
        "wave": wave,
        "duration_s": duration_s,
        "spawn_offset_m": spawn_offset_m,
        "absolute_initial_root_height_m": float(initial_center[2]),
        **{f"initial_{key}": value for key, value in initial_contacts.items()},
        "initial_penetration_free": initial_clear,
        "finite": finite,
        "max_abs_root_roll_deg": float(np.max(np.abs(values["root_roll_deg"]))),
        "max_abs_root_pitch_deg": float(np.max(np.abs(values["root_pitch_deg"]))),
        "max_root_tilt_deg": float(np.max(values["root_tilt_deg"])),
        "max_segment_tilt_deg": float(np.max(values["max_segment_tilt_deg"])),
        "max_body_pitch_joint_deg": float(np.max(values["max_body_pitch_joint_deg"])),
        "max_body_roll_joint_deg": float(np.max(values["max_body_roll_joint_deg"])),
        "minimum_chassis_z_m": float(np.min(values["minimum_chassis_z_m"])),
        "minimum_dynamic_floor_distance_m": float(
            np.nanmin(values["minimum_floor_distance_m"])
        ),
        "maximum_dynamic_self_contact_count": int(np.max(values["self_contact_count"])),
        "minimum_dynamic_self_distance_m": float(
            np.nanmin(values["minimum_self_distance_m"])
            if np.isfinite(values["minimum_self_distance_m"]).any()
            else float("nan")
        ),
        "forward_displacement_m": float(values["centroid_x_m"][-1] - values["centroid_x_m"][0]),
        "lateral_displacement_m": float(values["centroid_y_m"][-1] - values["centroid_y_m"][0]),
        "video": str(video_path) if render_video else "",
        "trace": str(trace_path),
    }
    result["pass"] = bool(
        result["initial_penetration_free"]
        and result["finite"]
        and result["max_abs_root_roll_deg"] <= ROOT_TILT_LIMIT_DEG
        and result["max_abs_root_pitch_deg"] <= ROOT_TILT_LIMIT_DEG
        and result["max_segment_tilt_deg"] <= SEGMENT_TILT_LIMIT_DEG
        and result["max_body_pitch_joint_deg"] <= SPINE_JOINT_LIMIT_DEG
        and result["max_body_roll_joint_deg"] <= SPINE_JOINT_LIMIT_DEG
        and result["minimum_chassis_z_m"] > 0.0
        and result["minimum_dynamic_floor_distance_m"] >= -DYNAMIC_PENETRATION_LIMIT_M
        and result["maximum_dynamic_self_contact_count"] == 0
    )
    result["limits"] = {
        "initial_penetration_m": INITIAL_PENETRATION_LIMIT_M,
        "dynamic_penetration_m": DYNAMIC_PENETRATION_LIMIT_M,
        "root_roll_pitch_deg": ROOT_TILT_LIMIT_DEG,
        "segment_tilt_deg": SEGMENT_TILT_LIMIT_DEG,
        "body_pitch_roll_joint_deg": SPINE_JOINT_LIMIT_DEG,
    }
    (OUT / f"{output_stem}_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(values["time_s"], values["root_roll_deg"], label="root roll")
    axes[0].plot(values["time_s"], values["root_pitch_deg"], label="root pitch")
    axes[0].axhline(ROOT_TILT_LIMIT_DEG, color="0.5", linestyle="--", linewidth=1)
    axes[0].axhline(-ROOT_TILT_LIMIT_DEG, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(values["time_s"], 1000.0 * values["centroid_z_m"], label="centroid")
    axes[1].plot(values["time_s"], 1000.0 * values["minimum_chassis_z_m"], label="minimum chassis")
    axes[1].set_ylabel("Height (mm)")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(values["time_s"], values["max_body_pitch_joint_deg"], label="body pitch")
    axes[2].plot(values["time_s"], values["max_body_roll_joint_deg"], label="body roll")
    axes[2].axhline(SPINE_JOINT_LIMIT_DEG, color="0.5", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Max |joint| (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    fig.savefig(OUT / f"{output_stem}_stability.png", dpi=220)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--wave", type=int, default=3)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--spawn-offset-mm", type=float, default=0.0)
    parser.add_argument("--model", type=Path, default=experiment.MODEL_PATH)
    parser.add_argument(
        "--controller",
        choices=("isaac_locked", "local_farms"),
        default="local_farms",
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    model_path = args.model.resolve()
    if args.scan:
        print(json.dumps(scan_offsets(DEFAULT_OFFSETS_M, model_path), indent=2))
        return
    report = run_gate(
        args.wave,
        args.duration,
        args.spawn_offset_mm / 1000.0,
        model_path,
        args.controller,
        render_video=not args.no_video,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
