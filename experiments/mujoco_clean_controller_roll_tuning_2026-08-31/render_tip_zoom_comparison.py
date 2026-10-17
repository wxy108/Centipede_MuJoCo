"""Render synchronized local tip-roll comparisons for the Centipede Model.

The left panel uses the frozen locked-head flat controller amplitude.  The
right panel uses the selected clean flat profile (leg-pitch amplitude 0.525x).
No model or controller source file is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parents[1]
CLEAN_RUNNER = ROOT / "run_clean_roll_tuning.py"
MODEL = (
    MODEL_ROOT
    / "experiments"
    / "mujoco_template_locked_head_2026-08-27"
    / "models"
    / "centipede_flat_locked_head_m29p0.xml"
)
VIDEOS = ROOT / "outputs" / "videos" / "tip_zoom"
FIGURES = ROOT / "outputs" / "figures"
DATA = ROOT / "outputs" / "data"
SEGMENT = 9


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


clean = load_module(CLEAN_RUNNER, "centipede_tip_zoom_clean")
tip = clean.tip


def name2id(model, objtype, name: str) -> int:
    value = mujoco.mj_name2id(model, objtype, name)
    if value < 0:
        raise RuntimeError(f"Missing {name}")
    return int(value)


def highlight_local_leg(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if f"link_leg_{SEGMENT}_L" in name:
            model.geom_rgba[geom_id] = (0.05, 0.95, 1.0, 1.0)
        elif f"link_leg_{SEGMENT}_R" in name:
            model.geom_rgba[geom_id] = (1.0, 0.15, 0.75, 1.0)


def make_state(pitch_scale: float):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    index = tip.LeanIndex(model)
    tip.configure_model(model, index)
    controller = tip.LeanController(model, index, tip.FLAT_WAVE)
    controller.leg_amplitudes[1] *= float(pitch_scale)
    highlight_local_leg(model)
    mujoco.mj_forward(model, data)
    body_ids = np.asarray(
        [
            name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{SEGMENT}"),
            name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_leg_{SEGMENT}_L_tip"),
            name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_leg_{SEGMENT}_R_tip"),
        ],
        dtype=int,
    )
    tip_qpos = np.asarray(
        [
            model.jnt_qposadr[
                name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"joint_tip_{SEGMENT}_{side}_roll",
                )
            ]
            for side in ("L", "R")
        ],
        dtype=int,
    )
    return model, data, controller, body_ids, tip_qpos


def scene_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    return option


def render_panel(renderer, camera, option, data, body_ids) -> np.ndarray:
    center = np.mean(data.xpos[body_ids], axis=0)
    camera.lookat[:] = center
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


def render_view(
    duration_s: float,
    view: str,
    azimuth: float,
    elevation: float,
) -> tuple[Path, np.ndarray]:
    states = [make_state(1.0), make_state(0.525)]
    renderers = [
        mujoco.Renderer(state[0], height=600, width=640) for state in states
    ]
    for renderer in renderers:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    cameras = []
    for _ in states:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.028 if view != "front" else 0.024
        camera.azimuth = azimuth
        camera.elevation = elevation
        cameras.append(camera)
    options = [scene_option(), scene_option()]
    path = VIDEOS / f"tip_zoom_baseline_vs_lp0525_{view}.mp4"
    writer = imageio.get_writer(
        path, fps=30, codec="libx264", quality=8, macro_block_size=1
    )
    next_frame = 0.0
    last_canvas = None
    try:
        while float(states[0][1].time) < duration_s - 1e-12:
            for model, data, controller, _, _ in states:
                controller.step(data)
                mujoco.mj_step(model, data)
            t = float(states[0][1].time)
            if t + 1e-12 < next_frame:
                continue
            panels = [
                render_panel(renderer, camera, option, state[1], state[3])
                for renderer, camera, option, state in zip(
                    renderers, cameras, options, states
                )
            ]
            canvas = Image.new("RGB", (1280, 720), (15, 15, 18))
            draw = ImageDraw.Draw(canvas)
            for column, (panel, state, scale) in enumerate(
                zip(panels, states, (1.0, 0.525))
            ):
                x = 640 * column
                canvas.paste(Image.fromarray(panel), (x, 90))
                angles = np.degrees(state[1].qpos[state[4]])
                title = "locked baseline" if column == 0 else "selected clean profile"
                draw.text((x + 18, 14), title, fill=(255, 255, 255))
                draw.text(
                    (x + 18, 38),
                    f"flat leg-pitch scale={scale:g} | segment {SEGMENT} | "
                    f"tip roll L={angles[0]:+5.1f} deg R={angles[1]:+5.1f} deg",
                    fill=(255, 220, 90),
                )
                draw.text(
                    (x + 18, 62),
                    "cyan=L leg, magenta=R leg; red markers are live contacts",
                    fill=(160, 235, 255),
                )
            draw.line((640, 0, 640, 720), fill=(255, 255, 255), width=2)
            draw.text((1190, 692), f"t={t:4.2f}s", fill=(255, 255, 255))
            last_canvas = np.asarray(canvas)
            writer.append_data(last_canvas)
            next_frame += 1.0 / 30.0
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    if last_canvas is None:
        raise RuntimeError("No frame rendered")
    return path, last_canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    before = tip.robot_sentinels()
    outputs = []
    frames = []
    for view, azimuth, elevation in (
        ("side", 90.0, -5.0),
        ("front", 180.0, -5.0),
        ("oblique", 135.0, -12.0),
    ):
        path, frame = render_view(args.duration, view, azimuth, elevation)
        outputs.append(str(path))
        frames.append(Image.fromarray(frame).resize((960, 540)))
    sheet = Image.new("RGB", (960, 1620), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame, (0, 540 * index))
    sheet_path = FIGURES / "tip_zoom_baseline_vs_lp0525_contact_sheet.png"
    sheet.save(sheet_path)
    integrity = []
    for path in outputs:
        reader = imageio.get_reader(path)
        metadata = reader.get_meta_data()
        frames_count = reader.count_frames()
        reader.close()
        integrity.append(
            {
                "video": path,
                "frames": int(frames_count),
                "fps": float(metadata.get("fps", float("nan"))),
            }
        )
    after = tip.robot_sentinels()
    payload = {
        "purpose": "local synchronized tip-roll comparison",
        "segment_pair": SEGMENT,
        "duration_s": args.duration,
        "left_panel": {"label": "locked baseline", "flat_leg_pitch_scale": 1.0},
        "right_panel": {
            "label": "selected clean profile",
            "flat_leg_pitch_scale": 0.525,
        },
        "model": str(MODEL),
        "model_sha256": tip.sha256(MODEL),
        "videos": outputs,
        "integrity": integrity,
        "contact_sheet": str(sheet_path),
        "robot_sentinels_unchanged": before == after,
    }
    output = DATA / "tip_zoom_comparison_audit.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
