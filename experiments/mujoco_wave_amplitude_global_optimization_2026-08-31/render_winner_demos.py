"""Render a synchronized three-view demo for every optimized wave winner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OPTIMIZER = ROOT / "run_wave_amplitude_global.py"
DATA = ROOT / "outputs" / "data"
VIDEOS = ROOT / "outputs" / "videos" / "wave_winners"
FIGURES = ROOT / "outputs" / "figures"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = load_module(OPTIMIZER, "centipede_wave_demo_optimizer")


def select_winners():
    final = json.loads(
        (DATA / "final_selection_summary.json").read_text(encoding="utf-8")
    )
    winners = {}
    for wave in opt.WAVES:
        item = final["waves"][str(wave)]
        selected = item["selected"]
        if selected is None:
            winners[wave] = {
                "row": item["diagnostic_best"],
                "feasible_selection": False,
                "random_demo_eligible": False,
                "label": "NO FEASIBLE GRID POINT - diagnostic only",
            }
        else:
            labels = {
                "long_safe_grid_global": "4 s global + 6 s pass",
                "long_safe_ranked_runner_up": "6 s-safe ranked runner-up",
                "no_6s_safe_among_4s_feasible": "4 s grid winner - 6 s gate fail",
            }
            winners[wave] = {
                "row": selected["grid_row"],
                "feasible_selection": True,
                "random_demo_eligible": bool(selected["long_safe"]),
                "label": labels[item["status"]],
            }
    return winners


def scene_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    return option


def render_demo(
    wave,
    winner,
    duration_s,
    fps,
    erasure_q,
    erasure_seed,
    feasible_selection=True,
    selection_label="finite-grid winner",
):
    evaluator = opt.FastEvaluator()
    model, data, index = evaluator.model, evaluator.data, evaluator.index
    controller = opt.ThreeAmplitudeController(
        model,
        index,
        wave,
        float(winner["body_yaw_amplitude_deg"]),
        float(winner["leg_yaw_amplitude_deg"]),
        float(winner["leg_pitch_amplitude_deg"]),
    )
    events, opportunities = opt.build_erasure_events(
        wave, erasure_q, erasure_seed, 1.5, duration_s
    )
    erasure = opt.ContactErasure(model, evaluator.segment_collision_geoms, events)
    original_rgba = model.geom_rgba.copy()
    renderer = mujoco.Renderer(model, height=360, width=640)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    cameras = []
    for _, azimuth, elevation in (
        ("front", 180.0, -10.0),
        ("side", 90.0, -12.0),
        ("oblique", 135.0, -18.0),
    ):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 0.17
        camera.azimuth = azimuth
        camera.elevation = elevation
        cameras.append(camera)
    option = scene_option()
    suffix = "nominal" if erasure_q == 0.0 else f"erase_q{erasure_q:.2f}_s{erasure_seed}"
    path = VIDEOS / f"wave_{wave:02d}_{suffix}_three_view.mp4"
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    next_frame = 0.0
    last_canvas = None
    erased_segment_time = np.zeros(19, dtype=float)
    dt = float(model.opt.timestep)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    try:
        while float(data.time) < duration_s - 0.5 * dt:
            active = erasure.update(float(data.time))
            if float(data.time) >= 1.5:
                erased_segment_time += active.astype(float) * dt
            model.geom_rgba[:] = original_rgba
            for segment in np.flatnonzero(active):
                model.geom_rgba[evaluator.segment_collision_geoms[int(segment)]] = (
                    1.0,
                    0.02,
                    0.02,
                    1.0,
                )
            controller.step(data)
            mujoco.mj_step(model, data)
            t = float(data.time)
            if t + 1e-12 < next_frame:
                continue
            center = opt.base.chassis_centroid(data, index)
            panels = []
            for camera in cameras:
                camera.lookat[:] = (float(center[0]), float(center[1]), 0.008)
                renderer.update_scene(data, camera=camera, scene_option=option)
                panels.append(Image.fromarray(renderer.render()))
            canvas = Image.new("RGB", (1920, 450), (17, 18, 20))
            draw = ImageDraw.Draw(canvas)
            for column, (label, panel) in enumerate(
                zip(("front", "side", "oblique"), panels)
            ):
                x = 640 * column
                canvas.paste(panel, (x, 90))
                draw.text((x + 16, 66), label, fill=(180, 225, 255))
            draw.text(
                (16, 12),
                f"Centipede Model | {selection_label} | wave={wave} | body yaw="
                f"{winner['body_yaw_amplitude_deg']:g} deg | leg yaw="
                f"{winner['leg_yaw_amplitude_deg']:g} deg | leg pitch="
                f"{winner['leg_pitch_amplitude_deg']:g} deg",
                fill=(255, 255, 255),
            )
            active_ids = np.flatnonzero(active).tolist()
            draw.text(
                (16, 38),
                f"flat | q={erasure_q:.2f} seed={erasure_seed} | "
                f"red=currently erased segment pairs {active_ids} | t={t:4.2f}s",
                fill=(255, 210, 90),
            )
            last_canvas = np.asarray(canvas)
            writer.append_data(last_canvas)
            next_frame += 1.0 / fps
    finally:
        model.geom_rgba[:] = original_rgba
        erasure.restore()
        writer.close()
        renderer.close()
    if last_canvas is None:
        raise RuntimeError("no demo frame rendered")
    return {
        "wave": wave,
        "feasible_selection": feasible_selection,
        "selection_label": selection_label,
        "erasure_q": erasure_q,
        "erasure_seed": erasure_seed,
        "events": len(events),
        "opportunities": opportunities,
        "realized_erased_load": float(erased_segment_time.sum())
        / max(19.0 * (duration_s - 1.5), 1e-12),
        "video": str(path),
        "last_frame": last_canvas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--random-waves", nargs="*", type=int, default=[3, 5, 7])
    parser.add_argument("--random-q", type=float, default=0.3)
    parser.add_argument("--random-seed", type=int, default=2)
    args = parser.parse_args()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    winners = select_winners()
    before = opt.tip.robot_sentinels()
    records = []
    nominal_frames = []
    for wave in opt.WAVES:
        print(f"rendering nominal wave {wave}", flush=True)
        selection = winners[wave]
        record = render_demo(
            wave,
            selection["row"],
            args.duration,
            args.fps,
            0.0,
            0,
            selection["feasible_selection"],
            selection["label"],
        )
        nominal_frames.append(Image.fromarray(record.pop("last_frame")))
        records.append(record)
    for wave in args.random_waves:
        selection = winners[wave]
        if not selection["random_demo_eligible"]:
            continue
        print(f"rendering random-contact wave {wave}", flush=True)
        record = render_demo(
            wave,
            selection["row"],
            args.duration,
            args.fps,
            args.random_q,
            args.random_seed,
            True,
            selection["label"],
        )
        record.pop("last_frame")
        records.append(record)

    sheet = Image.new("RGB", (1280, 2700), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(nominal_frames):
        resized = frame.resize((1280, 300))
        sheet.paste(resized, (0, 300 * index))
        wave = index + 1
        suffix = "" if winners[wave]["feasible_selection"] else " - DIAGNOSTIC / INFEASIBLE"
        draw.text(
            (8, 300 * index + 6),
            f"wave {wave}{suffix}",
            fill=(255, 255, 255),
        )
    sheet_path = FIGURES / "wave_winner_three_view_contact_sheet.png"
    sheet.save(sheet_path)

    integrity = []
    for record in records:
        reader = imageio.get_reader(record["video"])
        metadata = reader.get_meta_data()
        frames = reader.count_frames()
        reader.close()
        integrity.append(
            {
                "video": record["video"],
                "frames": int(frames),
                "fps": float(metadata.get("fps", float("nan"))),
            }
        )
    payload = {
        "purpose": "three-view flat demos of final selections and infeasible diagnostics",
        "model": str(opt.MODEL),
        "model_sha256": opt.tip.sha256(opt.MODEL),
        "duration_s": args.duration,
        "fps": args.fps,
        "records": records,
        "integrity": integrity,
        "contact_sheet": str(sheet_path),
        "local_tip_zoom_contact_sheet": str(
            ROOT.parent
            / "mujoco_clean_controller_roll_tuning_2026-08-31"
            / "outputs"
            / "figures"
            / "tip_zoom_baseline_vs_lp0525_contact_sheet.png"
        ),
        "robot_sentinels_unchanged": before == opt.tip.robot_sentinels(),
    }
    (DATA / "wave_winner_demo_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
