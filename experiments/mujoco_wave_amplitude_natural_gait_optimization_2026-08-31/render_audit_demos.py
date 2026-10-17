"""High-clarity diagnostic demos with visual-only geometry and erasure marks."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_SITE = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
if TRUSTED_SITE.exists():
    sys.path.insert(0, str(TRUSTED_SITE))

import numpy as np

PROJECT_PACKAGES = PROJECT_ROOT / ".python_packages"
if PROJECT_PACKAGES.exists():
    sys.path.append(str(PROJECT_PACKAGES))

import imageio.v2 as imageio
import mujoco
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REJECTED_ROOT = (
    PROJECT_ROOT / "experiments" / "mujoco_wave_amplitude_global_optimization_2026-08-31"
)
OPTIMIZER = REJECTED_ROOT / "run_wave_amplitude_global.py"
VIDEOS = ROOT / "outputs" / "videos" / "audit_demos"
DATA = ROOT / "outputs" / "data"
WIDTH = 960
HEIGHT = 540
FPS = 30
DURATION_S = 4.5


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = load_module(OPTIMIZER, "centipede_audit_demo_optimizer")


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


FONT = font(24)
FONT_SMALL = font(19)
FONT_BOLD = font(27, bold=True)


def clean_case(wave: float) -> dict[str, float | str]:
    return {
        "label": "accepted clean amplitudes" if math.isclose(wave, 1.5) else "clean amplitudes transferred to wave 3 (diagnostic)",
        "wave": wave,
        "body": math.degrees(opt.base.BODY_AMPLITUDE_RAD * opt.tip.BODY_AMPLITUDE_SCALE),
        "leg_yaw": math.degrees(opt.base.LEG_YAW_AMPLITUDE_RAD * opt.tip.ctrl.LEG_YAW_AMPLITUDE_SCALE),
        "leg_pitch": math.degrees(opt.base.LEG_PITCH_AMPLITUDE_RAD * opt.tip.ctrl.LEG_PITCH_AMPLITUDE_SCALE) * 0.525,
    }


def rejected_w3() -> dict[str, float | str]:
    return {
        "label": "REJECTED displacement-grid wave 3",
        "wave": 3.0,
        "body": 16.0,
        "leg_yaw": 60.0,
        "leg_pitch": 42.0,
    }


def subtree_visual_geoms(model: mujoco.MjModel, segment: int) -> np.ndarray:
    roots = {
        opt.base.name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_leg_{segment}_{side}_0")
        for side in ("L", "R")
    }
    descendants = set()
    for body_id in range(model.nbody):
        cursor = body_id
        while cursor > 0:
            if cursor in roots:
                descendants.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return np.asarray(
        [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in descendants
            and int(model.geom_group[geom_id]) == 2
        ],
        dtype=int,
    )


class DemoState:
    def __init__(self, case, q: float = 0.0, seed: int = 2):
        self.case = case
        self.evaluator = opt.FastEvaluator()
        self.model = self.evaluator.model
        self.data = self.evaluator.data
        self.index = self.evaluator.index
        self.controller = opt.ThreeAmplitudeController(
            self.model,
            self.index,
            float(case["wave"]),
            float(case["body"]),
            float(case["leg_yaw"]),
            float(case["leg_pitch"]),
        )
        events, opportunities = opt.build_erasure_events(
            int(round(float(case["wave"]))), q, seed, 1.5, DURATION_S
        )
        self.events = events
        self.opportunities = opportunities
        self.erasure = opt.ContactErasure(
            self.model, self.evaluator.segment_collision_geoms, events
        )
        self.visual_geoms = [subtree_visual_geoms(self.model, i) for i in range(19)]
        self.original_rgba = self.model.geom_rgba.copy()
        self.original_group = self.model.geom_group.copy()
        self.ground = self.evaluator.ground
        # Move physical collision surfaces to an invisible render group.  This
        # changes visualization only; collision filtering remains untouched.
        collision = np.flatnonzero(
            (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        )
        collision = collision[collision != self.ground]
        self.model.geom_group[collision] = 5
        self.model.geom_rgba[self.ground] = (0.72, 0.76, 0.80, 1.0)
        self.model.vis.headlight.active = 1
        self.model.vis.headlight.ambient[:] = (0.42, 0.42, 0.42)
        self.model.vis.headlight.diffuse[:] = (0.72, 0.72, 0.72)
        self.model.vis.headlight.specular[:] = (0.08, 0.08, 0.08)
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[0] = 1
        self.option.geomgroup[2] = 1
        self.option.sitegroup[:] = 0
        self.active = np.zeros(19, dtype=bool)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def advance(self, target_t: float) -> None:
        dt = float(self.model.opt.timestep)
        while float(self.data.time) < target_t - 0.5 * dt:
            self.active = self.erasure.update(float(self.data.time))
            self.controller.step(self.data)
            mujoco.mj_step(self.model, self.data)

    def apply_colors(self) -> None:
        self.model.geom_rgba[:] = self.original_rgba
        self.model.geom_rgba[self.ground] = (0.72, 0.76, 0.80, 1.0)
        for segment in np.flatnonzero(self.active):
            geoms = self.visual_geoms[int(segment)]
            self.model.geom_rgba[geoms] = (1.0, 0.02, 0.02, 1.0)

    def render(self, azimuth: float, elevation: float, distance: float, focus_segment=None):
        self.apply_colors()
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        if focus_segment is None:
            center = opt.base.chassis_centroid(self.data, self.index)
        else:
            center = self.data.xpos[int(self.index.chassis_body_ids[focus_segment])]
        camera.lookat[:] = (float(center[0]), float(center[1]), max(0.002, float(center[2]) - 0.001))
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        return Image.fromarray(self.renderer.render())

    def close(self) -> None:
        self.model.geom_rgba[:] = self.original_rgba
        self.model.geom_group[:] = self.original_group
        self.erasure.restore()
        self.renderer.close()


def title(draw, case, y: int, prefix: str = ""):
    draw.text((22, y), f"{prefix}{case['label']}", font=FONT_BOLD, fill=(245, 248, 252))
    draw.text(
        (22, y + 34),
        f"wave={float(case['wave']):g} | body={float(case['body']):.1f} deg | leg yaw={float(case['leg_yaw']):.1f} deg | leg pitch={float(case['leg_pitch']):.1f} deg",
        font=FONT_SMALL,
        fill=(191, 211, 226),
    )


def write_video(path: Path, frames) -> int:
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=9, macro_block_size=1)
    count = 0
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
            count += 1
    finally:
        writer.close()
    return count


def three_view_frames(state: DemoState, erasure_label: bool = False):
    for frame_id in range(int(round(DURATION_S * FPS)) + 1):
        t = frame_id / FPS
        state.advance(t)
        panels = [
            state.render(180.0, -32.0, 0.16),
            state.render(90.0, -28.0, 0.16),
            state.render(135.0, -32.0, 0.16),
        ]
        canvas = Image.new("RGB", (3 * WIDTH, HEIGHT + 120), (20, 25, 30))
        draw = ImageDraw.Draw(canvas)
        for i, (name, panel) in enumerate(zip(("FRONT", "SIDE", "OBLIQUE"), panels)):
            canvas.paste(panel, (i * WIDTH, 120))
            draw.text((i * WIDTH + 22, 88), name, font=FONT_SMALL, fill=(160, 219, 246))
        title(draw, state.case, 12)
        ids = np.flatnonzero(state.active).tolist()
        if erasure_label:
            draw.rounded_rectangle((1960, 18, 2850, 100), radius=16, fill=(70, 8, 8), outline=(255, 70, 70), width=3)
            draw.rectangle((1990, 39, 2030, 79), fill=(255, 5, 5))
            draw.text((2050, 33), f"RED = CONTACT ERASED | active pairs: {ids or 'none'}", font=FONT, fill=(255, 235, 235))
        draw.text((2740, 88), f"t={t:4.2f}s", font=FONT_SMALL, fill=(230, 230, 230))
        yield canvas


def local_comparison_frames(clean: DemoState, rejected: DemoState):
    for frame_id in range(int(round(DURATION_S * FPS)) + 1):
        t = frame_id / FPS
        clean.advance(t)
        rejected.advance(t)
        panels = [
            clean.render(90.0, -22.0, 0.040, focus_segment=9),
            rejected.render(90.0, -22.0, 0.040, focus_segment=9),
            clean.render(135.0, -28.0, 0.040, focus_segment=9),
            rejected.render(135.0, -28.0, 0.040, focus_segment=9),
        ]
        canvas = Image.new("RGB", (2 * WIDTH, 2 * HEIGHT + 170), (20, 25, 30))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(panels[0], (0, 170))
        canvas.paste(panels[1], (WIDTH, 170))
        canvas.paste(panels[2], (0, 170 + HEIGHT))
        canvas.paste(panels[3], (WIDTH, 170 + HEIGHT))
        title(draw, clean.case, 12, "LEFT: ")
        title(draw, rejected.case, 78, "RIGHT: ")
        draw.text((22, 145), "SIDE ZOOM", font=FONT_SMALL, fill=(160, 219, 246))
        draw.text((22, 145 + HEIGHT), "OBLIQUE ZOOM", font=FONT_SMALL, fill=(160, 219, 246))
        draw.text((1750, 145), f"t={t:4.2f}s", font=FONT_SMALL, fill=(230, 230, 230))
        yield canvas


def main() -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    audit = []

    state = DemoState(clean_case(1.5))
    try:
        path = VIDEOS / "accepted_clean_w1p5_three_view_clear.mp4"
        count = write_video(path, three_view_frames(state))
        audit.append({"video": str(path), "frames": count, "purpose": "accepted clean three-view"})
    finally:
        state.close()

    state = DemoState(rejected_w3())
    try:
        path = VIDEOS / "rejected_w3_three_view_clear.mp4"
        count = write_video(path, three_view_frames(state))
        audit.append({"video": str(path), "frames": count, "purpose": "rejected gait negative control"})
    finally:
        state.close()

    clean = DemoState(clean_case(3.0))
    rejected = DemoState(rejected_w3())
    try:
        path = VIDEOS / "clean_vs_rejected_w3_local_zoom.mp4"
        count = write_video(path, local_comparison_frames(clean, rejected))
        audit.append({"video": str(path), "frames": count, "purpose": "matched-wave local zoom comparison"})
    finally:
        clean.close()
        rejected.close()

    erased = DemoState(clean_case(3.0), q=0.35, seed=2)
    try:
        path = VIDEOS / "clean_w3_q035_erasure_marked_three_view.mp4"
        count = write_video(path, three_view_frames(erased, erasure_label=True))
        audit.append(
            {
                "video": str(path),
                "frames": count,
                "purpose": "random-contact marking proof",
                "events": len(erased.events),
                "opportunities": erased.opportunities,
                "color_contract": "bright red visible leg subtree = contact currently erased",
            }
        )
    finally:
        erased.close()

    path = DATA / "audit_demo_manifest.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
