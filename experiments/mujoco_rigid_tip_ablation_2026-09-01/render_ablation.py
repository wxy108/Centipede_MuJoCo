"""Recorded-pose three-view and local videos for compliant-vs-rigid tip."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import run_ablation as ab


np, mujoco = ab.np, ab.mujoco
ROOT, DATA, PREVIOUS = ab.ROOT, ab.OUT, ab.PREVIOUS
sys.modules["run_search"] = ab.near
sys.modules["run_pd_pilot"] = ab.pilot
source_render = ab.load(PREVIOUS / "render_demo.py", "rigid_tip_source_renderer")
demo = source_render.demo
VIDEOS = ROOT / "outputs" / "videos"
TIP_VISUAL = re.compile(r"^link_leg_\d+_[LR]_tip_visual$")

PAIRS = {
    2.5: (
        PREVIOUS / "outputs" / "common" / "legyaw_refine_k0p29_w2p5_t6_dt0p00025.json",
        DATA / "rigid_tip_legyaw_refine_k0p29_w2p5_t6_dt0p00025.json",
    ),
    3.0: (
        PREVIOUS / "outputs" / "common" / "basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json",
        DATA / "rigid_tip_basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json",
    ),
}


class RigidState:
    def __init__(self, record):
        self.record = record
        self.model = mujoco.MjModel.from_xml_path(str(ab.MODEL))
        self.index = ab.RigidTipIndex(self.model)
        ab.tip.configure_model(self.model, self.index)
        ab.pilot.apply_impedance(self.model, self.index, record["config"])
        self.data = mujoco.MjData(self.model)
        self.ground = int(ab.tip.stage1.ground_geom_id(self.model))
        self.original_rgba = self.model.geom_rgba.copy()
        self.original_group = self.model.geom_group.copy()
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if TIP_VISUAL.fullmatch(name):
                self.original_rgba[geom_id] = (0.10, 0.72, 0.94, 1.0)
        collision = np.flatnonzero(
            (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        )
        collision = collision[collision != self.ground]
        self.model.geom_group[collision] = 5
        self.model.geom_rgba[self.ground] = (0.72, 0.76, 0.80, 1.0)
        self.model.geom_matid[self.ground] = -1
        self.model.light_diffuse[:] *= 0.35
        self.model.light_specular[:] = 0
        self.model.light_castshadow[:] = 0
        self.model.vis.headlight.ambient[:] = 0.30
        self.model.vis.headlight.diffuse[:] = 0.45
        self.model.vis.headlight.specular[:] = 0
        chassis = set(map(int, self.index.chassis_body_ids))
        for geom_id in range(self.model.ngeom):
            if (
                int(self.model.geom_bodyid[geom_id]) in chassis
                and self.original_group[geom_id] == 2
                and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            ):
                self.original_rgba[geom_id] = (0.48, 0.15, 0.065, 1.0)
        self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[0] = 1
        self.option.geomgroup[2] = 1
        self.option.sitegroup[:] = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def apply_colors(self):
        self.model.geom_rgba[:] = self.original_rgba
        self.model.geom_rgba[self.ground] = (0.72, 0.76, 0.80, 1.0)

    def close(self):
        self.renderer.close()


def read_record(path: Path):
    record = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as archive:
        trace = {key: archive[key] for key in archive.files}
    return record, trace


def replay_rigid(state: RigidState, trace, t: float):
    index = int(np.argmin(np.abs(trace["t"] - t)))
    state.data.qpos[:] = trace["qpos"][index]
    state.data.qvel[:] = 0
    state.data.time = float(trace["t"][index])
    mujoco.mj_forward(state.model, state.data)


def displacement(trace, t: float):
    index = int(np.argmin(np.abs(trace["t"] - t)))
    start = int(np.argmin(np.abs(trace["t"] - 1.5)))
    return ab.base.FORWARD_SIGN * float(
        trace["xpos"][index, :, 0].mean() - trace["xpos"][start, :, 0].mean()
    ) * 1000


def make(wave: float, zoom: bool):
    compliant_path, rigid_path = PAIRS[wave]
    compliant_record, compliant_trace = read_record(compliant_path)
    rigid_record, rigid_trace = read_record(rigid_path)
    compliant_state = source_render.state(compliant_record)
    rigid_state = RigidState(rigid_record)
    views = (
        [("SIDE LOCAL", 90, -14, 0.042, 9), ("OBLIQUE LOCAL", 135, -24, 0.042, 9)]
        if zoom
        else [
            ("FRONT", 180, -22, 0.12, None),
            ("SIDE", 90, -20, 0.105, None),
            ("OBLIQUE", 135, -28, 0.12, None),
        ]
    )
    start = 1.0 if zoom else 0.0
    end = 6.0
    suffix = "local" if zoom else "three_views"
    output = VIDEOS / f"rigid_tip_w{str(wave).replace('.', 'p')}_{suffix}.mp4"
    VIDEOS.mkdir(parents=True, exist_ok=True)
    writer = demo.imageio.get_writer(
        output, fps=30, codec="libx264", quality=9, macro_block_size=1
    )
    frames = int(round((end - start) * 30)) + 1
    try:
        for frame in range(frames):
            t = start + frame / 30
            canvas = demo.Image.new(
                "RGB", (len(views) * 640, 880), (23, 28, 34)
            )
            draw = demo.ImageDraw.Draw(canvas)
            source_render.visual.replay_pose(compliant_state, compliant_trace, t)
            replay_rigid(rigid_state, rigid_trace, t)
            for row, (state, record, trace, label, color_note) in enumerate(
                (
                    (
                        compliant_state,
                        compliant_record,
                        compliant_trace,
                        "COMPLIANT TIP REFERENCE",
                        "orange tip = passive roll DOF",
                    ),
                    (
                        rigid_state,
                        rigid_record,
                        rigid_trace,
                        "RIGID ONE-PIECE LEG",
                        "cyan tip = render label only; zero distal DOF",
                    ),
                )
            ):
                y = row * 440
                failed = (
                    []
                    if row == 0
                    else record["failed_applicable_gates"]
                )
                status = (
                    "REFERENCE: ALL ORIGINAL GATES PASSED"
                    if row == 0
                    else (
                        "APPLICABLE GATES PASSED"
                        if not failed
                        else "FAILED: " + ", ".join(failed)
                    )
                )
                progress = (
                    f"D={displacement(trace, t):.1f}mm"
                    if t >= 1.5
                    else "SETTLING / CPG START"
                )
                draw.text(
                    (14, y + 5),
                    f"{label} | wave={wave:g} | t={t:.2f}s | {progress}",
                    font=demo.FONT_SMALL,
                    fill=(239, 239, 235),
                )
                draw.text(
                    (14, y + 28),
                    f"{status} | {color_note}",
                    font=demo.FONT_SMALL,
                    fill=(239, 200, 150),
                )
                for column, (name, azimuth, elevation, distance, segment) in enumerate(views):
                    draw.text(
                        (column * 640 + 14, y + 52),
                        name + " | contact erasure OFF",
                        font=demo.FONT_SMALL,
                        fill=(150, 211, 235),
                    )
                    canvas.paste(
                        source_render.panel(
                            state, azimuth, elevation, distance, segment
                        ),
                        (column * 640, y + 80),
                    )
            writer.append_data(np.asarray(canvas))
            if frame in (0, min(90, frames - 1)):
                canvas.save(output.with_name(output.stem + f"_frame{frame:03d}.png"))
    finally:
        writer.close()
        compliant_state.close()
        rigid_state.close()
    reader = demo.imageio.get_reader(output)
    try:
        count = reader.count_frames()
        shape = list(reader.get_data(0).shape)
    finally:
        reader.close()
    metadata = {
        "path": str(output),
        "wave": wave,
        "frames": count,
        "shape": shape,
        "source_records": [str(compliant_path), str(rigid_path)],
        "physical_pose_recording_dt_s": 0.001,
        "smoothing": False,
        "contact_erasure": False,
        "render_only_tip_color": "compliant orange; rigid cyan",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=float, required=True, choices=[2.5, 3.0])
    parser.add_argument("--zoom", action="store_true")
    args = parser.parse_args()
    make(args.wave, args.zoom)
