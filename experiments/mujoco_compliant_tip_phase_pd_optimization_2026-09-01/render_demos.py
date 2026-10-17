"""Three-view and local comparisons from recorded compliant-tip poses."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / "experiments" / "mujoco_nearground_pd_search_2026-09-01"
DATA = ROOT / "outputs" / "data"
VIDEOS = ROOT / "outputs" / "videos"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


experiment = load(ROOT / "optimize.py", "compliant_phase_render_source")
np, mujoco = experiment.np, experiment.mujoco
sys.modules["run_search"] = experiment.near
source = load(PREVIOUS / "render_demo.py", "compliant_phase_source_renderer")
demo = source.demo

PAIRS = {
    2.5: (
        PREVIOUS / "outputs" / "common" / "legyaw_refine_k0p29_w2p5_t6_dt0p00025.json",
        DATA / "recovery_sobol_027_w2p5_t12_dt0p00025.json",
    ),
    3.0: (
        PREVIOUS / "outputs" / "common" / "basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json",
        DATA / "selected_w3_w3_t12_dt0p00025.json",
    ),
}


def read(path: Path):
    record = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as archive:
        trace = {key: archive[key] for key in archive.files}
    return record, trace


def recolor(state, rgba):
    chassis = set(map(int, state.index.chassis_body_ids))
    for geom_id in range(state.model.ngeom):
        if (
            int(state.model.geom_bodyid[geom_id]) in chassis
            and state.original_group[geom_id] == 2
            and state.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        ):
            state.original_rgba[geom_id] = rgba


def displacement(trace, time_s: float) -> float:
    index = int(np.argmin(np.abs(trace["t"] - time_s)))
    start = int(np.argmin(np.abs(trace["t"] - 1.5)))
    return experiment.opt.base.FORWARD_SIGN * float(
        trace["xpos"][index, :, 0].mean() - trace["xpos"][start, :, 0].mean()
    ) * 1000.0


def make(wave: float, zoom: bool) -> Path:
    old_path, new_path = PAIRS[wave]
    old_record, old_trace = read(old_path)
    new_record, new_trace = read(new_path)
    old_state, new_state = source.state(old_record), source.state(new_record)
    recolor(old_state, (0.38, 0.40, 0.43, 1.0))
    recolor(new_state, (0.10, 0.36, 0.66, 1.0))
    views = (
        [("SIDE LOCAL", 90, -14, 0.042, 9), ("OBLIQUE LOCAL", 135, -24, 0.042, 9)]
        if zoom
        else [
            ("FRONT", 180, -22, 0.12, None),
            ("SIDE", 90, -20, 0.105, None),
            ("OBLIQUE", 135, -28, 0.12, None),
        ]
    )
    start, end = (1.0, 6.0) if zoom else (0.0, 6.0)
    suffix = "local" if zoom else "three_views"
    output = VIDEOS / f"compliant_tip_phase_w{str(wave).replace('.', 'p')}_{suffix}.mp4"
    VIDEOS.mkdir(parents=True, exist_ok=True)
    writer = demo.imageio.get_writer(
        output, fps=30, codec="libx264", quality=9, macro_block_size=1
    )
    frames = int(round((end - start) * 30)) + 1
    try:
        for frame in range(frames):
            time_s = start + frame / 30.0
            canvas = demo.Image.new("RGB", (len(views) * 640, 880), (23, 28, 34))
            draw = demo.ImageDraw.Draw(canvas)
            source.visual.replay_pose(old_state, old_trace, time_s)
            source.visual.replay_pose(new_state, new_trace, time_s)
            fraction = float(new_record["phase_collocation"]["leg_phase_fraction"])
            offset = float(new_record["phase_collocation"]["leg_phase_offset_deg"])
            rows = (
                (
                    old_state,
                    old_trace,
                    "VALIDATED SHARED-PHASE REFERENCE",
                    "f=0 | long + half-dt certified",
                ),
                (
                    new_state,
                    new_trace,
                    "COMPLIANT TIP + POSITION-AWARE PHASE",
                    f"f={fraction:.3f}, offset={offset:.1f} deg | 6s nominal pass; robustness unresolved",
                ),
            )
            for row, (state, trace, label, status) in enumerate(rows):
                y = row * 440
                progress = (
                    f"D={displacement(trace, time_s):.1f} mm"
                    if time_s >= 1.5 else "SETTLING / CPG START"
                )
                draw.text(
                    (14, y + 5),
                    f"{label} | wave={wave:g} | t={time_s:.2f}s | {progress}",
                    font=demo.FONT_SMALL,
                    fill=(239, 239, 235),
                )
                draw.text(
                    (14, y + 28),
                    status + " | compliant tip unchanged | erasure OFF",
                    font=demo.FONT_SMALL,
                    fill=(239, 200, 150),
                )
                for column, (name, azimuth, elevation, distance, segment) in enumerate(views):
                    draw.text(
                        (column * 640 + 14, y + 52),
                        name + " | recorded 1 kHz pose; no smoothing",
                        font=demo.FONT_SMALL,
                        fill=(150, 211, 235),
                    )
                    canvas.paste(
                        source.panel(state, azimuth, elevation, distance, segment),
                        (column * 640, y + 80),
                    )
            writer.append_data(np.asarray(canvas))
            if frame in (0, min(90, frames - 1)):
                canvas.save(output.with_name(output.stem + f"_frame{frame:03d}.png"))
    finally:
        writer.close()
        old_state.close()
        new_state.close()
    reader = demo.imageio.get_reader(output)
    try:
        count = reader.count_frames()
        shape = list(reader.get_data(0).shape)
    finally:
        reader.close()
    output.with_suffix(".json").write_text(json.dumps({
        "path": str(output),
        "wave": wave,
        "source_records": [str(old_path), str(new_path)],
        "frames": count,
        "shape": shape,
        "start_s": start,
        "end_s": end,
        "physical_pose_recording_dt_s": 0.001,
        "smoothing": False,
        "contact_erasure": False,
        "tip_topology": "original compliant roll tip in both rows",
        "render_only_changes": "lighting, chassis albedo, shadow removal, world grid",
    }, indent=2))
    print(output, flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--waves", nargs="+", type=float, default=(2.5, 3.0))
    args = parser.parse_args()
    for requested in args.waves:
        make(requested, False)
        make(requested, True)
