"""
batch_terrain_sweep.py
======================
Batch simulation sweep for terrain roughness ML dataset.

For each terrain configuration, runs N trials with randomised initial yaw
rotation. Each trial saves a compressed NPZ with full sensor timeseries and
metadata. A master index JSON is written at the end.

Directory layout produced:
    8-Dataset/
        sweep_index.json          ← master index of all trials
        flat/
            trial_000.npz
            trial_001.npz
            ...
        moderate/
            trial_000.npz
            ...
        rough/
            trial_000.npz
            ...

Each NPZ contains:
    time            (T,)          simulation time vector
    sensordata      (T, N_sens)   all MuJoCo sensordata at each recorded step
    sensor_names    (N_sens,)     sensor name strings (for column lookup)
    com_pos         (T, 3)        COM position
    com_vel         (T, 3)        COM velocity
    metadata        scalar        JSON string: terrain label, roughness_index,
                                  rotation_deg, seed, duration, dt_record

Usage:
    python batch_terrain_sweep.py
    python batch_terrain_sweep.py --n-trials 20 --duration 8
    python batch_terrain_sweep.py --dry-run
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(os.path.abspath(__file__)).parent
BASE_DIR    = SCRIPT_DIR.parent          # Centipede_MUJOCO-main (inner)
MODEL_PATH  = BASE_DIR / "3-Model" / "2-Centipede_FARMS" / "centipede.xml"
TERRAIN_DIR = BASE_DIR / "7-Terrain"
CONTROL_DIR = BASE_DIR / "1-Control"
DATASET_DIR = BASE_DIR / "8-Dataset"

# Add dirs to path for direct imports
sys.path.insert(0, str(CONTROL_DIR))
sys.path.insert(0, str(TERRAIN_DIR))

# ── Terrain sweep config ───────────────────────────────────────────────────────
TERRAIN_PNG_DIR = (BASE_DIR / "7-Terrain" / "terrain_test_with_pits"
                   / "terrain_test_with_pits")

TERRAINS = [
    {
        "label":           "flat",
        "png":             None,          # use default flat ground plane
        "roughness_index": 0.0,
    },
    {
        "label":           "rough",
        "png":             str(TERRAIN_PNG_DIR / "terrain_rough_perlin_roughness_1.0_centered_pits.png"),
        "roughness_index": 1.4,
    },
]

DEFAULT_N_TRIALS   = 50
DEFAULT_DURATION   = 10.0   # seconds
DEFAULT_DT_RECORD  = 0.01   # 100 Hz recording
DEFAULT_ROT_RANGE  = 180.0   # ± degrees → full 360° coverage


# ── Terrain generation & patching ─────────────────────────────────────────────

def patch_terrain(terrain_cfg: dict, dry_run: bool = False):
    """Patch XML terrain. Flat = restore plane geom. Others = swap hfield file."""
    import re

    label = terrain_cfg["label"]
    png   = terrain_cfg["png"]

    if dry_run:
        if png is None:
            print(f"  [DRY] Would restore flat plane geom")
        else:
            print(f"  [DRY] Would patch XML → {Path(png).name}")
        return

    xml_path = MODEL_PATH
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    if png is None:
        # Flat: switch hfield geom back to plane with proper size
        content = re.sub(
            r'(<geom\b[^>]*\bname="ground"[^>]*\b)type="hfield"',
            r'\1type="plane"',
            content)
        content = re.sub(r'\s*hfield="terrain"', '', content)
        # Restore size if missing
        if 'name="ground"' in content:
            if not re.search(r'name="ground"[^/]*\bsize=', content):
                content = re.sub(
                    r'(<geom\b[^>]*\bname="ground")',
                    r'\1 size="2 2 0.01"',
                    content)
        print(f"  Ground → flat plane")
    else:
        # Rough: update hfield file path
        folder  = Path(png).parent.name
        parent  = Path(png).parent.parent.name
        rel_png = f"../../../7-Terrain/{parent}/{folder}/{Path(png).name}"
        pattern = r'(<hfield\b[^>]*\bname="terrain"[^>]*\bfile=")[^"]*(")'
        content, n = re.subn(pattern, rf'\g<1>{rel_png}\g<2>', content)
        if n == 0:
            raise RuntimeError("No hfield 'terrain' in XML — run patch_farms_xml.py first")
        # Ensure ground geom is hfield type
        content = re.sub(
            r'(<geom\b[^>]*\bname="ground"[^>]*\b)type="plane"',
            r'\1type="hfield"',
            content)
        if 'hfield="terrain"' not in content.split('name="ground"')[1][:200]:
            content = re.sub(
                r'(<geom\b[^>]*\bname="ground")',
                r'\1 hfield="terrain"',
                content)
        print(f"  XML patched → {rel_png}")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(content)


# ── Rotation helper ────────────────────────────────────────────────────────────

def yaw_quaternion(angle_rad: float) -> np.ndarray:
    """Unit quaternion [w, x, y, z] for a rotation around Z by angle_rad."""
    return np.array([
        np.cos(angle_rad / 2), 0.0, 0.0, np.sin(angle_rad / 2)
    ])


def apply_initial_rotation(model, data, yaw_deg: float):
    """
    Apply a random yaw rotation to the freejoint initial state.
    The freejoint qpos layout is [x, y, z, qw, qx, qy, qz].
    """
    import mujoco

    # Find freejoint id
    fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if fj_id < 0:
        return   # no freejoint found

    qpos_adr = model.jnt_qposadr[fj_id]
    q = yaw_quaternion(np.deg2rad(yaw_deg))
    # qpos[adr+3:adr+7] = [qw, qx, qy, qz]
    data.qpos[qpos_adr + 3] = q[0]
    data.qpos[qpos_adr + 4] = q[1]
    data.qpos[qpos_adr + 5] = q[2]
    data.qpos[qpos_adr + 6] = q[3]
    mujoco.mj_forward(model, data)


# ── Single trial runner ────────────────────────────────────────────────────────

def run_trial(model, data, ctrl, duration: float, dt_record: float,
              yaw_deg: float, video_path: str = None,
              fps: int = 30) -> dict:
    """
    Run one simulation trial with given initial yaw rotation.
    Optionally renders an MP4 video.
    Returns dict with recorded arrays.
    """
    import mujoco
    import mediapy as media

    # Reset and apply rotation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    apply_initial_rotation(model, data, yaw_deg)

    n_steps      = int(duration / model.opt.timestep)
    step_every   = max(1, int(dt_record / model.opt.timestep))
    frame_every  = max(1, int(1.0 / (fps * model.opt.timestep)))
    n_sensors    = model.nsensordata

    # Pre-allocate sensor arrays
    max_frames = n_steps // step_every + 2
    times      = np.zeros(max_frames)
    sensordata = np.zeros((max_frames, n_sensors), dtype=np.float32)
    com_pos    = np.zeros((max_frames, 3), dtype=np.float32)
    com_vel    = np.zeros((max_frames, 3), dtype=np.float32)

    # Set up offscreen renderer if video requested
    renderer  = None
    frames    = []
    cam       = None
    scene_opt = None
    if video_path is not None:
        width, height = 640, 480   # reduced resolution for speed
        model.vis.global_.offwidth  = width
        model.vis.global_.offheight = height
        renderer  = mujoco.Renderer(model, height, width)
        cam       = mujoco.MjvCamera()
        cam.distance  = 0.2
        cam.azimuth   = 60
        cam.elevation = -35
        scene_opt = mujoco.MjvOption()
        scene_opt.geomgroup[0] = True   # terrain hfield — show
        scene_opt.geomgroup[1] = False  # unused
        scene_opt.geomgroup[2] = True   # visual meshes — show
        scene_opt.geomgroup[3] = False  # foot spheres — hide

    rec_frame = 0
    b0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Record sensor data
        if step % step_every == 0 and rec_frame < max_frames:
            times[rec_frame]      = data.time
            sensordata[rec_frame] = data.sensordata.copy()
            com_pos[rec_frame]    = data.subtree_com[b0_id].copy()
            com_vel[rec_frame]    = data.subtree_linvel[b0_id].copy()
            rec_frame += 1

        # Capture video frame
        if renderer is not None and step % frame_every == 0:
            cam.lookat[:] = data.subtree_com[b0_id]
            renderer.update_scene(data, cam, scene_option=scene_opt)
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        media.write_video(video_path, frames, fps=fps)

    return {
        "time":       times[:rec_frame],
        "sensordata": sensordata[:rec_frame],
        "com_pos":    com_pos[:rec_frame],
        "com_vel":    com_vel[:rec_frame],
        "n_frames":   rec_frame,
    }


def get_sensor_names(model) -> list[str]:
    """Return list of sensor names in sensordata order."""
    import mujoco
    names = []
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        # Each sensor may have multiple outputs (e.g. framepos=3 values)
        dim = model.sensor_dim[i]
        if dim == 1:
            names.append(name or f"sensor_{i}")
        else:
            for d in range(dim):
                names.append(f"{name or f'sensor_{i}'}_{d}")
    return names


# ── Main sweep ─────────────────────────────────────────────────────────────────

def run_sweep(n_trials: int, duration: float, dt_record: float,
              rot_range: float, dry_run: bool, resume: bool,
              save_video: bool = True):
    import mujoco
    from farms_controller import FARMSTravelingWaveController
    from farms_kinematics import FARMSModelIndex

    config_path = CONTROL_DIR / "farms_config.yaml"

    print("=" * 60)
    print("TERRAIN ROUGHNESS SWEEP — ML DATASET COLLECTION")
    print("=" * 60)
    print(f"  Terrains   : {[t['label'] for t in TERRAINS]}")
    print(f"  Trials     : {n_trials} per terrain  ({n_trials * len(TERRAINS)} total)")
    print(f"  Duration   : {duration}s per trial")
    print(f"  Recording  : {dt_record}s interval ({1/dt_record:.0f} Hz)")
    print(f"  Rotation   : ±{rot_range}° random yaw")
    print(f"  Output     : {DATASET_DIR}")
    print(f"  Dry run    : {dry_run}")
    print()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    master_index  = []
    rng           = np.random.default_rng(0)
    total_trials  = n_trials * len(TERRAINS)
    trial_count   = 0
    t_start_wall  = time.time()

    for terrain_cfg in TERRAINS:
        label     = terrain_cfg["label"]
        out_dir   = DATASET_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*50}")
        print(f"TERRAIN: {label.upper()}  "
              f"(roughness={terrain_cfg['roughness_index']:.3f})")
        print(f"{'─'*50}")

        # Generate terrain and patch XML
        try:
            patch_terrain(terrain_cfg, dry_run=dry_run)
        except Exception as e:
            print(f"  ERROR setting up terrain: {e}")
            continue

        if dry_run:
            for i in range(n_trials):
                trial_count += 1
                yaw = rng.uniform(-rot_range, rot_range)
                print(f"  [DRY] Trial {i:03d}: yaw={yaw:.1f}°")
            continue

        # Load model (after XML has been patched)
        print(f"  Loading model...")
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data  = mujoco.MjData(model)
        ctrl  = FARMSTravelingWaveController(model, str(config_path))

        sensor_names = get_sensor_names(model)
        print(f"  Sensors: {model.nsensordata} values  "
              f"({model.nsensor} sensors)")

        for i in range(n_trials):
            trial_count += 1
            out_path = out_dir / f"trial_{i:03d}.npz"

            # Resume: skip if file exists and is valid
            if resume and out_path.exists():
                print(f"  Trial {i:03d}/{n_trials-1}: SKIP (exists)")
                master_index.append({
                    "file": str(out_path.relative_to(DATASET_DIR)),
                    "label": label,
                    "trial": i,
                    "skipped": True,
                })
                continue

            yaw_deg    = float(rng.uniform(-rot_range, rot_range))
            video_path = None
            if save_video:
                vid_dir    = DATASET_DIR / "videos" / label
                vid_dir.mkdir(parents=True, exist_ok=True)
                video_path = str(vid_dir / f"trial_{i:03d}.mp4")
            t0 = time.time()

            try:
                result = run_trial(model, data, ctrl, duration,
                                   dt_record, yaw_deg,
                                   video_path=video_path)
                elapsed = time.time() - t0

                # Build metadata dict
                meta = {
                    "label":           label,
                    "roughness_index": terrain_cfg["roughness_index"],
                    "png":             terrain_cfg["png"] or "flat",
                    "trial":           i,
                    "yaw_deg":         round(yaw_deg, 3),
                    "duration":        duration,
                    "dt_record":       dt_record,
                    "n_frames":        result["n_frames"],
                    "timestamp":       datetime.now().isoformat(),
                }

                # Save NPZ
                np.savez_compressed(
                    out_path,
                    time        = result["time"],
                    sensordata  = result["sensordata"],
                    com_pos     = result["com_pos"],
                    com_vel     = result["com_vel"],
                    sensor_names= np.array(sensor_names),
                    metadata    = np.array(json.dumps(meta)),
                )

                master_index.append({
                    "file":            str(out_path.relative_to(DATASET_DIR)),
                    "label":           label,
                    "roughness_index": terrain_cfg["roughness_index"],
                    "trial":           i,
                    "yaw_deg":         round(yaw_deg, 3),
                    "n_frames":        result["n_frames"],
                })

                wall_elapsed = time.time() - t_start_wall
                eta = wall_elapsed / trial_count * (total_trials - trial_count)
                print(f"  Trial {i:03d}/{n_trials-1}: "
                      f"{result['n_frames']} frames  "
                      f"yaw={yaw_deg:+.1f}°  "
                      f"{elapsed:.1f}s  "
                      f"ETA {eta/60:.1f}min")

            except Exception as e:
                print(f"  Trial {i:03d}: ERROR — {e}")
                traceback.print_exc()

    # Save master index
    index_path = DATASET_DIR / "sweep_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "created":    datetime.now().isoformat(),
            "n_trials":   n_trials,
            "duration":   duration,
            "dt_record":  dt_record,
            "rot_range":  rot_range,
            "terrains":   TERRAINS,
            "trials":     master_index,
        }, f, indent=2)

    wall_total = time.time() - t_start_wall
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  Total trials : {len(master_index)}")
    print(f"  Wall time    : {wall_total/60:.1f} min")
    print(f"  Index        : {index_path}")
    print(f"  Dataset      : {DATASET_DIR}")
    print(f"{'='*60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Batch terrain sweep for ML dataset collection")
    p.add_argument("--n-trials",  type=int,   default=DEFAULT_N_TRIALS,
                   help=f"Trials per terrain (default: {DEFAULT_N_TRIALS})")
    p.add_argument("--duration",  type=float, default=DEFAULT_DURATION,
                   help=f"Simulation duration in seconds (default: {DEFAULT_DURATION})")
    p.add_argument("--dt-record", type=float, default=DEFAULT_DT_RECORD,
                   help=f"Recording interval in seconds (default: {DEFAULT_DT_RECORD})")
    p.add_argument("--rot-range", type=float, default=DEFAULT_ROT_RANGE,
                   help=f"±degrees initial yaw randomisation (default: {DEFAULT_ROT_RANGE})")
    p.add_argument("--dry-run",   action="store_true",
                   help="Preview without running simulations")
    p.add_argument("--no-video",  action="store_true",
                   help="Skip video rendering (faster)")
    p.add_argument("--resume",    action="store_true",
                   help="Skip trials where NPZ already exists")
    args = p.parse_args()

    run_sweep(
        n_trials   = args.n_trials,
        duration   = args.duration,
        dt_record  = args.dt_record,
        rot_range  = args.rot_range,
        dry_run    = args.dry_run,
        resume     = args.resume,
        save_video = not args.no_video,
    )


if __name__ == "__main__":
    main()
