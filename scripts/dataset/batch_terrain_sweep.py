"""
batch_terrain_sweep.py
======================
Batch simulation sweep for terrain roughness ML dataset.

Generates terrains inline via generate_terrain_multifreq.py with amplitudes
scaled to centipede morphology (~7 mm leg length). Includes real-time flip
detection that marks unstable trials in metadata.

Directory layout produced:
    8-Dataset/
        sweep_index.json
        flat/          trial_000.npz  ...
        very_gentle/   trial_000.npz  ...
        gentle/        trial_000.npz  ...
        moderate/      trial_000.npz  ...
        challenging/   trial_000.npz  ...
        rough/         trial_000.npz  ...
        videos/flat/   trial_000.mp4  ...

Each NPZ contains:
    time            (T,)          simulation time vector
    sensordata      (T, N_sens)   all MuJoCo sensordata at each recorded step
    sensor_names    (N_sens,)     sensor name strings
    com_pos         (T, 3)        COM position
    com_vel         (T, 3)        COM velocity
    metadata        scalar        JSON string with terrain, roughness,
                                  rotation, seed, duration, flip info

Usage:
    python batch_terrain_sweep.py --n-trials 50 --duration 10
    python batch_terrain_sweep.py --n-trials 1 --duration 5 --seed -1
    python batch_terrain_sweep.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(os.path.abspath(__file__)).parent
BASE_DIR    = SCRIPT_DIR.parent.parent    # Centipede_MUJOCO-main
MODEL_PATH  = BASE_DIR / "models" / "farms" / "centipede.xml"
TERRAIN_DIR = BASE_DIR / "terrain" / "generator"
CONTROL_DIR = BASE_DIR / "controllers" / "farms"
DATASET_DIR = BASE_DIR / "outputs" / "dataset"

sys.path.insert(0, str(CONTROL_DIR))
sys.path.insert(0, str(TERRAIN_DIR))


# ── Terrain roughness ladder ──────────────────────────────────────────────────
# Amplitudes scaled to centipede morphology:
#   Leg length ~ 7 mm, COM height ~ 5 mm
#   Peak terrain height ~ 2*low_amp + 2*high_amp
#   Target: levels from 0x to ~1x leg length max obstacle
#
# low_amp  = landscape undulation amplitude (metres)
# high_amp = surface roughness amplitude (metres)
# hf       = high-frequency band centre (cycles/metre)

TERRAINS = [
    {"label": "flat",        "low": 0.0,    "high": 0.0,    "hf": 12, "roughness_index": 0.0},
    {"label": "very_gentle", "low": 0.0005, "high": 0.0003, "hf": 12, "roughness_index": 0.05},
    {"label": "gentle",      "low": 0.001,  "high": 0.0005, "hf": 12, "roughness_index": 0.10},
    {"label": "moderate",    "low": 0.0015, "high": 0.0008, "hf": 15, "roughness_index": 0.15},
    {"label": "challenging", "low": 0.002,  "high": 0.001,  "hf": 18, "roughness_index": 0.20},
    {"label": "rough",       "low": 0.003,  "high": 0.0015, "hf": 20, "roughness_index": 0.30},
]

DEFAULT_N_TRIALS   = 50
DEFAULT_DURATION   = 10.0
DEFAULT_DT_RECORD  = 0.01   # 100 Hz
DEFAULT_ROT_RANGE  = 180.0

# Flip detection
FLIP_CONTACT_THRESH  = 0.01
FLIP_FRAC_THRESH     = 0.05
FLIP_SUSTAINED_STEPS = 50


# ── Terrain generation & XML patching ─────────────────────────────────────────

def generate_and_patch_terrain(terrain_cfg: dict, terrain_seed: int = 42,
                               dry_run: bool = False):
    """Generate terrain PNG via generate_terrain_multifreq, then patch XML."""
    label = terrain_cfg["label"]

    if dry_run:
        if terrain_cfg["low"] == 0 and terrain_cfg["high"] == 0:
            print(f"  [DRY] {label}: flat plane")
        else:
            print(f"  [DRY] {label}: low={terrain_cfg['low']:.4f} "
                  f"high={terrain_cfg['high']:.4f}")
        return

    xml_path = MODEL_PATH
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    if terrain_cfg["low"] == 0 and terrain_cfg["high"] == 0:
        # Flat: restore plane geom
        content = re.sub(
            r'(<geom\b[^>]*\bname="ground"[^>]*\b)type="hfield"',
            r'\1type="plane"', content)
        content = re.sub(r'\s*hfield="terrain"', '', content)
        if 'name="ground"' in content:
            if not re.search(r'name="ground"[^/]*\bsize=', content):
                content = re.sub(
                    r'(<geom\b[^>]*\bname="ground")',
                    r'\1 size="2 2 0.01"', content)
        print(f"  Ground -> flat plane")
    else:
        # Generate heightmap
        try:
            from generate_terrain_multifreq import (load_config,
                                                     generate_terrain,
                                                     save_terrain)
        except ImportError:
            raise RuntimeError(
                f"Cannot import generate_terrain_multifreq from {TERRAIN_DIR}")

        cfg = load_config(str(TERRAIN_DIR / "terrain_config.yaml"))
        hmap, meta = generate_terrain(
            cfg,
            low_amplitude=terrain_cfg["low"],
            high_amplitude=terrain_cfg["high"],
            high_freq_centre=terrain_cfg["hf"],
            seed=terrain_seed,
        )

        out_dir = TERRAIN_DIR / "terrain_output" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "1.png"
        save_terrain(hmap, meta, str(out_dir), cfg, preview=False)
        print(f"  Generated: {label}  "
              f"roughness={meta.get('roughness_index', '?'):.4f}")

        # Compute relative path from XML dir to PNG
        xml_dir = MODEL_PATH.parent
        try:
            rel_png = os.path.relpath(png_path, xml_dir).replace("\\", "/")
        except ValueError:
            rel_png = str(png_path).replace("\\", "/")

        pattern = r'(<hfield\b[^>]*\bname="terrain"[^>]*\bfile=")[^"]*(")'
        content, n = re.subn(pattern, rf'\g<1>{rel_png}\g<2>', content)
        if n == 0:
            raise RuntimeError(
                "No hfield 'terrain' in XML — run patch_farms_xml.py first")

        content = re.sub(
            r'(<geom\b[^>]*\bname="ground"[^>]*\b)type="plane"',
            r'\1type="hfield"', content)
        if 'hfield="terrain"' not in content.split('name="ground"')[1][:200]:
            content = re.sub(
                r'(<geom\b[^>]*\bname="ground")',
                r'\1 hfield="terrain"', content)
        print(f"  XML patched -> {rel_png}")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(content)


# ── Helpers ────────────────────────────────────────────────────────────────────

def yaw_quaternion(angle_rad: float) -> np.ndarray:
    return np.array([
        np.cos(angle_rad / 2), 0.0, 0.0, np.sin(angle_rad / 2)])


def apply_initial_rotation(model, data, yaw_deg: float):
    import mujoco
    fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if fj_id < 0:
        return
    qpos_adr = model.jnt_qposadr[fj_id]
    q = yaw_quaternion(np.deg2rad(yaw_deg))
    data.qpos[qpos_adr + 3: qpos_adr + 7] = q
    mujoco.mj_forward(model, data)


def _find_touch_indices(model):
    """Return sensordata indices for touch sensors."""
    import mujoco
    indices = []
    adr = 0
    for i in range(model.nsensor):
        dim = model.sensor_dim[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
        if "touch" in name.lower():
            indices.extend(range(adr, adr + dim))
        adr += dim
    return indices


def get_sensor_names(model) -> list:
    import mujoco
    names = []
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        dim = model.sensor_dim[i]
        if dim == 1:
            names.append(name or f"sensor_{i}")
        else:
            for d in range(dim):
                names.append(f"{name or f'sensor_{i}'}_{d}")
    return names


# ── Single trial runner ───────────────────────────────────────────────────────

def run_trial(model, data, ctrl, duration: float, dt_record: float,
              yaw_deg: float, touch_indices: list,
              video_path: str = None, fps: int = 30) -> dict:
    """Run one trial with flip detection."""
    import mujoco
    import mediapy as media

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    apply_initial_rotation(model, data, yaw_deg)

    n_steps     = int(duration / model.opt.timestep)
    step_every  = max(1, int(dt_record / model.opt.timestep))
    frame_every = max(1, int(1.0 / (fps * model.opt.timestep)))
    n_sensors   = model.nsensordata

    max_frames  = n_steps // step_every + 2
    times       = np.zeros(max_frames)
    sensordata  = np.zeros((max_frames, n_sensors), dtype=np.float32)
    com_pos     = np.zeros((max_frames, 3), dtype=np.float32)
    com_vel     = np.zeros((max_frames, 3), dtype=np.float32)

    # Video
    renderer = None
    frames   = []
    cam      = None
    scene_opt = None
    if video_path is not None:
        w, h = 640, 480
        model.vis.global_.offwidth  = w
        model.vis.global_.offheight = h
        renderer  = mujoco.Renderer(model, h, w)
        cam       = mujoco.MjvCamera()
        cam.distance  = 0.2
        cam.azimuth   = 60
        cam.elevation = -35
        scene_opt = mujoco.MjvOption()
        scene_opt.geomgroup[0] = True
        scene_opt.geomgroup[1] = False
        scene_opt.geomgroup[2] = True
        scene_opt.geomgroup[3] = False

    # Flip state
    n_touch = len(touch_indices)
    flip_run = 0
    flip_detected = False
    flip_time = None

    rec_frame = 0
    b0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Flip detection
        if n_touch > 0 and not flip_detected:
            touch_vals = data.sensordata[touch_indices]
            frac = np.sum(np.abs(touch_vals) > FLIP_CONTACT_THRESH) / n_touch
            if frac < FLIP_FRAC_THRESH:
                flip_run += 1
                if flip_run >= FLIP_SUSTAINED_STEPS:
                    flip_detected = True
                    flip_time = float(data.time)
            else:
                flip_run = 0

        # Record
        if step % step_every == 0 and rec_frame < max_frames:
            times[rec_frame]      = data.time
            sensordata[rec_frame] = data.sensordata.copy()
            com_pos[rec_frame]    = data.subtree_com[b0_id].copy()
            com_vel[rec_frame]    = data.subtree_linvel[b0_id].copy()
            rec_frame += 1

        # Video frame
        if renderer is not None and step % frame_every == 0:
            cam.lookat[:] = data.subtree_com[b0_id]
            renderer.update_scene(data, cam, scene_option=scene_opt)
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        media.write_video(video_path, frames, fps=fps)

    # Overall contact fraction
    if n_touch > 0 and rec_frame > 0:
        touch_all = sensordata[:rec_frame, touch_indices]
        contact_frac = float(np.mean(np.abs(touch_all) > FLIP_CONTACT_THRESH))
    else:
        contact_frac = -1.0

    return {
        "time":         times[:rec_frame],
        "sensordata":   sensordata[:rec_frame],
        "com_pos":      com_pos[:rec_frame],
        "com_vel":      com_vel[:rec_frame],
        "n_frames":     rec_frame,
        "flipped":      flip_detected,
        "flip_time":    flip_time,
        "contact_frac": contact_frac,
    }


# ── Main sweep ─────────────────────────────────────────────────────────────────

def run_sweep(n_trials: int, duration: float, dt_record: float,
              rot_range: float, dry_run: bool, resume: bool,
              save_video: bool = True, seed: int = 0):
    import mujoco
    from farms_controller import FARMSTravelingWaveController

    config_path = CONTROL_DIR / "farms_config.yaml"

    print("=" * 60)
    print("TERRAIN ROUGHNESS SWEEP — ML DATASET COLLECTION")
    print("=" * 60)
    print(f"  Terrains   : {[t['label'] for t in TERRAINS]}")
    print(f"  Trials     : {n_trials} per terrain  "
          f"({n_trials * len(TERRAINS)} total)")
    print(f"  Duration   : {duration}s per trial")
    print(f"  Recording  : {dt_record}s interval ({1/dt_record:.0f} Hz)")
    print(f"  Rotation   : +/-{rot_range} deg random yaw")
    print(f"  Output     : {DATASET_DIR}")
    print(f"  Seed       : {seed}")
    print(f"  Dry run    : {dry_run}")
    print()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    master_index    = []
    rng             = np.random.default_rng(None if seed == -1 else seed)
    total_trials    = n_trials * len(TERRAINS)
    trial_count     = 0
    n_flipped_total = 0
    t_start_wall    = time.time()

    for terrain_cfg in TERRAINS:
        label   = terrain_cfg["label"]
        out_dir = DATASET_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'_'*50}")
        print(f"TERRAIN: {label.upper()}  "
              f"(roughness={terrain_cfg['roughness_index']:.3f}  "
              f"low={terrain_cfg['low']:.4f}  "
              f"high={terrain_cfg['high']:.4f})")
        print(f"{'_'*50}")

        try:
            generate_and_patch_terrain(terrain_cfg, terrain_seed=42,
                                       dry_run=dry_run)
        except Exception as e:
            print(f"  ERROR setting up terrain: {e}")
            traceback.print_exc()
            continue

        if dry_run:
            for i in range(n_trials):
                trial_count += 1
                yaw = rng.uniform(-rot_range, rot_range)
                print(f"  [DRY] Trial {i:03d}: yaw={yaw:.1f} deg")
            continue

        print(f"  Loading model...")
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data  = mujoco.MjData(model)
        ctrl  = FARMSTravelingWaveController(model, str(config_path))

        sensor_names  = get_sensor_names(model)
        touch_indices = _find_touch_indices(model)
        print(f"  Sensors: {model.nsensordata} values  "
              f"({model.nsensor} sensors, {len(touch_indices)} touch)")

        n_flipped_terrain = 0

        for i in range(n_trials):
            trial_count += 1
            out_path = out_dir / f"trial_{i:03d}.npz"

            if resume and out_path.exists():
                print(f"  Trial {i:03d}/{n_trials-1}: SKIP (exists)")
                master_index.append({
                    "file": str(out_path.relative_to(DATASET_DIR)),
                    "label": label, "trial": i, "skipped": True,
                })
                continue

            yaw_deg    = float(rng.uniform(-rot_range, rot_range))
            video_path = None
            if save_video:
                vid_dir = DATASET_DIR / "videos" / label
                vid_dir.mkdir(parents=True, exist_ok=True)
                video_path = str(vid_dir / f"trial_{i:03d}.mp4")
            t0 = time.time()

            try:
                result = run_trial(model, data, ctrl, duration,
                                   dt_record, yaw_deg, touch_indices,
                                   video_path=video_path)
                elapsed = time.time() - t0

                if result["flipped"]:
                    n_flipped_terrain += 1
                    n_flipped_total += 1

                meta = {
                    "label":           label,
                    "roughness_index": terrain_cfg["roughness_index"],
                    "low_amp":         terrain_cfg["low"],
                    "high_amp":        terrain_cfg["high"],
                    "high_freq":       terrain_cfg["hf"],
                    "trial":           i,
                    "yaw_deg":         round(yaw_deg, 3),
                    "duration":        duration,
                    "dt_record":       dt_record,
                    "n_frames":        result["n_frames"],
                    "flipped":         result["flipped"],
                    "flip_time":       result["flip_time"],
                    "contact_frac":    round(result["contact_frac"], 4),
                    "timestamp":       datetime.now().isoformat(),
                }

                np.savez_compressed(
                    out_path,
                    time         = result["time"],
                    sensordata   = result["sensordata"],
                    com_pos      = result["com_pos"],
                    com_vel      = result["com_vel"],
                    sensor_names = np.array(sensor_names),
                    metadata     = np.array(json.dumps(meta)),
                )

                master_index.append({
                    "file":            str(out_path.relative_to(DATASET_DIR)),
                    "label":           label,
                    "roughness_index": terrain_cfg["roughness_index"],
                    "trial":           i,
                    "yaw_deg":         round(yaw_deg, 3),
                    "n_frames":        result["n_frames"],
                    "flipped":         result["flipped"],
                    "contact_frac":    round(result["contact_frac"], 4),
                })

                wall_elapsed = time.time() - t_start_wall
                eta = wall_elapsed / trial_count * (total_trials - trial_count)
                flip_tag = "  FLIP" if result["flipped"] else ""
                print(f"  Trial {i:03d}/{n_trials-1}: "
                      f"{result['n_frames']}fr  "
                      f"yaw={yaw_deg:+.1f}  "
                      f"contact={result['contact_frac']:.0%}  "
                      f"{elapsed:.1f}s  "
                      f"ETA {eta/60:.1f}min{flip_tag}")

            except Exception as e:
                print(f"  Trial {i:03d}: ERROR -- {e}")
                traceback.print_exc()

        if n_flipped_terrain > 0:
            print(f"  >> {n_flipped_terrain}/{n_trials} flipped on {label}")

    # Save index
    index_path = DATASET_DIR / "sweep_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "created":    datetime.now().isoformat(),
            "n_trials":   n_trials,
            "duration":   duration,
            "dt_record":  dt_record,
            "rot_range":  rot_range,
            "seed":       seed,
            "terrains":   TERRAINS,
            "n_flipped":  n_flipped_total,
            "trials":     master_index,
        }, f, indent=2)

    wall_total = time.time() - t_start_wall
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  Total trials : {len(master_index)}")
    print(f"  Flipped      : {n_flipped_total}")
    print(f"  Wall time    : {wall_total/60:.1f} min")
    print(f"  Index        : {index_path}")
    print(f"  Dataset      : {DATASET_DIR}")
    print(f"{'='*60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Batch terrain sweep for ML dataset collection")
    p.add_argument("--n-trials",  type=int,   default=DEFAULT_N_TRIALS)
    p.add_argument("--duration",  type=float, default=DEFAULT_DURATION)
    p.add_argument("--dt-record", type=float, default=DEFAULT_DT_RECORD)
    p.add_argument("--rot-range", type=float, default=DEFAULT_ROT_RANGE)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--no-video",  action="store_true")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--seed",      type=int,   default=0,
                   help="RNG seed for yaw angles (-1 = random)")
    args = p.parse_args()

    run_sweep(
        n_trials   = args.n_trials,
        duration   = args.duration,
        dt_record  = args.dt_record,
        rot_range  = args.rot_range,
        dry_run    = args.dry_run,
        resume     = args.resume,
        save_video = not args.no_video,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
