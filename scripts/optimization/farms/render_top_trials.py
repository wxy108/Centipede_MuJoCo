#!/usr/bin/env python3
"""
render_top_trials.py — Render videos of the top-N trials from a MuJoCo
three-eval optimization run.

Reads `all_trials.csv` from a three-eval run directory, sorts non-buckled
trials by `cost`, picks the top N, and produces:
  - one MP4 per trial in `<run_dir>/top_videos/rank_NN_trial_TTT_cost_CCCC.mp4`
  - one YAML per trial in `<run_dir>/top_yamls/rank_NN_trial_TTT.yaml`

Each video is a SINGLE rollout on the chosen terrain at the chosen body-wave
frequency. By default we render rough λ=18mm at 1 Hz (matches eval B of the
three-eval optimizer — the slowest, most stable evaluation). Override with
--terrain flat / --rough-frequency / --duration to inspect other regimes.

Important caveat: the three-eval optimizer averaged cost over 3 evals (flat 2Hz,
rough 1Hz, rough 2Hz). A SINGLE replay can only show ONE of those regimes. The
cost number in the filename is the OPTIMIZER's average cost, not the cost of
the regime you're watching. To see all three regimes, run this script three
times with different --terrain / --rough-frequency / --duration combinations.

Usage:
    python scripts/optimization/farms/render_top_trials.py \\
        outputs/optimization/impedance_three_eval_*/all_trials.csv \\
        --top 10 --duration 8

    # Flat ground at 2Hz (eval A — bio-comparable CoT regime):
    python scripts/optimization/farms/render_top_trials.py \\
        outputs/optimization/impedance_three_eval_*/all_trials.csv \\
        --top 10 --terrain flat --rough-frequency 2.0 --duration 8

    # Rough λ=18mm at 2Hz (eval C — hardest regime):
    python scripts/optimization/farms/render_top_trials.py \\
        outputs/optimization/impedance_three_eval_*/all_trials.csv \\
        --top 10 --terrain rough --rough-frequency 2.0 --duration 8
"""
import argparse
import copy
import csv
import glob
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))

import mujoco                                                # noqa: E402
from impedance_controller import ImpedanceTravelingWaveController  # noqa: E402
from kinematics import FARMSModelIndex                       # noqa: E402
from wavelength_sweep import (                               # noqa: E402
    generate_single_wavelength_terrain,
    save_wavelength_terrain,
    patch_xml_terrain,
)

DEFAULT_XML = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
DEFAULT_CFG = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")

BUCKLED_THR = 1e4
MAX_ROOT_PITCH_DEG = 45.0
MAX_ROOT_ROLL_DEG  = 45.0


# ══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

def f(row, key, default=0.0):
    v = row.get(key, "")
    try:
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


def row_to_params(row):
    p = {}
    for k in ("body_kp", "body_kv",
              "hip_yaw_kp", "hip_yaw_kv",
              "hip_pitch_kp", "hip_pitch_kv",
              "tibia_kp", "tibia_kv",
              "tarsus_kp", "tarsus_kv",
              "pitch_kp", "pitch_kv"):
        if k in row and row[k] != "":
            try: p[k] = float(row[k])
            except ValueError: pass
    return p


def patch_config_with_params(base_cfg, params):
    """Deep-copy base_cfg, write the trial's gains into the impedance section."""
    cfg = copy.deepcopy(base_cfg)
    imp = cfg.setdefault("impedance", {})
    imp["body_kp"] = float(params["body_kp"])
    imp["body_kv"] = float(params["body_kv"])
    if "pitch_kp" in params: imp["pitch_kp"] = float(params["pitch_kp"])
    if "pitch_kv" in params: imp["pitch_kv"] = float(params["pitch_kv"])
    leg = imp.setdefault("leg", {})
    kp = list(leg.get("kp", [0.5, 0.06, 0.13, 0.13]))
    kv = list(leg.get("kv", [0.001, 0.003, 0.0005, 0.001]))
    kp[0] = float(params["hip_yaw_kp"]);   kv[0] = float(params["hip_yaw_kv"])
    kp[1] = float(params["hip_pitch_kp"]); kv[1] = float(params["hip_pitch_kv"])
    if "tibia_kp" in params:
        kp[2] = float(params["tibia_kp"]);  kv[2] = float(params["tibia_kv"])
    if "tarsus_kp" in params:
        kp[3] = float(params["tarsus_kp"]); kv[3] = float(params["tarsus_kv"])
    leg["kp"] = kp; leg["kv"] = kv
    return cfg


def export_yaml(cfg, out_path):
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


# ══════════════════════════════════════════════════════════════════════════════
# Terrain XML setup (flat = base, rough = patched hfield)
# ══════════════════════════════════════════════════════════════════════════════

def get_xml_for_terrain(args, run_dir):
    """Return the MJCF XML path matching the user's --terrain choice."""
    if args.terrain == "flat":
        return args.model
    # rough λ patch
    wavelength_m = args.terrain_wavelength_mm * 1e-3
    h, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m=wavelength_m,
        amplitude_m=args.terrain_amplitude,
        seed=args.terrain_seed)
    png_path = save_wavelength_terrain(h, wavelength_m, args.terrain_seed, run_dir)
    z_max = max(2.0 * args.terrain_amplitude, 1e-3)
    patched = patch_xml_terrain(args.model, png_path, z_max=z_max)
    rough_path = (args.model
                  + f".render_rough_wl{int(args.terrain_wavelength_mm)}.xml")
    shutil.copy(patched, rough_path)
    print(f"[terrain] rough XML: {rough_path}  "
          f"(λ={args.terrain_wavelength_mm:.1f}mm  amp={args.terrain_amplitude*1000:.1f}mm  "
          f"rms={rms_m*1000:.2f}mm  peak={peak_m*1000:.2f}mm)")
    return rough_path


# ══════════════════════════════════════════════════════════════════════════════
# Single-trial render
# ══════════════════════════════════════════════════════════════════════════════

def render_one_trial(xml_path, cfg_path, duration, freq_override,
                      video_path, fps=30, camera_mode="chase"):
    """Run one rollout, capture frames, write MP4. Returns dict of metrics."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    ctrl  = ImpedanceTravelingWaveController(model, cfg_path)
    idx   = FARMSModelIndex(model)
    if freq_override is not None:
        ctrl.set_frequency(float(freq_override))

    dt      = model.opt.timestep
    n_steps = int(duration / dt)
    settle  = ctrl.settle_time + getattr(ctrl, "ramp_time", 0.0)

    # Renderer setup
    try:
        import mediapy   # noqa: F401
    except ImportError:
        print("  [video] mediapy not installed — pip install mediapy")
        return None
    vid_w = min(1280, int(getattr(model.vis.global_, "offwidth", 1280)))
    vid_h = min(720,  int(getattr(model.vis.global_, "offheight", 720)))
    renderer = mujoco.Renderer(model, height=vid_h, width=vid_w)
    cam = mujoco.MjvCamera()
    if camera_mode == "static":
        cam.distance  = 0.40
        cam.azimuth   = 60
        cam.elevation = -25
    else:
        # chase
        cam.distance  = 0.30
        cam.azimuth   = 60
        cam.elevation = -30
    cam.lookat[:] = idx.com_pos(data)

    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
    pitch_qposadr = []
    for j in range(model.njnt):
        nm = model.joint(j).name or ""
        if "joint_pitch_body" in nm: pitch_qposadr.append(model.jnt_qposadr[j])

    frames = []
    last_frame_t = -np.inf
    video_dt = 1.0 / max(fps, 1)
    start_pos = None
    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if step_i == int(settle / dt):
            start_pos = data.xpos[root_body].copy()

        # Buckle detection
        if step_i > 0 and step_i % 200 == 0:
            R = data.xmat[root_body].reshape(3, 3)
            root_pitch = math.degrees(math.asin(-R[2, 0]))
            root_roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            if abs(root_pitch) > MAX_ROOT_PITCH_DEG:
                buckled = True
                buckle_reason = f"root_pitch={root_pitch:.1f}@{data.time:.1f}s"
                break
            if abs(root_roll) > MAX_ROOT_ROLL_DEG:
                buckled = True
                buckle_reason = f"root_roll={root_roll:.1f}@{data.time:.1f}s"
                break

        # Capture frame at video_dt rate
        if camera_mode == "chase":
            cam.lookat[:] = idx.com_pos(data)
        if data.time - last_frame_t >= video_dt:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

    # Distance / speed
    end_pos = data.xpos[root_body].copy()
    if start_pos is not None:
        dist = float(math.hypot(end_pos[0] - start_pos[0],
                                end_pos[1] - start_pos[1]))
    else:
        dist = 0.0
    active_time = max(data.time - settle, 1e-3)
    speed = dist / active_time

    # Save MP4
    if frames:
        try:
            import mediapy
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            mediapy.write_video(video_path, frames, fps=fps)
        except Exception as e:
            print(f"  [video] save failed: {e}")
            frames = []

    return dict(buckled=buckled, buckle_reason=buckle_reason,
                speed_mps=speed, distance_m=dist, sim_time=float(data.time),
                n_frames=len(frames))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_csv():
    """Default: most recent three_eval run. Falls back to bio runs if none."""
    pat1 = os.path.join(PROJECT_ROOT, "outputs", "optimization",
                        "impedance_three_eval_*", "all_trials.csv")
    pat2 = os.path.join(PROJECT_ROOT, "outputs", "optimization",
                        "impedance_bio_*", "all_trials.csv")
    matches = sorted(glob.glob(pat1) + glob.glob(pat2),
                     key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default=None,
                       help="all_trials.csv (defaults to most recent run).")
    parser.add_argument("--top",         type=int,   default=10)
    parser.add_argument("--duration",    type=float, default=8.0,
                       help="Sim seconds per video.")
    parser.add_argument("--terrain",     choices=["flat","rough"], default="rough")
    parser.add_argument("--terrain-wavelength-mm", type=float, default=18.0)
    parser.add_argument("--terrain-amplitude",     type=float, default=0.01)
    parser.add_argument("--terrain-seed",          type=int,   default=42)
    parser.add_argument("--rough-frequency", type=float, default=1.0,
                       help="Body-wave frequency override (Hz). Default 1.0 "
                            "matches eval B of the three-eval optimizer (slow "
                            "stable rough). Use 2.0 to render eval C (fast "
                            "rough) or eval A (with --terrain flat).")
    parser.add_argument("--video-fps",   type=int,   default=30)
    parser.add_argument("--camera-mode", choices=["chase","static"],
                       default="chase")
    parser.add_argument("--model",       default=DEFAULT_XML,
                       help="Base MJCF (used for both flat and rough patching).")
    parser.add_argument("--config",      default=DEFAULT_CFG,
                       help="YAML to use as base for trial-specific patching.")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip trials whose video file already exists.")
    parser.add_argument("--include-buckled", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv()
    if not csv_path or not os.path.isfile(csv_path):
        print("ERROR: no all_trials.csv found.", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(csv_path, "r", newline="",
                                      encoding="utf-8")))
    filtered = []
    for r in rows:
        cost = f(r, "cost", default=BUCKLED_THR)
        if not args.include_buckled and cost >= BUCKLED_THR * 0.99:
            continue
        filtered.append(r)
    filtered.sort(key=lambda r: f(r, "cost", default=BUCKLED_THR))
    top = filtered[:args.top]
    if not top:
        print("No trials match the filters.", file=sys.stderr)
        return 1

    run_dir   = os.path.dirname(csv_path)
    video_dir = os.path.join(run_dir, "top_videos")
    yaml_dir  = os.path.join(run_dir, "top_yamls")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(yaml_dir,  exist_ok=True)

    print(f"\n[render] reading {csv_path}")
    print(f"[render] {len(rows)} trials in CSV  ·  {len(filtered)} match "
          f"filters  ·  rendering top {len(top)}")
    print(f"[render] terrain={args.terrain}  freq={args.rough_frequency:.1f} Hz  "
          f"duration={args.duration}s  camera={args.camera_mode}")
    print(f"[render] output: {video_dir}")

    # Load base config
    with open(args.config, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    # Generate the (one) terrain XML up front
    xml_path = get_xml_for_terrain(args, run_dir)

    t0_batch = time.time()
    n_done = n_skipped = n_failed = 0

    for rank, r in enumerate(top, start=1):
        trial_num = int(f(r, "trial"))
        cost      = f(r, "cost")
        params    = row_to_params(r)

        video_name = (f"rank_{rank:02d}_trial_{trial_num:04d}"
                       f"_cost_{cost:+07.2f}.mp4")
        video_path = os.path.join(video_dir, video_name)
        yaml_path  = os.path.join(yaml_dir,
                                    f"rank_{rank:02d}_trial_{trial_num:04d}.yaml")

        print(f"\n[render {rank}/{len(top)}] trial #{trial_num}  cost={cost:+.3f}",
              flush=True)

        if args.skip_existing and os.path.isfile(video_path):
            print(f"          [skip] video exists")
            n_skipped += 1
            continue

        # Patch and write the trial's YAML (so people can copy it later)
        cfg = patch_config_with_params(base_cfg, params)
        export_yaml(cfg, yaml_path)

        # Render
        try:
            cfg_path_tmp = yaml_path   # reuse the saved YAML directly
            t_start = time.time()
            result = render_one_trial(
                xml_path        = xml_path,
                cfg_path        = cfg_path_tmp,
                duration        = args.duration,
                freq_override   = args.rough_frequency,
                video_path      = video_path,
                fps             = args.video_fps,
                camera_mode     = args.camera_mode)
            wall = time.time() - t_start
            if result is None:
                print(f"          ✗ video render failed (mediapy?) ({wall:.1f}s)")
                n_failed += 1
                continue
            if result["buckled"]:
                print(f"          ⚠ BUCKLED in replay: {result['buckle_reason']}  "
                      f"({wall:.1f}s, {result['n_frames']} frames)")
            size_kb = (os.path.getsize(video_path) / 1024
                       if os.path.exists(video_path) else 0)
            print(f"          ✓ {video_name}  ({size_kb:.0f} KB, "
                  f"{wall:.1f}s, {result['n_frames']} frames, "
                  f"speed={result['speed_mps']*1000:.1f} mm/s)")
            n_done += 1
        except Exception as e:
            print(f"          ✗ exception: {e}")
            import traceback; traceback.print_exc()
            n_failed += 1

    elapsed = time.time() - t0_batch
    print(f"\n[render] done — {n_done} rendered, {n_skipped} skipped, "
          f"{n_failed} failed  ·  {elapsed/60:.1f} min total")
    print(f"[render] videos in: {video_dir}")
    print(f"[render] yamls   in: {yaml_dir}")
    print(f"\nWhen you've picked a winner:")
    print(f"  cp {yaml_dir}/rank_<NN>_trial_<TTT>.yaml configs/farms_controller.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
