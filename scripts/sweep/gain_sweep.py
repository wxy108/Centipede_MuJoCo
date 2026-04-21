#!/usr/bin/env python3
"""
gain_sweep.py — Separate 1-D sweeps of body and leg impedance scale factors.

Generates a single terrain (fixed wavelength), then runs two 1-D sweeps:
  1. Body sweep: scale body_kp/kv, pitch_kp/kv, roll_kp/kv  (legs at 1.0×)
  2. Leg sweep:  scale leg.kp/kv                              (body at 1.0×)

Each cell runs N trials with random spawn yaw (0-360°) for statistical
robustness on rough terrain.

Outputs:
  - Per-trial MP4 video:  videos/{label}/trial{T}_yaw{Y}.mp4
  - results.json with CoT, survival, max_pitch/roll for every trial
  - summary CSV aggregated per cell

Usage:
  python scripts/sweep/gain_sweep.py --wavelength 18 --duration 10 --video --n-trials 3
  python scripts/sweep/gain_sweep.py --wavelength 18 --duration 10 --video --n-trials 3 --sweep body
  python scripts/sweep/gain_sweep.py --wavelength 18 --duration 10 --video --n-trials 3 --sweep leg
  python scripts/sweep/gain_sweep.py --wavelength 18 --duration 10 --video --n-trials 3 \
      --scales "1.0,0.9,0.7,0.5,0.3,0.1,0.05,0.01"
"""

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import mujoco

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "terrain", "generator"))

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex

import yaml
from generate import (resolve_morphology, heightmap_to_png, _spectral_band)
from scipy.ndimage import gaussian_filter
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
XML_PATH    = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")
TERRAIN_CFG = os.path.join(PROJECT_ROOT, "configs", "terrain.yaml")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "gain_sweep")

VID_FPS  = 30
VID_W    = 1280
VID_H    = 720
CAM_DISTANCE  = 0.20
CAM_AZIMUTH   = 60
CAM_ELEVATION = -35
MAX_PITCH_DEG = 35
MAX_ROLL_DEG  = 60

# Default scale factors
DEFAULT_SCALES = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# Quality thresholds for early-skip and final ranking
MIN_SPEED_MM_S   = 1.0     # Below this = "doesn't move" (mm/s)
MIN_DISTANCE_MM  = 5.0     # Below this = stuck (mm, over full duration)
QUALITY_TAG_SKIP = "SKIP"  # Tag for cells skipped after trial-1 check


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain generation (reuse from wavelength_sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_single_wavelength_terrain(wavelength_m, amplitude_m, seed,
                                       image_size=1024, world_half=0.5,
                                       n_components=8):
    rng = np.random.default_rng(seed)
    world_size = world_half * 2.0
    freq = 1.0 / wavelength_m
    freq_min = freq * 0.9
    freq_max = freq * 1.1

    h = _spectral_band(
        image_size, world_size,
        freq_min=freq_min, freq_max=freq_max,
        n_components=n_components,
        amplitude=amplitude_m,
        orientation_spread=1.0,
        rng=rng,
    )

    px_per_cycle = wavelength_m / world_size * image_size
    sigma = max(0.25 * px_per_cycle, 0.5)
    h = gaussian_filter(h, sigma=sigma)
    h -= h.mean()

    cur_peak = max(abs(h.min()), abs(h.max()))
    if cur_peak > 1e-12:
        h = h * (amplitude_m / cur_peak)

    rms_m  = float(np.std(h))
    peak_m = float(max(abs(h.min()), abs(h.max())))
    return h, rms_m, peak_m


def patch_xml_terrain(xml_path, png_path, z_max):
    from lxml import etree
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    asset = root.find("asset")
    hfield = asset.find("hfield[@name='terrain']")
    if hfield is not None:
        hfield.set("file", os.path.abspath(png_path).replace("\\", "/"))
        hfield.set("size", f"0.500 0.500 {z_max:.4f} 0.001")

    arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32)
    nrow, ncol = arr.shape
    cy, cx = nrow // 2, ncol // 2
    r = 8
    patch = arr[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    terrain_h = (float(patch.max()) / 255.0) * z_max
    spawn_z = terrain_h + 0.015

    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            body.set("pos", f"0 0 {spawn_z:.4f}")
            break

    tmp_xml = xml_path + ".gain_sweep_tmp.xml"
    tree.write(tmp_xml, xml_declaration=True, encoding="utf-8", pretty_print=False)
    return tmp_xml


# ═══════════════════════════════════════════════════════════════════════════════
# Config manipulation
# ═══════════════════════════════════════════════════════════════════════════════

def load_baseline_config():
    """Load the current config as baseline gains."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    imp = cfg.get("impedance", {})
    return {
        'body_kp':  float(imp.get("body_kp", 0.05)),
        'body_kv':  float(imp.get("body_kv", 0.01)),
        'pitch_kp': float(imp.get("pitch_kp", 0.003477)),
        'pitch_kv': float(imp.get("pitch_kv", 0.000914)),
        'roll_kp':  float(imp.get("roll_kp", 0.004)),
        'roll_kv':  float(imp.get("roll_kv", 0.0016)),
        'leg_kp':   list(imp.get("leg", {}).get("kp", [0.127, 0.0147, 0.127, 0.127])),
        'leg_kv':   list(imp.get("leg", {}).get("kv", [0.00056, 0.000114, 4.36e-5, 0.00091])),
        'settle_time': float(imp.get("settle_time", 1.0)),
        'ramp_time':   float(imp.get("ramp_time", 1.0)),
    }


def write_scaled_config(baseline, body_scale, leg_scale, out_path,
                        pitch_scale=None):
    """Write a temporary config YAML with scaled gains.

    pitch_scale: if provided, overrides body_scale for pitch gains only.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ps = pitch_scale if pitch_scale is not None else body_scale

    imp = cfg.setdefault("impedance", {})
    imp["body_kp"]  = baseline["body_kp"]  * body_scale
    imp["body_kv"]  = baseline["body_kv"]  * body_scale
    imp["pitch_kp"] = baseline["pitch_kp"] * ps
    imp["pitch_kv"] = baseline["pitch_kv"] * ps
    imp["roll_kp"]  = baseline["roll_kp"]  * body_scale
    imp["roll_kv"]  = baseline["roll_kv"]  * body_scale

    leg = imp.setdefault("leg", {})
    leg["kp"] = [v * leg_scale for v in baseline["leg_kp"]]
    leg["kv"] = [v * leg_scale for v in baseline["leg_kv"]]

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Video / Renderer helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _try_make_renderer(model):
    try:
        renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
        return renderer, True
    except Exception:
        return None, False


def _make_camera(idx, data):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = idx.com_pos(data)
    cam.distance  = CAM_DISTANCE
    cam.azimuth   = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def _save_video(frames, path, fps=VID_FPS):
    import mediapy
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mediapy.write_video(path, frames, fps=fps)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def set_initial_yaw(model, data, yaw_rad):
    """Rotate the centipede's initial orientation around z-axis."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            data.qpos[qadr + 3] = math.cos(yaw_rad / 2.0)
            data.qpos[qadr + 4] = 0.0
            data.qpos[qadr + 5] = 0.0
            data.qpos[qadr + 6] = math.sin(yaw_rad / 2.0)
            break
    mujoco.mj_forward(model, data)


def run_simulation(xml_path, config_path, duration, yaw_rad=0.0, video_path=None):
    """Run one simulation with optional initial yaw. Returns dict of metrics."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    set_initial_yaw(model, data, yaw_rad)

    ctrl = ImpedanceTravelingWaveController(model, config_path)
    idx  = FARMSModelIndex(model)

    pitch_jnt_ids = []
    roll_jnt_ids  = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_jnt_ids.append(j)
        if nm and 'joint_roll_body' in nm:
            roll_jnt_ids.append(j)

    n_actuators = model.nu
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    gravity = abs(model.opt.gravity[2])
    n_steps = int(duration / model.opt.timestep)
    dt = model.opt.timestep

    settle_time = ctrl.settle_time + ctrl.ramp_time

    # Find root body
    root_body = None
    for b in range(model.nbody):
        if model.body(b).name and 'root' in model.body(b).name.lower():
            root_body = b
            break
    if root_body is None:
        for b in range(model.nbody):
            jnt_start = model.body_jntadr[b]
            if jnt_start >= 0 and model.jnt_type[jnt_start] == mujoco.mjtJoint.mjJNT_FREE:
                root_body = b
                break

    # Video
    renderer, frames, vid_cam = None, [], None
    vid_dt = 1.0 / VID_FPS
    last_frame_t = -1.0
    cam_azimuth_fixed = CAM_AZIMUTH + math.degrees(yaw_rad)
    if video_path:
        renderer, ok = _try_make_renderer(model)
        if ok:
            vid_cam = _make_camera(idx, data)
            vid_cam.azimuth = cam_azimuth_fixed
        else:
            video_path = None

    # Storage
    energy_sum = 0.0
    pitch_angles = []
    roll_angles  = []
    start_pos = None
    buckled = False
    buckle_reason = ""
    max_body_torque = 0.0
    max_leg_torque  = 0.0

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Record start position after settling
        if start_pos is None and data.time >= settle_time:
            start_pos = idx.com_pos(data).copy()

        # Energy + angles (after settling)
        if data.time >= settle_time:
            for a in range(n_actuators):
                tau = abs(data.actuator_force[a])
                jid = model.actuator_trnid[a, 0]
                if 0 <= jid < model.njnt:
                    dof = model.jnt_dofadr[jid]
                    energy_sum += tau * abs(data.qvel[dof]) * dt
                # Track peak torques
                act_name = model.actuator(a).name
                if 'leg' in act_name or 'hip' in act_name or 'tibia' in act_name or 'tarsus' in act_name:
                    max_leg_torque = max(max_leg_torque, tau)
                else:
                    max_body_torque = max(max_body_torque, tau)

            if step_i % 100 == 0:
                for jid in pitch_jnt_ids:
                    pitch_angles.append(abs(math.degrees(data.qpos[model.jnt_qposadr[jid]])))
                for jid in roll_jnt_ids:
                    roll_angles.append(abs(math.degrees(data.qpos[model.jnt_qposadr[jid]])))

        # Video
        if renderer and video_path and data.time - last_frame_t >= vid_dt - 1e-6:
            vid_cam.lookat[:] = idx.com_pos(data)
            renderer.update_scene(data, camera=vid_cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

        # Failure check
        if step_i % 200 == 0 and step_i > 0:
            for jid in pitch_jnt_ids:
                q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                if q_deg > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch({q_deg:.1f} t={data.time:.1f}s)"
                    break
            if not buckled:
                for jid in roll_jnt_ids:
                    q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                    if q_deg > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll({q_deg:.1f} t={data.time:.1f}s)"
                        break
            if buckled:
                break

            # Check if body flipped
            if root_body is not None:
                body_z = data.xpos[root_body, 2]
                if body_z < -0.01:
                    buckled = True
                    buckle_reason = f"fell(z={body_z:.3f} t={data.time:.1f}s)"
                    break

    # Save video
    if renderer and frames and video_path:
        _save_video(frames, video_path)
    if renderer:
        renderer.close()

    # Compute metrics
    end_pos = idx.com_pos(data)
    if start_pos is None:
        start_pos = np.zeros(3)
    disp = end_pos - start_pos
    distance = np.linalg.norm(disp[:2])
    effective_time = max(data.time - settle_time, 0.01)

    if distance < 1e-6 or buckled:
        cot = 1e6
    else:
        cot = energy_sum / (total_mass * gravity * distance)

    return {
        'survived':         not buckled,
        'buckle_reason':    buckle_reason,
        'yaw_deg':          float(math.degrees(yaw_rad)),
        'cot':              float(cot),
        'forward_speed':    float(distance / effective_time),
        'distance':         float(distance),
        'max_pitch_deg':    float(max(pitch_angles)) if pitch_angles else 0.0,
        'max_roll_deg':     float(max(roll_angles)) if roll_angles else 0.0,
        'mean_pitch_deg':   float(np.mean(pitch_angles)) if pitch_angles else 0.0,
        'mean_roll_deg':    float(np.mean(roll_angles)) if roll_angles else 0.0,
        'max_body_torque':  float(max_body_torque),
        'max_leg_torque':   float(max_leg_torque),
        'sim_time':         float(data.time),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_trials(trial_results):
    """Compute mean/std/median of key metrics across trials."""
    survived = [r for r in trial_results if r['survived']]
    n_survived = len(survived)
    n_total = len(trial_results)

    if not survived:
        return {
            'n_trials': n_total, 'n_survived': 0, 'survival_rate': 0.0,
            'cot_mean': float('inf'), 'cot_std': 0,
            'speed_mean': 0, 'speed_std': 0,
            'max_pitch_mean': 0, 'max_roll_mean': 0,
            'max_body_torque_mean': 0, 'max_leg_torque_mean': 0,
        }

    cots   = [r['cot'] for r in survived]
    speeds = [r['forward_speed'] for r in survived]
    pitches = [r['max_pitch_deg'] for r in survived]
    rolls   = [r['max_roll_deg'] for r in survived]
    body_taus = [r['max_body_torque'] for r in survived]
    leg_taus  = [r['max_leg_torque'] for r in survived]

    return {
        'n_trials':       n_total,
        'n_survived':     n_survived,
        'survival_rate':  n_survived / n_total,
        'cot_mean':       float(np.mean(cots)),
        'cot_std':        float(np.std(cots)),
        'speed_mean':     float(np.mean(speeds)),
        'speed_std':      float(np.std(speeds)),
        'max_pitch_mean': float(np.mean(pitches)),
        'max_pitch_std':  float(np.std(pitches)),
        'max_roll_mean':  float(np.mean(rolls)),
        'max_roll_std':   float(np.std(rolls)),
        'max_body_torque_mean': float(np.mean(body_taus)),
        'max_leg_torque_mean':  float(np.mean(leg_taus)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wavelength", type=float, default=18.0,
                        help="Terrain wavelength in mm (default: 18)")
    parser.add_argument("--amplitude", type=float, default=0.01,
                        help="Terrain amplitude in metres (default: 0.01)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Trials per scale factor (random yaw, default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true",
                        help="Save MP4 video for every trial")
    parser.add_argument("--sweep", type=str, default="both",
                        choices=["body", "leg", "both"],
                        help="Which sweep to run (default: both)")
    parser.add_argument("--scales", type=str, default=None,
                        help="Custom scale factors, comma-separated "
                             "(e.g. '1.0,0.5,0.1'). Default: built-in list")
    args = parser.parse_args()

    # Parse scales
    if args.scales:
        scales = sorted([float(s.strip()) for s in args.scales.split(",")],
                        reverse=True)
    else:
        scales = DEFAULT_SCALES

    baseline = load_baseline_config()

    # ── Generate terrain ──────────────────────────────────────────────────
    with open(TERRAIN_CFG, encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    img_size = int(t_cfg["world"]["image_size"])
    world_half = float(t_cfg["world"]["size"])

    wl_m = args.wavelength / 1000.0
    h_m, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m=wl_m, amplitude_m=args.amplitude,
        seed=args.seed, image_size=img_size, world_half=world_half,
    )
    z_max = max(2.0 * peak_m, 0.005)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"gsweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save terrain
    terrain_dir = os.path.join(run_dir, "terrain")
    os.makedirs(terrain_dir, exist_ok=True)
    arr = heightmap_to_png(h_m)
    png_path = os.path.join(terrain_dir, "1.png")
    Image.fromarray(arr, mode="L").save(png_path)

    tmp_xml = patch_xml_terrain(XML_PATH, png_path, z_max)

    # ── Build sweep plan ──────────────────────────────────────────────────
    # Each entry: (label, body_scale, leg_scale)
    sweep_plan = []

    if args.sweep in ("body", "both"):
        for s in scales:
            sweep_plan.append((f"body_x{s}", s, 1.0))

    if args.sweep in ("leg", "both"):
        for s in scales:
            # Skip (1.0, 1.0) duplicate if body sweep already has it
            if args.sweep == "both" and s == 1.0:
                continue
            sweep_plan.append((f"leg_x{s}", 1.0, s))

    n_cells = len(sweep_plan)
    total_sims = n_cells * args.n_trials

    # Pre-generate random yaw angles (reproducible)
    rng = np.random.default_rng(args.seed + 9999)
    all_yaws = rng.uniform(0, 2 * math.pi, size=(n_cells, args.n_trials))

    # Video check
    can_video = False
    if args.video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed — skipping video.")

    # ── Print header ──────────────────────────────────────────────────────
    print("=" * 72)
    print("Gain Sweep — Body & Leg Impedance Scale Factors")
    print("=" * 72)
    print(f"  Terrain:     wavelength={args.wavelength:.0f}mm  "
          f"amplitude={args.amplitude*1000:.1f}mm")
    print(f"  Duration:    {args.duration}s  "
          f"(settle={baseline['settle_time']}s + ramp={baseline['ramp_time']}s)")
    print(f"  Sweep:       {args.sweep}")
    print(f"  Scales:      {scales}")
    print(f"  Cells:       {n_cells}")
    print(f"  Trials/cell: {args.n_trials} (random yaw 0-360°)")
    print(f"  Total sims:  {total_sims}")
    print(f"  Video:       {'ON' if can_video else 'OFF'}")
    print(f"  Output:      {run_dir}")
    print()
    print(f"  Baseline gains:")
    print(f"    body  kp={baseline['body_kp']:.4f}  kv={baseline['body_kv']:.4f}")
    print(f"    pitch kp={baseline['pitch_kp']:.6f}  kv={baseline['pitch_kv']:.6f}")
    print(f"    roll  kp={baseline['roll_kp']:.4f}  kv={baseline['roll_kv']:.4f}")
    print(f"    leg   kp={baseline['leg_kp']}")
    print(f"    leg   kv={baseline['leg_kv']}")
    print("=" * 72)
    print()

    # ── Run sweep ─────────────────────────────────────────────────────────
    all_results = []
    cell_aggregates = []
    t_start = time.time()
    sim_count = 0

    for cell_i, (label, bs, ls) in enumerate(sweep_plan):
        actual_body_kp = baseline['body_kp'] * bs
        actual_leg_kp0 = baseline['leg_kp'][0] * ls
        print(f"[{cell_i+1:2d}/{n_cells}] {label:15s}  "
              f"(body_kp={actual_body_kp:.6f}, leg_kp[0]={actual_leg_kp0:.6f})")

        # Write temp config (pitch tied to body)
        tmp_cfg = os.path.join(run_dir, f"config_{label}.yaml")
        write_scaled_config(baseline, bs, ls, tmp_cfg, pitch_scale=bs)

        cell_trials = []
        cell_skipped = False

        for t in range(args.n_trials):
            sim_count += 1
            yaw = float(all_yaws[cell_i, t])
            yaw_deg = math.degrees(yaw)

            vid_path = None
            if can_video:
                vid_dir = os.path.join(run_dir, "videos", label)
                os.makedirs(vid_dir, exist_ok=True)
                vid_path = os.path.join(vid_dir,
                                        f"trial{t:02d}_yaw{yaw_deg:.0f}.mp4")

            try:
                metrics = run_simulation(tmp_xml, tmp_cfg, args.duration,
                                         yaw_rad=yaw, video_path=vid_path)
            except Exception as e:
                metrics = {
                    'survived': False, 'buckle_reason': str(e),
                    'yaw_deg': yaw_deg, 'cot': 1e6, 'forward_speed': 0,
                    'distance': 0, 'max_pitch_deg': 0, 'mean_pitch_deg': 0,
                    'max_roll_deg': 0, 'mean_roll_deg': 0,
                    'max_body_torque': 0, 'max_leg_torque': 0,
                    'sim_time': 0,
                }

            metrics['label'] = label
            metrics['body_scale'] = bs
            metrics['leg_scale'] = ls
            metrics['trial_idx'] = t
            metrics['body_kp'] = actual_body_kp
            metrics['body_kv'] = baseline['body_kv'] * bs
            metrics['leg_kp'] = [v * ls for v in baseline['leg_kp']]
            cell_trials.append(metrics)
            all_results.append(metrics)

            status = ("OK" if metrics['survived']
                      else f"FAIL:{metrics['buckle_reason']}")
            elapsed_total = time.time() - t_start
            eta = (elapsed_total / sim_count) * (total_sims - sim_count)
            print(f"    [{sim_count:3d}/{total_sims}] "
                  f"yaw={yaw_deg:5.1f}°  "
                  f"CoT={metrics['cot']:8.1f}  "
                  f"speed={metrics['forward_speed']*1000:5.1f}mm/s  "
                  f"body_tau={metrics['max_body_torque']*1000:.2f}mNm  "
                  f"leg_tau={metrics['max_leg_torque']*1000:.2f}mNm  "
                  f"{status}  "
                  f"(ETA {eta/60:.0f}min)", flush=True)

            # ── Early-skip check after first trial ────────────────────
            # If trial 0 already buckled OR is completely immobile,
            # skip remaining trials — this scale is clearly bad.
            if t == 0:
                t0_failed  = not metrics['survived']
                t0_stuck   = (metrics['distance'] * 1000 < MIN_DISTANCE_MM
                              and metrics['survived'])
                if t0_failed or t0_stuck:
                    reason = ("UNSTABLE" if t0_failed
                              else f"STUCK({metrics['distance']*1000:.1f}mm)")
                    skipped_count = args.n_trials - 1
                    sim_count += skipped_count  # account for skipped sims in ETA
                    print(f"    !! {reason} on trial 0 — "
                          f"skipping {skipped_count} remaining trials")
                    cell_skipped = True
                    # Fill in dummy results for skipped trials
                    for skip_t in range(1, args.n_trials):
                        skip_m = metrics.copy()
                        skip_m['trial_idx'] = skip_t
                        skip_m['yaw_deg'] = float(math.degrees(all_yaws[cell_i, skip_t]))
                        skip_m['buckle_reason'] = f"skipped({reason})"
                        skip_m['video_path'] = ''
                        cell_trials.append(skip_m)
                        all_results.append(skip_m)
                    break

        # Aggregate this cell
        agg = aggregate_trials(cell_trials)
        agg['label'] = label
        agg['body_scale'] = bs
        agg['leg_scale'] = ls
        agg['skipped'] = cell_skipped
        cell_aggregates.append(agg)

        tag_str = f" [{QUALITY_TAG_SKIP}]" if cell_skipped else ""
        print(f"    => CoT={agg['cot_mean']:.1f}±{agg['cot_std']:.1f}  "
              f"speed={agg['speed_mean']*1000:.1f}±{agg['speed_std']*1000:.1f}mm/s  "
              f"survived={agg['n_survived']}/{agg['n_trials']}{tag_str}")
        print()

        # Clean up temp config
        try:
            os.remove(tmp_cfg)
        except OSError:
            pass

    total_time = time.time() - t_start

    # ── Save results ──────────────────────────────────────────────────────
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            'timestamp':     timestamp,
            'sweep_type':    args.sweep,
            'wavelength_mm': args.wavelength,
            'amplitude_m':   args.amplitude,
            'duration':      args.duration,
            'n_trials':      args.n_trials,
            'scales':        scales,
            'baseline':      baseline,
            'all_trials':    all_results,
            'elapsed_s':     total_time,
        }, f, indent=2, default=str)

    # Raw CSV — one row per trial
    raw_csv = os.path.join(run_dir, "results_all_trials.csv")
    with open(raw_csv, "w", encoding="utf-8") as f:
        headers = ['label', 'body_scale', 'leg_scale', 'trial_idx', 'yaw_deg',
                   'survived', 'cot', 'forward_speed', 'distance',
                   'max_pitch_deg', 'mean_pitch_deg',
                   'max_roll_deg', 'mean_roll_deg',
                   'max_body_torque', 'max_leg_torque', 'sim_time']
        f.write(",".join(headers) + "\n")
        for r in all_results:
            vals = [str(r.get(h, '')) for h in headers]
            f.write(",".join(vals) + "\n")

    # Aggregated CSV — one row per cell
    agg_csv = os.path.join(run_dir, "results_aggregated.csv")
    with open(agg_csv, "w", encoding="utf-8") as f:
        f.write("label,body_scale,leg_scale,n_survived,survival_rate,"
                "cot_mean,cot_std,speed_mean,speed_std,"
                "max_pitch_mean,max_roll_mean,"
                "max_body_torque_mean,max_leg_torque_mean\n")
        for agg in cell_aggregates:
            f.write(f"{agg['label']},{agg['body_scale']},{agg['leg_scale']},"
                    f"{agg['n_survived']},{agg['survival_rate']:.2f},"
                    f"{agg['cot_mean']:.2f},{agg['cot_std']:.2f},"
                    f"{agg['speed_mean']:.6f},{agg['speed_std']:.6f},"
                    f"{agg.get('max_pitch_mean',0):.2f},"
                    f"{agg.get('max_roll_mean',0):.2f},"
                    f"{agg.get('max_body_torque_mean',0):.6f},"
                    f"{agg.get('max_leg_torque_mean',0):.6f}\n")

    # ── Classify each cell ───────────────────────────────────────────────
    # Categories:
    #   GOOD     = survived all trials, mean speed > threshold
    #   MARGINAL = survived some trials, or low speed
    #   STUCK    = survived but barely moves
    #   UNSTABLE = buckled in all trials
    #   SKIPPED  = early-skipped (unstable or stuck on trial 0)
    for agg in cell_aggregates:
        if agg.get('skipped', False):
            if agg['n_survived'] == 0:
                agg['quality'] = 'UNSTABLE'
            else:
                agg['quality'] = 'STUCK'
        elif agg['n_survived'] == 0:
            agg['quality'] = 'UNSTABLE'
        elif agg['speed_mean'] * 1000 < MIN_SPEED_MM_S:
            agg['quality'] = 'STUCK'
        elif agg['survival_rate'] < 1.0:
            agg['quality'] = 'MARGINAL'
        else:
            agg['quality'] = 'GOOD'

    # ── Print full table ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"DONE  ({total_time/60:.1f} min, {total_sims} simulations, "
          f"{sum(1 for a in cell_aggregates if a.get('skipped'))} cells skipped)")
    print("=" * 72)

    for sweep_label, filter_prefix in [("BODY SWEEP", "body_x"),
                                        ("LEG SWEEP", "leg_x")]:
        cells = [a for a in cell_aggregates if a['label'].startswith(filter_prefix)]
        if not cells:
            continue
        print(f"\n  {sweep_label}:")
        print(f"  {'Scale':>8s}  {'Surv':>5s}  {'Quality':>9s}  {'CoT':>8s}  "
              f"{'Speed mm/s':>10s}  {'Pitch°':>7s}  {'Roll°':>7s}  "
              f"{'BodyTau':>8s}  {'LegTau':>8s}")
        for c in cells:
            scale = c['body_scale'] if 'body' in c['label'] else c['leg_scale']
            surv = f"{c['n_survived']}/{c['n_trials']}"
            cot_s = f"{c['cot_mean']:.1f}" if c['cot_mean'] < 1e5 else "INF"
            q = c['quality']
            print(f"  {scale:>8.4f}  {surv:>5s}  {q:>9s}  {cot_s:>8s}  "
                  f"{c['speed_mean']*1000:>10.1f}  "
                  f"{c.get('max_pitch_mean',0):>7.1f}  "
                  f"{c.get('max_roll_mean',0):>7.1f}  "
                  f"{c.get('max_body_torque_mean',0)*1000:>8.2f}  "
                  f"{c.get('max_leg_torque_mean',0)*1000:>8.2f}")

    # ── Recommended parameter ranges & videos to check ────────────────────
    good_cells = [a for a in cell_aggregates if a['quality'] == 'GOOD']
    marginal_cells = [a for a in cell_aggregates if a['quality'] == 'MARGINAL']
    worth_checking = good_cells + marginal_cells

    print()
    print("=" * 72)
    print("RECOMMENDED PARAMETER RANGES")
    print("=" * 72)

    if not worth_checking:
        print("\n  No cells passed quality check! All were UNSTABLE or STUCK.")
        print("  Consider narrowing the scale range or checking baseline config.")
    else:
        # Rank by a composite score: speed (higher=better) / CoT (lower=better)
        # Simple ranking: sort by speed descending among GOOD cells
        for sweep_label, filter_prefix in [("BODY", "body_x"),
                                            ("LEG", "leg_x")]:
            sw_good = sorted(
                [c for c in worth_checking if c['label'].startswith(filter_prefix)],
                key=lambda c: c['speed_mean'], reverse=True)
            if not sw_good:
                continue

            scales_good = [c['body_scale'] if 'body' in c['label'] else c['leg_scale']
                           for c in sw_good]
            best = sw_good[0]
            best_scale = scales_good[0]
            lo = min(scales_good)
            hi = max(scales_good)

            print(f"\n  {sweep_label} gains:")
            print(f"    Viable range:  ×{lo} — ×{hi}")
            print(f"    Best speed:    ×{best_scale}  "
                  f"({best['speed_mean']*1000:.1f} mm/s, "
                  f"CoT={best['cot_mean']:.1f})")

            # Lowest CoT among viable
            best_cot = min(sw_good, key=lambda c: c['cot_mean'])
            bc_scale = (best_cot['body_scale'] if 'body' in best_cot['label']
                        else best_cot['leg_scale'])
            print(f"    Best CoT:      ×{bc_scale}  "
                  f"({best_cot['speed_mean']*1000:.1f} mm/s, "
                  f"CoT={best_cot['cot_mean']:.1f})")

        # ── List specific video folders to check ──────────────────────
        if can_video:
            print(f"\n  ▸ VIDEOS TO CHECK (GOOD + MARGINAL only):")
            vid_base = os.path.join(run_dir, "videos")
            for c in sorted(worth_checking,
                            key=lambda c: c['speed_mean'], reverse=True):
                scale = (c['body_scale'] if 'body' in c['label']
                         else c['leg_scale'])
                q_tag = f"[{c['quality']}]"
                print(f"    {q_tag:>12s}  {c['label']:15s}  "
                      f"speed={c['speed_mean']*1000:.1f}mm/s  "
                      f"CoT={c['cot_mean']:.1f}  "
                      f"→ {os.path.join(vid_base, c['label'])}/")

    # ── File locations ────────────────────────────────────────────────────
    print(f"\n  Results JSON:     {json_path}")
    print(f"  All trials CSV:   {raw_csv}")
    print(f"  Aggregated CSV:   {agg_csv}")
    if can_video:
        print(f"  Videos:           {os.path.join(run_dir, 'videos')}/")
    print(f"\n  Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
