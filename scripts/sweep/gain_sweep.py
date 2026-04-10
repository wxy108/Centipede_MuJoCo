#!/usr/bin/env python3
"""
gain_sweep.py — Grid search for softest stable impedance gains.

Generates a single terrain (fixed wavelength), then sweeps over a 2D grid
of gain scale factors:  body_scale × leg_scale.

Each scale factor multiplies ALL gains in that group:
  body_scale → body_kp, body_kv, pitch_kp, pitch_kv, roll_kp, roll_kv
  leg_scale  → leg.kp[0..3], leg.kv[0..3]

Outputs:
  - Per-trial MP4 video:  videos/body{B}_leg{L}.mp4
  - results.json with CoT, survival, max_pitch/roll for every combo
  - summary CSV sorted by softness (lowest scale product first)

Usage:
  python scripts/sweep/gain_sweep.py --wavelength 18 --duration 8 --video
  python scripts/sweep/gain_sweep.py --wavelength 18 --duration 8 --video \
      --body-scales 1.0,0.7,0.5,0.3,0.2,0.1 \
      --leg-scales 1.0,0.7,0.5,0.3,0.2,0.1
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
VID_W    = 640
VID_H    = 480
CAM_DISTANCE  = 0.20
CAM_AZIMUTH   = 60
CAM_ELEVATION = -35
MAX_PITCH_DEG = 60


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

def run_simulation(xml_path, config_path, duration, video_path=None):
    """Run one simulation. Returns dict of metrics."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

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
    if video_path:
        renderer, ok = _try_make_renderer(model)
        if ok:
            vid_cam = _make_camera(idx, data)
        else:
            video_path = None

    # Storage
    energy_sum = 0.0
    pitch_angles = []
    roll_angles  = []
    start_pos = None
    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Record start position after settling
        if start_pos is None and data.time >= settle_time:
            start_pos = idx.com_pos(data).copy()

        # Energy + angles (after settling)
        if data.time >= settle_time:
            for a in range(n_actuators):
                jid = model.actuator_trnid[a, 0]
                dof = model.jnt_dofadr[jid]
                energy_sum += abs(data.ctrl[a] * data.qvel[dof]) * dt

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
        'survived':       not buckled,
        'buckle_reason':  buckle_reason,
        'cot':            float(cot),
        'forward_speed':  float(distance / effective_time),
        'distance':       float(distance),
        'max_pitch_deg':  float(max(pitch_angles)) if pitch_angles else 0.0,
        'max_roll_deg':   float(max(roll_angles)) if roll_angles else 0.0,
        'mean_pitch_deg': float(np.mean(pitch_angles)) if pitch_angles else 0.0,
        'mean_roll_deg':  float(np.mean(roll_angles)) if roll_angles else 0.0,
        'sim_time':       float(data.time),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wavelength", type=float, default=18.0,
                        help="Terrain wavelength in mm (default: 18)")
    parser.add_argument("--amplitude", type=float, default=0.008,
                        help="Terrain amplitude in metres (default: 0.008)")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true",
                        help="Save MP4 video for each combo")
    parser.add_argument("--body-scales", type=str,
                        default="1.0,0.8,0.6,0.4,0.3,0.2,0.1",
                        help="Comma-separated body gain scale factors (yaw+roll)")
    parser.add_argument("--leg-scales", type=str,
                        default="1.0,0.8,0.6,0.4,0.3,0.2,0.1",
                        help="Comma-separated leg gain scale factors")
    parser.add_argument("--pitch-scales", type=str, default=None,
                        help="Pitch-only sweep: comma-separated scale factors. "
                             "When set, body+leg stay at 1.0× and only pitch varies.")
    args = parser.parse_args()

    baseline = load_baseline_config()

    # ── Determine sweep mode ──────────────────────────────────────────────
    if args.pitch_scales:
        # Pitch-only 1D sweep
        pitch_scales = sorted([float(x) for x in args.pitch_scales.split(",")], reverse=True)
        sweep_mode = "pitch"
        sweep_combos = [(1.0, 1.0, ps) for ps in pitch_scales]  # (body, leg, pitch)
        n_combos = len(pitch_scales)
    else:
        # Body × Leg 2D grid (pitch tied to body)
        body_scales = sorted([float(x) for x in args.body_scales.split(",")], reverse=True)
        leg_scales  = sorted([float(x) for x in args.leg_scales.split(",")], reverse=True)
        sweep_mode = "body_leg"
        sweep_combos = [(bs, ls, bs) for bs in body_scales for ls in leg_scales]
        n_combos = len(sweep_combos)

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

    # ── Print header ──────────────────────────────────────────────────────
    print("=" * 70)
    print("Gain Sweep — Finding Softest Stable Parameters")
    print("=" * 70)
    print(f"  Terrain: wavelength={args.wavelength}mm  amplitude={args.amplitude*1000:.1f}mm")
    print(f"  Duration: {args.duration}s  (settle={baseline['settle_time']}s + ramp={baseline['ramp_time']}s)")
    if sweep_mode == "pitch":
        print(f"  Mode: PITCH-ONLY sweep (body+leg fixed at 1.0×)")
        print(f"  Pitch scales: {pitch_scales}")
    else:
        print(f"  Mode: Body × Leg grid")
        print(f"  Body scales: {body_scales}")
        print(f"  Leg scales:  {leg_scales}")
    print(f"  Total combos: {n_combos}")
    print(f"  Video: {'ON' if args.video else 'OFF'}")
    print(f"  Output: {run_dir}")
    print()
    print(f"  Baseline gains:")
    print(f"    body  kp={baseline['body_kp']:.4f}  kv={baseline['body_kv']:.4f}")
    print(f"    pitch kp={baseline['pitch_kp']:.6f}  kv={baseline['pitch_kv']:.6f}")
    print(f"    roll  kp={baseline['roll_kp']:.4f}  kv={baseline['roll_kv']:.4f}")
    print(f"    leg   kp={baseline['leg_kp']}")
    print(f"    leg   kv={baseline['leg_kv']}")
    print("=" * 70)

    results = []
    combo_i = 0
    t_start = time.time()

    for bs, ls, ps in sweep_combos:
        combo_i += 1

        if sweep_mode == "pitch":
            tag = f"pitch{ps:.2f}"
            label = f"pitch×{ps:.2f}"
        else:
            tag = f"body{bs:.2f}_leg{ls:.2f}"
            label = f"body×{bs:.2f} leg×{ls:.2f}"

        # Write temp config
        tmp_cfg = os.path.join(run_dir, f"config_{tag}.yaml")
        write_scaled_config(baseline, bs, ls, tmp_cfg, pitch_scale=ps)

        # Video path
        vid_path = None
        if args.video:
            vid_path = os.path.join(run_dir, "videos", f"{tag}.mp4")

        # ETA
        elapsed = time.time() - t_start
        if combo_i > 1:
            eta_s = elapsed / (combo_i - 1) * (n_combos - combo_i + 1)
            eta_str = f"ETA {eta_s/60:.0f}min"
        else:
            eta_str = ""

        print(f"[{combo_i:3d}/{n_combos}] {label:25s}  ", end="", flush=True)

        metrics = run_simulation(tmp_xml, tmp_cfg, args.duration, video_path=vid_path)

        status = "OK" if metrics['survived'] else f"FAIL({metrics['buckle_reason']})"
        print(f"  {status:30s}  CoT={metrics['cot']:10.2f}  "
              f"speed={metrics['forward_speed']*1000:.1f}mm/s  "
              f"pitch={metrics['max_pitch_deg']:.1f}°  "
              f"roll={metrics['max_roll_deg']:.1f}°  {eta_str}")

        result = {
            'body_scale':  bs,
            'leg_scale':   ls,
            'pitch_scale': ps,
            'body_kp':     baseline['body_kp'] * bs,
            'body_kv':     baseline['body_kv'] * bs,
            'pitch_kp':    baseline['pitch_kp'] * ps,
            'pitch_kv':    baseline['pitch_kv'] * ps,
            'roll_kp':     baseline['roll_kp'] * bs,
            'roll_kv':     baseline['roll_kv'] * bs,
            'leg_kp':      [v * ls for v in baseline['leg_kp']],
            'leg_kv':      [v * ls for v in baseline['leg_kv']],
            **metrics,
        }
        results.append(result)

        # Clean up temp config
        try:
            os.remove(tmp_cfg)
        except OSError:
            pass

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        'timestamp':     timestamp,
        'sweep_mode':    sweep_mode,
        'wavelength_mm': args.wavelength,
        'amplitude_m':   args.amplitude,
        'duration':      args.duration,
        'baseline':      baseline,
        'results':       results,
    }
    if sweep_mode == "pitch":
        output['pitch_scales'] = pitch_scales
    else:
        output['body_scales'] = body_scales
        output['leg_scales']  = leg_scales

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Summary CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ['body_scale', 'leg_scale', 'pitch_scale', 'survived', 'cot',
                    'forward_speed_mm_s', 'max_pitch_deg', 'max_roll_deg',
                    'mean_pitch_deg', 'mean_roll_deg', 'distance_mm',
                    'body_kp', 'pitch_kp', 'roll_kp']
        f.write(",".join(headers) + "\n")
        # Sort: pitch mode → by pitch_scale ascending; body_leg → by product ascending
        if sweep_mode == "pitch":
            sort_key = lambda x: x['pitch_scale']
        else:
            sort_key = lambda x: x['body_scale'] * x['leg_scale']
        for r in sorted(results, key=sort_key):
            f.write(f"{r['body_scale']:.2f},{r['leg_scale']:.2f},{r['pitch_scale']:.2f},"
                    f"{r['survived']},{r['cot']:.4f},"
                    f"{r['forward_speed']*1000:.2f},"
                    f"{r['max_pitch_deg']:.2f},{r['max_roll_deg']:.2f},"
                    f"{r['mean_pitch_deg']:.2f},{r['mean_roll_deg']:.2f},"
                    f"{r['distance']*1000:.2f},"
                    f"{r['body_kp']:.6f},{r['pitch_kp']:.6f},{r['roll_kp']:.6f}\n")

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    survived = [r for r in results if r['survived']]
    failed   = [r for r in results if not r['survived']]
    print(f"  Survived: {len(survived)}/{n_combos}")
    print(f"  Failed:   {len(failed)}/{n_combos}")

    if survived:
        if sweep_mode == "pitch":
            # Find softest surviving pitch scale
            softest = min(survived, key=lambda r: r['pitch_scale'])
            print(f"\n  SOFTEST STABLE PITCH:")
            print(f"    pitch_scale={softest['pitch_scale']:.2f}  (body+leg fixed at 1.0×)")
            print(f"    pitch_kp={softest['pitch_kp']:.6f}  pitch_kv={softest['pitch_kv']:.6f}")
            print(f"    CoT={softest['cot']:.2f}  speed={softest['forward_speed']*1000:.1f}mm/s")
            print(f"    max_pitch={softest['max_pitch_deg']:.1f}°  max_roll={softest['max_roll_deg']:.1f}°")
        else:
            # Find softest surviving combo
            softest = min(survived, key=lambda r: r['body_scale'] * r['leg_scale'])
            print(f"\n  SOFTEST STABLE:")
            print(f"    body_scale={softest['body_scale']:.2f}  leg_scale={softest['leg_scale']:.2f}")
            print(f"    body_kp={softest['body_kp']:.6f}  pitch_kp={softest['pitch_kp']:.6f}  roll_kp={softest['roll_kp']:.6f}")
            print(f"    leg_kp={softest['leg_kp']}")
            print(f"    CoT={softest['cot']:.2f}  speed={softest['forward_speed']*1000:.1f}mm/s")
            print(f"    max_pitch={softest['max_pitch_deg']:.1f}°  max_roll={softest['max_roll_deg']:.1f}°")

        # Best CoT among survived
        best_cot = min(survived, key=lambda r: r['cot'])
        print(f"\n  BEST CoT:")
        if sweep_mode == "pitch":
            print(f"    pitch_scale={best_cot['pitch_scale']:.2f}")
        else:
            print(f"    body_scale={best_cot['body_scale']:.2f}  leg_scale={best_cot['leg_scale']:.2f}")
        print(f"    CoT={best_cot['cot']:.2f}  speed={best_cot['forward_speed']*1000:.1f}mm/s")

    print(f"\n  Results: {json_path}")
    print(f"  Summary: {csv_path}")
    if args.video:
        print(f"  Videos:  {os.path.join(run_dir, 'videos')}/")

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time/60:.1f} min")

    # Clean up tmp XML
    try:
        os.remove(tmp_xml)
    except OSError:
        pass


if __name__ == "__main__":
    main()
