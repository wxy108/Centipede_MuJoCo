#!/usr/bin/env python3
"""
Wavelength sweep — frequency response of centipede locomotion (Bode-plot analogy).

For each wavelength, runs N trials with the centipede spawned at random yaw
angles (0-360°) to average out directional bias.

Metrics per trial:
  - Cost of Transport (CoT) = Σ|τ·ω|·dt / (m·g·d)
  - Forward speed, pitch, roll
  - Phase lag: terrain-slope → body-pitch cross-spectral phase

Aggregated per wavelength: mean, std, median, min, max across trials.

Usage
-----
  python scripts/sweep/wavelength_sweep.py --n-points 20 --n-trials 15 --duration 5 --amplitude 0.004
  python scripts/sweep/wavelength_sweep.py --n-points 20 --n-trials 15 --duration 5 --amplitude 0.004 --video

Design
------
  - Fixed amplitude across all wavelengths (same physical height)
  - Single-frequency terrain at each wavelength (narrow ±10% band)
  - Wavelengths log-spaced from L_w down to L_ell/2
  - N random yaw rotations per wavelength (random heading 0-360°)
  - Robot morphology read from configs/terrain.yaml
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import mujoco

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "terrain", "generator"))

from impedance_controller import ImpedanceTravelingWaveController
from kinematics import FARMSModelIndex

import yaml
from generate import (resolve_morphology, heightmap_to_png, _spectral_band)
from scipy.ndimage import gaussian_filter
from PIL import Image

XML_PATH    = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")
TERRAIN_CFG = os.path.join(PROJECT_ROOT, "configs", "terrain.yaml")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "wavelength_sweep")

# Failure thresholds
MAX_PITCH_DEG = 35.0
MAX_ROLL_DEG  = 60.0

# Video settings
VID_W, VID_H = 1280, 720
VID_FPS      = 30
CAM_DISTANCE = 0.20
CAM_AZIMUTH  = 60
CAM_ELEVATION = -35


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain generation: single-wavelength
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

    # Adaptive Gaussian smooth: sigma = 0.25× pixels-per-cycle
    # Removes sub-wavelength pixelation without attenuating the signal.
    px_per_cycle = wavelength_m / world_size * image_size
    sigma = max(0.25 * px_per_cycle, 0.5)
    h = gaussian_filter(h, sigma=sigma)
    h -= h.mean()

    # Rescale so peak amplitude matches the requested amplitude exactly
    cur_peak = max(abs(h.min()), abs(h.max()))
    if cur_peak > 1e-12:
        h = h * (amplitude_m / cur_peak)

    rms_m  = float(np.std(h))
    peak_m = float(max(abs(h.min()), abs(h.max())))
    return h, rms_m, peak_m


def save_wavelength_terrain(h_m, wavelength_m, seed, output_dir):
    arr = heightmap_to_png(h_m)
    tag = f"wl{wavelength_m*1000:.1f}mm_s{seed}"
    out_dir = os.path.join(output_dir, "terrains", tag)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "1.png")
    Image.fromarray(arr, mode="L").save(png_path)
    return png_path


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain height sampler (for phase measurement)
# ═══════════════════════════════════════════════════════════════════════════════

class TerrainSampler:
    def __init__(self, model):
        hf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hf_id < 0:
            raise ValueError("Heightfield 'terrain' not found")
        self.nrow = model.hfield_nrow[hf_id]
        self.ncol = model.hfield_ncol[hf_id]
        self.x_half = model.hfield_size[hf_id, 0]
        self.y_half = model.hfield_size[hf_id, 1]
        self.z_top  = model.hfield_size[hf_id, 2]
        n_data = self.nrow * self.ncol
        start = model.hfield_adr[hf_id]
        self.data = model.hfield_data[start:start + n_data].reshape(
            self.nrow, self.ncol).copy()
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
        self.offset = model.geom_pos[geom_id].copy() if geom_id >= 0 else np.zeros(3)

    def get_height(self, x, y):
        lx = x - self.offset[0]
        ly = y - self.offset[1]
        u = np.clip((lx + self.x_half) / (2.0 * self.x_half), 0, 1 - 1e-10)
        v = np.clip((ly + self.y_half) / (2.0 * self.y_half), 0, 1 - 1e-10)
        col = u * (self.ncol - 1)
        row = v * (self.nrow - 1)
        c0, r0 = int(col), int(row)
        c1, r1 = min(c0 + 1, self.ncol - 1), min(r0 + 1, self.nrow - 1)
        fc, fr = col - c0, row - r0
        h = (self.data[r0, c0] * (1 - fc) * (1 - fr) +
             self.data[r0, c1] * fc * (1 - fr) +
             self.data[r1, c0] * (1 - fc) * fr +
             self.data[r1, c1] * fc * fr)
        return self.offset[2] + h * self.z_top

    def get_slope_along(self, x, y, heading_rad, dx_m=0.002):
        """Terrain slope along the robot's forward direction."""
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        h_fwd  = self.get_height(x + dx_m * cos_h, y + dx_m * sin_h)
        h_back = self.get_height(x - dx_m * cos_h, y - dx_m * sin_h)
        return (h_fwd - h_back) / (2.0 * dx_m)


def compute_phase_lag(terrain_slope_ts, body_pitch_ts, dt_sample):
    n = len(terrain_slope_ts)
    if n < 16:
        return {'phase_lag_deg': float('nan'), 'coherence': 0.0,
                'dominant_freq_hz': 0.0}
    slope = terrain_slope_ts - np.mean(terrain_slope_ts)
    pitch = body_pitch_ts - np.mean(body_pitch_ts)
    S = np.fft.rfft(slope)
    P = np.fft.rfft(pitch)
    freqs = np.fft.rfftfreq(n, d=dt_sample)
    Sxy = S * np.conj(P)
    Sxx = np.abs(S) ** 2
    Syy = np.abs(P) ** 2
    if len(Sxx) < 2:
        return {'phase_lag_deg': float('nan'), 'coherence': 0.0,
                'dominant_freq_hz': 0.0}
    peak_idx = np.argmax(Sxx[1:]) + 1
    phase_rad = np.angle(Sxy[peak_idx])
    phase_deg = float(np.degrees(phase_rad))
    denom = Sxx[peak_idx] * Syy[peak_idx]
    coherence = float(np.abs(Sxy[peak_idx]) ** 2 / denom) if denom > 1e-30 else 0.0
    return {
        'phase_lag_deg': phase_deg,
        'coherence': coherence,
        'dominant_freq_hz': float(freqs[peak_idx]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Video helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _try_make_renderer(model):
    try:
        import mediapy  # noqa
        return mujoco.Renderer(model, height=VID_H, width=VID_W), True
    except (ImportError, Exception):
        return None, False

def _make_tracking_camera(idx, data):
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
# XML patching
# ═══════════════════════════════════════════════════════════════════════════════

def patch_xml_terrain(xml_path, png_path, z_max):
    """Patch the base XML to include a heightfield terrain.

    The base centipede.xml ships with NO hfield (flat plane only) so that
    ``run.py`` works out-of-the-box.  This function adds (or updates) the
    hfield asset + geom, sets spawn height, and writes a temp XML.
    """
    from lxml import etree
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    abs_png = os.path.abspath(png_path).replace("\\", "/")

    # ── hfield asset ──────────────────────────────────────────────────────
    asset = root.find("asset")
    hfield = asset.find("hfield[@name='terrain']")
    if hfield is not None:
        hfield.set("file", abs_png)
        hfield.set("size", f"0.500 0.500 {z_max:.4f} 0.001")
    else:
        etree.SubElement(asset, "hfield", {
            "name": "terrain",
            "file": abs_png,
            "size": f"0.500 0.500 {z_max:.4f} 0.001",
            "nrow": "1024", "ncol": "1024",
        })

    # ── hfield geom in worldbody ──────────────────────────────────────────
    worldbody = root.find("worldbody")
    terrain_geom = worldbody.find("geom[@name='terrain_geom']")
    if terrain_geom is None:
        etree.SubElement(worldbody, "geom", {
            "type": "hfield", "name": "terrain_geom",
            "hfield": "terrain", "pos": "0 0 0",
            "conaffinity": "1", "condim": "3",
            "friction": "1.6 0.005 0.0001",
        })

    # ── spawn height ──────────────────────────────────────────────────────
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

    tmp_xml = xml_path + ".sweep_tmp.xml"
    tree.write(tmp_xml, xml_declaration=True, encoding="utf-8", pretty_print=False)
    return tmp_xml


# ═══════════════════════════════════════════════════════════════════════════════
# Set initial yaw rotation
# ═══════════════════════════════════════════════════════════════════════════════

def set_initial_yaw(model, data, yaw_rad):
    """
    Rotate the centipede's initial orientation around the z-axis.

    The freejoint qpos layout is [x, y, z, qw, qx, qy, qz].
    Rotation about z by angle θ → quaternion [cos(θ/2), 0, 0, sin(θ/2)].
    """
    # Find the freejoint
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            # Set quaternion (indices 3-6 of freejoint qpos)
            data.qpos[qadr + 3] = math.cos(yaw_rad / 2.0)   # qw
            data.qpos[qadr + 4] = 0.0                         # qx
            data.qpos[qadr + 5] = 0.0                         # qy
            data.qpos[qadr + 6] = math.sin(yaw_rad / 2.0)   # qz
            break

    # Forward kinematics to propagate the new orientation
    mujoco.mj_forward(model, data)


def get_body_heading(model, data):
    """Get the current yaw heading of the root body (rad)."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            qw = data.qpos[qadr + 3]
            qx = data.qpos[qadr + 4]
            qy = data.qpos[qadr + 5]
            qz = data.qpos[qadr + 6]
            # Extract yaw from quaternion
            return math.atan2(2.0 * (qw * qz + qx * qy),
                              1.0 - 2.0 * (qy * qy + qz * qz))
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Single simulation run
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(xml_path, config_path, duration, yaw_rad=0.0, video_path=None):
    """
    Run one simulation with the centipede rotated to yaw_rad.

    Returns dict of locomotion metrics.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Set initial rotation BEFORE controller init
    set_initial_yaw(model, data, yaw_rad)

    ctrl = ImpedanceTravelingWaveController(model, config_path)
    idx  = FARMSModelIndex(model)
    terrain = TerrainSampler(model)

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
    SETTLE_STEPS = 6000   # 3.0 s at dt=0.0005 — 1 s settle + 2 s sequential ramp

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

    # Video — set camera azimuth once from spawn yaw (not every frame)
    renderer, frames, vid_cam = None, [], None
    vid_dt = 1.0 / VID_FPS
    last_frame_t = -1.0
    cam_azimuth_fixed = CAM_AZIMUTH + math.degrees(yaw_rad)
    if video_path:
        renderer, ok = _try_make_renderer(model)
        if ok:
            vid_cam = _make_tracking_camera(idx, data)
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

    PHASE_SAMPLE_INTERVAL = 50
    terrain_slope_ts = []
    body_pitch_ts    = []
    phase_sample_dt  = PHASE_SAMPLE_INTERVAL * dt

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if step_i == SETTLE_STEPS and root_body is not None:
            start_pos = data.xpos[root_body].copy()

        # Energy
        if step_i > SETTLE_STEPS:
            for a in range(n_actuators):
                tau = abs(data.actuator_force[a])
                jnt_id = model.actuator_trnid[a, 0]
                if 0 <= jnt_id < model.njnt:
                    dof_adr = model.jnt_dofadr[jnt_id]
                    omega = abs(data.qvel[dof_adr])
                    energy_sum += tau * omega * dt

        # Phase sampling
        if step_i > SETTLE_STEPS and step_i % PHASE_SAMPLE_INTERVAL == 0:
            com = idx.com_pos(data)
            heading = get_body_heading(model, data)
            slope = terrain.get_slope_along(com[0], com[1], heading)
            terrain_slope_ts.append(slope)
            if pitch_jnt_ids:
                mean_pitch = np.mean([
                    data.qpos[model.jnt_qposadr[jid]] for jid in pitch_jnt_ids
                ])
            else:
                mean_pitch = 0.0
            body_pitch_ts.append(mean_pitch)

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

        # Record pitch/roll
        if step_i % 100 == 0:
            for jid in pitch_jnt_ids:
                pitch_angles.append(abs(math.degrees(
                    data.qpos[model.jnt_qposadr[jid]])))
            for jid in roll_jnt_ids:
                roll_angles.append(abs(math.degrees(
                    data.qpos[model.jnt_qposadr[jid]])))

    # Save video
    if video_path and frames:
        _save_video(frames, video_path)

    # Metrics
    end_pos = data.xpos[root_body].copy() if root_body is not None else None
    if start_pos is not None and end_pos is not None:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
    else:
        distance = 0.0

    settle_time = SETTLE_STEPS * dt
    effective_time = max(data.time - settle_time, 0.01)
    forward_speed = distance / effective_time

    if distance > 0.001:
        cot = energy_sum / (total_mass * gravity * distance)
    else:
        cot = float('inf')

    phase_info = compute_phase_lag(
        np.array(terrain_slope_ts), np.array(body_pitch_ts), phase_sample_dt)

    return {
        'survived':         not buckled,
        'buckle_reason':    buckle_reason,
        'yaw_deg':          float(math.degrees(yaw_rad)),
        'sim_time':         float(data.time),
        'distance_m':       float(distance),
        'forward_speed':    float(forward_speed),
        'cot':              float(min(cot, 1e6)),
        'energy_J':         float(energy_sum),
        'max_pitch_deg':    float(max(pitch_angles)) if pitch_angles else 0,
        'mean_pitch_deg':   float(np.mean(pitch_angles)) if pitch_angles else 0,
        'max_roll_deg':     float(max(roll_angles)) if roll_angles else 0,
        'mean_roll_deg':    float(np.mean(roll_angles)) if roll_angles else 0,
        'total_mass_kg':    float(total_mass),
        'phase_lag_deg':    float(phase_info['phase_lag_deg']),
        'phase_coherence':  float(phase_info['coherence']),
        'phase_freq_hz':    float(phase_info['dominant_freq_hz']),
        'video_path':       video_path or '',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate batch results
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_trials(trial_results):
    """Compute mean/std/median/min/max of key metrics across trials."""
    survived = [r for r in trial_results if r['survived']]
    n_survived = len(survived)
    n_total = len(trial_results)

    if not survived:
        return {
            'n_trials': n_total, 'n_survived': 0, 'survival_rate': 0.0,
            'cot_mean': float('inf'), 'cot_std': 0, 'cot_median': float('inf'),
            'speed_mean': 0, 'speed_std': 0,
            'max_pitch_mean': 0, 'max_roll_mean': 0,
            'phase_lag_mean': float('nan'), 'phase_lag_std': float('nan'),
        }

    cots   = [r['cot'] for r in survived]
    speeds = [r['forward_speed'] for r in survived]
    pitches = [r['max_pitch_deg'] for r in survived]
    rolls   = [r['max_roll_deg'] for r in survived]
    phases  = [r['phase_lag_deg'] for r in survived
               if not math.isnan(r['phase_lag_deg'])]

    return {
        'n_trials':       n_total,
        'n_survived':     n_survived,
        'survival_rate':  n_survived / n_total,
        'cot_mean':       float(np.mean(cots)),
        'cot_std':        float(np.std(cots)),
        'cot_median':     float(np.median(cots)),
        'cot_min':        float(np.min(cots)),
        'cot_max':        float(np.max(cots)),
        'speed_mean':     float(np.mean(speeds)),
        'speed_std':      float(np.std(speeds)),
        'max_pitch_mean': float(np.mean(pitches)),
        'max_pitch_std':  float(np.std(pitches)),
        'max_roll_mean':  float(np.mean(rolls)),
        'max_roll_std':   float(np.std(rolls)),
        'phase_lag_mean': float(np.mean(phases)) if phases else float('nan'),
        'phase_lag_std':  float(np.std(phases)) if phases else float('nan'),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-points",  type=int,   default=20,
                        help="Number of wavelengths to test (log-spaced)")
    parser.add_argument("--n-trials",  type=int,   default=15,
                        help="Number of random-rotation trials per wavelength")
    parser.add_argument("--duration",  type=float, default=8.0,
                        help="Simulation duration per trial (seconds). "
                             "Includes 1 s settle + 2 s head→tail ramp + ~5 s active gait.")
    parser.add_argument("--amplitude", type=float, default=0.004,
                        help="Fixed terrain height amplitude (metres)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--wl-min",    type=float, default=None,
                        help="Min wavelength (m). Default: L_ell / 2")
    parser.add_argument("--wl-max",    type=float, default=None,
                        help="Max wavelength (m). Default: L_w")
    parser.add_argument("--video",     action="store_true",
                        help="Save an MP4 video for each trial")
    parser.add_argument("--wavelengths", type=str, default=None,
                        help="Explicit wavelengths in mm, comma-separated "
                             "(e.g. '4,7,10,20,50'). Overrides --n-points/wl-min/wl-max.")
    args = parser.parse_args()

    # Load morphology
    with open(TERRAIN_CFG, encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    lengths = resolve_morphology(t_cfg)
    img_size = int(t_cfg["world"]["image_size"])
    world_half = float(t_cfg["world"]["size"])
    sigma = float(t_cfg["world"]["smooth_sigma"])

    if args.wavelengths:
        # Explicit wavelength list (mm → m), sorted long-to-short
        wavelengths = sorted([float(w.strip()) / 1000.0
                              for w in args.wavelengths.split(",")], reverse=True)
        wavelengths = np.array(wavelengths)
        args.n_points = len(wavelengths)
    else:
        wl_min = args.wl_min if args.wl_min else lengths["L_ell"] / 2.0
        wl_max = args.wl_max if args.wl_max else lengths["L_w"]
        wavelengths = np.logspace(np.log10(wl_max), np.log10(wl_min), args.n_points)

    # Pre-generate random yaw angles for all trials (reproducible)
    rng = np.random.default_rng(args.seed + 9999)
    all_yaws = rng.uniform(0, 2 * math.pi, size=(args.n_points, args.n_trials))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"sweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Video check
    can_video = False
    if args.video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed -- skipping video.")

    total_sims = args.n_points * args.n_trials
    print("=" * 70)
    print("Wavelength Sweep — Batch Random Rotation")
    print("=" * 70)
    print(f"  Morphology: L_w={lengths['L_w']*1000:.0f}mm  "
          f"L_b={lengths['L_b']*1000:.1f}mm  "
          f"L_s={lengths['L_s']*1000:.1f}mm  "
          f"L_ell={lengths['L_ell']*1000:.1f}mm")
    print(f"  Wavelengths: {args.n_points} points, "
          f"{wavelengths[0]*1000:.1f}mm -> {wavelengths[-1]*1000:.1f}mm")
    print(f"  Trials per wavelength: {args.n_trials} (random yaw 0-360)")
    print(f"  Total simulations: {total_sims}")
    print(f"  Duration: {args.duration}s each, amplitude: {args.amplitude*1000:.1f}mm")
    print(f"  Video: {'ON' if can_video else 'OFF'}")
    print(f"  Output: {run_dir}")
    print()

    scales = {
        'L_w': lengths['L_w'], 'L_b': lengths['L_b'],
        'L_s': lengths['L_s'], 'L_ell': lengths['L_ell'],
    }

    # Results: per-wavelength aggregated + all raw trials
    wavelength_results = []
    all_trial_results = []
    t0 = time.time()
    sim_count = 0

    for i, wl in enumerate(wavelengths):
        nearest = min(scales.items(), key=lambda kv: abs(kv[1] - wl))
        scale_tag = (f"(near {nearest[0]})"
                     if abs(nearest[1] - wl) / nearest[1] < 0.3 else "")

        print(f"[WL {i+1:2d}/{args.n_points}] "
              f"lam = {wl*1000:7.1f} mm  (f = {1/wl:7.1f} cyc/m)  {scale_tag}")

        # Generate terrain (same for all trials at this wavelength)
        h_m, rms_m, peak_m = generate_single_wavelength_terrain(
            wavelength_m=wl, amplitude_m=args.amplitude,
            seed=args.seed + i, image_size=img_size,
            world_half=world_half,
        )
        png_path = save_wavelength_terrain(h_m, wl, args.seed + i, run_dir)
        z_max = max(2.0 * peak_m, 0.005)
        tmp_xml = patch_xml_terrain(XML_PATH, png_path, z_max)

        trial_results = []

        for t in range(args.n_trials):
            sim_count += 1
            yaw = float(all_yaws[i, t])
            yaw_deg = math.degrees(yaw)

            vid_path = None
            if can_video:
                vid_dir = os.path.join(run_dir, "videos",
                                       f"wl_{wl*1000:.0f}mm")
                os.makedirs(vid_dir, exist_ok=True)
                vid_path = os.path.join(vid_dir, f"trial_{t:02d}_yaw{yaw_deg:.0f}.mp4")

            try:
                metrics = run_simulation(tmp_xml, CONFIG_PATH, args.duration,
                                         yaw_rad=yaw, video_path=vid_path)
            except Exception as e:
                metrics = {
                    'survived': False, 'buckle_reason': str(e),
                    'yaw_deg': yaw_deg, 'cot': 1e6, 'forward_speed': 0,
                    'distance_m': 0, 'max_pitch_deg': 0, 'mean_pitch_deg': 0,
                    'max_roll_deg': 0, 'mean_roll_deg': 0, 'energy_J': 0,
                    'sim_time': 0, 'total_mass_kg': 0,
                    'phase_lag_deg': float('nan'), 'phase_coherence': 0,
                    'phase_freq_hz': 0, 'video_path': '',
                }

            metrics['wavelength_m'] = float(wl)
            metrics['wavelength_mm'] = float(wl * 1000)
            metrics['trial_idx'] = t
            trial_results.append(metrics)
            all_trial_results.append(metrics)

            status = ("OK" if metrics['survived']
                      else f"FAIL:{metrics['buckle_reason']}")
            elapsed_total = time.time() - t0
            eta = (elapsed_total / sim_count) * (total_sims - sim_count)
            print(f"  [{sim_count:3d}/{total_sims}] "
                  f"yaw={yaw_deg:5.1f}  "
                  f"CoT={metrics['cot']:7.1f}  "
                  f"speed={metrics['forward_speed']*1000:5.1f}mm/s  "
                  f"{status}  "
                  f"(ETA {eta/60:.0f}min)", flush=True)

        # Clean up temp XML after all trials for this wavelength
        if os.path.exists(tmp_xml):
            os.remove(tmp_xml)

        # Aggregate this wavelength
        agg = aggregate_trials(trial_results)
        agg['wavelength_m'] = float(wl)
        agg['wavelength_mm'] = float(wl * 1000)
        agg['frequency'] = float(1.0 / wl)
        agg['amplitude_m'] = float(args.amplitude)
        agg['rms_m'] = float(rms_m)
        wavelength_results.append(agg)

        print(f"  => CoT: {agg['cot_mean']:.1f} +/- {agg['cot_std']:.1f}  "
              f"speed: {agg['speed_mean']*1000:.1f} +/- {agg['speed_std']*1000:.1f}mm/s  "
              f"survived: {agg['n_survived']}/{agg['n_trials']}")
        print()

    elapsed = time.time() - t0

    # ── Save results ──────────────────────────────────────────────────────
    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, 'w') as f:
        json.dump({
            'timestamp':    timestamp,
            'n_points':     args.n_points,
            'n_trials':     args.n_trials,
            'duration':     args.duration,
            'amplitude':    args.amplitude,
            'wl_min':       float(wavelengths[-1]),
            'wl_max':       float(wavelengths[0]),
            'morphology':   {k: float(v) for k, v in lengths.items()},
            'elapsed_s':    elapsed,
            'wavelength_results': wavelength_results,
            'all_trials':   all_trial_results,
        }, f, indent=2)

    # Aggregated CSV (one row per wavelength — for plotting)
    out_csv = os.path.join(run_dir, "results_aggregated.csv")
    with open(out_csv, 'w') as f:
        headers = ['wavelength_mm', 'frequency', 'n_survived', 'survival_rate',
                    'cot_mean', 'cot_std', 'cot_median', 'cot_min', 'cot_max',
                    'speed_mean', 'speed_std',
                    'max_pitch_mean', 'max_pitch_std',
                    'max_roll_mean', 'max_roll_std',
                    'phase_lag_mean', 'phase_lag_std']
        f.write(','.join(headers) + '\n')
        for r in wavelength_results:
            vals = [str(r.get(h, '')) for h in headers]
            f.write(','.join(vals) + '\n')

    # Raw CSV (one row per trial)
    out_raw = os.path.join(run_dir, "results_all_trials.csv")
    with open(out_raw, 'w') as f:
        headers = ['wavelength_mm', 'trial_idx', 'yaw_deg', 'survived',
                    'cot', 'forward_speed', 'distance_m',
                    'max_pitch_deg', 'mean_pitch_deg',
                    'max_roll_deg', 'mean_roll_deg',
                    'energy_J', 'phase_lag_deg', 'phase_coherence']
        f.write(','.join(headers) + '\n')
        for r in all_trial_results:
            vals = [str(r.get(h, '')) for h in headers]
            f.write(','.join(vals) + '\n')

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"DONE  ({elapsed/60:.1f} min, {total_sims} simulations)")
    print(f"{'=' * 70}")

    all_survived = [r for r in wavelength_results if r['n_survived'] > 0]
    if all_survived:
        cots = [r['cot_mean'] for r in all_survived]
        print(f"\n  Mean CoT range: {min(cots):.1f} - {max(cots):.1f}")
        speeds = [r['speed_mean'] * 1000 for r in all_survived]
        print(f"  Mean speed range: {min(speeds):.1f} - {max(speeds):.1f} mm/s")

    print(f"\n  Morphology reference lines:")
    for name, val in sorted(scales.items(), key=lambda kv: -kv[1]):
        print(f"    {name:>5s} = {val*1000:>7.1f} mm")

    print(f"\n  Results JSON:     {out_json}")
    print(f"  Aggregated CSV:   {out_csv}")
    print(f"  All trials CSV:   {out_raw}")
    if can_video:
        print(f"  Videos:           {os.path.join(run_dir, 'videos')}/")
    print(f"\n  To plot: python analysis/wavelength_sweep/plot_bode.py")


if __name__ == "__main__":
    main()
