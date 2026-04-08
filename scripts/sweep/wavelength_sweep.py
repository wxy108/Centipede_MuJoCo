#!/usr/bin/env python3
"""
Wavelength sweep — frequency response of centipede locomotion (Bode-plot analogy).

Generates single-wavelength terrains from λ_max (world-scale) down to λ_min
(leg-scale), runs a simulation on each, and records:

  Magnitude (gain):
    - CoT(λ) / CoT_flat   — locomotion cost amplification
    - Raw CoT, speed, pitch, roll

  Phase:
    - Cross-spectral phase lag between terrain slope under the body and
      the measured body pitch angle.  At low frequencies (long λ) the body
      conforms → phase ≈ 0.  At high frequencies (short λ) the body can't
      follow → phase grows, eventually decoupling.

Output: JSON + CSV with all metrics, optional per-trial MP4 video.

Usage
-----
  python scripts/sweep/wavelength_sweep.py
  python scripts/sweep/wavelength_sweep.py --n-points 30 --duration 8 --amplitude 0.005
  python scripts/sweep/wavelength_sweep.py --video          # save per-trial MP4s
  python scripts/sweep/wavelength_sweep.py --video --n-points 5  # quick visual check

Design
------
  - Fixed amplitude across all wavelengths (same physical height)
  - Single-frequency terrain at each wavelength (narrow ±10% band)
  - Wavelengths spaced logarithmically from L_w down to L_ell/2
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
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
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
MIN_HEIGHT_M  = -0.01

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
                                       sigma=1.5, n_components=8):
    """
    Generate terrain with features at exactly one wavelength.

    Parameters
    ----------
    wavelength_m : float — target wavelength in metres
    amplitude_m  : float — RMS amplitude in metres
    seed         : int
    """
    rng = np.random.default_rng(seed)
    world_size = world_half * 2.0
    freq = 1.0 / wavelength_m

    # Narrow band: ±10% around exact frequency
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

    h = gaussian_filter(h, sigma=sigma)
    h -= h.mean()

    rms_m  = float(np.std(h))
    peak_m = float(max(abs(h.min()), abs(h.max())))

    return h, rms_m, peak_m


def save_wavelength_terrain(h_m, wavelength_m, seed, output_dir):
    """Save PNG and return path."""
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
    """Bilinear-interpolated heightfield lookup from MuJoCo model."""

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

    def get_slope_along(self, x, y, dx_m=0.002):
        """Terrain slope (dz/dx) at (x,y) along the forward x-direction."""
        h_fwd  = self.get_height(x + dx_m, y)
        h_back = self.get_height(x - dx_m, y)
        return (h_fwd - h_back) / (2.0 * dx_m)


def compute_phase_lag(terrain_slope_ts, body_pitch_ts, dt_sample):
    """
    Compute phase lag between terrain slope signal and body pitch response.

    Uses cross-spectral density at the dominant frequency of the terrain
    slope signal.  Returns phase in degrees (positive = pitch lags slope).

    Parameters
    ----------
    terrain_slope_ts : 1D array — terrain slope (dz/dx) at body COM over time
    body_pitch_ts    : 1D array — mean body pitch angle (rad) over time
    dt_sample        : float — sample interval (seconds)

    Returns
    -------
    dict with: phase_lag_deg, coherence, dominant_freq_hz
    """
    n = len(terrain_slope_ts)
    if n < 16:
        return {'phase_lag_deg': float('nan'), 'coherence': 0.0,
                'dominant_freq_hz': 0.0}

    # Remove DC
    slope = terrain_slope_ts - np.mean(terrain_slope_ts)
    pitch = body_pitch_ts - np.mean(body_pitch_ts)

    # FFT
    S = np.fft.rfft(slope)
    P = np.fft.rfft(pitch)
    freqs = np.fft.rfftfreq(n, d=dt_sample)

    # Cross-spectral density
    Sxy = S * np.conj(P)

    # Auto-spectral densities
    Sxx = np.abs(S) ** 2
    Syy = np.abs(P) ** 2

    # Find dominant frequency of the terrain slope signal (skip DC at idx 0)
    if len(Sxx) < 2:
        return {'phase_lag_deg': float('nan'), 'coherence': 0.0,
                'dominant_freq_hz': 0.0}
    peak_idx = np.argmax(Sxx[1:]) + 1

    # Phase at dominant frequency
    phase_rad = np.angle(Sxy[peak_idx])
    phase_deg = float(np.degrees(phase_rad))

    # Coherence at dominant frequency (γ² = |Sxy|² / (Sxx × Syy))
    denom = Sxx[peak_idx] * Syy[peak_idx]
    if denom > 1e-30:
        coherence = float(np.abs(Sxy[peak_idx]) ** 2 / denom)
    else:
        coherence = 0.0

    return {
        'phase_lag_deg':    phase_deg,
        'coherence':        coherence,
        'dominant_freq_hz': float(freqs[peak_idx]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Video helper
# ═══════════════════════════════════════════════════════════════════════════════

def _try_make_renderer(model):
    """Try to create a MuJoCo offscreen renderer. Returns (renderer, True) or (None, False)."""
    try:
        import mediapy  # noqa: F401 — just check it's importable
        renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
        return renderer, True
    except (ImportError, Exception):
        return None, False


def _make_tracking_camera(idx, data):
    cam = mujoco.MjvCamera()
    com = idx.com_pos(data)
    cam.lookat[:] = com
    cam.distance  = CAM_DISTANCE
    cam.azimuth   = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def _save_video(frames, path, fps=VID_FPS):
    import mediapy
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    mediapy.write_video(path, frames, fps=fps)


# ═══════════════════════════════════════════════════════════════════════════════
# XML patching (in-memory)
# ═══════════════════════════════════════════════════════════════════════════════

def patch_xml_terrain(xml_path, png_path, z_max):
    """
    Patch centipede.xml to use the given terrain PNG.
    Returns the patched XML as a string for mujoco.MjModel.from_xml_string().
    """
    from lxml import etree

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    # Update hfield
    asset = root.find("asset")
    hfield = asset.find("hfield[@name='terrain']")
    if hfield is not None:
        hfield.set("file", os.path.abspath(png_path).replace("\\", "/"))
        hfield.set("size", f"0.500 0.500 {z_max:.4f} 0.001")

    # Update spawn z
    arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32)
    nrow, ncol = arr.shape
    cy, cx = nrow // 2, ncol // 2
    r = 8
    patch = arr[max(0,cy-r):cy+r+1, max(0,cx-r):cx+r+1]
    terrain_h = (float(patch.max()) / 255.0) * z_max
    spawn_z = terrain_h + 0.015

    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            body.set("pos", f"0 0 {spawn_z:.4f}")
            break

    # Write to temp file (MuJoCo needs file path for hfield)
    tmp_xml = xml_path + ".sweep_tmp.xml"
    tree.write(tmp_xml, xml_declaration=True, encoding="utf-8", pretty_print=False)
    return tmp_xml


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation + metrics
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(xml_path, config_path, duration, video_path=None):
    """
    Run one simulation and return locomotion metrics + phase data.

    Parameters
    ----------
    xml_path    : path to patched XML
    config_path : path to controller YAML
    duration    : simulation seconds
    video_path  : if not None, save MP4 here

    Returns dict with: survived, forward_speed, cot, max_pitch_deg,
        mean_pitch_deg, max_roll_deg, mean_roll_deg, sim_time,
        phase_lag_deg, phase_coherence, phase_freq_hz
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    ctrl = ImpedanceTravelingWaveController(model, config_path)
    idx  = FARMSModelIndex(model)
    terrain = TerrainSampler(model)

    # Resolve joint/actuator IDs
    pitch_jnt_ids = []
    roll_jnt_ids  = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_jnt_ids.append(j)
        if nm and 'joint_roll_body' in nm:
            roll_jnt_ids.append(j)

    n_actuators = model.nu

    # Total mass for CoT
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    gravity = abs(model.opt.gravity[2])

    n_steps = int(duration / model.opt.timestep)
    dt = model.opt.timestep
    SETTLE_STEPS = 200

    # Find root body (freejoint)
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

    # ── Video setup ───────────────────────────────────────────────────────
    renderer = None
    frames = []
    vid_cam = None
    vid_dt = 1.0 / VID_FPS
    last_frame_t = -1.0

    if video_path:
        renderer, ok = _try_make_renderer(model)
        if ok:
            vid_cam = _make_tracking_camera(idx, data)
        else:
            video_path = None  # silently skip

    # ── Storage ───────────────────────────────────────────────────────────
    energy_sum = 0.0
    pitch_angles = []
    roll_angles  = []
    start_pos = None
    buckled = False
    buckle_reason = ""

    # Phase measurement time-series (sampled at ~100 Hz)
    PHASE_SAMPLE_INTERVAL = 50  # every 50 physics steps
    terrain_slope_ts = []
    body_pitch_ts    = []
    phase_sample_dt  = PHASE_SAMPLE_INTERVAL * dt

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Record start position after settling
        if step_i == SETTLE_STEPS and root_body is not None:
            start_pos = data.xpos[root_body].copy()

        # Energy: Σ |τ_i · ω_i| · dt  (skip transient)
        if step_i > SETTLE_STEPS:
            for a in range(n_actuators):
                tau = abs(data.actuator_force[a])
                jnt_id = model.actuator_trnid[a, 0]
                if 0 <= jnt_id < model.njnt:
                    dof_adr = model.jnt_dofadr[jnt_id]
                    omega = abs(data.qvel[dof_adr])
                    energy_sum += tau * omega * dt

        # ── Phase sampling ────────────────────────────────────────────────
        if step_i > SETTLE_STEPS and step_i % PHASE_SAMPLE_INTERVAL == 0:
            # Terrain slope under body COM (forward direction)
            com = idx.com_pos(data)
            slope_x = terrain.get_slope_along(com[0], com[1])
            terrain_slope_ts.append(slope_x)

            # Mean body pitch angle (average across all pitch joints)
            if pitch_jnt_ids:
                mean_pitch = np.mean([
                    data.qpos[model.jnt_qposadr[jid]] for jid in pitch_jnt_ids
                ])
            else:
                mean_pitch = 0.0
            body_pitch_ts.append(mean_pitch)

        # ── Video frame capture ───────────────────────────────────────────
        if renderer and video_path and data.time - last_frame_t >= vid_dt - 1e-6:
            vid_cam.lookat[:] = idx.com_pos(data)
            renderer.update_scene(data, camera=vid_cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

        # Failure check every 200 steps
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

        # Record pitch/roll extremes every 100 steps
        if step_i % 100 == 0:
            for jid in pitch_jnt_ids:
                pitch_angles.append(abs(math.degrees(
                    data.qpos[model.jnt_qposadr[jid]])))
            for jid in roll_jnt_ids:
                roll_angles.append(abs(math.degrees(
                    data.qpos[model.jnt_qposadr[jid]])))

    # ── Save video ────────────────────────────────────────────────────────
    if video_path and frames:
        _save_video(frames, video_path)

    # ── Compute locomotion metrics ────────────────────────────────────────
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

    # ── Compute phase lag ─────────────────────────────────────────────────
    phase_info = compute_phase_lag(
        np.array(terrain_slope_ts),
        np.array(body_pitch_ts),
        phase_sample_dt,
    )

    return {
        'survived':         not buckled,
        'buckle_reason':    buckle_reason,
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
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-points",  type=int,   default=20,
                        help="Number of wavelengths to test (log-spaced)")
    parser.add_argument("--duration",  type=float, default=8.0,
                        help="Simulation duration per wavelength (seconds)")
    parser.add_argument("--amplitude", type=float, default=0.005,
                        help="Fixed terrain RMS amplitude (metres)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--wl-min",    type=float, default=None,
                        help="Min wavelength (m). Default: L_ell / 2")
    parser.add_argument("--wl-max",    type=float, default=None,
                        help="Max wavelength (m). Default: L_w")
    parser.add_argument("--video",     action="store_true",
                        help="Save an MP4 video for each wavelength trial")
    args = parser.parse_args()

    # Load morphology
    with open(TERRAIN_CFG, encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    lengths = resolve_morphology(t_cfg)
    img_size = int(t_cfg["world"]["image_size"])
    world_half = float(t_cfg["world"]["size"])
    sigma = float(t_cfg["world"]["smooth_sigma"])

    wl_min = args.wl_min if args.wl_min else lengths["L_ell"] / 2.0
    wl_max = args.wl_max if args.wl_max else lengths["L_w"]

    # Log-spaced wavelengths from large to small
    wavelengths = np.logspace(np.log10(wl_max), np.log10(wl_min), args.n_points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"sweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Check video capability once
    can_video = False
    if args.video:
        try:
            import mediapy  # noqa: F401
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed — skipping video recording.")
            print("           Install with: pip install mediapy")

    print("=" * 70)
    print("Wavelength Sweep — Centipede Frequency Response (Bode plot)")
    print("=" * 70)
    print(f"  Morphology: L_w={lengths['L_w']*1000:.0f}mm  "
          f"L_b={lengths['L_b']*1000:.1f}mm  "
          f"L_s={lengths['L_s']*1000:.1f}mm  "
          f"L_ell={lengths['L_ell']*1000:.1f}mm")
    print(f"  Wavelength range: {wl_max*1000:.0f}mm -> {wl_min*1000:.1f}mm  "
          f"({args.n_points} points, log-spaced)")
    print(f"  Fixed amplitude: {args.amplitude*1000:.1f}mm RMS")
    print(f"  Duration: {args.duration}s per trial")
    print(f"  Video: {'ON' if can_video else 'OFF'}")
    print(f"  Output: {run_dir}")
    print()

    # Mark morphology scales on the sweep
    scales = {
        'L_w':   lengths['L_w'],
        'L_b':   lengths['L_b'],
        'L_s':   lengths['L_s'],
        'L_ell': lengths['L_ell'],
    }

    results = []
    t0 = time.time()

    for i, wl in enumerate(wavelengths):
        # Check which morphology scale this wavelength is near
        nearest = min(scales.items(), key=lambda kv: abs(kv[1] - wl))
        scale_tag = f"(near {nearest[0]})" if abs(nearest[1] - wl) / nearest[1] < 0.3 else ""

        print(f"[{i+1:3d}/{args.n_points}] λ = {wl*1000:8.1f} mm  "
              f"(f = {1/wl:7.1f} cyc/m)  {scale_tag}")

        # Generate terrain
        h_m, rms_m, peak_m = generate_single_wavelength_terrain(
            wavelength_m=wl,
            amplitude_m=args.amplitude,
            seed=args.seed + i,
            image_size=img_size,
            world_half=world_half,
            sigma=sigma,
        )

        # Save terrain PNG
        png_path = save_wavelength_terrain(h_m, wl, args.seed + i, run_dir)

        # z_max: at least 2× peak so terrain fits in hfield range
        z_max = max(2.0 * peak_m, 0.005)

        # Patch XML
        tmp_xml = patch_xml_terrain(XML_PATH, png_path, z_max)

        # Video path for this trial
        vid_path = None
        if can_video:
            vid_dir = os.path.join(run_dir, "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"wl_{wl*1000:.0f}mm.mp4")

        # Run simulation
        try:
            metrics = run_simulation(tmp_xml, CONFIG_PATH, args.duration,
                                     video_path=vid_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            metrics = {'survived': False, 'buckle_reason': str(e),
                       'cot': 1e6, 'forward_speed': 0, 'distance_m': 0,
                       'max_pitch_deg': 0, 'mean_pitch_deg': 0,
                       'max_roll_deg': 0, 'mean_roll_deg': 0,
                       'energy_J': 0, 'sim_time': 0, 'total_mass_kg': 0,
                       'phase_lag_deg': float('nan'), 'phase_coherence': 0,
                       'phase_freq_hz': 0, 'video_path': ''}
        finally:
            # Clean up temp XML
            if os.path.exists(tmp_xml):
                os.remove(tmp_xml)

        entry = {
            'wavelength_m':  float(wl),
            'wavelength_mm': float(wl * 1000),
            'frequency':     float(1.0 / wl),
            'amplitude_m':   float(args.amplitude),
            'rms_m':         float(rms_m),
            'z_max':         float(z_max),
            **metrics,
        }
        results.append(entry)

        if metrics['survived']:
            phase_str = (f"  phase={metrics['phase_lag_deg']:.1f}deg"
                         if not math.isnan(metrics.get('phase_lag_deg', float('nan')))
                         else "  phase=N/A")
            print(f"    -> CoT={metrics['cot']:.2f}  "
                  f"speed={metrics['forward_speed']*1000:.1f}mm/s  "
                  f"pitch={metrics['max_pitch_deg']:.1f}deg"
                  f"{phase_str}  "
                  f"dist={metrics['distance_m']*1000:.0f}mm")
        else:
            print(f"    -> FAIL: {metrics['buckle_reason']}")

    elapsed = time.time() - t0

    # ── Save results ──────────────────────────────────────────────────────
    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, 'w') as f:
        json.dump({
            'timestamp':   timestamp,
            'n_points':    args.n_points,
            'duration':    args.duration,
            'amplitude':   args.amplitude,
            'wl_min':      float(wl_min),
            'wl_max':      float(wl_max),
            'morphology':  {k: float(v) for k, v in lengths.items()},
            'elapsed_s':   elapsed,
            'results':     results,
        }, f, indent=2)

    # CSV for easy plotting
    out_csv = os.path.join(run_dir, "results.csv")
    with open(out_csv, 'w') as f:
        headers = ['wavelength_mm', 'frequency', 'cot', 'forward_speed',
                    'max_pitch_deg', 'mean_pitch_deg', 'max_roll_deg',
                    'mean_roll_deg', 'distance_m', 'energy_J', 'survived',
                    'phase_lag_deg', 'phase_coherence', 'phase_freq_hz']
        f.write(','.join(headers) + '\n')
        for r in results:
            vals = [str(r.get(h, '')) for h in headers]
            f.write(','.join(vals) + '\n')

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"DONE  ({elapsed:.0f}s, {args.n_points} wavelengths)")
    print(f"{'=' * 70}")

    survived = [r for r in results if r['survived']]
    failed   = [r for r in results if not r['survived']]

    print(f"\n  Survived: {len(survived)}/{len(results)}")
    if failed:
        fail_strs = [f"{r['wavelength_mm']:.1f}mm" for r in failed]
        print(f"  Failed at λ: {', '.join(fail_strs)}")

    if survived:
        # Find critical transitions
        by_wl = sorted(survived, key=lambda r: r['wavelength_m'], reverse=True)
        print(f"\n  CoT range: {min(r['cot'] for r in survived):.2f} - "
              f"{max(r['cot'] for r in survived):.2f}")
        print(f"  Speed range: {min(r['forward_speed'] for r in survived)*1000:.1f} - "
              f"{max(r['forward_speed'] for r in survived)*1000:.1f} mm/s")

        # Phase summary
        phase_valid = [r for r in survived if not math.isnan(r.get('phase_lag_deg', float('nan')))]
        if phase_valid:
            print(f"\n  Phase lag range: "
                  f"{min(r['phase_lag_deg'] for r in phase_valid):.1f}deg - "
                  f"{max(r['phase_lag_deg'] for r in phase_valid):.1f}deg")
            hi_coh = [r for r in phase_valid if r['phase_coherence'] > 0.3]
            if hi_coh:
                print(f"  High-coherence trials (gamma^2 > 0.3): {len(hi_coh)}/{len(phase_valid)}")

        # Mark morphology scales
        print(f"\n  Morphology reference lines for plotting:")
        for name, val in sorted(scales.items(), key=lambda kv: -kv[1]):
            print(f"    {name:>5s} = {val*1000:>7.1f} mm  (f = {1/val:.1f} cyc/m)")

    print(f"\n  Results: {out_json}")
    print(f"  CSV:     {out_csv}")
    if can_video:
        print(f"  Videos:  {os.path.join(run_dir, 'videos')}/")
    print(f"\n  Bode plot guide:")
    print(f"    Magnitude: plot CoT(lam)/CoT_flat vs wavelength (log-x)")
    print(f"    Phase:     plot phase_lag_deg vs wavelength (log-x)")
    print(f"    Draw vertical lines at L_w, L_b, L_s, L_ell.")


if __name__ == "__main__":
    main()
