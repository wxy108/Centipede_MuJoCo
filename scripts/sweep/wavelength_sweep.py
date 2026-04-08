#!/usr/bin/env python3
"""
Wavelength sweep — frequency response of centipede locomotion.

Generates single-wavelength terrains from λ_max (world-scale) down to λ_min
(leg-scale), runs a simulation on each, and records locomotion metrics:
  - Cost of Transport (CoT) = Σ|τ·ω|·dt / (m·g·d)
  - Forward speed
  - Max/mean pitch angle
  - Max/mean roll angle
  - Survival (did it fold?)

Output: JSON + CSV with (wavelength, CoT, speed, pitch, roll, survived)
        → plot CoT vs wavelength to find critical transitions.

Usage
-----
  python scripts/sweep/wavelength_sweep.py
  python scripts/sweep/wavelength_sweep.py --n-points 30 --duration 8 --amplitude 0.005
  python scripts/sweep/wavelength_sweep.py --headless   (no video, faster)

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

def run_simulation(xml_path, config_path, duration):
    """
    Run one simulation and return locomotion metrics.

    Returns dict with: survived, forward_speed, cot, max_pitch_deg,
                       mean_pitch_deg, max_roll_deg, mean_roll_deg, sim_time
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    ctrl = ImpedanceTravelingWaveController(model, config_path)

    # Resolve joint/actuator IDs
    pitch_jnt_ids = []
    roll_jnt_ids  = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_jnt_ids.append(j)
        if nm and 'joint_roll_body' in nm:
            roll_jnt_ids.append(j)

    # Get all actuator IDs for energy computation
    n_actuators = model.nu

    # Get total mass for CoT
    total_mass = 0.0
    for i in range(model.nbody):
        total_mass += model.body_mass[i]
    gravity = abs(model.opt.gravity[2])  # typically 9.81

    n_steps = int(duration / model.opt.timestep)
    dt = model.opt.timestep

    # Find root body (freejoint) once before the loop
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

    # Storage
    energy_sum = 0.0       # Σ |τ·ω| · dt
    pitch_angles = []
    roll_angles  = []
    start_pos = None
    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Record start position (after a few settling steps)
        if step_i == 200 and root_body is not None:
            start_pos = data.xpos[root_body].copy()

        # Energy: Σ |τ_i · ω_i| · dt for all actuators
        if step_i > 200:  # skip transient
            for a in range(n_actuators):
                # actuator_force is the force/torque applied
                # For general actuators: force = gain * ctrl
                # Joint velocity from the actuator's transmission
                tau = abs(data.actuator_force[a])
                # Get the joint this actuator acts on
                jnt_id = model.actuator_trnid[a, 0]
                if jnt_id >= 0 and jnt_id < model.njnt:
                    dof_adr = model.jnt_dofadr[jnt_id]
                    omega = abs(data.qvel[dof_adr])
                    energy_sum += tau * omega * dt

        # Failure check every 200 steps
        if step_i % 200 == 0 and step_i > 0:
            for jid in pitch_jnt_ids:
                q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                if q_deg > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch({q_deg:.1f}° t={data.time:.1f}s)"
                    break
            if not buckled:
                for jid in roll_jnt_ids:
                    q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                    if q_deg > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll({q_deg:.1f}° t={data.time:.1f}s)"
                        break
            if buckled:
                break

        # Record pitch/roll every 100 steps
        if step_i % 100 == 0:
            for jid in pitch_jnt_ids:
                pitch_angles.append(abs(math.degrees(
                    data.qpos[model.jnt_qposadr[jid]])))
            for jid in roll_jnt_ids:
                roll_angles.append(abs(math.degrees(
                    data.qpos[model.jnt_qposadr[jid]])))

    # Compute metrics
    end_pos = None
    if root_body is not None:
        end_pos = data.xpos[root_body].copy()

    if start_pos is not None and end_pos is not None:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
    else:
        distance = 0.0

    settle_time = 200 * dt
    effective_time = max(data.time - settle_time, 0.01)
    forward_speed = distance / effective_time

    # CoT = total_energy / (m * g * distance)
    if distance > 0.001:
        cot = energy_sum / (total_mass * gravity * distance)
    else:
        cot = float('inf')

    return {
        'survived':       not buckled,
        'buckle_reason':  buckle_reason,
        'sim_time':       float(data.time),
        'distance_m':     float(distance),
        'forward_speed':  float(forward_speed),
        'cot':            float(min(cot, 1e6)),
        'energy_J':       float(energy_sum),
        'max_pitch_deg':  float(max(pitch_angles)) if pitch_angles else 0,
        'mean_pitch_deg': float(np.mean(pitch_angles)) if pitch_angles else 0,
        'max_roll_deg':   float(max(roll_angles)) if roll_angles else 0,
        'mean_roll_deg':  float(np.mean(roll_angles)) if roll_angles else 0,
        'total_mass_kg':  float(total_mass),
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

    print("=" * 70)
    print("Wavelength Sweep — Centipede Frequency Response")
    print("=" * 70)
    print(f"  Morphology: L_w={lengths['L_w']*1000:.0f}mm  "
          f"L_b={lengths['L_b']*1000:.1f}mm  "
          f"L_s={lengths['L_s']*1000:.1f}mm  "
          f"L_ell={lengths['L_ell']*1000:.1f}mm")
    print(f"  Wavelength range: {wl_max*1000:.0f}mm → {wl_min*1000:.1f}mm  "
          f"({args.n_points} points, log-spaced)")
    print(f"  Fixed amplitude: {args.amplitude*1000:.1f}mm RMS")
    print(f"  Duration: {args.duration}s per trial")
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

        # Run simulation
        try:
            metrics = run_simulation(tmp_xml, CONFIG_PATH, args.duration)
        except Exception as e:
            print(f"    ERROR: {e}")
            metrics = {'survived': False, 'buckle_reason': str(e),
                       'cot': 1e6, 'forward_speed': 0, 'distance_m': 0,
                       'max_pitch_deg': 0, 'mean_pitch_deg': 0,
                       'max_roll_deg': 0, 'mean_roll_deg': 0,
                       'energy_J': 0, 'sim_time': 0, 'total_mass_kg': 0}
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
            print(f"    → CoT={metrics['cot']:.2f}  "
                  f"speed={metrics['forward_speed']*1000:.1f}mm/s  "
                  f"pitch={metrics['max_pitch_deg']:.1f}°  "
                  f"dist={metrics['distance_m']*1000:.0f}mm")
        else:
            print(f"    → FAIL: {metrics['buckle_reason']}")

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
                    'mean_roll_deg', 'distance_m', 'energy_J', 'survived']
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
        print(f"\n  CoT range: {min(r['cot'] for r in survived):.2f} – "
              f"{max(r['cot'] for r in survived):.2f}")
        print(f"  Speed range: {min(r['forward_speed'] for r in survived)*1000:.1f} – "
              f"{max(r['forward_speed'] for r in survived)*1000:.1f} mm/s")

        # Mark morphology scales
        print(f"\n  Morphology reference lines for plotting:")
        for name, val in sorted(scales.items(), key=lambda kv: -kv[1]):
            print(f"    {name:>5s} = {val*1000:>7.1f} mm  (f = {1/val:.1f} cyc/m)")

    print(f"\n  Results: {out_json}")
    print(f"  CSV:     {out_csv}")
    print(f"\n  To plot: use wavelength_mm as x-axis (log scale),")
    print(f"           CoT / speed / pitch as y-axis.")
    print(f"           Draw vertical lines at L_w, L_b, L_s, L_ell.")


if __name__ == "__main__":
    main()
