#!/usr/bin/env python3
"""
Wavelength test — quick 1-trial-per-wavelength sanity check.

Simpler companion to `wavelength_sweep.py`:
  - Sparse wavelength sampling (default: 6 points covering L_w down to sub-L_ell).
  - Exactly one trial per wavelength (fixed yaw = 0).
  - Optional MP4 per wavelength for visual inspection.
  - Writes a small CSV + JSON summary.

Usage
-----
  # default sparse sweep with videos
  python scripts/sweep/wavelength_test.py --video

  # custom wavelengths (mm)
  python scripts/sweep/wavelength_test.py --wavelengths 200,80,40,20,10,5

  # custom amplitude / duration
  python scripts/sweep/wavelength_test.py --duration 6 --amplitude 0.01
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "terrain", "generator"))

# Reuse heavy machinery from the full sweep so the two scripts stay in sync.
from wavelength_sweep import (
    generate_single_wavelength_terrain,
    save_wavelength_terrain,
    patch_xml_terrain,
    run_simulation,
    XML_PATH,
    CONFIG_PATH,
    TERRAIN_CFG,
    OUTPUT_DIR,
)

from generate import resolve_morphology

# Default sparse wavelengths (mm): spans above L_w down past L_ell.
# Roughly log-spaced; adjust via --wavelengths if needed.
DEFAULT_WAVELENGTHS_MM = [200.0, 80.0, 40.0, 20.0, 10.0, 5.0]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wavelengths", type=str, default=None,
                   help="Comma-separated wavelengths in mm "
                        f"(default: {DEFAULT_WAVELENGTHS_MM})")
    p.add_argument("--duration",  type=float, default=8.0,
                   help="Sim duration per trial (s). Includes settle + ramp.")
    p.add_argument("--amplitude", type=float, default=0.01,
                   help="Terrain amplitude (m). Default: 0.01 m (10 mm).")
    p.add_argument("--yaw-deg",   type=float, default=0.0,
                   help="Fixed spawn yaw (degrees). Default 0.")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--video",     action="store_true",
                   help="Save one MP4 per wavelength.")
    p.add_argument("--no-video",  action="store_true",
                   help="Force-disable video even if mediapy is installed.")
    args = p.parse_args()

    # ── Wavelength list ───────────────────────────────────────────────────────
    if args.wavelengths:
        wls_mm = [float(w.strip()) for w in args.wavelengths.split(",") if w.strip()]
    else:
        wls_mm = list(DEFAULT_WAVELENGTHS_MM)
    wls_mm = sorted(wls_mm, reverse=True)         # long → short
    wavelengths = np.array(wls_mm) / 1000.0       # metres

    # ── Load morphology (for reporting only) ──────────────────────────────────
    with open(TERRAIN_CFG, encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    lengths   = resolve_morphology(t_cfg)
    img_size  = int(t_cfg["world"]["image_size"])
    world_half = float(t_cfg["world"]["size"])

    # ── Video availability check ──────────────────────────────────────────────
    can_video = False
    if args.video and not args.no_video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed — skipping video.")

    # ── Output dir ────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"test_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    yaw_rad = math.radians(args.yaw_deg)

    print("=" * 70)
    print("Wavelength Test — 1 trial per wavelength")
    print("=" * 70)
    print(f"  Morphology: L_w={lengths['L_w']*1000:.0f}mm  "
          f"L_b={lengths['L_b']*1000:.1f}mm  "
          f"L_s={lengths['L_s']*1000:.1f}mm  "
          f"L_ell={lengths['L_ell']*1000:.1f}mm")
    print(f"  Wavelengths ({len(wls_mm)}): "
          f"{', '.join(f'{w:.1f}' for w in wls_mm)} mm")
    print(f"  Amplitude:  {args.amplitude*1000:.1f} mm  "
          f"Duration: {args.duration:.1f} s  "
          f"Yaw: {args.yaw_deg:.1f} deg")
    print(f"  Video: {'ON' if can_video else 'OFF'}")
    print(f"  Output: {run_dir}")
    print()

    results = []
    t0 = time.time()

    for i, wl in enumerate(wavelengths):
        print(f"[{i+1:2d}/{len(wavelengths)}] "
              f"lambda = {wl*1000:7.1f} mm  (f = {1/wl:7.1f} cyc/m)")

        # Terrain
        h_m, rms_m, peak_m = generate_single_wavelength_terrain(
            wavelength_m=wl, amplitude_m=args.amplitude,
            seed=args.seed + i, image_size=img_size, world_half=world_half,
        )
        png_path = save_wavelength_terrain(h_m, wl, args.seed + i, run_dir)
        z_max = max(2.0 * peak_m, 0.005)
        tmp_xml = patch_xml_terrain(XML_PATH, png_path, z_max)

        # Video path
        vid_path = None
        if can_video:
            vid_dir = os.path.join(run_dir, "videos")
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"wl_{wl*1000:.0f}mm.mp4")

        try:
            metrics = run_simulation(
                tmp_xml, CONFIG_PATH, args.duration,
                yaw_rad=yaw_rad, video_path=vid_path,
            )
        except Exception as e:
            metrics = {
                'survived': False, 'buckle_reason': str(e),
                'yaw_deg': args.yaw_deg, 'cot': 1e6, 'forward_speed': 0,
                'distance_m': 0, 'max_pitch_deg': 0, 'mean_pitch_deg': 0,
                'max_roll_deg': 0, 'mean_roll_deg': 0, 'energy_J': 0,
                'sim_time': 0, 'total_mass_kg': 0,
                'phase_lag_deg': float('nan'), 'phase_coherence': 0,
                'phase_freq_hz': 0, 'video_path': '',
            }

        metrics['wavelength_m']  = float(wl)
        metrics['wavelength_mm'] = float(wl * 1000)
        metrics['rms_m']         = float(rms_m)
        metrics['peak_m']        = float(peak_m)
        results.append(metrics)

        if os.path.exists(tmp_xml):
            os.remove(tmp_xml)

        status = ("OK" if metrics['survived']
                  else f"FAIL:{metrics['buckle_reason']}")
        print(f"    CoT={metrics['cot']:7.1f}  "
              f"speed={metrics['forward_speed']*1000:6.1f} mm/s  "
              f"max_pitch={metrics['max_pitch_deg']:5.1f} deg  "
              f"max_roll={metrics['max_roll_deg']:5.1f} deg  "
              f"{status}", flush=True)

    elapsed = time.time() - t0

    # ── Save JSON + CSV ───────────────────────────────────────────────────────
    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, 'w') as f:
        json.dump({
            'timestamp':    timestamp,
            'duration':     args.duration,
            'amplitude':    args.amplitude,
            'yaw_deg':      args.yaw_deg,
            'wavelengths_mm': wls_mm,
            'morphology':   {k: float(v) for k, v in lengths.items()},
            'elapsed_s':    elapsed,
            'results':      results,
        }, f, indent=2)

    out_csv = os.path.join(run_dir, "results.csv")
    headers = ['wavelength_mm', 'survived', 'cot', 'forward_speed',
               'distance_m', 'max_pitch_deg', 'mean_pitch_deg',
               'max_roll_deg', 'mean_roll_deg',
               'phase_lag_deg', 'phase_coherence', 'energy_J']
    with open(out_csv, 'w') as f:
        f.write(','.join(headers) + '\n')
        for r in results:
            f.write(','.join(str(r.get(h, '')) for h in headers) + '\n')

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"DONE  ({elapsed:.1f} s, {len(wavelengths)} simulations)")
    print("=" * 70)
    survived = [r for r in results if r['survived']]
    if survived:
        speeds = [r['forward_speed'] * 1000 for r in survived]
        cots   = [r['cot'] for r in survived]
        print(f"  Survived: {len(survived)}/{len(results)}")
        print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} mm/s")
        print(f"  CoT range:   {min(cots):.1f} - {max(cots):.1f}")
    else:
        print("  No surviving trials.")
    print(f"\n  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")
    if can_video:
        print(f"  Videos: {os.path.join(run_dir, 'videos')}/")


if __name__ == "__main__":
    main()
