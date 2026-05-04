#!/usr/bin/env python3
"""
pitch_roll_grid_sweep.py — grid sweep over pitch/roll impedance gains on
two fixed terrain wavelengths (default 80 mm and 40 mm).

Design
------
- 2D grid over (pitch_kp, roll_kp), log-spaced around user-confirmed centres:
      pitch_kp centre = 0.004       (range = ±0.6 decades → [0.001, 0.016])
      roll_kp  centre = 0.002       (range = ±0.6 decades → [0.0005, 0.008])
  kv is "bonded" to kp via fixed ratios at those centres:
      pitch_kv = pitch_kp × 0.25    (so 0.004 → 0.001)
      roll_kv  = roll_kp  × 0.50    (so 0.002 → 0.001)
  Override centres with --pitch-center / --roll-center, span with
  --range-decades, ratios with --pitch-ratio / --roll-ratio, or set
  kv == kp with --equal-ratio.
- Default grid is 5×5 → 25 combos × 2 wavelengths = 50 simulations.
- Each combo runs 1 trial per wavelength (fixed yaw = 0) with the current
  config otherwise unchanged (soft CPG, heading servo, H=1/R=3 taper, etc.).

Usage
-----
  # default 5×5 grid on λ ∈ {80, 40} mm, video off
  python scripts/sweep/pitch_roll_grid_sweep.py

  # finer 7×7 grid, 6 s duration, videos on
  python scripts/sweep/pitch_roll_grid_sweep.py --grid 7 --duration 6 --video

  # different wavelengths
  python scripts/sweep/pitch_roll_grid_sweep.py --wavelengths 130,80,40,20
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import yaml

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "terrain", "generator"))

from wavelength_sweep import (  # noqa: E402
    generate_single_wavelength_terrain,
    save_wavelength_terrain,
    patch_xml_terrain,
    run_simulation,
    XML_PATH,
    CONFIG_PATH,
    TERRAIN_CFG,
    OUTPUT_DIR,
)

# Current "good" centre values (user-confirmed):
#     pitch_kp=0.004  pitch_kv=0.001  → ratio 0.25
#     roll_kp =0.002  roll_kv =0.001  → ratio 0.5
# The grid is built relative to these centres with a multiplicative span
# (±RANGE_DECADES around the centre, log-spaced).
DEFAULT_PITCH_CENTER = 0.004
DEFAULT_ROLL_CENTER  = 0.002
DEFAULT_PITCH_RATIO  = 0.001 / 0.004   # = 0.25
DEFAULT_ROLL_RATIO   = 0.001 / 0.002   # = 0.5
DEFAULT_RANGE_DECADES = 0.6            # ±0.6 decades → ×0.25 .. ×4


def make_temp_config(pitch_kp, pitch_kv, roll_kp, roll_kv):
    """Write a temp YAML with pitch/roll gains overridden."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    imp = cfg.setdefault("impedance", {})
    imp["pitch_kp"] = float(pitch_kp)
    imp["pitch_kv"] = float(pitch_kv)
    imp["roll_kp"]  = float(roll_kp)
    imp["roll_kv"]  = float(roll_kv)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".yaml", prefix="_pr_grid_",
        dir=os.path.dirname(CONFIG_PATH))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def try_plot_heatmaps(grid_results, pitch_kps, roll_kps, wavelengths_mm,
                      run_dir):
    """Save CoT and speed heatmaps per wavelength."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    for wl_mm in wavelengths_mm:
        sub = [r for r in grid_results if abs(r['wavelength_mm'] - wl_mm) < 1e-6]
        nP = len(pitch_kps)
        nR = len(roll_kps)

        cot   = np.full((nP, nR), np.nan)
        speed = np.full((nP, nR), np.nan)
        surv  = np.zeros((nP, nR), dtype=bool)

        for r in sub:
            i = pitch_kps.index(r['pitch_kp'])
            j = roll_kps.index(r['roll_kp'])
            surv[i, j] = r['survived']
            if r['survived']:
                cot[i, j]   = r['cot']
                speed[i, j] = r['forward_speed'] * 1000  # mm/s

        for name, mat, unit in (('cot', cot, ''),
                                ('speed', speed, 'mm/s')):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, origin='lower', aspect='auto', cmap='viridis')
            ax.set_xticks(range(nR))
            ax.set_yticks(range(nP))
            ax.set_xticklabels([f"{k:.4f}" for k in roll_kps], rotation=45,
                               fontsize=8)
            ax.set_yticklabels([f"{k:.4f}" for k in pitch_kps], fontsize=8)
            ax.set_xlabel("roll_kp")
            ax.set_ylabel("pitch_kp")
            ax.set_title(f"{name} @ λ={wl_mm:.0f} mm  {unit}")
            # Mark non-survived cells
            for i in range(nP):
                for j in range(nR):
                    if not surv[i, j]:
                        ax.text(j, i, "✗", ha='center', va='center',
                                color='red', fontsize=10)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            out = os.path.join(run_dir, f"heatmap_{name}_wl{int(wl_mm)}mm.png")
            fig.savefig(out, dpi=120)
            plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid",      type=int,   default=5,
                   help="Grid size per axis (pitch_kp × roll_kp). Default 5.")
    p.add_argument("--pitch-center", type=float, default=DEFAULT_PITCH_CENTER,
                   help=f"pitch_kp centre (default {DEFAULT_PITCH_CENTER}).")
    p.add_argument("--roll-center",  type=float, default=DEFAULT_ROLL_CENTER,
                   help=f"roll_kp centre (default {DEFAULT_ROLL_CENTER}).")
    p.add_argument("--range-decades", type=float, default=DEFAULT_RANGE_DECADES,
                   help=f"Log-span per side in decades (default "
                        f"{DEFAULT_RANGE_DECADES}, i.e. ×0.25 to ×4 about centre).")
    p.add_argument("--pitch-ratio", type=float, default=DEFAULT_PITCH_RATIO,
                   help=f"pitch_kv / pitch_kp (default {DEFAULT_PITCH_RATIO:.4f}).")
    p.add_argument("--roll-ratio",  type=float, default=DEFAULT_ROLL_RATIO,
                   help=f"roll_kv / roll_kp (default {DEFAULT_ROLL_RATIO:.4f}).")
    p.add_argument("--equal-ratio", action="store_true",
                   help="Set both ratios to 1 (kv == kp).")
    p.add_argument("--wavelengths", type=str, default="80,40",
                   help="Terrain wavelengths (mm), comma-separated. Default '80,40'.")
    p.add_argument("--duration",  type=float, default=8.0,
                   help="Simulation duration (s). Default 8.")
    p.add_argument("--amplitude", type=float, default=0.01,
                   help="Terrain amplitude (m). Default 0.01 (10 mm).")
    p.add_argument("--yaw-deg",   type=float, default=0.0,
                   help="Fixed spawn yaw (deg). Default 0.")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--video",     action="store_true",
                   help="Save an MP4 per (combo × wavelength). Produces a lot "
                        "of videos — use only with small grids.")
    args = p.parse_args()

    if args.equal_ratio:
        args.pitch_ratio = 1.0
        args.roll_ratio  = 1.0

    # ── Build log-spaced grid centred on the "good" values ───────────────────
    d = float(args.range_decades)
    pitch_vals = np.logspace(
        math.log10(args.pitch_center) - d,
        math.log10(args.pitch_center) + d,
        args.grid,
    )
    roll_vals = np.logspace(
        math.log10(args.roll_center) - d,
        math.log10(args.roll_center) + d,
        args.grid,
    )
    pitch_kps = [float(v) for v in pitch_vals]
    roll_kps  = [float(v) for v in roll_vals]

    wl_mm = [float(w.strip()) for w in args.wavelengths.split(",") if w.strip()]
    wavelengths = np.array(sorted(wl_mm, reverse=True)) / 1000.0

    # ── Video availability ────────────────────────────────────────────────────
    can_video = False
    if args.video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed — skipping video.")

    # ── Terrain config ────────────────────────────────────────────────────────
    with open(TERRAIN_CFG, encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    img_size   = int(t_cfg["world"]["image_size"])
    world_half = float(t_cfg["world"]["size"])

    # ── Output dir ────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"pitch_roll_grid_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    total_sims = args.grid * args.grid * len(wavelengths)
    print("=" * 70)
    print("Pitch × Roll Grid Sweep")
    print("=" * 70)
    print(f"  grid: {args.grid} × {args.grid}  "
          f"(±{d:.2f} decades, log-spaced)")
    print(f"  pitch_kp centre = {args.pitch_center:.4f}  "
          f"range: [{pitch_kps[0]:.5f}, {pitch_kps[-1]:.5f}]")
    print(f"  roll_kp  centre = {args.roll_center:.4f}  "
          f"range: [{roll_kps[0]:.5f}, {roll_kps[-1]:.5f}]")
    print(f"  pitch_kv = pitch_kp × {args.pitch_ratio:.4f}")
    print(f"  roll_kv  = roll_kp  × {args.roll_ratio:.4f}")
    print(f"  wavelengths (mm): {wl_mm}")
    print(f"  amplitude: {args.amplitude*1000:.1f} mm  "
          f"duration: {args.duration:.1f}s  yaw: {args.yaw_deg:.1f} deg")
    print(f"  total simulations: {total_sims}")
    print(f"  video: {'ON' if can_video else 'OFF'}")
    print(f"  output: {run_dir}")
    print()

    # ── Pre-generate terrain per wavelength (one PNG, reused across combos) ──
    terrain_xmls = {}   # wl_m → tmp_xml_path
    for i, wl in enumerate(wavelengths):
        h_m, rms_m, peak_m = generate_single_wavelength_terrain(
            wavelength_m=wl, amplitude_m=args.amplitude,
            seed=args.seed + i, image_size=img_size, world_half=world_half,
        )
        png_path = save_wavelength_terrain(h_m, wl, args.seed + i, run_dir)
        z_max = max(2.0 * peak_m, 0.005)
        terrain_xmls[float(wl)] = patch_xml_terrain(XML_PATH, png_path, z_max)

    yaw_rad = math.radians(args.yaw_deg)
    all_results = []
    t0 = time.time()
    sim_count = 0

    for i_p, pitch_kp in enumerate(pitch_kps):
        pitch_kv = pitch_kp * args.pitch_ratio
        for i_r, roll_kp in enumerate(roll_kps):
            roll_kv = roll_kp * args.roll_ratio
            tmp_cfg = make_temp_config(pitch_kp, pitch_kv, roll_kp, roll_kv)

            for wl in wavelengths:
                sim_count += 1
                tmp_xml = terrain_xmls[float(wl)]

                vid_path = None
                if can_video:
                    vid_dir = os.path.join(run_dir, "videos")
                    os.makedirs(vid_dir, exist_ok=True)
                    vid_path = os.path.join(
                        vid_dir,
                        f"wl{wl*1000:.0f}_pkp{pitch_kp:.5f}_rkp{roll_kp:.5f}.mp4",
                    )

                try:
                    metrics = run_simulation(
                        tmp_xml, tmp_cfg, args.duration,
                        yaw_rad=yaw_rad, video_path=vid_path,
                    )
                except Exception as e:
                    metrics = {
                        'survived': False, 'buckle_reason': str(e),
                        'yaw_deg': args.yaw_deg, 'cot': 1e6,
                        'forward_speed': 0, 'distance_m': 0,
                        'max_pitch_deg': 0, 'mean_pitch_deg': 0,
                        'max_roll_deg': 0, 'mean_roll_deg': 0,
                        'energy_J': 0, 'sim_time': 0, 'total_mass_kg': 0,
                        'phase_lag_deg': float('nan'), 'phase_coherence': 0,
                        'phase_freq_hz': 0, 'video_path': '',
                    }

                metrics.update({
                    'wavelength_m':  float(wl),
                    'wavelength_mm': float(wl * 1000),
                    'pitch_kp':      float(pitch_kp),
                    'pitch_kv':      float(pitch_kv),
                    'roll_kp':       float(roll_kp),
                    'roll_kv':       float(roll_kv),
                })
                all_results.append(metrics)

                status = ("OK" if metrics['survived']
                          else f"FAIL:{metrics['buckle_reason']}")
                elapsed = time.time() - t0
                eta = (elapsed / sim_count) * (total_sims - sim_count)
                print(f"[{sim_count:3d}/{total_sims}] "
                      f"λ={wl*1000:5.0f}mm  "
                      f"pitch_kp={pitch_kp:.5f}  roll_kp={roll_kp:.5f}  "
                      f"CoT={metrics['cot']:6.1f}  "
                      f"v={metrics['forward_speed']*1000:5.1f}mm/s  "
                      f"{status}  (ETA {eta/60:.0f}min)",
                      flush=True)

            if os.path.exists(tmp_cfg):
                os.remove(tmp_cfg)

    # Clean up terrain tmp XMLs
    for path in terrain_xmls.values():
        if os.path.exists(path):
            os.remove(path)

    elapsed = time.time() - t0

    # ── Save JSON + CSV ───────────────────────────────────────────────────────
    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            'timestamp':    timestamp,
            'grid':         args.grid,
            'pitch_center': args.pitch_center,
            'roll_center':  args.roll_center,
            'range_decades': args.range_decades,
            'pitch_ratio':  args.pitch_ratio,
            'roll_ratio':   args.roll_ratio,
            'duration':     args.duration,
            'amplitude':    args.amplitude,
            'yaw_deg':      args.yaw_deg,
            'wavelengths_mm': wl_mm,
            'pitch_kps':    pitch_kps,
            'roll_kps':     roll_kps,
            'elapsed_s':    elapsed,
            'results':      all_results,
        }, f, indent=2)

    out_csv = os.path.join(run_dir, "results.csv")
    headers = ['wavelength_mm', 'pitch_kp', 'pitch_kv', 'roll_kp', 'roll_kv',
               'survived', 'cot', 'forward_speed', 'distance_m',
               'max_pitch_deg', 'mean_pitch_deg', 'max_roll_deg',
               'mean_roll_deg', 'phase_lag_deg', 'phase_coherence', 'energy_J']
    with open(out_csv, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(h, '')) for h in headers) + "\n")

    # Heatmaps
    plotted = try_plot_heatmaps(all_results, pitch_kps, roll_kps, wl_mm, run_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DONE  ({elapsed/60:.1f} min, {total_sims} simulations)")
    print("=" * 70)
    for wl_val in wl_mm:
        sub = [r for r in all_results
               if abs(r['wavelength_mm'] - wl_val) < 1e-6 and r['survived']]
        if sub:
            best = min(sub, key=lambda r: r['cot'])
            print(f"  λ={wl_val:.0f}mm  best CoT = {best['cot']:.2f}  "
                  f"(pitch_kp={best['pitch_kp']:.5f}, "
                  f"roll_kp={best['roll_kp']:.5f}, "
                  f"speed={best['forward_speed']*1000:.1f} mm/s)")
        else:
            print(f"  λ={wl_val:.0f}mm  no surviving combos")

    print(f"\n  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")
    if plotted:
        print(f"  heatmaps: {run_dir}/heatmap_*.png")


if __name__ == "__main__":
    main()
