#!/usr/bin/env python3
"""
Terrain roughness sweep — Option A + per-level z_max.

Each level has:
  - Three band amplitudes (low/mid/high) → unique spatial patterns per level
  - A per-level z_max → controls physical height in MuJoCo

The generator normalises every PNG to full 0-255 range (so MuJoCo gets
maximum height resolution with no staircase artifacts).  MuJoCo internally
normalises hfield data: min pixel → 0, max pixel → z_max.  So the per-level
z_max is the sole control of physical roughness.

  L0_flat  :  1 trial   (flat ground, no terrain PNG)
  L1_mild  : 50 trials  (unique hills, z_max = small)
  L2_mod   : 50 trials  (unique hills, z_max = medium)
  L3_rough : 50 trials  (unique hills, z_max = large)

Usage
-----
  python terrain_sweep.py --test       # 1 trial per terrain (4 runs)
  python terrain_sweep.py              # full 151-trial sweep
  python terrain_sweep.py --no-video   # skip video (faster)
  python terrain_sweep.py --trials 50  # custom trial count
"""

import argparse
import os
import subprocess
import sys
import json
import numpy as np
from datetime import datetime

# ── project layout ────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
TERRAIN_DIR   = os.path.join(PROJECT_ROOT, "terrain", "generator")
CONTROL_DIR   = os.path.join(PROJECT_ROOT, "controllers", "farms")
XML_PATH      = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")

TERRAIN_GEN   = os.path.join(TERRAIN_DIR, "generate.py")
TERRAIN_PATCH = os.path.join(TERRAIN_DIR, "patch_xml.py")
FARMS_RUN     = os.path.join(CONTROL_DIR, "run.py")
TERRAIN_OUT   = os.path.join(PROJECT_ROOT, "terrain", "output")

# ── terrain definitions ───────────────────────────────────────────────────────
# Each level has:
#   low_amp, mid_amp, high_amp → band amplitudes (control spatial pattern)
#   z_max → hfield z_max in metres (controls physical height in MuJoCo)
#
# Every PNG is normalised to full 0-255 range.  Different amplitudes create
# different terrain CHARACTER (ratio of hills vs mounds vs bumps).
# z_max controls the physical SCALE of the terrain.
#
# Centipede dimensions: leg ~7mm, COM ~5mm, body length ~100mm
#
# (label, low_amp, mid_amp, high_amp, z_max, n_trials, randomise_yaw)
TERRAINS = [
    ("L0_flat",  0.000,  0.000,  0.000,  0.000,   1, False),  # flat ground
    ("L1_mild",  0.004,  0.002,  0.001,  0.010,  50, True),   # z_max=10mm, gentle
    ("L2_mod",   0.006,  0.003,  0.002,  0.040,  50, True),   # z_max=40mm, moderate
    ("L3_rough", 0.010,  0.005,  0.003,  0.070,  50, True),   # z_max=70mm, challenging
]

# How many of the 50 trials per terrain also get a video saved
N_VIDEO_TRIALS = 10

SEED = 0   # terrain shape seed (fixed)


# ── helpers ───────────────────────────────────────────────────────────────────
def png_path(low_amp, mid_amp, high_amp):
    tag = f"low{low_amp:.4f}_mid{mid_amp:.4f}_high{high_amp:.4f}_s{SEED}"
    return os.path.join(TERRAIN_OUT, tag, "1.png")


def run(cmd, label="", check=True, fatal=True):
    if label:
        print(f"  -> {label}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if check and result.returncode != 0:
        print(f"\n[sweep] FAILED (exit {result.returncode})")
        if fatal:
            sys.exit(result.returncode)
    return result.returncode == 0


# ── pipeline steps ────────────────────────────────────────────────────────────
def generate_terrain(label, low_amp, mid_amp, high_amp):
    if low_amp == 0.0 and mid_amp == 0.0 and high_amp == 0.0:
        print(f"  skip generate {label} (flat — no PNG needed)")
        return
    png = png_path(low_amp, mid_amp, high_amp)
    if os.path.isfile(png):
        print(f"  skip generate {label} (exists)")
        return
    run([sys.executable, TERRAIN_GEN,
         "--low-amp",  str(low_amp),
         "--mid-amp",  str(mid_amp),
         "--high-amp", str(high_amp),
         "--seed",     str(SEED)],
        label=f"generate {label}")


def patch_terrain(low_amp, mid_amp, high_amp, z_max):
    if low_amp == 0.0 and mid_amp == 0.0 and high_amp == 0.0:
        run([sys.executable, TERRAIN_PATCH, "--flat-ground"],
            label="patch flat ground")
    else:
        run([sys.executable, TERRAIN_PATCH,
             "--terrain", png_path(low_amp, mid_amp, high_amp),
             "--z-max",   str(z_max),
             "--terrain-only"],
            label=f"patch terrain (z_max={z_max:.4f})")


def patch_rotation(yaw_deg):
    run([sys.executable, TERRAIN_PATCH,
         "--rotation-deg", f"{yaw_deg:.2f}",
         "--terrain-only"],          # skip redundant sensor re-patch
        label=f"patch rotation {yaw_deg:.1f} deg")


def run_trial(out_dir, duration, save_video):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, FARMS_RUN,
           "--headless",
           "--duration", str(duration),
           "--output",   os.path.join(out_dir, "data.npz")]
    if save_video:
        cmd += ["--video", os.path.join(out_dir, "video.mp4")]
    return run(cmd, label=f"simulate -> {os.path.relpath(out_dir, PROJECT_ROOT)}",
               fatal=False)  # don't kill entire sweep on a single trial failure


# ── sweep ─────────────────────────────────────────────────────────────────────
def run_sweep(n_trials_override, duration, save_video, test_mode, base_seed):
    stamp     = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    mode_tag  = "test" if test_mode else "full"
    sweep_dir = os.path.join(PROJECT_ROOT, "outputs", "data", f"sweep_{mode_tag}_{stamp}")

    rng = np.random.default_rng(base_seed)

    # Build run plan
    plan = []
    for label, low_amp, mid_amp, high_amp, z_max, n_trials_default, randomise in TERRAINS:
        if test_mode:
            n = 1
        elif n_trials_override is not None:
            n = 1 if not randomise else n_trials_override
        else:
            n = n_trials_default
        plan.append((label, low_amp, mid_amp, high_amp, z_max, n, randomise))

    total = sum(n for _, _, _, _, _, n, _ in plan)

    print(f"\n{'='*60}")
    print(f"  Terrain sweep  --  {total} trials total")
    print(f"  {'TEST MODE' if test_mode else 'FULL'}"
          f"  |  {duration}s/trial"
          f"  |  video={'on' if save_video else 'off'}")
    for label, low_amp, mid_amp, high_amp, z_max, n, rand in plan:
        print(f"    {label:<12} low={low_amp:.4f} mid={mid_amp:.4f} "
              f"high={high_amp:.4f} z_max={z_max:.3f}  "
              f"{n} trial{'s' if n>1 else ''}  "
              f"{'random yaw' if rand else 'fixed yaw=0'}")
    print(f"  Output -> {sweep_dir}")
    print(f"{'='*60}")

    # Save original XML so we can restore it after the sweep
    original_xml = None
    if os.path.isfile(XML_PATH):
        with open(XML_PATH, "r", encoding="utf-8") as f:
            original_xml = f.read()

    results = []
    done = 0

    for label, low_amp, mid_amp, high_amp, z_max, n_trials, randomise in plan:
        print(f"\n{'─'*60}")
        print(f"  {label}  (low={low_amp}, mid={mid_amp}, high={high_amp}, z_max={z_max})")
        print(f"{'─'*60}")

        generate_terrain(label, low_amp, mid_amp, high_amp)
        patch_terrain(low_amp, mid_amp, high_amp, z_max)

        for trial in range(n_trials):
            done += 1
            yaw = float(rng.uniform(0, 360)) if randomise else 0.0
            out_dir = os.path.join(sweep_dir, label, f"trial_{trial:03d}")

            print(f"\n  [{done}/{total}]  {label}  trial {trial:03d}"
                  f"  yaw={yaw:.1f} deg")

            patch_rotation(yaw)
            # save video for first N_VIDEO_TRIALS trials of every terrain
            save_this_video = save_video or (trial < N_VIDEO_TRIALS)
            ok = run_trial(out_dir, duration, save_this_video)

            results.append({
                "terrain":   label,
                "low_amp":   low_amp,
                "mid_amp":   mid_amp,
                "high_amp":  high_amp,
                "z_max":     z_max,
                "trial":     trial,
                "yaw_deg":   yaw,
                "data":      os.path.join(out_dir, "data.npz"),
                "video":     os.path.join(out_dir, "video.mp4") if (save_video or trial < N_VIDEO_TRIALS) else None,
                "success":   ok,
            })

    # Save manifest
    manifest = os.path.join(sweep_dir, "manifest.json")
    with open(manifest, "w") as f:
        json.dump(results, f, indent=2)

    # Restore original XML
    if original_xml is not None:
        with open(XML_PATH, "w", encoding="utf-8") as f:
            f.write(original_xml)
        print("[sweep] XML restored to original state.")

    n_ok = sum(r["success"] for r in results)
    n_fail = total - n_ok
    print(f"\n{'='*60}")
    print(f"  DONE  {n_ok}/{total} succeeded"
          + (f"  ({n_fail} failed)" if n_fail else ""))
    print(f"  Manifest -> {manifest}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test",     action="store_true",
                   help="Test mode: 1 trial per terrain (4 runs)")
    p.add_argument("--trials",   type=int, default=None,
                   help="Override trials per rough terrain (default: 50)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="Simulation duration per trial in seconds (default: 5)")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video recording (faster)")
    p.add_argument("--seed",     type=int, default=0,
                   help="Random seed for yaw angles (default: 0)")
    args = p.parse_args()

    run_sweep(
        n_trials_override = args.trials,
        duration          = args.duration,
        save_video        = not args.no_video,
        test_mode         = args.test,
        base_seed         = args.seed,
    )


if __name__ == "__main__":
    main()
