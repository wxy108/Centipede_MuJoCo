#!/usr/bin/env python3
"""
Terrain browser — generate the full sweep, then open any terrain in the MuJoCo viewer.

Usage
-----
  # Step 1 — generate all sweep terrains (~900 PNGs, runs once):
  python terrain_browse.py --generate

  # Step 2 — list all available terrains:
  python terrain_browse.py --list

  # Step 3 — open one in the viewer (by index from --list):
  python terrain_browse.py --index 42

  # Or pick directly by parameters:
  python terrain_browse.py --low-amp 0.0003 --high-amp 0.0002 --high-freq 5.0 --seed 0

  # Run headless + save video instead of opening viewer:
  python terrain_browse.py --index 42 --headless --duration 10
"""

import argparse
import os
import subprocess
import sys
import glob

# ── project layout ────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
TERRAIN_DIR   = os.path.join(PROJECT_ROOT, "terrain", "generator")
CONTROL_DIR   = os.path.join(PROJECT_ROOT, "controllers", "farms")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "videos", "terrain_browse")

TERRAIN_GEN   = os.path.join(TERRAIN_DIR, "generate.py")
TERRAIN_PATCH = os.path.join(TERRAIN_DIR, "patch_xml.py")
FARMS_RUN     = os.path.join(CONTROL_DIR, "run.py")
TERRAIN_OUT   = os.path.join(PROJECT_ROOT, "terrain", "output")


# ── helpers ───────────────────────────────────────────────────────────────────
def list_terrains():
    """Return sorted list of all terrain folder paths."""
    folders = sorted(glob.glob(os.path.join(TERRAIN_OUT, "*/1.png")))
    return [os.path.dirname(p) for p in folders]


def png_from_params(low_amp, high_amp, high_freq, seed):
    tag = f"low{low_amp:.4f}_high{high_amp:.4f}_hf{high_freq:.1f}_s{seed}"
    return os.path.join(TERRAIN_OUT, tag, "1.png")


def run(cmd, label=""):
    if label:
        print(f"\n  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[terrain_browse] ERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)


def patch_and_run(png_path, headless=False, duration=10.0, video=None):
    """Patch the XML to use png_path, then launch the simulation."""
    if not os.path.isfile(png_path):
        print(f"[terrain_browse] PNG not found: {png_path}")
        print("  → Run  python terrain_browse.py --generate  first.")
        sys.exit(1)

    # Patch XML
    run([sys.executable, TERRAIN_PATCH,
         "--terrain", png_path,
         "--terrain-only"],
        label=f"Patching XML → {os.path.basename(os.path.dirname(png_path))}")

    # Build farms_run command
    cmd = [sys.executable, FARMS_RUN]
    if headless:
        cmd += ["--headless", "--duration", str(duration)]
        if video:
            cmd += ["--video", video]
        else:
            tag = os.path.basename(os.path.dirname(png_path))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            cmd += ["--video", os.path.join(OUTPUT_DIR, f"{tag}.mp4")]
    # viewer mode: no extra flags — opens interactive window

    mode = "headless" if headless else "viewer"
    run(cmd, label=f"Running simulation ({mode})")


# ── commands ──────────────────────────────────────────────────────────────────
def cmd_generate():
    """Generate all sweep terrains."""
    print("[terrain_browse] Generating full sweep …")
    run([sys.executable, TERRAIN_GEN, "--sweep"], label="generate_terrain_multifreq --sweep")
    terrains = list_terrains()
    print(f"\n[terrain_browse] Done — {len(terrains)} terrains in {TERRAIN_OUT}")


def cmd_list():
    """Print numbered list of all available terrains."""
    terrains = list_terrains()
    if not terrains:
        print("[terrain_browse] No terrains found. Run  python terrain_browse.py --generate  first.")
        return
    print(f"\n{'idx':>4}  {'folder name'}")
    print(f"{'─'*4}  {'─'*55}")
    for i, folder in enumerate(terrains):
        print(f"{i:>4}  {os.path.basename(folder)}")
    print(f"\n{len(terrains)} terrains total.")
    print(f"Open one with:  python terrain_browse.py --index <idx>")


def cmd_open_index(index, headless, duration, video):
    terrains = list_terrains()
    if not terrains:
        print("[terrain_browse] No terrains found. Run --generate first.")
        sys.exit(1)
    if index < 0 or index >= len(terrains):
        print(f"[terrain_browse] --index must be 0–{len(terrains)-1}  (got {index})")
        sys.exit(1)
    png = os.path.join(terrains[index], "1.png")
    print(f"[terrain_browse] Terrain {index}: {os.path.basename(terrains[index])}")
    patch_and_run(png, headless=headless, duration=duration, video=video)


def cmd_open_params(low_amp, high_amp, high_freq, seed, headless, duration, video):
    png = png_from_params(low_amp, high_amp, high_freq, seed)
    # Auto-generate if missing
    if not os.path.isfile(png):
        print(f"[terrain_browse] PNG not found — generating …")
        run([sys.executable, TERRAIN_GEN,
             "--low-amp",   str(low_amp),
             "--high-amp",  str(high_amp),
             "--high-freq", str(high_freq),
             "--seed",      str(seed)],
            label="generate_terrain_multifreq")
    patch_and_run(png, headless=headless, duration=duration, video=video)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Actions
    p.add_argument("--generate",  action="store_true", help="Generate all sweep terrains")
    p.add_argument("--list",      action="store_true", help="List all available terrains")
    p.add_argument("--index",     type=int, default=None,
                   help="Open terrain by index (from --list)")

    # Direct parameter selection
    p.add_argument("--low-amp",   type=float, default=None)
    p.add_argument("--high-amp",  type=float, default=None)
    p.add_argument("--high-freq", type=float, default=5.0)
    p.add_argument("--seed",      type=int,   default=0)

    # Run options
    p.add_argument("--headless",  action="store_true",
                   help="Run headless and save video (default: open viewer)")
    p.add_argument("--duration",  type=float, default=10.0,
                   help="Simulation duration in seconds (headless only)")
    p.add_argument("--video",     type=str,   default=None,
                   help="Video output path (headless only, auto-named if omitted)")

    args = p.parse_args()

    if args.generate:
        cmd_generate()
    elif args.list:
        cmd_list()
    elif args.index is not None:
        cmd_open_index(args.index, args.headless, args.duration, args.video)
    elif args.low_amp is not None and args.high_amp is not None:
        cmd_open_params(args.low_amp, args.high_amp, args.high_freq, args.seed,
                        args.headless, args.duration, args.video)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
