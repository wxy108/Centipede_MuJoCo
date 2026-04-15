#!/usr/bin/env python3
"""
Flat-ground wave-number tuning harness.

Runs each requested wave-number on FLAT ground (no hfield bumps) so you can
tune the body-yaw amplitude and leg-yaw amplitudes for that wave-number.

Today every wave-number shares the same body/leg yaw amplitudes from
configs/farms_controller.yaml.  Use `--body-amps`/`--leg-amps` to try
different values per k, watch the videos, inspect the sensor CSVs, and then
feed the chosen values back into wave_number_sweep.py via the same flags.

Usage
-----
  # Just run every k at the config default (record video + sensors)
  python scripts/sweep/flat_ground_tune.py --video

  # Try custom body-yaw amplitudes per k
  python scripts/sweep/flat_ground_tune.py --video \\
      --body-amps "1.5:0.8,2:0.7,2.5:0.65,3:0.6,3.5:0.55"

  # Also override leg amplitudes per k (4 DOFs separated by '|')
  python scripts/sweep/flat_ground_tune.py --video \\
      --body-amps "1.5:0.8,2:0.7,2.5:0.65,3:0.6,3.5:0.55" \\
      --leg-amps  "1.5:0.6|0.3|0|0,2:0.55|0.3|0|0,3:0.5|0.3|0|0"

Outputs (per run)
-----------------
  outputs/flat_ground_tune/tune_<timestamp>/
    videos/k{k}/trial_{t:02d}.mp4
    sensors/k{k}/trial_{t:02d}.csv
    results.json
    results.csv
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
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))

from wavelength_sweep import patch_xml_terrain, XML_PATH, CONFIG_PATH  # noqa: E402
from wave_number_sweep import (   # noqa: E402
    _load_base_config,
    make_config_override,
    write_tmp_config,
    parse_body_amp_table,
    parse_leg_amp_table,
    run_simulation,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "flat_ground_tune")

DEFAULT_WAVE_NUMBERS = [1.5, 2.0, 2.5, 3.0, 3.5]


# ═══════════════════════════════════════════════════════════════════════════════
# Flat terrain PNG
# ═══════════════════════════════════════════════════════════════════════════════

def make_flat_png(path, image_size=1024, gray=0):
    """Write a constant-grayscale PNG for the hfield.

    gray=0 means every hfield cell has normalised height 0, so the ground
    surface sits exactly at world z=0 regardless of z_max. This avoids the
    ~0.5 mm offset you'd get with gray=128.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((image_size, image_size), int(gray), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wave-numbers", type=str,
                   default=",".join(str(x) for x in DEFAULT_WAVE_NUMBERS),
                   help="Comma-separated list of body-wave wave_numbers to test")
    p.add_argument("--n-trials",   type=int,   default=1,
                   help="Trials per wave-number (yaw=0 for trial 0, random after)")
    p.add_argument("--duration",   type=float, default=10.0,
                   help="Seconds per trial. Includes 2 s settle + ~8 s active gait.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--video",      action="store_true")
    p.add_argument("--no-sensors", action="store_true")
    p.add_argument("--body-amps",  type=str, default=None,
                   help="Per-k body yaw amplitude override, e.g. '1.5:0.6,2:0.55'")
    p.add_argument("--leg-amps",   type=str, default=None,
                   help="Per-k leg amplitudes (4 DOFs, | separated), "
                        "e.g. '1.5:0.6|0.3|0|0,2:0.5|0.25|0|0'")
    args = p.parse_args()

    wave_numbers = [float(x) for x in args.wave_numbers.split(",")]
    body_amp_table = parse_body_amp_table(args.body_amps, wave_numbers)
    leg_amp_table  = parse_leg_amp_table(args.leg_amps,  wave_numbers)

    base_cfg = _load_base_config()
    default_body_amp = float(base_cfg["body_wave"]["amplitude"])
    default_leg_amps = list(base_cfg["leg_wave"]["amplitudes"])

    # Video support check
    can_video = False
    if args.video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed -- skipping video.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"tune_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Flat heightmap + patched XML (reused across all wave numbers)
    flat_png = os.path.join(run_dir, "flat.png")
    make_flat_png(flat_png, image_size=1024, gray=128)
    # Tiny z_max so the constant grayscale contributes ~0 height.
    tmp_xml = patch_xml_terrain(XML_PATH, flat_png, z_max=0.001)

    # Pre-sample random yaws: trial 0 always yaw=0, rest uniform
    rng = np.random.default_rng(args.seed + 7777)
    yaws = np.zeros((len(wave_numbers), args.n_trials))
    if args.n_trials > 1:
        yaws[:, 1:] = rng.uniform(0, 2 * math.pi,
                                  size=(len(wave_numbers), args.n_trials - 1))

    total_sims = len(wave_numbers) * args.n_trials
    print("=" * 70)
    print("Flat-ground Wave-number Tune")
    print("=" * 70)
    print(f"  wave_numbers: {wave_numbers}")
    print(f"  n_trials per k: {args.n_trials}  (trial 0 = yaw 0°)")
    print(f"  duration   : {args.duration}s")
    print(f"  video      : {'ON' if can_video else 'OFF'}")
    print(f"  sensors    : {'ON' if not args.no_sensors else 'OFF'}")
    print(f"  output     : {run_dir}")
    print()

    results = []
    t0 = time.time()
    sim_count = 0

    for ki, k in enumerate(wave_numbers):
        body_amp = body_amp_table[k] if body_amp_table[k] is not None else default_body_amp
        leg_amps = leg_amp_table[k]  if leg_amp_table[k]  is not None else default_leg_amps

        # Per-k temp config
        ovr_cfg = make_config_override(base_cfg, k,
                                       body_amp=body_amp,
                                       leg_amps=leg_amps)
        cfg_tag = f"flat_k{str(k).replace('.', 'p')}"
        cfg_tmp_path = write_tmp_config(ovr_cfg, cfg_tag)

        print(f"[k={k}]  body_amp={body_amp:.3f}  leg_amps={leg_amps}")

        for t in range(args.n_trials):
            sim_count += 1
            yaw = float(yaws[ki, t])
            yaw_deg = math.degrees(yaw)

            vid_path = None
            if can_video:
                vid_dir = os.path.join(run_dir, "videos", f"k{k}")
                os.makedirs(vid_dir, exist_ok=True)
                vid_path = os.path.join(vid_dir, f"trial_{t:02d}_yaw{yaw_deg:.0f}.mp4")

            sensor_path = None
            if not args.no_sensors:
                s_dir = os.path.join(run_dir, "sensors", f"k{k}")
                sensor_path = os.path.join(s_dir, f"trial_{t:02d}.csv")

            try:
                m = run_simulation(tmp_xml, cfg_tmp_path, args.duration,
                                   yaw_rad=yaw, video_path=vid_path,
                                   sensor_csv_path=sensor_path)
            except Exception as e:
                m = {
                    "survived": False, "buckle_reason": str(e),
                    "yaw_deg": yaw_deg, "cot": 1e6, "forward_speed": 0,
                    "distance_m": 0, "max_pitch_deg": 0, "mean_pitch_deg": 0,
                    "max_roll_deg": 0, "mean_roll_deg": 0, "energy_J": 0,
                    "sim_time": 0, "total_mass_kg": 0,
                    "phase_lag_deg": float("nan"), "phase_coherence": 0,
                    "phase_freq_hz": 0, "video_path": "", "sensor_csv": "",
                }
            m["wave_number"] = float(k)
            m["body_amp"]    = float(body_amp)
            m["leg_amps"]    = [float(a) for a in leg_amps]
            m["trial_idx"]   = t
            results.append(m)

            status = "OK" if m["survived"] else f"FAIL:{m['buckle_reason']}"
            eta = (time.time() - t0) / sim_count * (total_sims - sim_count)
            print(f"  [{sim_count:3d}/{total_sims}]  "
                  f"yaw={yaw_deg:5.1f}  "
                  f"speed={m['forward_speed']*1000:6.2f}mm/s  "
                  f"CoT={m['cot']:7.2f}  "
                  f"pitch_max={m['max_pitch_deg']:5.1f}°  "
                  f"roll_max={m['max_roll_deg']:5.1f}°  "
                  f"{status}  (ETA {eta/60:.0f}min)", flush=True)

        print()

    # Clean up temp configs + XML
    for k in wave_numbers:
        cfg_tag = f"flat_k{str(k).replace('.', 'p')}"
        p_tmp = os.path.join(os.path.dirname(CONFIG_PATH),
                             f".farms_controller_{cfg_tag}.tmp.yaml")
        if os.path.exists(p_tmp):
            os.remove(p_tmp)
    if os.path.exists(tmp_xml):
        os.remove(tmp_xml)

    elapsed = time.time() - t0

    # ── Save results ──────────────────────────────────────────────────────
    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp":    timestamp,
            "wave_numbers": wave_numbers,
            "n_trials":     args.n_trials,
            "duration":     args.duration,
            "default_body_amp": default_body_amp,
            "default_leg_amps": default_leg_amps,
            "body_amp_table":   body_amp_table,
            "leg_amp_table":    leg_amp_table,
            "elapsed_s":    elapsed,
            "trials":       results,
        }, f, indent=2)

    out_csv = os.path.join(run_dir, "results.csv")
    with open(out_csv, "w") as f:
        hdr = ["wave_number", "trial_idx", "yaw_deg", "body_amp",
               "survived", "cot", "forward_speed", "distance_m",
               "max_pitch_deg", "mean_pitch_deg",
               "max_roll_deg", "mean_roll_deg",
               "energy_J", "video_path", "sensor_csv"]
        f.write(",".join(hdr) + "\n")
        for r in results:
            f.write(",".join(str(r.get(h, "")) for h in hdr) + "\n")

    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed/60:.1f} min, {total_sims} sims)")
    print(f"{'=' * 70}")
    print(f"  JSON   : {out_json}")
    print(f"  CSV    : {out_csv}")
    if can_video:
        print(f"  Videos : {os.path.join(run_dir, 'videos')}/")
    if not args.no_sensors:
        print(f"  Sensors: {os.path.join(run_dir, 'sensors')}/")
    print("\nNext: watch the videos, pick best amps per k, then pass to")
    print("wave_number_sweep.py via --body-amps / --leg-amps using the same syntax.")


if __name__ == "__main__":
    main()
