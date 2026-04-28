"""
sweep_compare.py — Sweep baseline-vs-RL comparison across terrain wavelengths.
==============================================================================

For each (wavelength, amplitude) point in the sweep grid, runs both:
  * BASELINE: identity-modulation action ([0..0, +1..+1]) — pure CPG
  * RL POLICY: trained PPO model (deterministic predict)

through the same `CentipedeEnv` with matched settle + warmup windows, then
collects the same aggregate metrics produced by `compare_baseline_rl.py`
(forward speed, peak F/W, root pitch/roll, displacement, straightness, etc.).

Outputs:
  * CSV with one row per wavelength × controller, all metrics
  * JSON with full results
  * Optional PNG plots of speed / force / attitude vs wavelength

Usage
-----
  # Default sweep: 10 wavelengths from 350mm down to 4mm at 10mm amplitude
  python scripts/rl/sweep_compare.py --rl-run outputs/rl/ppo_20260427_164013/

  # Custom wavelength list
  python scripts/rl/sweep_compare.py \\
      --rl-run outputs/rl/ppo_20260427_164013/ \\
      --wavelengths 350,130,65,36,24,18,14,10,7,4 \\
      --amplitude 0.01 --duration 10 --warmup 2.0

  # Save outputs to a specific directory
  python scripts/rl/sweep_compare.py \\
      --rl-run outputs/rl/ppo_20260427_164013/ \\
      --output-dir outputs/rl/ppo_20260427_164013/sweep_compare/
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("ERROR: stable-baselines3 required.")
    sys.exit(1)

from centipede_env import CentipedeEnv, CentipedeEnvConfig
from compare_baseline_rl import run_episode, BASELINE_ACTION


# ── Default sweep grids (override via CLI) ───────────────────────────────
DEFAULT_WAVELENGTHS_MM = [350, 130, 65, 36, 24, 18, 14, 10, 7, 4]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rl-run", required=True,
                   help="Training run dir (contains best/best_model.zip + vec_normalize.pkl)")
    p.add_argument("--wavelengths", type=str, default=None,
                   help="Comma-separated wavelengths in mm. "
                        f"Default: {','.join(str(w) for w in DEFAULT_WAVELENGTHS_MM)}")
    p.add_argument("--amplitude", type=float, default=0.01,
                   help="Terrain amplitude (m) — held constant across the sweep")
    p.add_argument("--terrain-seed", type=int, default=42)
    p.add_argument("--v-cmd", type=float, default=0.025,
                   help="Velocity command (m/s) the policy should track")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Episode length AFTER the env's settle (seconds)")
    p.add_argument("--warmup", type=float, default=2.0,
                   help="Skip the first N seconds of policy time when computing metrics")
    p.add_argument("--final", action="store_true",
                   help="Use ppo_policy.zip instead of best/best_model.zip")
    p.add_argument("--output-dir", default=None,
                   help="Where to write CSV/JSON/PNG. Default: <rl-run>/sweep_compare/")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib plotting (CSV/JSON still written)")
    return p.parse_args()


def make_env(wavelength_mm, amplitude_m, args):
    cfg = CentipedeEnvConfig(
        rl_step_dt=0.02,
        episode_seconds=args.duration,
        terrain_wavelength_lo=float(wavelength_mm),
        terrain_wavelength_hi=float(wavelength_mm),
        terrain_amplitude_lo=float(amplitude_m),
        terrain_amplitude_hi=float(amplitude_m),
        terrain_pool_size=1,
        terrain_pool_resample_episodes=10**9,
        v_cmd_lo=args.v_cmd,
        v_cmd_hi=args.v_cmd,
        enable_video=False,
    )
    return CentipedeEnv(cfg, worker_id=0)


# Metrics we extract from each run_episode summary into the per-row CSV
METRIC_KEYS = [
    "v_mean_mm_s",
    "v_std_mm_s",
    "speed_err_rms_mm_s",
    "peak_fw_max",
    "peak_fw_p99",
    "peak_fw_mean",
    "root_pitch_p99",
    "root_roll_p99",
    "displacement_mm",
    "path_length_mm",
    "straightness",
    "reward_total",
    "n_samples",
]


def run_sweep(args):
    # ── Resolve wavelengths ─────────────────────────────────────────────
    if args.wavelengths:
        wls = [float(s) for s in args.wavelengths.split(",") if s.strip()]
    else:
        wls = [float(w) for w in DEFAULT_WAVELENGTHS_MM]

    # ── Load RL model + VecNormalize ONCE (env-independent) ─────────────
    if args.final:
        model_path = os.path.join(args.rl_run, "ppo_policy.zip")
    else:
        cand = os.path.join(args.rl_run, "best", "best_model.zip")
        model_path = cand if os.path.exists(cand) \
                     else os.path.join(args.rl_run, "ppo_policy.zip")
    vec_path = os.path.join(args.rl_run, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(vec_path):
        print(f"WARN: vec_normalize.pkl missing — RL obs will not be normalized.")

    print(f"[sweep] RL model:  {model_path}")
    print(f"[sweep] vecnorm:   {vec_path}")
    print(f"[sweep] wavelengths (mm): {wls}")
    print(f"[sweep] amplitude: {args.amplitude*1000:.1f} mm   "
          f"v_cmd: {args.v_cmd*1000:.1f} mm/s   "
          f"duration: {args.duration}s   warmup: {args.warmup}s")
    print()

    # ── Output dir ──────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(args.rl_run, "sweep_compare")
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, "sweep_compare.csv")
    json_path = os.path.join(out_dir, "sweep_compare.json")

    # ── Run sweep ───────────────────────────────────────────────────────
    rows = []   # list of dicts (one per (wavelength, controller))
    structured = []  # for JSON: list of {wl, baseline, rl}

    t_sweep_start = time.time()
    for i, wl in enumerate(wls):
        t_wl_start = time.time()
        print(f"[{i+1:>2}/{len(wls)}] wavelength = {wl:.0f} mm")

        # ----- BASELINE (unmodulated CPG) -----
        env_b = make_env(wl, args.amplitude, args)
        baseline = run_episode(
            env_b,
            action_fn=lambda _obs: BASELINE_ACTION,
            duration=args.duration,
            warmup=args.warmup,
        )
        env_b.close()

        # ----- RL POLICY -----
        env_r = make_env(wl, args.amplitude, args)
        vec_env = DummyVecEnv([lambda: env_r])
        if os.path.exists(vec_path):
            vec_env = VecNormalize.load(vec_path, vec_env)
            vec_env.training    = False
            vec_env.norm_reward = False
        model = PPO.load(model_path, env=vec_env, device="cpu")

        def rl_action_fn(_inner_obs):
            norm_obs = vec_env.normalize_obs(np.asarray([_inner_obs]))
            action, _ = model.predict(norm_obs, deterministic=True)
            return action[0]

        rl = run_episode(
            env_r,
            action_fn=rl_action_fn,
            duration=args.duration,
            warmup=args.warmup,
        )
        env_r.close()

        # ----- Print mini summary -----
        dv = rl["v_mean_mm_s"] - baseline["v_mean_mm_s"]
        dpfw = rl["peak_fw_p99"] - baseline["peak_fw_p99"]
        dpitch = rl["root_pitch_p99"] - baseline["root_pitch_p99"]
        droll  = rl["root_roll_p99"]  - baseline["root_roll_p99"]
        wall = time.time() - t_wl_start
        print(f"        v(B,R)=({baseline['v_mean_mm_s']:6.1f},{rl['v_mean_mm_s']:6.1f}) mm/s   Δ={dv:+5.1f}  "
              f"|  pFW p99(B,R)=({baseline['peak_fw_p99']:5.2f},{rl['peak_fw_p99']:5.2f}) Δ={dpfw:+5.2f}  "
              f"|  pitch p99(B,R)=({baseline['root_pitch_p99']:4.2f},{rl['root_pitch_p99']:4.2f}) Δ={dpitch:+5.2f}  "
              f"[{wall:.1f}s]")

        # ----- Store rows -----
        for ctrl_name, summary in (("baseline", baseline), ("rl", rl)):
            row = {
                "wavelength_mm": wl,
                "amplitude_mm":  args.amplitude * 1000,
                "v_cmd_mm_s":    args.v_cmd * 1000,
                "controller":    ctrl_name,
            }
            for k in METRIC_KEYS:
                row[k] = summary[k]
            rows.append(row)

        structured.append({
            "wavelength_mm": wl,
            "amplitude_mm":  args.amplitude * 1000,
            "baseline":      {k: baseline[k] for k in METRIC_KEYS},
            "rl":            {k: rl[k] for k in METRIC_KEYS},
        })

    t_sweep = time.time() - t_sweep_start
    print(f"\n[sweep] done in {t_sweep:.1f}s")

    # ── Write CSV ───────────────────────────────────────────────────────
    fieldnames = ["wavelength_mm", "amplitude_mm", "v_cmd_mm_s",
                  "controller"] + METRIC_KEYS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[saved] CSV  → {csv_path}")

    # ── Write JSON ──────────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump({"config": {"rl_run":     args.rl_run,
                              "amplitude":  args.amplitude,
                              "v_cmd":      args.v_cmd,
                              "duration":   args.duration,
                              "warmup":     args.warmup,
                              "terrain_seed": args.terrain_seed},
                   "results": structured}, f, indent=2)
    print(f"[saved] JSON → {json_path}")

    # ── Print combined table ────────────────────────────────────────────
    print_summary_table(structured)

    # ── Plot ────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            plot_sweep(structured, out_dir, args)
        except Exception as e:
            print(f"[plot] skipped ({e})")


def print_summary_table(results):
    print("\n" + "=" * 110)
    print("Wavelength sweep: BASELINE vs RL (same terrain, same v_cmd, matched warmup)")
    print("=" * 110)
    hdr = "{:>6} | {:>14} | {:>14} | {:>13} | {:>13} | {:>13} | {:>13}".format(
        "wl mm", "v_mean (mm/s)", "Δv (mm/s)", "pFW p99", "ΔpFW p99",
        "pitch p99 deg", "Δpitch p99")
    print(hdr)
    print("-" * 110)
    for r in results:
        b = r["baseline"]; rl = r["rl"]
        dv = rl["v_mean_mm_s"] - b["v_mean_mm_s"]
        dpfw = rl["peak_fw_p99"] - b["peak_fw_p99"]
        dpitch = rl["root_pitch_p99"] - b["root_pitch_p99"]
        print("{:>6.0f} | {:>6.1f} → {:>5.1f} | {:>+13.1f} | "
              "{:>5.2f} → {:>4.2f} | {:>+13.2f} | "
              "{:>5.2f} → {:>4.2f} | {:>+13.2f}".format(
                  r["wavelength_mm"],
                  b["v_mean_mm_s"], rl["v_mean_mm_s"], dv,
                  b["peak_fw_p99"], rl["peak_fw_p99"], dpfw,
                  b["root_pitch_p99"], rl["root_pitch_p99"], dpitch))
    print("=" * 110)


def plot_sweep(results, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wls = [r["wavelength_mm"] for r in results]
    base = {k: [r["baseline"][k] for r in results] for k in METRIC_KEYS}
    rl   = {k: [r["rl"][k]       for r in results] for k in METRIC_KEYS}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    # 1) Forward speed
    ax = axes[0, 0]
    ax.plot(wls, base["v_mean_mm_s"], "o-", label="baseline (CPG)", color="#888")
    ax.plot(wls, rl["v_mean_mm_s"],   "s-", label="RL policy",      color="#c0392b")
    ax.axhline(args.v_cmd * 1000, ls="--", color="black", lw=0.8, label="v_cmd")
    ax.set_ylabel("forward speed (mm/s)")
    ax.set_title("Forward speed vs terrain wavelength")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2) Peak F/W p99
    ax = axes[0, 1]
    ax.plot(wls, base["peak_fw_p99"], "o-", label="baseline (CPG)", color="#888")
    ax.plot(wls, rl["peak_fw_p99"],   "s-", label="RL policy",      color="#c0392b")
    ax.set_ylabel("peak F/W p99 (× body weight)")
    ax.set_title("Peak ground reaction (p99) vs wavelength")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3) Root pitch p99
    ax = axes[1, 0]
    ax.plot(wls, base["root_pitch_p99"], "o-", label="baseline (CPG)", color="#888")
    ax.plot(wls, rl["root_pitch_p99"],   "s-", label="RL policy",      color="#c0392b")
    ax.set_ylabel("root pitch p99 (deg)")
    ax.set_xlabel("terrain wavelength (mm)")
    ax.set_title("Body pitch (p99) vs wavelength")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4) Root roll p99
    ax = axes[1, 1]
    ax.plot(wls, base["root_roll_p99"], "o-", label="baseline (CPG)", color="#888")
    ax.plot(wls, rl["root_roll_p99"],   "s-", label="RL policy",      color="#c0392b")
    ax.set_ylabel("root roll p99 (deg)")
    ax.set_xlabel("terrain wavelength (mm)")
    ax.set_title("Body roll (p99) vs wavelength")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # X-axis log-scale (wavelengths span 4–350 mm)
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xticks(wls)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))

    fig.suptitle(
        f"Baseline vs RL across terrain wavelengths "
        f"(amp={args.amplitude*1000:.1f} mm, v_cmd={args.v_cmd*1000:.1f} mm/s, "
        f"warmup={args.warmup}s)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = os.path.join(out_dir, "sweep_compare.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[saved] PNG  → {png_path}")


def main():
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
