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
import math
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
from compare_baseline_rl import run_episode, BASELINE_ACTION, save_trajectory_npz


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
    p.add_argument("--model-path", default=None,
                   help="Direct path to a specific .zip policy file. "
                        "Overrides --final and the default best/ lookup. "
                        "Useful for testing midpoint checkpoints "
                        "(e.g., outputs/rl/<run>/checkpoints/ppo_3000000_steps.zip).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write CSV/JSON/PNG. Default: <rl-run>/sweep_compare/")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib plotting (CSV/JSON still written)")
    p.add_argument("--video", action="store_true",
                   help="Save mp4 videos for both baseline and RL at each wavelength "
                        "(into <output-dir>/videos/). Slower and writes ~2 mp4 per wl.")
    p.add_argument("--no-save-trajectories", action="store_true",
                   help="Skip writing per-(wl,controller) NPZ trajectories. "
                        "By default trajectories are saved into <output-dir>/trajectories/")
    p.add_argument("--n-trials", type=int, default=1,
                   help="Number of random-yaw trials per wavelength (default 1). "
                        "Trials use different starting headings (0–360°) to average "
                        "out direction-dependent terrain effects, mirroring the "
                        "scripts/sweep/wavelength_sweep.py protocol.")
    p.add_argument("--yaw-seed", type=int, default=42,
                   help="Seed for random yaw generation (reproducible)")
    return p.parse_args()


def make_env(wavelength_mm, amplitude_m, args, enable_video=False):
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
        enable_video=enable_video,
    )
    return CentipedeEnv(cfg, worker_id=0)


def save_episode_video(env, mp4_path):
    """Pull frames from a finished episode and write an mp4. Returns True on
    success, False otherwise.  Tries `_last_episode_frames` first (in case the
    env auto-reset between the last step and now), then `get_video_frames()`.
    """
    frames = []
    if hasattr(env, "_last_episode_frames") and env._last_episode_frames:
        frames = env._last_episode_frames
    if not frames:
        try:
            frames = env.get_video_frames()
        except Exception:
            frames = []
    if not frames:
        return False
    try:
        import mediapy
        os.makedirs(os.path.dirname(mp4_path) or ".", exist_ok=True)
        fps = int(round(1.0 / env.cfg.video_dt))
        mediapy.write_video(mp4_path, frames, fps=fps)
        return True
    except Exception as e:
        print(f"        [video] save failed for {mp4_path}: {e}")
        return False


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
    if args.model_path:
        model_path = args.model_path
    elif args.final:
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
    print(f"[sweep] n_trials: {args.n_trials} per wavelength "
          f"(yaw_seed={args.yaw_seed})")
    print()

    # ── Generate per-(wl, trial) random yaws ──────────────────────────────
    rng = np.random.default_rng(args.yaw_seed)
    yaw_grid = rng.uniform(0.0, 2.0 * math.pi,
                           size=(len(wls), args.n_trials))

    # ── Output dir ──────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(args.rl_run, "sweep_compare")
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, "sweep_compare.csv")
    json_path = os.path.join(out_dir, "sweep_compare.json")
    vid_dir   = os.path.join(out_dir, "videos") if args.video else None
    if vid_dir:
        os.makedirs(vid_dir, exist_ok=True)
        print(f"[sweep] videos → {vid_dir}")
    traj_dir  = (None if args.no_save_trajectories
                 else os.path.join(out_dir, "trajectories"))
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
        print(f"[sweep] trajectories → {traj_dir}")

    # ── Run sweep ───────────────────────────────────────────────────────
    rows = []        # one row per (wavelength, trial, controller) — long format
    structured = []  # for JSON: list of {wl, baseline:{...stats}, rl:{...stats}, trials:[...]}

    t_sweep_start = time.time()
    for i, wl in enumerate(wls):
        per_wl_baseline = {k: [] for k in METRIC_KEYS}
        per_wl_rl       = {k: [] for k in METRIC_KEYS}
        trials_log      = []

        for tidx in range(args.n_trials):
            t_t_start = time.time()
            yaw_rad = float(yaw_grid[i, tidx])
            yaw_deg = math.degrees(yaw_rad)
            tag = (f"wl{int(wl)}_t{tidx}_yaw{int(yaw_deg):03d}"
                   if args.n_trials > 1 else f"wl{int(wl)}")
            print(f"[{i+1:>2}/{len(wls)}] wavelength={wl:.0f} mm   "
                  f"trial {tidx+1}/{args.n_trials}   yaw={yaw_deg:5.1f}°")

            # ----- BASELINE (unmodulated CPG) -----
            env_b = make_env(wl, args.amplitude, args, enable_video=bool(vid_dir))
            baseline = run_episode(
                env_b,
                action_fn=lambda _obs: BASELINE_ACTION,
                duration=args.duration,
                warmup=args.warmup,
                initial_yaw_rad=yaw_rad,
            )
            if vid_dir:
                save_episode_video(env_b, os.path.join(vid_dir, f"{tag}_baseline.mp4"))
            if traj_dir:
                save_trajectory_npz(baseline["trajectory"],
                                    os.path.join(traj_dir, f"{tag}_baseline.npz"))
            env_b.close()

            # ----- RL POLICY -----
            env_r = make_env(wl, args.amplitude, args, enable_video=bool(vid_dir))
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
                initial_yaw_rad=yaw_rad,
            )
            if vid_dir:
                save_episode_video(env_r, os.path.join(vid_dir, f"{tag}_rl.mp4"))
            if traj_dir:
                save_trajectory_npz(rl["trajectory"],
                                    os.path.join(traj_dir, f"{tag}_rl.npz"))
            env_r.close()

            # ----- Print mini summary -----
            dv     = rl["v_mean_mm_s"]    - baseline["v_mean_mm_s"]
            dpfw   = rl["peak_fw_p99"]    - baseline["peak_fw_p99"]
            dpitch = rl["root_pitch_p99"] - baseline["root_pitch_p99"]
            wall   = time.time() - t_t_start
            print(f"        v(B,R)=({baseline['v_mean_mm_s']:6.1f},"
                  f"{rl['v_mean_mm_s']:6.1f}) mm/s  Δ={dv:+5.1f}  | "
                  f"pFW p99=({baseline['peak_fw_p99']:5.2f},"
                  f"{rl['peak_fw_p99']:5.2f}) Δ={dpfw:+5.2f}  | "
                  f"pitch p99=({baseline['root_pitch_p99']:4.2f},"
                  f"{rl['root_pitch_p99']:4.2f}) Δ={dpitch:+5.2f}  [{wall:.1f}s]")

            # ----- Per-trial CSV rows -----
            for ctrl_name, summary in (("baseline", baseline), ("rl", rl)):
                row = {
                    "wavelength_mm": wl,
                    "amplitude_mm":  args.amplitude * 1000,
                    "v_cmd_mm_s":    args.v_cmd * 1000,
                    "trial_idx":     tidx,
                    "yaw_deg":       yaw_deg,
                    "controller":    ctrl_name,
                }
                for k in METRIC_KEYS:
                    row[k] = summary[k]
                rows.append(row)

            for k in METRIC_KEYS:
                per_wl_baseline[k].append(baseline[k])
                per_wl_rl      [k].append(rl[k])

            trials_log.append({
                "trial_idx": tidx,
                "yaw_deg":   yaw_deg,
                "baseline":  {k: baseline[k] for k in METRIC_KEYS},
                "rl":        {k: rl[k]       for k in METRIC_KEYS},
            })

        # ----- Aggregate across trials for this wavelength -----
        agg_b = {f"{k}_mean": float(np.mean(per_wl_baseline[k])) for k in METRIC_KEYS}
        agg_b.update({f"{k}_std": float(np.std(per_wl_baseline[k])) for k in METRIC_KEYS})
        agg_r = {f"{k}_mean": float(np.mean(per_wl_rl[k])) for k in METRIC_KEYS}
        agg_r.update({f"{k}_std": float(np.std(per_wl_rl[k])) for k in METRIC_KEYS})

        structured.append({
            "wavelength_mm": wl,
            "amplitude_mm":  args.amplitude * 1000,
            "n_trials":      args.n_trials,
            "baseline":      agg_b,
            "rl":            agg_r,
            "trials":        trials_log,
        })

    t_sweep = time.time() - t_sweep_start
    print(f"\n[sweep] done in {t_sweep:.1f}s")

    # ── Write CSV ───────────────────────────────────────────────────────
    fieldnames = ["wavelength_mm", "amplitude_mm", "v_cmd_mm_s",
                  "trial_idx", "yaw_deg", "controller"] + METRIC_KEYS
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
    n_trials = results[0].get("n_trials", 1) if results else 1
    print("\n" + "=" * 130)
    print(f"Wavelength sweep: BASELINE vs RL  (same terrain, same v_cmd, matched warmup, "
          f"n_trials={n_trials} per wavelength)")
    print("=" * 130)
    hdr = ("{:>6} | {:>20} | {:>9} | {:>20} | {:>9} | {:>20} | {:>9}").format(
        "wl mm", "v_mean (mm/s) B → R", "Δv mean",
        "pFW p99 B → R", "Δp99",
        "pitch p99 B → R", "Δp99")
    print(hdr)
    print("-" * 130)
    for r in results:
        b = r["baseline"]; rl = r["rl"]
        dv     = rl["v_mean_mm_s_mean"]    - b["v_mean_mm_s_mean"]
        dpfw   = rl["peak_fw_p99_mean"]    - b["peak_fw_p99_mean"]
        dpitch = rl["root_pitch_p99_mean"] - b["root_pitch_p99_mean"]
        print("{:>6.0f} | "
              "{:>6.1f}±{:<4.1f} → {:>5.1f}±{:<3.1f} | {:>+9.1f} | "
              "{:>5.2f}±{:<4.2f} → {:>4.2f}±{:<3.2f} | {:>+9.2f} | "
              "{:>5.2f}±{:<4.2f} → {:>4.2f}±{:<3.2f} | {:>+9.2f}".format(
                  r["wavelength_mm"],
                  b["v_mean_mm_s_mean"],    b["v_mean_mm_s_std"],
                  rl["v_mean_mm_s_mean"],   rl["v_mean_mm_s_std"], dv,
                  b["peak_fw_p99_mean"],    b["peak_fw_p99_std"],
                  rl["peak_fw_p99_mean"],   rl["peak_fw_p99_std"], dpfw,
                  b["root_pitch_p99_mean"], b["root_pitch_p99_std"],
                  rl["root_pitch_p99_mean"],rl["root_pitch_p99_std"], dpitch))
    print("=" * 130)


def plot_sweep(results, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wls = [r["wavelength_mm"] for r in results]
    # Helpers to pull mean/std for any metric
    def vals(side, key, suffix):
        return [r[side][f"{key}_{suffix}"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    # 1) Forward speed
    ax = axes[0, 0]
    ax.errorbar(wls, vals("baseline", "v_mean_mm_s", "mean"),
                yerr=vals("baseline", "v_mean_mm_s", "std"),
                fmt="o-", label="baseline (CPG)", color="#888", capsize=3)
    ax.errorbar(wls, vals("rl", "v_mean_mm_s", "mean"),
                yerr=vals("rl", "v_mean_mm_s", "std"),
                fmt="s-", label="RL policy", color="#c0392b", capsize=3)
    ax.axhline(args.v_cmd * 1000, ls="--", color="black", lw=0.8, label="v_cmd")
    ax.set_ylabel("forward speed (mm/s)")
    ax.set_title("Forward speed vs terrain wavelength")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 2) Peak F/W p99
    ax = axes[0, 1]
    ax.errorbar(wls, vals("baseline", "peak_fw_p99", "mean"),
                yerr=vals("baseline", "peak_fw_p99", "std"),
                fmt="o-", label="baseline (CPG)", color="#888", capsize=3)
    ax.errorbar(wls, vals("rl", "peak_fw_p99", "mean"),
                yerr=vals("rl", "peak_fw_p99", "std"),
                fmt="s-", label="RL policy", color="#c0392b", capsize=3)
    ax.set_ylabel("peak F/W p99 (× body weight)")
    ax.set_title("Peak ground reaction (p99) vs wavelength")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3) Root pitch p99
    ax = axes[1, 0]
    ax.errorbar(wls, vals("baseline", "root_pitch_p99", "mean"),
                yerr=vals("baseline", "root_pitch_p99", "std"),
                fmt="o-", label="baseline (CPG)", color="#888", capsize=3)
    ax.errorbar(wls, vals("rl", "root_pitch_p99", "mean"),
                yerr=vals("rl", "root_pitch_p99", "std"),
                fmt="s-", label="RL policy", color="#c0392b", capsize=3)
    ax.set_ylabel("root pitch p99 (deg)")
    ax.set_xlabel("terrain wavelength (mm)")
    ax.set_title("Body pitch (p99) vs wavelength")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 4) Root roll p99
    ax = axes[1, 1]
    ax.errorbar(wls, vals("baseline", "root_roll_p99", "mean"),
                yerr=vals("baseline", "root_roll_p99", "std"),
                fmt="o-", label="baseline (CPG)", color="#888", capsize=3)
    ax.errorbar(wls, vals("rl", "root_roll_p99", "mean"),
                yerr=vals("rl", "root_roll_p99", "std"),
                fmt="s-", label="RL policy", color="#c0392b", capsize=3)
    ax.set_ylabel("root roll p99 (deg)")
    ax.set_xlabel("terrain wavelength (mm)")
    ax.set_title("Body roll (p99) vs wavelength")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

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
