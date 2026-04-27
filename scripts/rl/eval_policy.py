"""
eval_policy.py — Replay a trained PPO policy and write a video.
================================================================

Loads `ppo_policy.zip` (or `best/best_model.zip`) plus
`vec_normalize.pkl` from a training run dir, runs one episode on a
specified terrain + velocity command, and saves an mp4 + metrics.

Usage
-----
  # default: best model, random terrain from training distribution,
  # random velocity command
  python scripts/rl/eval_policy.py --run-dir outputs/rl/ppo_<TS>/

  # specific terrain + speed
  python scripts/rl/eval_policy.py --run-dir outputs/rl/ppo_<TS>/ \\
      --terrain-wavelength 18 --terrain-amplitude 0.01 \\
      --v-cmd 0.025 --duration 12 --video custom_eval.mp4

  # use the FINAL model rather than best-by-eval
  python scripts/rl/eval_policy.py --run-dir outputs/rl/ppo_<TS>/ --final
"""

import argparse
import os
import sys
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("ERROR: stable-baselines3 not installed.\n"
          "Install with:  pip install \"stable-baselines3[extra]\"")
    sys.exit(1)

from centipede_env import CentipedeEnv, CentipedeEnvConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="Training run directory (contains ppo_policy.zip etc.)")
    p.add_argument("--final",   action="store_true",
                   help="Use ppo_policy.zip (final) instead of best/best_model.zip")
    p.add_argument("--terrain-wavelength", type=float, default=18.0)
    p.add_argument("--terrain-amplitude",  type=float, default=0.01)
    p.add_argument("--terrain-seed",       type=int,   default=42)
    p.add_argument("--v-cmd",      type=float, default=None,
                   help="Target body-x velocity in m/s (random if omitted)")
    p.add_argument("--duration",   type=float, default=10.0)
    p.add_argument("--rl-step-dt", type=float, default=0.02)
    p.add_argument("--video",      type=str,   default="eval.mp4")
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Pick which model to load ─────────────────────────────────────
    if args.final:
        model_path = os.path.join(args.run_dir, "ppo_policy.zip")
    else:
        candidate = os.path.join(args.run_dir, "best", "best_model.zip")
        model_path = candidate if os.path.exists(candidate) else \
                     os.path.join(args.run_dir, "ppo_policy.zip")
    vec_path = os.path.join(args.run_dir, "vec_normalize.pkl")
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)
    print(f"[eval] model: {model_path}")
    print(f"[eval] vecnorm: {vec_path}")

    # ── Build env (single-env, with one fixed terrain) ───────────────
    cfg = CentipedeEnvConfig(
        rl_step_dt=args.rl_step_dt,
        episode_seconds=args.duration,
        # Pin terrain to a single wavelength + amplitude
        terrain_wavelength_lo=args.terrain_wavelength,
        terrain_wavelength_hi=args.terrain_wavelength,
        terrain_amplitude_lo=args.terrain_amplitude,
        terrain_amplitude_hi=args.terrain_amplitude,
        terrain_pool_size=1,
        terrain_pool_resample_episodes=10**9,   # never resample
        # Velocity command (we override after reset if --v-cmd is set)
        v_cmd_lo=(args.v_cmd if args.v_cmd is not None else 0.005),
        v_cmd_hi=(args.v_cmd if args.v_cmd is not None else 0.040),
        enable_video=True,
    )

    base_env = CentipedeEnv(cfg, worker_id=0)
    vec_env = DummyVecEnv([lambda: base_env])
    if os.path.exists(vec_path):
        vec_env = VecNormalize.load(vec_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"[eval] WARN: {vec_path} missing; running without obs/rwd normalization")

    model = PPO.load(model_path, env=vec_env)

    # ── Single episode ───────────────────────────────────────────────
    obs = vec_env.reset()

    # If user pinned --v-cmd, force the underlying env's command
    if args.v_cmd is not None:
        try:
            inner = vec_env.envs[0] if hasattr(vec_env, "envs") \
                    else vec_env.unwrapped.envs[0]
            inner._v_cmd = float(args.v_cmd)
        except Exception:
            pass
    print(f"[eval] v_cmd = {base_env._v_cmd*1000:.1f} mm/s, "
          f"terrain wl={args.terrain_wavelength}mm  amp={args.terrain_amplitude*1000:.1f}mm")

    metrics = {"v_body_x": [], "speed_err": [], "peak_fw": [],
               "root_pitch": [], "root_roll": [], "reward": []}

    done = np.array([False])
    total_r = 0.0
    n_steps = 0
    while not done[0]:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, info = vec_env.step(action)
        total_r += float(reward[0])
        n_steps += 1
        info_d = info[0]
        for k in metrics.keys():
            if k == "reward":
                metrics[k].append(float(reward[0]))
            elif k in info_d:
                metrics[k].append(float(info_d[k]))

    # ── Save video + metrics ─────────────────────────────────────────
    frames = base_env.get_video_frames()
    if frames and args.video:
        try:
            import mediapy
            os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
            fps = int(round(1.0 / args.rl_step_dt))
            mediapy.write_video(args.video, frames, fps=fps)
            print(f"[eval] video → {args.video}  ({len(frames)} frames @ {fps} fps)")
        except Exception as e:
            print(f"[eval] video save failed: {e}")

    # Summary
    print(f"\n=== Episode summary ===")
    print(f"  steps           : {n_steps}")
    print(f"  total reward    : {total_r:.2f}")
    if metrics["v_body_x"]:
        v_arr = np.asarray(metrics["v_body_x"])
        e_arr = np.asarray(metrics["speed_err"])
        f_arr = np.asarray(metrics["peak_fw"])
        p_arr = np.asarray(metrics["root_pitch"])
        r_arr = np.asarray(metrics["root_roll"])
        print(f"  v_cmd           : {base_env._v_cmd*1000:6.1f} mm/s")
        print(f"  v_actual mean   : {v_arr.mean()*1000:6.1f} mm/s")
        print(f"  speed err RMS   : {np.sqrt(np.mean(e_arr**2))*1000:6.1f} mm/s")
        print(f"  peak F/W (max)  : {f_arr.max():6.2f}")
        print(f"  root pitch p99  : {np.percentile(np.abs(p_arr), 99):6.2f} deg")
        print(f"  root roll  p99  : {np.percentile(np.abs(r_arr), 99):6.2f} deg")

    base_env.close()


if __name__ == "__main__":
    main()
