"""
train_ppo.py — Train a PPO policy on the centipede CPG-modulation task.
=======================================================================

  Action space   : Box(36,)  — per-segment phase nudge + amplitude scale
  Observation    : Box(284,) — proprioception only
  Velocity cmd   : random per-episode in [5, 40] mm/s
  Terrain        : random wavelength [10, 30] mm, amplitude [5, 12] mm,
                   pool of 8 patched XMLs per worker, regenerated every
                   200 episodes.
  Reward         : speed-tracking + alive bonus − action² − force overshoot
                   − buckle penalty.

Outputs
-------
  outputs/rl/<run_tag>/
    ppo_policy.zip          # final PPO model
    ppo_policy_best.zip     # best-by-eval model (auto-saved by callback)
    vec_normalize.pkl       # observation/reward normalizer state
    tensorboard/            # TB logs
    checkpoints/            # periodic checkpoints

Usage
-----
  python scripts/rl/train_ppo.py                       # 8 workers, 5 M steps
  python scripts/rl/train_ppo.py --n-envs 16 --total-steps 20_000_000
  tensorboard --logdir outputs/rl/<run_tag>/tensorboard
"""

import argparse
import os
import sys
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (SubprocVecEnv, DummyVecEnv,
                                                   VecNormalize, VecMonitor)
    from stable_baselines3.common.callbacks import (CheckpointCallback,
                                                     EvalCallback)
except ImportError:
    print("ERROR: stable-baselines3 not installed.\n"
          "Install with:  pip install \"stable-baselines3[extra]\" "
          "gymnasium tensorboard")
    sys.exit(1)

from centipede_env import CentipedeEnv, CentipedeEnvConfig, make_env


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-envs",        type=int,   default=8)
    p.add_argument("--total-steps",   type=int,   default=5_000_000)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps",       type=int,   default=2048,
                   help="Steps per env between policy updates")
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--n-epochs",      type=int,   default=10)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-range",    type=float, default=0.2)
    p.add_argument("--ent-coef",      type=float, default=0.005)
    p.add_argument("--net-arch",      type=str,   default="256,256",
                   help="Hidden layer sizes, comma-separated")
    p.add_argument("--checkpoint-freq", type=int, default=200_000,
                   help="Save checkpoint every N steps (per-worker)")
    p.add_argument("--eval-freq",     type=int,   default=100_000,
                   help="Evaluate on a single env every N steps")
    p.add_argument("--no-subproc",    action="store_true",
                   help="Use DummyVecEnv (single-process, easier debugging)")
    p.add_argument("--resume-from",   default=None,
                   help="Path to a previous ppo_policy.zip to resume from")
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--rl-step-dt",      type=float, default=0.02)
    p.add_argument("--device", default="cpu",
                   help="Torch device for the policy (default 'cpu').  "
                        "PPO with a small MLP runs FASTER on CPU than GPU "
                        "due to per-step transfer overhead — use 'cuda' "
                        "only for big nets or CNN policies.")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Output dir ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROJECT_ROOT, "outputs", "rl", f"ppo_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    tb_dir       = os.path.join(run_dir, "tensorboard")
    ckpt_dir     = os.path.join(run_dir, "checkpoints")
    best_dir     = os.path.join(run_dir, "best")
    os.makedirs(tb_dir,   exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    print(f"{'='*72}")
    print(f"PPO training run: {run_dir}")
    print(f"{'='*72}")
    print(f"  n_envs       : {args.n_envs}")
    print(f"  total_steps  : {args.total_steps:,}")
    print(f"  episode_s    : {args.episode_seconds}")
    print(f"  rl_step_dt   : {args.rl_step_dt}")
    print(f"  net_arch     : {args.net_arch}")
    print()

    # ── Environment config ──────────────────────────────────────────────
    cfg = CentipedeEnvConfig(
        episode_seconds=args.episode_seconds,
        rl_step_dt=args.rl_step_dt,
    )

    # ── VecEnv ──────────────────────────────────────────────────────────
    env_fns = [make_env(rank=i, config=cfg, seed=args.seed)
               for i in range(args.n_envs)]
    if args.no_subproc:
        vec_env = DummyVecEnv(env_fns)
    else:
        # On Linux use fork — much lower IPC overhead than spawn (3-5x faster).
        # On Windows fork is unsupported, fall back to spawn.
        import platform
        start_method = "fork" if platform.system() == "Linux" else "spawn"
        print(f"[vec_env] SubprocVecEnv start_method={start_method}")
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0,
                           gamma=args.gamma)

    # Eval env (single, deterministic, no normalization for monitoring)
    eval_cfg = CentipedeEnvConfig(
        episode_seconds=args.episode_seconds,
        rl_step_dt=args.rl_step_dt,
    )
    eval_env = DummyVecEnv([make_env(rank=999, config=eval_cfg,
                                     seed=args.seed + 1000)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    # ── PPO ─────────────────────────────────────────────────────────────
    net_arch = [int(h) for h in args.net_arch.split(",") if h.strip()]
    policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[resume] Loading {args.resume_from}")
        model = PPO.load(args.resume_from, env=vec_env,
                         tensorboard_log=tb_dir, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir,
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    # ── Callbacks ───────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // max(args.n_envs, 1)),
        save_path=ckpt_dir,
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=max(1, args.eval_freq // max(args.n_envs, 1)),
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )

    # ── Train ───────────────────────────────────────────────────────────
    try:
        model.learn(total_timesteps=args.total_steps,
                    callback=[ckpt_cb, eval_cb],
                    progress_bar=True)
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving partial model")

    # ── Save final ──────────────────────────────────────────────────────
    final_path = os.path.join(run_dir, "ppo_policy.zip")
    model.save(final_path)
    vec_env.save(os.path.join(run_dir, "vec_normalize.pkl"))
    print(f"\n[done] Saved final policy to {final_path}")
    print(f"[done] Best-eval policy is in {best_dir}/best_model.zip")
    print(f"[done] TensorBoard:  tensorboard --logdir {tb_dir}")
    print(f"[done] To replay:    python scripts/rl/eval_policy.py "
          f"--run-dir {run_dir}")


if __name__ == "__main__":
    main()
