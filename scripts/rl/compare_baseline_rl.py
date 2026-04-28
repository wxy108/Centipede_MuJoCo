"""
compare_baseline_rl.py — Apples-to-apples comparison of unmodulated CPG vs.
trained RL policy on the same terrain, with matched warmup windows.

Both episodes are run through the SAME `CentipedeEnv`:
  * BASELINE: action vector [0..0, +1..+1] — zero phase nudge, max amplitude
              scale (1.0).  Equivalent to the unmodulated CPG; the
              ModulationController step() reduces to the parent's torque
              calculation.
  * RL POLICY: actions queried from the loaded PPO model on each obs.

Both episodes start with the env's 2-second settle phase (identical), then
run `--duration` seconds of policy-controlled time.  Metrics are computed
ONLY on samples where `t >= --warmup` (skipping the first N seconds of
policy time so transients from the settle-to-policy transition don't
distort the comparison).

Usage
-----
  python scripts/rl/compare_baseline_rl.py \\
      --rl-run outputs/rl/ppo_20260427_093255/ \\
      --terrain-wavelength 18 --terrain-amplitude 0.01 --terrain-seed 42 \\
      --v-cmd 0.025 --duration 12 --warmup 2.0

  # Save metrics to JSON for plotting later
  python scripts/rl/compare_baseline_rl.py \\
      --rl-run outputs/rl/ppo_20260427_093255/ \\
      --output-json comparison_18mm.json
"""

import argparse
import json
import os
import sys

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
from kinematics import N_BODY_JOINTS
from modulation_controller import ACTION_DIM, AMP_SCALE_LO, AMP_SCALE_HI


# ── Identity-modulation action for the BASELINE episode ──────────────────
# The ModulationController maps the second half of the action [-1, 1] to
# amp_scale ∈ [LO, HI] linearly.  With LO=0.7, HI=1.0:
#   amp_scale = 0.85 + 0.15 * a_amp
# To get amp_scale = 1.0 (full CPG amplitude, no attenuation), we need
# a_amp = +1.0.  Phase nudges are zero.
N_MOD_SEG = ACTION_DIM // 2
BASELINE_ACTION = np.concatenate([
    np.zeros(N_MOD_SEG,  dtype=np.float32),    # phase nudges all zero
    np.ones (N_MOD_SEG,  dtype=np.float32),    # amp scales = AMP_SCALE_HI = 1.0
])


def run_episode(env, action_fn, duration, warmup):
    """Run one episode collecting per-step metrics + raw sensor trajectories;
    skip pre-warmup samples for both.

    `action_fn(obs) -> np.ndarray` is called every RL step to produce an
    action.  For the baseline this returns BASELINE_ACTION ignoring obs;
    for the RL policy it calls model.predict.

    Returns a dict containing:
      - aggregate scalar metrics (v_mean_mm_s, peak_fw_p99, …)
      - per-step scalar metrics (`per_step` sub-dict, post-warmup)
      - raw trajectory arrays (`trajectory` sub-dict, post-warmup):
            t        (T,)            time (s)
            q_yaw    (T, N_BODY)     body yaw joint angles (rad)
            q_pitch  (T, N_PITCH)    body pitch joint angles (rad), if model has them
            qd_yaw   (T, N_BODY)     body yaw joint velocities (rad/s)
            action   (T, A)          action issued each step
            com      (T, 3)          COM xyz (m)
            dt       float           RL step dt (s)
    """
    obs, info = env.reset()
    n_steps   = int(duration / env.cfg.rl_step_dt)
    n_warmup  = int(warmup   / env.cfg.rl_step_dt)

    keys = ["v_body_x", "peak_fw", "root_pitch", "root_roll", "speed_err"]
    rec  = {k: [] for k in keys}
    rec["reward"] = []
    rec["t"]      = []

    com_path = []           # for path-length / straightness
    com_path.append(env.idx.com_pos(env.data).copy())

    # ── Pre-resolve qpos/qvel addresses for fast vectorized access ──────
    body_qposadrs = env.model.jnt_qposadr[env.idx.body_jnt_ids]
    body_qveladrs = env.model.jnt_dofadr[env.idx.body_jnt_ids]
    has_pitch     = (hasattr(env.idx, "pitch_jnt_ids")
                     and len(env.idx.pitch_jnt_ids) > 0)
    if has_pitch:
        pitch_qposadrs = env.model.jnt_qposadr[env.idx.pitch_jnt_ids]

    traj_t, traj_qy, traj_qp, traj_qdy = [], [], [], []
    traj_act, traj_com = [], []

    for i in range(n_steps):
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        com_path.append(env.idx.com_pos(env.data).copy())

        if i >= n_warmup:
            for k in keys:
                if k in info:
                    rec[k].append(float(info[k]))
            rec["reward"].append(float(reward))
            rec["t"].append(env._cur_step * env.cfg.rl_step_dt)

            # Trajectory recordings (vectorized — one numpy slice per step)
            traj_t.append(env._cur_step * env.cfg.rl_step_dt)
            traj_qy.append(env.data.qpos[body_qposadrs].copy())
            traj_qdy.append(env.data.qvel[body_qveladrs].copy())
            if has_pitch:
                traj_qp.append(env.data.qpos[pitch_qposadrs].copy())
            traj_act.append(np.asarray(action, dtype=float).copy())
            traj_com.append(env.idx.com_pos(env.data).copy())

        if terminated or truncated:
            break

    com_path = np.asarray(com_path)[:, :2]   # XY only
    seg_lens = np.linalg.norm(np.diff(com_path, axis=0), axis=1)

    # Aggregate scalars (post-warmup window)
    summary = {
        "n_samples":         len(rec["v_body_x"]),
        "v_mean_mm_s":       float(np.mean(rec["v_body_x"])  * 1000),
        "v_std_mm_s":        float(np.std (rec["v_body_x"])  * 1000),
        "speed_err_rms_mm_s": float(np.sqrt(np.mean(np.square(rec["speed_err"]))) * 1000)
                              if rec["speed_err"] else 0.0,
        "peak_fw_max":       float(np.max(rec["peak_fw"]))    if rec["peak_fw"]   else 0.0,
        "peak_fw_p99":       float(np.percentile(rec["peak_fw"], 99))
                              if rec["peak_fw"]   else 0.0,
        "peak_fw_mean":      float(np.mean(rec["peak_fw"]))   if rec["peak_fw"]   else 0.0,
        "root_pitch_p99":    float(np.percentile(np.abs(rec["root_pitch"]), 99))
                              if rec["root_pitch"] else 0.0,
        "root_roll_p99":     float(np.percentile(np.abs(rec["root_roll"]),  99))
                              if rec["root_roll"]  else 0.0,
        "reward_total":      float(np.sum(rec["reward"]))     if rec["reward"]   else 0.0,
        "displacement_mm":   float(np.linalg.norm(com_path[-1] - com_path[0]) * 1000),
        "path_length_mm":    float(seg_lens.sum() * 1000),
        "straightness":      float(np.linalg.norm(com_path[-1] - com_path[0])
                                   / max(seg_lens.sum(), 1e-9)),
    }
    summary["per_step"] = rec
    summary["trajectory"] = {
        "t":       np.asarray(traj_t),
        "q_yaw":   np.asarray(traj_qy),
        "q_pitch": np.asarray(traj_qp) if has_pitch else None,
        "qd_yaw":  np.asarray(traj_qdy),
        "action":  np.asarray(traj_act),
        "com":     np.asarray(traj_com),
        "dt":      float(env.cfg.rl_step_dt),
    }
    return summary


def save_trajectory_npz(traj, path):
    """Persist a `trajectory` sub-dict (returned by run_episode) to NPZ.

    The analysis script reads this format directly.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path,
             t=traj["t"],
             q_yaw=traj["q_yaw"],
             q_pitch=traj["q_pitch"] if traj["q_pitch"] is not None else np.array([]),
             qd_yaw=traj["qd_yaw"],
             action=traj["action"],
             com=traj["com"],
             dt=np.array([traj["dt"]]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rl-run", required=True,
                   help="Training run dir (contains best/best_model.zip + vec_normalize.pkl)")
    p.add_argument("--terrain-wavelength", type=float, default=18.0)
    p.add_argument("--terrain-amplitude",  type=float, default=0.01)
    p.add_argument("--terrain-seed",       type=int,   default=42)
    p.add_argument("--v-cmd",      type=float, default=0.025,
                   help="Velocity command (m/s) the policy should track")
    p.add_argument("--duration",   type=float, default=12.0,
                   help="Episode length AFTER the env's settle (seconds)")
    p.add_argument("--warmup",     type=float, default=2.0,
                   help="Skip the first N seconds of policy-active time when "
                        "computing metrics, to discard settle-to-policy transients")
    p.add_argument("--output-json", default=None,
                   help="Optional path to dump full metrics JSON")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save trajectories + metrics. "
                        "Defaults to <rl-run>/comparison_wl<W>_v<V>/")
    p.add_argument("--no-save-trajectories", action="store_true",
                   help="Skip writing baseline_trajectory.npz / rl_trajectory.npz")
    p.add_argument("--final", action="store_true",
                   help="Use ppo_policy.zip instead of best/best_model.zip")
    return p.parse_args()


def make_env_for_eval(args):
    cfg = CentipedeEnvConfig(
        rl_step_dt=0.02,
        episode_seconds=args.duration,
        terrain_wavelength_lo=args.terrain_wavelength,
        terrain_wavelength_hi=args.terrain_wavelength,
        terrain_amplitude_lo=args.terrain_amplitude,
        terrain_amplitude_hi=args.terrain_amplitude,
        terrain_pool_size=1,
        terrain_pool_resample_episodes=10**9,
        v_cmd_lo=args.v_cmd,
        v_cmd_hi=args.v_cmd,
        enable_video=False,
    )
    return CentipedeEnv(cfg, worker_id=0)


def print_comparison(b, r):
    print("\n" + "=" * 78)
    print("Baseline (unmodulated CPG)  vs.  RL policy")
    print("=" * 78)
    fmt = "{:30s} | {:>14} | {:>14} | {:>14}"
    print(fmt.format("Metric", "Baseline", "RL policy", "Δ (RL − Base)"))
    print("-" * 78)

    rows = [
        ("samples (post-warmup)",      "n_samples",         "{:14d}",   1.0),
        ("forward speed (mm/s)",       "v_mean_mm_s",       "{:14.2f}", 1.0),
        ("speed std (mm/s)",           "v_std_mm_s",        "{:14.2f}", 1.0),
        ("speed err RMS (mm/s)",       "speed_err_rms_mm_s","{:14.2f}", 1.0),
        ("peak F/W (max)",             "peak_fw_max",       "{:14.2f}", 1.0),
        ("peak F/W (p99)",             "peak_fw_p99",       "{:14.2f}", 1.0),
        ("peak F/W (mean)",            "peak_fw_mean",      "{:14.2f}", 1.0),
        ("root pitch p99 (deg)",       "root_pitch_p99",    "{:14.2f}", 1.0),
        ("root roll  p99 (deg)",       "root_roll_p99",     "{:14.2f}", 1.0),
        ("displacement (mm)",          "displacement_mm",   "{:14.2f}", 1.0),
        ("path length (mm)",           "path_length_mm",    "{:14.2f}", 1.0),
        ("straightness (0..1)",        "straightness",      "{:14.4f}", 1.0),
        ("total reward (post-warmup)", "reward_total",      "{:14.2f}", 1.0),
    ]
    for label, key, fspec, _ in rows:
        bv = b[key]
        rv = r[key]
        if isinstance(bv, int):
            cell = lambda v: fspec.format(v)
        else:
            cell = lambda v: fspec.format(v)
        delta = rv - bv if not isinstance(bv, int) else rv - bv
        print(fmt.format(label, cell(bv), cell(rv), cell(delta)))
    print("=" * 78)

    # One-line interpretation
    speed_gain  = r["v_mean_mm_s"]    - b["v_mean_mm_s"]
    track_err_b = abs(b["v_mean_mm_s"]/1000 - r["per_step"]["v_body_x"][0]) if False else None
    print(f"\nReading:")
    if abs(speed_gain) < 1.0:
        print(f"  • Forward speed nearly identical ({speed_gain:+.1f} mm/s).")
    elif speed_gain > 0:
        print(f"  • RL policy is {speed_gain:.1f} mm/s FASTER than the CPG.")
    else:
        print(f"  • RL policy is {-speed_gain:.1f} mm/s SLOWER than the CPG.")
    if r["peak_fw_p99"] < b["peak_fw_p99"]:
        print(f"  • RL policy reduced peak F/W p99 by {b['peak_fw_p99']-r['peak_fw_p99']:.1f}× body weight.")
    elif r["peak_fw_p99"] > b["peak_fw_p99"] * 1.2:
        print(f"  • WARNING: RL policy increased peak F/W p99 by "
              f"{r['peak_fw_p99']-b['peak_fw_p99']:.1f}× body weight.")
    if abs(r["straightness"] - b["straightness"]) > 0.05:
        if r["straightness"] > b["straightness"]:
            print(f"  • RL policy walks straighter (Δ straightness +{r['straightness']-b['straightness']:.3f}).")
        else:
            print(f"  • RL policy walks less straight (Δ straightness {r['straightness']-b['straightness']:+.3f}).")


def main():
    args = parse_args()

    # ── Load RL model ───────────────────────────────────────────────────
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

    print(f"[compare] terrain wl={args.terrain_wavelength}mm  "
          f"amp={args.terrain_amplitude*1000:.1f}mm  seed={args.terrain_seed}")
    print(f"[compare] v_cmd={args.v_cmd*1000:.1f} mm/s  "
          f"duration={args.duration}s  warmup={args.warmup}s")
    print(f"[compare] RL model: {model_path}")
    print()

    # ── BASELINE episode (env with constant identity-modulation action) ─
    print("[1/2] Running BASELINE (unmodulated CPG)…")
    base_env_inner = make_env_for_eval(args)
    baseline = run_episode(
        base_env_inner,
        action_fn=lambda _obs: BASELINE_ACTION,
        duration=args.duration,
        warmup=args.warmup,
    )
    base_env_inner.close()

    # ── RL POLICY episode (env wrapped in VecNormalize as during training)
    print("[2/2] Running RL POLICY…")
    rl_env_inner = make_env_for_eval(args)
    vec_env = DummyVecEnv([lambda: rl_env_inner])
    if os.path.exists(vec_path):
        vec_env = VecNormalize.load(vec_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env, device="cpu")

    # We need to run the inner env step-by-step manually because run_episode
    # operates on the raw env.  Instead: provide an action_fn that converts
    # raw obs (from inner env) → normalized obs → policy.predict → action.
    def rl_action_fn(_inner_obs):
        # Get the latest VecNormalize-normalized observation by asking the
        # wrapper to normalize the inner env's observation.
        norm_obs = vec_env.normalize_obs(np.asarray([_inner_obs]))
        action, _ = model.predict(norm_obs, deterministic=True)
        return action[0]

    rl_summary = run_episode(
        rl_env_inner,
        action_fn=rl_action_fn,
        duration=args.duration,
        warmup=args.warmup,
    )
    rl_env_inner.close()

    print_comparison(baseline, rl_summary)

    # ── Default output dir ──────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(
        args.rl_run,
        f"comparison_wl{int(args.terrain_wavelength)}_v{int(args.v_cmd*1000)}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Always save metrics JSON into out_dir ───────────────────────────
    metrics_json = args.output_json or os.path.join(out_dir, "metrics.json")
    skipped_keys = {"per_step", "trajectory"}     # drop bulky arrays from JSON
    with open(metrics_json, "w") as f:
        json.dump({"baseline": {k: v for k, v in baseline.items()    if k not in skipped_keys},
                   "rl":       {k: v for k, v in rl_summary.items() if k not in skipped_keys},
                   "config":   {k: getattr(args, k) for k in
                                ["terrain_wavelength", "terrain_amplitude",
                                 "terrain_seed", "v_cmd", "duration", "warmup"]}},
                  f, indent=2)
    print(f"\n[saved] metrics → {metrics_json}")

    # ── Save raw trajectories for downstream analysis ───────────────────
    if not args.no_save_trajectories:
        b_npz = os.path.join(out_dir, "baseline_trajectory.npz")
        r_npz = os.path.join(out_dir, "rl_trajectory.npz")
        save_trajectory_npz(baseline   ["trajectory"], b_npz)
        save_trajectory_npz(rl_summary ["trajectory"], r_npz)
        print(f"[saved] trajectory → {b_npz}")
        print(f"[saved] trajectory → {r_npz}")


if __name__ == "__main__":
    main()
