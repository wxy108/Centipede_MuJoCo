# Centipede RL — CPG-modulation policy training

## What this trains

A residual policy that **modulates** the existing CPG-driven gait per body
segment. The action is small — phase nudge and amplitude scale on each of
18 body segments — so the policy can't override the CPG, only locally warp
it to handle terrain and follow a velocity command.

| Component             | Choice                                                |
|-----------------------|-------------------------------------------------------|
| Action (36-D)         | Per-segment Δφ_i ∈ [−π/4, π/4], ε_i ∈ [0.5, 1.5]     |
| Observation (~284-D)  | Proprioception only — joints, IMU, foot contacts, CPG phase, v_cmd |
| Reward                | Speed-tracking + alive bonus − action² − force overshoot − buckle |
| Velocity command      | Random per-episode in [5, 40] mm/s, body-x direction  |
| Terrain               | Random wavelength [10, 30] mm, amplitude [5, 12] mm   |
| Algorithm             | PPO with MlpPolicy [256, 256]                         |
| Episode length        | 10 s = 500 RL steps                                   |
| RL step               | 20 ms (40 MuJoCo substeps at 0.5 ms)                  |

The Bayesian-optimized impedance gains in `configs/farms_controller.yaml`
are the **base controller** — RL only sits on top.  All low-level kp/kv
values are frozen.

## Files

| File                             | Purpose                                |
|----------------------------------|----------------------------------------|
| `modulation_controller.py`       | Subclass of `ImpedanceTravelingWaveController` that takes a 36-D action and rewrites body yaw torques with modulated targets |
| `centipede_env.py`               | Gymnasium env — terrain pool, velocity-command sampling, reward, observation |
| `train_ppo.py`                   | Training loop with parallel SubprocVecEnv workers |
| `eval_policy.py`                 | Replay a trained policy with video + metrics |
| `README.md`                      | This file                              |

Nothing in `controllers/farms/` or `configs/` was modified.  The RL
controller is a clean subclass; the env reads the existing YAML.

## Setup (one-time)

The current Conda env (`CentipedeTracking`) needs three extra packages
on top of what you already have:

```
pip install "stable-baselines3[extra]" gymnasium tensorboard
```

That's it — `mujoco`, `mediapy`, `numpy`, `scipy`, `pyyaml` are already
installed from the earlier optimization work.

## Train

Default 8 workers, 5 M timesteps (~2–4 hours on your RTX 5070 laptop):

```
python scripts/rl/train_ppo.py
```

Larger run on a multi-core box (e.g. lab PC):

```
python scripts/rl/train_ppo.py --n-envs 16 --total-steps 20_000_000
```

Single-process for easy debugging (no parallel envs, slower):

```
python scripts/rl/train_ppo.py --no-subproc --n-envs 1 --total-steps 200_000
```

Resume from a previous run:

```
python scripts/rl/train_ppo.py \
    --resume-from outputs/rl/ppo_<TS>/ppo_policy.zip \
    --total-steps 5_000_000
```

Key CLI flags:

| Flag                | Default       | What it does                          |
|---------------------|---------------|---------------------------------------|
| `--n-envs`          | 8             | Parallel sim workers (use up to your physical cores) |
| `--total-steps`     | 5,000,000     | Total environment steps to train      |
| `--learning-rate`   | 3e-4          | PPO LR                                |
| `--n-steps`         | 2048          | Steps per env between gradient updates |
| `--batch-size`      | 256           | Minibatch                             |
| `--ent-coef`        | 0.005         | Entropy bonus                         |
| `--net-arch`        | "256,256"     | Hidden layer sizes                    |
| `--episode-seconds` | 10.0          | Episode length                        |
| `--rl-step-dt`      | 0.02          | Policy decision interval (s)          |

Outputs land in `outputs/rl/ppo_<timestamp>/`:

```
ppo_policy.zip          # final model
best/best_model.zip     # best by eval reward
vec_normalize.pkl       # observation normalizer state (LOAD with the policy)
checkpoints/            # periodic snapshots
tensorboard/            # TB logs
```

While training:

```
tensorboard --logdir outputs/rl/ppo_<TS>/tensorboard
```

## Evaluate / Replay

After training:

```
python scripts/rl/eval_policy.py --run-dir outputs/rl/ppo_<TS>/
```

This loads `best/best_model.zip` + `vec_normalize.pkl`, runs one episode
on default 18 mm / 10 mm terrain at a randomly-sampled velocity command,
saves `eval.mp4`, prints summary metrics.

Custom evaluation (specific terrain + speed):

```
python scripts/rl/eval_policy.py \
    --run-dir outputs/rl/ppo_<TS>/ \
    --terrain-wavelength 18 --terrain-amplitude 0.01 \
    --v-cmd 0.025 --duration 12 --video custom.mp4
```

`--final` uses the final-step policy instead of the best-by-eval one.

## Pipeline at a glance

```
                                 ┌─────────────────────────────┐
                                 │  configs/farms_controller   │
                                 │  .yaml                      │
                                 │  (Bayesian-tuned kp/kv —    │
                                 │   FROZEN during RL)         │
                                 └──────────────┬──────────────┘
                                                │
                                                ▼
   ┌─────────────────┐    obs    ┌────────────────────────────────┐
   │  PPO MlpPolicy  │ ◄──────── │ CentipedeEnv (gymnasium)       │
   │  [256, 256]     │           │  ┌──────────────────────────┐  │
   │                 │           │  │ ModulationController      │  │
   │  obs ∈ R^284    │ ─────────►│  │  (subclass of impedance   │  │
   │  act ∈ R^36     │  action   │  │   controller — adds       │  │
   │                 │           │  │   per-segment Δφ + ε)     │  │
   └─────────────────┘           │  └──────────────────────────┘  │
                                 │  ┌──────────────────────────┐  │
                                 │  │ MuJoCo (40 substeps)     │  │
                                 │  └──────────────────────────┘  │
                                 │  reward = speed-track + alive  │
                                 │           − action²            │
                                 │           − force overshoot    │
                                 │           − buckle penalty     │
                                 └────────────────────────────────┘
```

Each env worker has its own pool of 8 patched-terrain XMLs, regenerated
every 200 episodes so the policy sees fresh wavelengths over the course
of training without paying the regeneration cost every reset.

## What to expect

**Early training** (first ~200k steps): policy outputs near-zero,
reward ≈ alive bonus + ~50% speed-tracking from the unmodulated CPG.

**Mid training** (~1 M steps): policy starts amplifying the gait at
appropriate segments, speed-tracking improves, fewer buckles.

**Late training** (~5 M steps): policy finds local phase shifts that
help feet land cleanly on bumps, peak-force events drop, sustained
forward speed at commanded values across the terrain distribution.

Concrete success criteria for the first run:

* `ep_rew_mean` in TB > 200 by 1 M steps (alive bonus alone gives ~25)
* `eval/mean_reward` consistently increasing
* Eval-time `speed err RMS` < 5 mm/s
* Eval-time `peak F/W` < 5 (rare buckles)

If after 2 M steps `ep_rew_mean` is still negative, the most likely
cause is action-magnitude penalty too high or speed reward too low.
Bump `w_speed_match` to 8.0 and lower `w_action_l2` to 0.02 in
`centipede_env.py` and restart.

## Hardware sizing

| Setup                        | n_envs | Wall-clock for 5 M steps |
|------------------------------|--------|--------------------------|
| Windows PC (i7 + RTX 5070)   | 8      | ~3–4 h                   |
| Lab PC (28 logical cores)    | 16     | ~2–3 h                   |
| Single-process debug         | 1      | ~30 h (don't actually run) |

PPO is CPU-bound (the policy network is tiny; the bottleneck is MuJoCo
stepping), so more cores = faster.

## Architecture notes

* **The CPG generates the rhythm; the policy nudges it.** With
  `Δφ_max = π/4` and `ε ∈ [0.5, 1.5]`, even a saturated policy can only
  warp the gait, never replace it.  This is the "embodied prior"
  formulation from the literature.
* **Action-magnitude penalty (`w_action_l2 = 0.05`) keeps the policy
  close to zero unless modulation buys real reward.**  This prevents
  the policy from drifting into noise.
* **Velocity command is in BODY-x frame.**  The centipede's body x-axis
  is forward-pointing, so this is "go forward at this speed."
* **Termination is hard.**  A buckle ends the episode + −50 reward.
  This shapes the policy strongly away from instability.

## Common debug paths

* `--no-subproc` runs the whole training in a single process — slow,
  but errors print stack traces directly instead of being lost in
  worker subprocesses.
* If env construction fails on import, run
  `python scripts/rl/centipede_env.py` (won't actually run anything but
  imports happen so syntax errors surface).
* If MuJoCo complains about the heightfield XML, check that
  `models/farms/centipede.xml` is intact (no leftover `.sweep_tmp.xml`
  artifacts from previous runs in the project root).

## When training succeeds — next steps

1. Compare against the baseline (CPG-only on the same terrain pool)
   using `eval_policy.py` and `controllers/farms/run_rough.py`.
2. Add terrain perception (next iteration): inject a downsampled
   heightmap patch into the observation, retrain.
3. Port to IsaacLab for ~10× wall-clock speedup once the env design
   is validated in MuJoCo.
