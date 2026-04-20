#!/usr/bin/env python3
"""
optimize_gait.py — Bayesian optimization for leg & body impedance gains
========================================================================
Finds gains that produce smooth, effective centipede locomotion on flat ground.

Search space (6 parameters):
  - body_kp:       body yaw impedance stiffness    [0.005 .. 0.10]
  - body_kv:       body yaw impedance damping       [0.001 .. 0.05]
  - leg_kp_scale:  uniform multiplier on base leg kp [0.05 .. 1.0]
                   → leg_kp = scale * [0.127, 0.0147, 0.127, 0.127]
  - leg_kv_scale:  uniform multiplier on base leg kv [0.05 .. 1.0]
                   → leg_kv = scale * [0.00056, 0.000114, 0.0000436, 0.000910]

Cost function (lower is better):
  1. Leg tracking error: RMS of (commanded - actual) across all active leg DOFs,
     averaged over all legs and time steps after settle. Measures how well legs
     follow the wave. Target: < 0.15 rad.

  2. Body jerk penalty: measures sudden angular accelerations in body yaw joints
     caused by leg reaction torques. Computed as the 95th percentile of
     |d²q/dt²| across all body segments and time. High jerk = legs twisting body.

  3. Forward speed reward (negative cost): faster is better, scaled by body length.

  4. Stability penalty: max pitch and roll angles. Buckle → cost = 1e6.

  5. Smoothness penalty: variance of body yaw joint velocities (qdot) — penalizes
     jerky body motion.

  Total cost = w_track * leg_tracking_rms
             + w_jerk  * body_jerk_95pct
             + w_smooth * body_qdot_variance
             - w_speed * forward_speed
             + w_pitch * max_pitch_penalty
             + w_roll  * max_roll_penalty

Usage:
    python scripts/optimization/farms/optimize_gait.py --n-trials 60 --duration 8
    python scripts/optimization/farms/optimize_gait.py --n-trials 100 --duration 10

Runs on flat ground (no terrain) so the optimization isolates gait quality from
terrain interaction.
"""

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import mujoco

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

import yaml

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import (
    FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF, ACTIVE_DOFS,
)

XML_PATH    = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "optimization", "gait_tune")

# ── Base leg gains (from original FARMS model, known to track well) ──
BASE_LEG_KP = np.array([0.1270, 0.0147, 0.1270, 0.1270])
BASE_LEG_KV = np.array([0.00056, 0.000114, 0.0000436, 0.000910])

# Failure thresholds
MAX_PITCH_DEG = 35.0
MAX_ROLL_DEG  = 60.0

# Cost weights (tuned to give each term O(1) contribution at typical values)
W_TRACK   = 10.0    # leg tracking error (rad) — most important
W_JERK    = 0.001   # body jerk (rad/s²) — penalize sudden twists
W_SMOOTH  = 100.0   # body qdot variance (rad/s)² — penalize jerky body wave
W_SPEED   = 50.0    # forward speed reward (m/s) — encourage locomotion
W_PITCH   = 0.5     # max pitch angle (deg)
W_ROLL    = 0.2     # max roll angle (deg)

# ═══════════════════════════════════════════════════════════════════
# Simulation + metrics
# ═══════════════════════════════════════════════════════════════════

def run_trial(config_path, duration, verbose=False):
    """
    Run one flat-ground simulation, return detailed metrics.
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctrl = ImpedanceTravelingWaveController(model, config_path)
    idx  = FARMSModelIndex(model)

    dt = model.opt.timestep
    n_steps = int(duration / dt)
    settle_steps = int(ctrl.settle_time / dt) + int(ctrl.ramp_time / dt)

    # ── Resolve body yaw joint addresses ──
    body_qpos_adr = []
    body_dof_adr  = []
    for i in range(N_BODY_JOINTS):
        jname = f"joint_body_{i+1}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        body_qpos_adr.append(model.jnt_qposadr[jid])
        body_dof_adr.append(model.jnt_dofadr[jid])

    # ── Resolve pitch/roll joint addresses ──
    pitch_qpos = []
    roll_qpos  = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if 'joint_pitch_body' in nm:
            pitch_qpos.append(model.jnt_qposadr[j])
        if 'joint_roll_body' in nm:
            roll_qpos.append(model.jnt_qposadr[j])

    # ── Resolve leg joint addresses ──
    leg_qpos_adr = np.zeros((N_LEGS, 2, N_LEG_DOF), dtype=int)
    for n in range(N_LEGS):
        for si, side in enumerate(('L', 'R')):
            for dof in range(N_LEG_DOF):
                jname = f"joint_leg_{n}_{side}_{dof}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                leg_qpos_adr[n, si, dof] = model.jnt_qposadr[jid]

    # ── Storage for time-series ──
    SAMPLE_EVERY = 20  # sample every 20 steps (every 10ms at dt=0.0005)
    n_samples = (n_steps - settle_steps) // SAMPLE_EVERY

    # Body yaw qdot history for jerk and smoothness
    body_qdot_hist = np.zeros((max(n_samples, 1), N_BODY_JOINTS))
    # Leg tracking error accumulator
    leg_error_sum = 0.0
    leg_error_count = 0

    # Position tracking for speed
    root_body_id = None
    for b in range(model.nbody):
        jnt_start = model.body_jntadr[b]
        if jnt_start >= 0 and model.jnt_type[jnt_start] == mujoco.mjtJoint.mjJNT_FREE:
            root_body_id = b
            break

    start_pos = None
    max_pitch = 0.0
    max_roll  = 0.0
    buckled   = False
    sample_idx = 0

    for step_i in range(n_steps):
        # Compute controller targets (we need them for tracking error)
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if step_i == settle_steps and root_body_id is not None:
            start_pos = data.xpos[root_body_id].copy()

        # ── Post-settle sampling ──
        if step_i > settle_steps and (step_i - settle_steps) % SAMPLE_EVERY == 0:
            if sample_idx < n_samples:
                # Body yaw qdot
                for j in range(N_BODY_JOINTS):
                    body_qdot_hist[sample_idx, j] = data.qvel[body_dof_adr[j]]

                # Leg tracking error (active DOFs only)
                for n in range(N_LEGS):
                    t = data.time
                    if ctrl.use_cpg and ctrl._cpg_initialized:
                        leg_base_phase = ctrl.leg_phases[n]
                    else:
                        leg_base_phase = ctrl.omega * t - ctrl._spatial_phase(n)
                    blend_n = ctrl._seg_blend(t, n, n_seg=N_LEGS)
                    for si, side in enumerate(('L', 'R')):
                        for dof in ctrl.active_dofs:
                            # Reconstruct commanded target
                            if blend_n <= 0:
                                target = ctrl.leg_dc_offsets[dof]
                            else:
                                phase = leg_base_phase + ctrl.leg_phase_offsets[dof]
                                wave = math.sin(phase)
                                sign = 1.0 if si == 0 else -1.0
                                target = (blend_n * sign * ctrl.leg_amps[dof] * wave
                                          + ctrl.leg_dc_offsets[dof])
                            actual = data.qpos[leg_qpos_adr[n, si, dof]]
                            leg_error_sum += (target - actual) ** 2
                            leg_error_count += 1

                sample_idx += 1

        # ── Failure check ──
        if step_i % 500 == 0 and step_i > 0:
            for qa in pitch_qpos:
                deg = abs(math.degrees(data.qpos[qa]))
                max_pitch = max(max_pitch, deg)
                if deg > MAX_PITCH_DEG:
                    buckled = True
                    break
            if not buckled:
                for qa in roll_qpos:
                    deg = abs(math.degrees(data.qpos[qa]))
                    max_roll = max(max_roll, deg)
                    if deg > MAX_ROLL_DEG:
                        buckled = True
                        break
            if buckled:
                break

    # ── Compute metrics ──
    if buckled:
        return {'cost': 1e6, 'buckled': True, 'reason': 'pitch/roll exceeded'}

    # Leg tracking RMS (rad)
    if leg_error_count > 0:
        leg_tracking_rms = math.sqrt(leg_error_sum / leg_error_count)
    else:
        leg_tracking_rms = 1.0

    # Body jerk: numerical d²q/dt² from qdot time series
    actual_samples = min(sample_idx, n_samples)
    body_qdot_used = body_qdot_hist[:actual_samples]
    if actual_samples > 2:
        dt_sample = SAMPLE_EVERY * dt
        # d(qdot)/dt = angular acceleration
        body_qddot = np.diff(body_qdot_used, axis=0) / dt_sample
        body_jerk_95 = float(np.percentile(np.abs(body_qddot), 95))
    else:
        body_jerk_95 = 0.0

    # Body qdot variance (smoothness)
    if actual_samples > 1:
        body_qdot_var = float(np.mean(np.var(body_qdot_used, axis=0)))
    else:
        body_qdot_var = 0.0

    # Forward speed
    end_pos = data.xpos[root_body_id].copy() if root_body_id is not None else None
    if start_pos is not None and end_pos is not None:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
    else:
        distance = 0.0
    effective_time = max(data.time - (settle_steps * dt), 0.01)
    forward_speed = distance / effective_time

    # Pitch/roll penalty
    pitch_penalty = (max_pitch / 20.0) ** 2 if max_pitch > 5.0 else 0.0
    roll_penalty  = (max_roll / 30.0) ** 2 if max_roll > 10.0 else 0.0

    # Total cost
    cost = (W_TRACK  * leg_tracking_rms
            + W_JERK   * body_jerk_95
            + W_SMOOTH * body_qdot_var
            - W_SPEED  * forward_speed
            + W_PITCH  * pitch_penalty
            + W_ROLL   * roll_penalty)

    metrics = {
        'cost':             float(cost),
        'buckled':          False,
        'leg_tracking_rms': float(leg_tracking_rms),
        'body_jerk_95':     float(body_jerk_95),
        'body_qdot_var':    float(body_qdot_var),
        'forward_speed':    float(forward_speed),
        'distance_m':       float(distance),
        'max_pitch_deg':    float(max_pitch),
        'max_roll_deg':     float(max_roll),
        'pitch_penalty':    float(pitch_penalty),
        'roll_penalty':     float(roll_penalty),
    }

    if verbose:
        print(f"  track={leg_tracking_rms:.4f} rad  "
              f"jerk95={body_jerk_95:.1f} rad/s²  "
              f"qdot_var={body_qdot_var:.6f}  "
              f"speed={forward_speed*1000:.1f} mm/s  "
              f"pitch={max_pitch:.1f}°  roll={max_roll:.1f}°  "
              f"cost={cost:.4f}")

    return metrics


def make_temp_config(body_kp, body_kv, leg_kp_scale, leg_kv_scale):
    """Create a temp YAML config with the trial's gain values."""
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['impedance']['body_kp'] = float(body_kp)
    cfg['impedance']['body_kv'] = float(body_kv)

    leg_kp = (BASE_LEG_KP * leg_kp_scale).tolist()
    leg_kv = (BASE_LEG_KV * leg_kv_scale).tolist()
    cfg['impedance']['leg'] = {
        'kp': [round(v, 8) for v in leg_kp],
        'kv': [round(v, 8) for v in leg_kv],
    }

    fd, path = tempfile.mkstemp(suffix='.yaml', prefix='gait_opt_')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return path


# ═══════════════════════════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════════════════════════

def make_objective(duration, n_repeats=1):
    """Return an Optuna objective function."""

    def objective(trial):
        body_kp      = trial.suggest_float("body_kp", 0.005, 0.10, log=True)
        body_kv      = trial.suggest_float("body_kv", 0.001, 0.05, log=True)
        leg_kp_scale = trial.suggest_float("leg_kp_scale", 0.05, 1.0)
        leg_kv_scale = trial.suggest_float("leg_kv_scale", 0.05, 1.0)

        tmp_cfg = make_temp_config(body_kp, body_kv, leg_kp_scale, leg_kv_scale)

        try:
            costs = []
            for rep in range(n_repeats):
                result = run_trial(tmp_cfg, duration, verbose=True)
                costs.append(result['cost'])

                # Report intermediate metrics as trial attributes
                if rep == 0:
                    for k, v in result.items():
                        if isinstance(v, (int, float)):
                            trial.set_user_attr(k, v)

            return float(np.mean(costs))
        finally:
            os.remove(tmp_cfg)

    return objective


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for leg & body impedance gains",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials",  type=int,   default=60,
                        help="Number of Optuna trials (default: 60)")
    parser.add_argument("--duration",  type=float, default=8.0,
                        help="Simulation duration in seconds (default: 8)")
    parser.add_argument("--n-repeats", type=int,   default=1,
                        help="Repeats per trial for noise averaging (default: 1)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume from existing study database")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Study storage
    db_path = os.path.join(OUTPUT_DIR, "gait_tune.db")
    storage = f"sqlite:///{db_path}"
    study_name = "gait_tune_v1"

    if args.resume:
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=TPESampler(seed=args.seed))
        print(f"Resuming study '{study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=TPESampler(seed=args.seed),
            load_if_exists=True,
        )

    # ── Seed with current config as first trial ──
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    current_leg_kp = np.array(cfg['impedance']['leg']['kp'])
    current_leg_kv = np.array(cfg['impedance']['leg']['kv'])
    # Compute approximate scale relative to base
    kp_scale = float(np.mean(current_leg_kp / BASE_LEG_KP))
    kv_scale = float(np.mean(current_leg_kv / BASE_LEG_KV))
    study.enqueue_trial({
        "body_kp": cfg['impedance']['body_kp'],
        "body_kv": cfg['impedance']['body_kv'],
        "leg_kp_scale": np.clip(kp_scale, 0.05, 1.0),
        "leg_kv_scale": np.clip(kv_scale, 0.05, 1.0),
    })
    # Also seed with the known-good full-strength gains
    study.enqueue_trial({
        "body_kp": 0.05,
        "body_kv": 0.01,
        "leg_kp_scale": 1.0,
        "leg_kv_scale": 1.0,
    })
    # And a softer body + medium legs
    study.enqueue_trial({
        "body_kp": 0.02,
        "body_kv": 0.005,
        "leg_kp_scale": 0.5,
        "leg_kv_scale": 0.5,
    })

    print(f"\n{'='*70}")
    print(f"  Gait gain optimization — {args.n_trials} trials, {args.duration}s each")
    print(f"  Search: body_kp [0.005..0.1], body_kv [0.001..0.05]")
    print(f"          leg_kp_scale [0.05..1.0], leg_kv_scale [0.05..1.0]")
    print(f"  Base leg kp: {BASE_LEG_KP.tolist()}")
    print(f"  Base leg kv: {BASE_LEG_KV.tolist()}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    t0 = time.time()
    study.optimize(make_objective(args.duration, args.n_repeats),
                   n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # ── Report best ──
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"  Best trial #{best.number}  (cost={best.value:.4f})")
    print(f"{'='*70}")
    print(f"  body_kp      = {best.params['body_kp']:.6f}")
    print(f"  body_kv      = {best.params['body_kv']:.6f}")
    print(f"  leg_kp_scale = {best.params['leg_kp_scale']:.4f}")
    print(f"  leg_kv_scale = {best.params['leg_kv_scale']:.4f}")
    best_leg_kp = BASE_LEG_KP * best.params['leg_kp_scale']
    best_leg_kv = BASE_LEG_KV * best.params['leg_kv_scale']
    print(f"  → leg_kp = {best_leg_kp.tolist()}")
    print(f"  → leg_kv = {best_leg_kv.tolist()}")
    print()
    for k, v in sorted(best.user_attrs.items()):
        print(f"  {k:20s} = {v}")
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Trials: {len(study.trials)}")

    # ── Save results ──
    results = {
        'best_params': best.params,
        'best_cost':   best.value,
        'best_attrs':  best.user_attrs,
        'best_leg_kp': best_leg_kp.tolist(),
        'best_leg_kv': best_leg_kv.tolist(),
        'n_trials':    len(study.trials),
        'duration':    args.duration,
        'elapsed_s':   elapsed,
        'all_trials':  [],
    }
    for t in study.trials:
        results['all_trials'].append({
            'number': t.number,
            'params': t.params,
            'value':  t.value,
            'attrs':  t.user_attrs,
            'state':  str(t.state),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # ── Generate YAML snippet ──
    print(f"\n  Suggested YAML update:")
    print(f"  impedance:")
    print(f"    body_kp: {best.params['body_kp']:.6f}")
    print(f"    body_kv: {best.params['body_kv']:.6f}")
    print(f"    leg:")
    print(f"      kp: {[round(v, 6) for v in best_leg_kp.tolist()]}")
    print(f"      kv: {[round(v, 8) for v in best_leg_kv.tolist()]}")


if __name__ == "__main__":
    main()
