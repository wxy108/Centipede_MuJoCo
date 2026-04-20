#!/usr/bin/env python3
"""
optimize_gait.py — Bayesian optimization for leg & body impedance gains
========================================================================
Phase 1 (diagnostic): run with --diagnose to evaluate current config,
    output video + detailed metrics CSV.  Use this to inspect the baseline.

Phase 2 (optimize): run without --diagnose to launch Optuna TPE search.
    Every trial saves a video + metrics row to a CSV, so you can review
    all candidates visually.

Search space (4 parameters):
  - body_kp:       [0.005 .. 0.10]   body yaw impedance stiffness
  - body_kv:       [0.001 .. 0.05]   body yaw impedance damping
  - leg_kp_scale:  [0.05 .. 1.0]     multiplier on base leg kp
  - leg_kv_scale:  [0.05 .. 1.0]     multiplier on base leg kv

Base leg gains (original FARMS, scale=1.0):
  kp = [0.1270, 0.0147, 0.1270, 0.1270]
  kv = [0.00056, 0.000114, 0.0000436, 0.000910]

Metrics recorded per trial:
  - leg_tracking_rms (rad): commanded vs actual joint angles
  - body_jerk_95 (rad/s²): 95th-pctl body yaw angular acceleration
  - body_qdot_var (rad/s)²: variance of body yaw velocities (smoothness)
  - forward_speed (m/s): net displacement / time
  - max_pitch_deg, max_roll_deg: stability
  - peak_leg_torque (N·m): max |tau| across all leg actuators

Usage:
  # Step 1: diagnose current config (single run, with video)
  python scripts/optimization/farms/optimize_gait.py --diagnose --duration 10

  # Step 2: optimize (60 trials, 8s each, videos for every trial)
  python scripts/optimization/farms/optimize_gait.py --n-trials 60 --duration 8

  # Step 3: optimize more / resume
  python scripts/optimization/farms/optimize_gait.py --n-trials 40 --duration 8 --resume
"""

import argparse
import csv
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

# ── Base leg gains (original FARMS model, known to track well) ──
BASE_LEG_KP = np.array([0.1270, 0.0147, 0.1270, 0.1270])
BASE_LEG_KV = np.array([0.00056, 0.000114, 0.0000436, 0.000910])

# Failure thresholds
MAX_PITCH_DEG = 35.0
MAX_ROLL_DEG  = 60.0

# Video settings
VID_W, VID_H = 1280, 720
VID_FPS      = 30
CAM_DISTANCE = 0.20
CAM_AZIMUTH  = 60
CAM_ELEVATION = -35


# ═══════════════════════════════════════════════════════════════════
# Video helpers
# ═══════════════════════════════════════════════════════════════════

def _try_make_renderer(model):
    try:
        import mediapy  # noqa: F401
        return mujoco.Renderer(model, height=VID_H, width=VID_W), True
    except (ImportError, Exception) as e:
        print(f"  [video] renderer unavailable: {e}")
        return None, False


def _save_video(frames, path, fps=VID_FPS):
    import mediapy
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mediapy.write_video(path, frames, fps=fps)
    print(f"  [video] saved {len(frames)} frames → {path}")


# ═══════════════════════════════════════════════════════════════════
# Single simulation run with full diagnostics
# ═══════════════════════════════════════════════════════════════════

def run_trial(config_path, duration, video_path=None, verbose=False):
    """
    Run one flat-ground simulation, return detailed metrics dict.
    Optionally record video.
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
    body_dof_adr = []
    for i in range(N_BODY_JOINTS):
        jname = f"joint_body_{i+1}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
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

    # ── Resolve leg actuator IDs for torque reading ──
    leg_act_ids = []
    for a in range(model.nu):
        nm = model.actuator(a).name
        if 'act_joint_leg_' in nm:
            leg_act_ids.append(a)

    # ── Root body for position tracking ──
    root_body_id = None
    for b in range(model.nbody):
        jnt_start = model.body_jntadr[b]
        if jnt_start >= 0 and model.jnt_type[jnt_start] == mujoco.mjtJoint.mjJNT_FREE:
            root_body_id = b
            break

    # ── Video setup ──
    renderer, frames = None, []
    vid_dt = 1.0 / VID_FPS
    last_frame_t = -1.0
    if video_path:
        renderer, ok = _try_make_renderer(model)
        if not ok:
            video_path = None

    # ── Storage ──
    SAMPLE_EVERY = 20  # every 10ms at dt=0.0005
    n_samples_max = (n_steps - settle_steps) // SAMPLE_EVERY + 1

    body_qdot_hist = []
    leg_error_sum = 0.0
    leg_error_count = 0
    peak_leg_torque = 0.0

    start_pos = None
    max_pitch = 0.0
    max_roll  = 0.0
    buckled   = False
    buckle_reason = ""

    # ── Main simulation loop ──
    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if step_i == settle_steps and root_body_id is not None:
            start_pos = data.xpos[root_body_id].copy()

        # ── Post-settle sampling ──
        if step_i > settle_steps and (step_i - settle_steps) % SAMPLE_EVERY == 0:
            # Body yaw qdot snapshot
            qdot_snap = [data.qvel[adr] for adr in body_dof_adr]
            body_qdot_hist.append(qdot_snap)

            # Leg tracking error (active DOFs only)
            t = data.time
            for n in range(N_LEGS):
                if ctrl.use_cpg and ctrl._cpg_initialized:
                    leg_base_phase = ctrl.leg_phases[n]
                else:
                    leg_base_phase = ctrl.omega * t - ctrl._spatial_phase(n)
                blend_n = ctrl._seg_blend(t, n, n_seg=N_LEGS)
                for si in range(2):
                    for dof in ctrl.active_dofs:
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

            # Peak leg torque
            for aid in leg_act_ids:
                tau = abs(data.actuator_force[aid])
                if tau > peak_leg_torque:
                    peak_leg_torque = tau

        # ── Video frame ──
        if renderer and video_path and data.time - last_frame_t >= vid_dt - 1e-6:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = idx.com_pos(data)
            cam.distance  = CAM_DISTANCE
            cam.azimuth   = CAM_AZIMUTH
            cam.elevation = CAM_ELEVATION
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

        # ── Failure check (every 500 steps) ──
        if step_i % 500 == 0 and step_i > 0:
            for qa in pitch_qpos:
                deg = abs(math.degrees(data.qpos[qa]))
                max_pitch = max(max_pitch, deg)
                if deg > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch={deg:.1f}° at t={data.time:.2f}s"
                    break
            if not buckled:
                for qa in roll_qpos:
                    deg = abs(math.degrees(data.qpos[qa]))
                    max_roll = max(max_roll, deg)
                    if deg > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll={deg:.1f}° at t={data.time:.2f}s"
                        break
            if buckled:
                break

    # ── Save video ──
    if video_path and frames:
        _save_video(frames, video_path)

    # ── Compute metrics ──
    if buckled:
        return {
            'cost': 1e6, 'buckled': True, 'buckle_reason': buckle_reason,
            'leg_tracking_rms': float('nan'), 'body_jerk_95': float('nan'),
            'body_qdot_var': float('nan'), 'forward_speed': 0.0,
            'distance_m': 0.0, 'max_pitch_deg': max_pitch,
            'max_roll_deg': max_roll, 'peak_leg_torque_Nm': peak_leg_torque,
            'video_path': video_path or '',
        }

    # Leg tracking RMS
    leg_tracking_rms = math.sqrt(leg_error_sum / max(leg_error_count, 1))

    # Body jerk (95th percentile of angular acceleration)
    body_qdot_arr = np.array(body_qdot_hist)
    if len(body_qdot_arr) > 2:
        dt_sample = SAMPLE_EVERY * dt
        body_qddot = np.diff(body_qdot_arr, axis=0) / dt_sample
        body_jerk_95 = float(np.percentile(np.abs(body_qddot), 95))
    else:
        body_jerk_95 = 0.0

    # Body qdot variance (smoothness)
    if len(body_qdot_arr) > 1:
        body_qdot_var = float(np.mean(np.var(body_qdot_arr, axis=0)))
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

    metrics = {
        'cost':               0.0,   # filled below
        'buckled':            False,
        'buckle_reason':      '',
        'leg_tracking_rms':   float(leg_tracking_rms),
        'body_jerk_95':       float(body_jerk_95),
        'body_qdot_var':      float(body_qdot_var),
        'forward_speed':      float(forward_speed),
        'forward_speed_mm_s': float(forward_speed * 1000),
        'distance_m':         float(distance),
        'max_pitch_deg':      float(max_pitch),
        'max_roll_deg':       float(max_roll),
        'peak_leg_torque_Nm': float(peak_leg_torque),
        'peak_leg_torque_mNm': float(peak_leg_torque * 1000),
        'video_path':         video_path or '',
    }

    if verbose:
        print(f"  leg_track_rms  = {leg_tracking_rms:.4f} rad")
        print(f"  body_jerk_95   = {body_jerk_95:.1f} rad/s²")
        print(f"  body_qdot_var  = {body_qdot_var:.6f} (rad/s)²")
        print(f"  forward_speed  = {forward_speed*1000:.1f} mm/s")
        print(f"  distance       = {distance*1000:.1f} mm")
        print(f"  max_pitch      = {max_pitch:.1f}°")
        print(f"  max_roll       = {max_roll:.1f}°")
        print(f"  peak_leg_tau   = {peak_leg_torque*1000:.2f} mN·m")

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Config generation
# ═══════════════════════════════════════════════════════════════════

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
# CSV logger
# ═══════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    'trial', 'body_kp', 'body_kv', 'leg_kp_scale', 'leg_kv_scale',
    'cost', 'buckled',
    'leg_tracking_rms', 'body_jerk_95', 'body_qdot_var',
    'forward_speed_mm_s', 'distance_m',
    'max_pitch_deg', 'max_roll_deg',
    'peak_leg_torque_mNm', 'video_path',
]


def init_csv(path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
    return path


def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({k: row.get(k, '') for k in CSV_FIELDS})


# ═══════════════════════════════════════════════════════════════════
# Diagnose mode: single run with current config
# ═══════════════════════════════════════════════════════════════════

def run_diagnose(duration):
    """Run one trial with current config, output video + full metrics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUT_DIR, f"diagnose_{timestamp}.mp4")

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC RUN — current config, {duration}s, flat ground")
    print(f"{'='*70}")

    # Print current config
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    imp = cfg['impedance']
    print(f"  body_kp = {imp['body_kp']}")
    print(f"  body_kv = {imp['body_kv']}")
    print(f"  leg kp  = {imp['leg']['kp']}")
    print(f"  leg kv  = {imp['leg']['kv']}")
    print()

    t0 = time.time()
    result = run_trial(CONFIG_PATH, duration, video_path=video_path, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  Simulation time: {elapsed:.1f}s")
    if result.get('video_path'):
        print(f"  Video: {result['video_path']}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, f"diagnose_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Metrics: {json_path}")

    print(f"\n{'='*70}")
    print(f"  Use these metric values to judge cost function weights.")
    print(f"  Then run: python {os.path.basename(__file__)} --n-trials 60 --duration 8")
    print(f"{'='*70}\n")

    return result


# ═══════════════════════════════════════════════════════════════════
# Optimize mode: Optuna Bayesian search
# ═══════════════════════════════════════════════════════════════════

def compute_cost(metrics, w_track, w_jerk, w_smooth, w_speed, w_pitch, w_roll):
    """Compute scalar cost from metrics dict."""
    if metrics.get('buckled'):
        return 1e6

    track = metrics['leg_tracking_rms']
    jerk  = metrics['body_jerk_95']
    smooth = metrics['body_qdot_var']
    speed = metrics['forward_speed']
    pitch = (metrics['max_pitch_deg'] / 20.0) ** 2 if metrics['max_pitch_deg'] > 5.0 else 0.0
    roll  = (metrics['max_roll_deg'] / 30.0) ** 2 if metrics['max_roll_deg'] > 10.0 else 0.0

    cost = (w_track * track
            + w_jerk * jerk
            + w_smooth * smooth
            - w_speed * speed
            + w_pitch * pitch
            + w_roll * roll)
    return cost


def run_optimize(args):
    """Run Optuna optimization."""
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV log for all trials
    csv_path = os.path.join(OUTPUT_DIR, f"trials_{timestamp}.csv")
    init_csv(csv_path)
    print(f"  Trial log: {csv_path}")

    # ── First: run diagnose to get baseline metric scales ──
    print("\n  Running baseline diagnostic to calibrate cost weights...")
    baseline = run_trial(CONFIG_PATH, args.duration, verbose=True)

    if baseline.get('buckled'):
        print("  WARNING: baseline config buckles! Using default weights.")
        w_track  = 10.0
        w_jerk   = 0.001
        w_smooth = 100.0
        w_speed  = 50.0
    else:
        # Auto-calibrate weights so each term contributes roughly equally
        # Target: each term ≈ 1.0 at baseline values
        b_track  = max(baseline['leg_tracking_rms'], 1e-6)
        b_jerk   = max(baseline['body_jerk_95'], 1e-6)
        b_smooth = max(baseline['body_qdot_var'], 1e-12)
        b_speed  = max(baseline['forward_speed'], 1e-6)

        w_track  = 1.0 / b_track
        w_jerk   = 1.0 / b_jerk
        w_smooth = 1.0 / b_smooth
        w_speed  = 1.0 / b_speed
        print(f"\n  Auto-calibrated weights:")
        print(f"    w_track  = {w_track:.4f}  (baseline track = {b_track:.4f} rad)")
        print(f"    w_jerk   = {w_jerk:.6f}  (baseline jerk  = {b_jerk:.1f} rad/s²)")
        print(f"    w_smooth = {w_smooth:.2f}  (baseline var   = {b_smooth:.6f})")
        print(f"    w_speed  = {w_speed:.2f}  (baseline speed = {b_speed*1000:.1f} mm/s)")

    w_pitch = 0.5
    w_roll  = 0.2

    # ── Optuna study ──
    db_path = os.path.join(OUTPUT_DIR, "gait_tune.db")
    storage = f"sqlite:///{db_path}"
    study_name = "gait_tune_v1"

    if args.resume:
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=TPESampler(seed=args.seed))
        print(f"\n  Resuming study with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction="minimize",
            sampler=TPESampler(seed=args.seed), load_if_exists=True,
        )

    # Seed trials
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    cur_kp = np.array(cfg['impedance']['leg']['kp'])
    cur_kv = np.array(cfg['impedance']['leg']['kv'])
    kp_sc = float(np.clip(np.mean(cur_kp / BASE_LEG_KP), 0.05, 1.0))
    kv_sc = float(np.clip(np.mean(cur_kv / BASE_LEG_KV), 0.05, 1.0))

    seeds = [
        {"body_kp": cfg['impedance']['body_kp'], "body_kv": cfg['impedance']['body_kv'],
         "leg_kp_scale": kp_sc, "leg_kv_scale": kv_sc},
        {"body_kp": 0.05,  "body_kv": 0.01,  "leg_kp_scale": 1.0,  "leg_kv_scale": 1.0},
        {"body_kp": 0.02,  "body_kv": 0.005, "leg_kp_scale": 0.5,  "leg_kv_scale": 0.5},
        {"body_kp": 0.03,  "body_kv": 0.008, "leg_kp_scale": 0.3,  "leg_kv_scale": 0.4},
        {"body_kp": 0.01,  "body_kv": 0.003, "leg_kp_scale": 0.7,  "leg_kv_scale": 0.7},
    ]
    for s in seeds:
        study.enqueue_trial(s)

    trial_counter = [0]

    def objective(trial):
        body_kp      = trial.suggest_float("body_kp", 0.005, 0.10, log=True)
        body_kv      = trial.suggest_float("body_kv", 0.001, 0.05, log=True)
        leg_kp_scale = trial.suggest_float("leg_kp_scale", 0.05, 1.0)
        leg_kv_scale = trial.suggest_float("leg_kv_scale", 0.05, 1.0)

        tmp_cfg = make_temp_config(body_kp, body_kv, leg_kp_scale, leg_kv_scale)

        # Video for every trial
        vid_name = f"trial_{trial.number:04d}.mp4"
        vid_path = os.path.join(OUTPUT_DIR, "videos", vid_name)
        os.makedirs(os.path.dirname(vid_path), exist_ok=True)

        try:
            leg_kp = BASE_LEG_KP * leg_kp_scale
            leg_kv = BASE_LEG_KV * leg_kv_scale
            print(f"\n── Trial {trial.number} ──")
            print(f"  body_kp={body_kp:.4f}  body_kv={body_kv:.4f}")
            print(f"  leg_kp_scale={leg_kp_scale:.3f}  leg_kv_scale={leg_kv_scale:.3f}")
            print(f"  → leg_kp={[round(v,4) for v in leg_kp.tolist()]}")

            result = run_trial(tmp_cfg, args.duration, video_path=vid_path, verbose=True)

            cost = compute_cost(result, w_track, w_jerk, w_smooth, w_speed,
                                w_pitch, w_roll)
            result['cost'] = cost

            # Save to CSV
            row = {
                'trial': trial.number,
                'body_kp': body_kp,
                'body_kv': body_kv,
                'leg_kp_scale': leg_kp_scale,
                'leg_kv_scale': leg_kv_scale,
            }
            row.update(result)
            append_csv(csv_path, row)

            # Store in Optuna
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    trial.set_user_attr(k, v)

            print(f"  COST = {cost:.4f}")
            return cost

        finally:
            os.remove(tmp_cfg)

    # ── Run ──
    print(f"\n{'='*70}")
    print(f"  Gait optimization — {args.n_trials} trials × {args.duration}s")
    print(f"  Search: body_kp[0.005..0.1] body_kv[0.001..0.05]")
    print(f"          leg_kp_scale[0.05..1.0] leg_kv_scale[0.05..1.0]")
    print(f"  Videos: {os.path.join(OUTPUT_DIR, 'videos')}")
    print(f"  CSV:    {csv_path}")
    print(f"{'='*70}\n")

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    elapsed = time.time() - t0

    # ── Report best ──
    best = study.best_trial
    best_leg_kp = BASE_LEG_KP * best.params['leg_kp_scale']
    best_leg_kv = BASE_LEG_KV * best.params['leg_kv_scale']

    print(f"\n{'='*70}")
    print(f"  BEST: trial #{best.number}  cost={best.value:.4f}")
    print(f"{'='*70}")
    print(f"  body_kp      = {best.params['body_kp']:.6f}")
    print(f"  body_kv      = {best.params['body_kv']:.6f}")
    print(f"  leg_kp_scale = {best.params['leg_kp_scale']:.4f}")
    print(f"  leg_kv_scale = {best.params['leg_kv_scale']:.4f}")
    print(f"  → leg_kp = {[round(v,6) for v in best_leg_kp.tolist()]}")
    print(f"  → leg_kv = {[round(v,8) for v in best_leg_kv.tolist()]}")
    print()
    for k, v in sorted(best.user_attrs.items()):
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.6f}")
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Videos dir: {os.path.join(OUTPUT_DIR, 'videos')}")
    print(f"  CSV log:    {csv_path}")

    # ── Save summary ──
    summary = {
        'best_params': best.params,
        'best_cost': best.value,
        'best_leg_kp': best_leg_kp.tolist(),
        'best_leg_kv': best_leg_kv.tolist(),
        'best_attrs': best.user_attrs,
        'weights': {'w_track': w_track, 'w_jerk': w_jerk, 'w_smooth': w_smooth,
                     'w_speed': w_speed, 'w_pitch': w_pitch, 'w_roll': w_roll},
        'baseline': {k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        'n_trials': len(study.trials),
        'duration': args.duration,
        'elapsed_s': elapsed,
    }
    json_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  YAML snippet for farms_controller.yaml:")
    print(f"  impedance:")
    print(f"    body_kp: {best.params['body_kp']:.6f}")
    print(f"    body_kv: {best.params['body_kv']:.6f}")
    print(f"    leg:")
    print(f"      kp: {[round(v, 6) for v in best_leg_kp.tolist()]}")
    print(f"      kv: {[round(v, 8) for v in best_leg_kv.tolist()]}")
    print()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnose",  action="store_true",
                        help="Run single diagnostic trial with current config")
    parser.add_argument("--n-trials",  type=int,   default=60)
    parser.add_argument("--duration",  type=float, default=8.0)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume existing Optuna study")
    args = parser.parse_args()

    if args.diagnose:
        run_diagnose(args.duration)
    else:
        run_optimize(args)


if __name__ == "__main__":
    main()
