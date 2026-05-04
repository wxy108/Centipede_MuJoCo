#!/usr/bin/env python3
"""
optimize_impedance_biological.py — bio-cost + multi-terrain Bayesian tuner
============================================================================
Sister of `optimize_impedance_bayesian.py`. Same TPE sampler, same search
space, same simulation runner — but with TWO substantive differences:

1. **Bio-faithful cost function** (matches Isaac Lab's
   optimize_impedance_biological.py exactly):
     − speed reward CAPPED at 25 mm/s (biological centipede max)
     + body_z RMS penalty (vertical body bounce)
     + body_roll RMS penalty (planarity)
     + torque-jerk penalty (smoothness of muscle commands)
     + cost-of-transport penalty (energy / distance)
   plus all the classical speed/displacement/wobble/tracking/compliance
   terms from the v3 cost.

2. **Multi-terrain evaluation per trial.** Each Optuna trial evaluates
   the same gains on THREE rough terrains (λ = 10 mm, 18 mm, 50 mm) with
   shared amplitude and seed, and AVERAGES the per-terrain cost. This
   selects for gains that walk well across a range of roughness scales,
   not just a single terrain.

3. **PD-tracking-correct controller assumed.** This optimizer is
   intended to run on the patched `impedance_controller.py` that uses
   `τ = kp·(target − q) + kv·(target_dot − qdot)`.

Tunes impedance gains on multi-terrain heightfields using Bayesian (TPE)
optimisation.  Cost combines straightness, speed (capped), tracking
RMSE, body-z bounce, body-roll, torque jerk, and cost-of-transport.

Search space
------------
Proximal (always tuned):
    body_kp,     body_kv       — body yaw joints 1..18
    hip_yaw_kp,  hip_yaw_kv    — leg DOF 0
    hip_pitch_kp,hip_pitch_kv  — leg DOF 1

Distal (tuned by default, disable with --no-tune-distal):
    tibia_kp,    tibia_kv      — leg DOF 2
    tarsus_kp,   tarsus_kv     — leg DOF 3

Body pitch (tuned by default, disable with --no-tune-pitch):
    pitch_kp,    pitch_kv      — body pitch joints (non-head)
                                  the direct knob for body compliance

Cost function
-------------
    if buckled: return BUCKLED_COST (1e5)

    cost =  w_body_track  * max(0, rmse_body_yaw_deg - body_tol_deg)
         +  w_leg_track   * max(0, rmse_leg_deg      - leg_tol_deg)
         +  w_wobble      * (1 - straightness)
         +  w_force       * max(0, peak_force_over_weight - force_limit)
         -  w_speed       * forward_speed_mps
         -  w_compliance  * min(pitch_compliance_deg, compliance_cap_deg)

    where
      straightness         = |COM(t_end) - COM(t_settle)| / COM_path_length ∈ [0,1]
      pitch_compliance_deg = time-avg of std(pitch_q - pitch_target) across the
                             ~20 body pitch joints — measures how non-uniformly
                             the body drapes over terrain (larger = more
                             compliant body shape, saturating reward at
                             --compliance-cap-deg).

Terrain
-------
The optimizer patches a heightfield into the centipede XML once at startup
and reuses it across all trials.  Defaults: wavelength = 18 mm, amplitude =
10 mm, seed = 42.  Pass --flat to optimize on a flat plane instead.

Artifacts
---------
  outputs/optimization/impedance_bay_<tag>_<timestamp>/
    study.db                    SQLite Optuna storage (resumable)
    all_trials.csv              one row per trial, all params + cost parts
    patched_model.xml           the exact XML used by every trial
    terrains/wl{λ}mm_s{seed}/   heightmap PNG + metadata
    best_params.yaml            valid farms_controller.yaml with the winner
    best_params.json            summary (trial #, cost, params)
    best_trial.npz              rich sensor dump of a replay of the winner
    best_trial.mp4              video of that replay (if mediapy installed)
    best_analysis/              analyze_sensor_data.py plots for the winner
    opt.log                     tee-d console output

Usage
-----
    # default: 18 mm rough terrain, 10-D search, seeded from current config
    python scripts/optimization/farms/optimize_impedance_bayesian.py \\
        --n-trials 250 --duration 10

    # seeded from a prior run's winner
    python scripts/optimization/farms/optimize_impedance_bayesian.py \\
        --n-trials 250 --duration 10 \\
        --seed-from outputs/optimization/impedance_bay_20260422_235624/best_params.yaml

    # flat ground (skip terrain generation)
    python scripts/optimization/farms/optimize_impedance_bayesian.py \\
        --n-trials 150 --duration 10 --flat
"""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import yaml

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna is required. Install with: "
          "pip install optuna --break-system-packages")
    sys.exit(1)

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))

import mujoco  # noqa: E402
from impedance_controller import ImpedanceTravelingWaveController  # noqa: E402
from kinematics import FARMSModelIndex  # noqa: E402
from sensor_recorder import SensorRecorder  # noqa: E402
from wavelength_sweep import (  # noqa: E402
    generate_single_wavelength_terrain,
    save_wavelength_terrain,
    patch_xml_terrain,
)

DEFAULT_XML = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
DEFAULT_CFG = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "optimization")

# ── Safety thresholds (buckling detection) ───────────────────────────────────
MAX_ROOT_PITCH_DEG  = 45.0
MAX_ROOT_ROLL_DEG   = 45.0
MAX_JOINT_PITCH_DEG = 35.0
MAX_JOINT_ROLL_DEG  = 35.0
BUCKLED_COST = 1e5

# ── Default leg kp/kv for config patching (used when base yaml lacks them) ──
DEFAULT_LEG_KP = [0.127, 0.0147, 0.127, 0.127]
DEFAULT_LEG_KV = [0.00056, 0.000114, 0.0000436, 0.000910]


# ══════════════════════════════════════════════════════════════════════════════
# Config patching
# ══════════════════════════════════════════════════════════════════════════════

def patch_config(base_cfg, params):
    """Deep-copy base_cfg and overwrite tuned parameters.  Accepts a partial
    or full parameter set — unknown keys are ignored and missing keys retain
    their base_cfg values."""
    cfg = copy.deepcopy(base_cfg)
    imp = cfg.setdefault("impedance", {})
    imp["body_kp"] = float(params["body_kp"])
    imp["body_kv"] = float(params["body_kv"])
    if "pitch_kp" in params:
        imp["pitch_kp"] = float(params["pitch_kp"])
    if "pitch_kv" in params:
        imp["pitch_kv"] = float(params["pitch_kv"])
    leg = imp.setdefault("leg", {})
    kp = list(leg.get("kp", DEFAULT_LEG_KP))
    kv = list(leg.get("kv", DEFAULT_LEG_KV))
    kp[0] = float(params["hip_yaw_kp"])
    kp[1] = float(params["hip_pitch_kp"])
    kv[0] = float(params["hip_yaw_kv"])
    kv[1] = float(params["hip_pitch_kv"])
    if "tibia_kp" in params:
        kp[2] = float(params["tibia_kp"])
        kv[2] = float(params["tibia_kv"])
    if "tarsus_kp" in params:
        kp[3] = float(params["tarsus_kp"])
        kv[3] = float(params["tarsus_kv"])
    leg["kp"] = kp
    leg["kv"] = kv
    return cfg


def write_tmp_yaml(cfg, tmp_dir, tag):
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"cfg_{tag}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Terrain setup (done once before the study starts)
# ══════════════════════════════════════════════════════════════════════════════

def setup_terrain(run_dir, wavelength_mm, amplitude_m, seed, base_xml):
    """Generate a heightfield + patched XML once.  The patched XML is left
    IN PLACE next to the original model so the existing
    ``<compiler meshdir="meshes">`` relative-path reference still resolves
    correctly on any machine (Windows or Lab PC).

    A reference copy of the patched XML and the heightfield PNG are
    saved inside ``run_dir`` for reproducibility, but those are NOT the
    ones used for simulation.  Regenerating with the same ``seed``
    reproduces the terrain deterministically.
    """
    wavelength_m = wavelength_mm * 1e-3
    h, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m=wavelength_m,
        amplitude_m=amplitude_m,
        seed=seed,
    )
    png_path = save_wavelength_terrain(h, wavelength_m, seed, run_dir)
    z_max = max(2.0 * amplitude_m, 1e-3)

    # patch_xml_terrain writes to `<base_xml>.sweep_tmp.xml` which sits
    # next to the original model -> the relative meshdir="meshes"
    # reference still resolves.  We use THIS path for every trial.
    patched_xml = patch_xml_terrain(base_xml, png_path, z_max=z_max)

    # Reference snapshot inside run_dir (NOT used for simulation — purely
    # for reproducibility / post-hoc inspection).  It will have a broken
    # meshdir if opened stand-alone, which is fine: the heightfield PNG
    # is also saved here and the seed is in best_params.json, so the
    # terrain can be regenerated deterministically.
    try:
        shutil.copy(patched_xml, os.path.join(run_dir, "patched_model.reference.xml"))
    except Exception:
        pass

    print(f"[terrain] wavelength={wavelength_mm:.1f}mm  "
          f"amplitude={amplitude_m*1000:.1f}mm  seed={seed}")
    print(f"[terrain] rms={rms_m*1000:.2f}mm  peak={peak_m*1000:.2f}mm")
    print(f"[terrain] simulation xml -> {patched_xml}")
    return patched_xml


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_one(xml_path, cfg_path, duration,
            record=False, video_path=None,
            sensor_npz_path=None, record_hz=200.0):
    """Run one simulation.  Returns (metrics, recorder)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    ctrl  = ImpedanceTravelingWaveController(model, cfg_path)
    idx   = FARMSModelIndex(model)

    dt      = model.opt.timestep
    n_steps = int(duration / dt)
    settle  = ctrl.settle_time + getattr(ctrl, "ramp_time", 0.0)

    recorder = None
    if record or sensor_npz_path is not None:
        recorder = SensorRecorder(
            model, data, ctrl,
            dt_record=1.0 / max(record_hz, 1.0),
            terrain_sampler=None,
            settle_time=settle,
        )

    frames = []
    renderer = None
    cam = None
    last_frame_t = -np.inf
    video_dt = 1.0 / 30.0
    if video_path:
        try:
            import mediapy  # noqa
            vid_w = min(1280, model.vis.global_.offwidth)
            vid_h = min(720, model.vis.global_.offheight)
            renderer = mujoco.Renderer(model, height=vid_h, width=vid_w)
            cam = mujoco.MjvCamera()
            cam.distance  = 0.25
            cam.azimuth   = 60
            cam.elevation = -35
            cam.lookat[:] = idx.com_pos(data)
        except Exception as e:
            print(f"  [video] disabled: {e}")
            video_path = None

    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

    pitch_qposadr = []
    roll_qposadr  = []
    for j in range(model.njnt):
        nm = model.joint(j).name or ""
        if "joint_pitch_body" in nm:
            pitch_qposadr.append(model.jnt_qposadr[j])
        if "joint_roll_body" in nm:
            roll_qposadr.append(model.jnt_qposadr[j])

    start_pos = None
    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if recorder is not None:
            recorder.maybe_record(model, data, ctrl)

        if step_i == int(settle / dt):
            start_pos = data.xpos[root_body].copy()

        if step_i > 0 and step_i % 200 == 0:
            R = data.xmat[root_body].reshape(3, 3)
            root_pitch_deg = math.degrees(math.asin(-R[2, 0]))
            root_roll_deg  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            if abs(root_pitch_deg) > MAX_ROOT_PITCH_DEG:
                buckled = True
                buckle_reason = f"root_pitch={root_pitch_deg:.1f}deg@{data.time:.1f}s"
                break
            if abs(root_roll_deg) > MAX_ROOT_ROLL_DEG:
                buckled = True
                buckle_reason = f"root_roll={root_roll_deg:.1f}deg@{data.time:.1f}s"
                break
            for a in pitch_qposadr:
                if abs(math.degrees(data.qpos[a])) > MAX_JOINT_PITCH_DEG:
                    buckled = True
                    buckle_reason = (f"pitch_jnt={math.degrees(data.qpos[a]):.1f}"
                                     f"@{data.time:.1f}s")
                    break
            if buckled:
                break
            for a in roll_qposadr:
                if abs(math.degrees(data.qpos[a])) > MAX_JOINT_ROLL_DEG:
                    buckled = True
                    buckle_reason = (f"roll_jnt={math.degrees(data.qpos[a]):.1f}"
                                     f"@{data.time:.1f}s")
                    break
            if buckled:
                break

        if renderer is not None and video_path and data.time - last_frame_t >= video_dt:
            cam.lookat[:] = idx.com_pos(data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

    end_pos = data.xpos[root_body].copy()
    if start_pos is not None:
        dist = float(math.hypot(end_pos[0] - start_pos[0],
                                end_pos[1] - start_pos[1]))
    else:
        dist = 0.0
    active_time = max(data.time - settle, 1e-3)
    speed = dist / active_time

    metrics = dict(
        buckled=buckled, buckle_reason=buckle_reason,
        sim_time=float(data.time), distance_m=dist,
        forward_speed_mps=speed,
    )

    if video_path and frames:
        try:
            import mediapy
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            mediapy.write_video(video_path, frames, fps=30)
        except Exception as e:
            print(f"  [video] save failed: {e}")

    if recorder is not None and sensor_npz_path is not None:
        recorder.save(sensor_npz_path)

    return metrics, recorder


# ══════════════════════════════════════════════════════════════════════════════
# Cost function  —  straightness-based
# ══════════════════════════════════════════════════════════════════════════════

def compute_cost(metrics, recorder, weights, limits, settle_s):
    """Score a completed trial using the straightness-dominated cost."""
    if metrics["buckled"]:
        return BUCKLED_COST, {"buckled": True,
                              "reason": metrics["buckle_reason"]}

    if recorder is None or len(recorder.times) < 10:
        return BUCKLED_COST, {"buckled": True,
                              "reason": "no recorder data"}

    t    = np.asarray(recorder.times, dtype=float)
    mask = t >= settle_s
    if mask.sum() < 5:
        return BUCKLED_COST, {"buckled": True,
                              "reason": "trial too short post-settle"}

    # ── Tracking (body + leg) ──────────────────────────────────────────────
    body_q   = np.asarray(recorder.body_yaw_q)[mask]
    body_tgt = np.asarray(recorder.body_yaw_tgt)[mask]
    leg_q    = np.asarray(recorder.leg_q)[mask]
    leg_tgt  = np.asarray(recorder.leg_tgt)[mask]
    rmse_body_deg = float(np.degrees(np.sqrt(np.mean((body_q - body_tgt) ** 2))))
    rmse_leg_deg  = float(np.degrees(np.sqrt(np.mean((leg_q  - leg_tgt ) ** 2))))

    # ── Straightness from COM trajectory (post-settle XY only) ─────────────
    com_xy = np.asarray(recorder.com_pos)[mask, :2]
    if len(com_xy) < 3:
        return BUCKLED_COST, {"buckled": True,
                              "reason": "too short for straightness"}
    displacement_m = float(np.linalg.norm(com_xy[-1] - com_xy[0]))
    seg_lens       = np.linalg.norm(np.diff(com_xy, axis=0), axis=1)
    path_length_m  = float(seg_lens.sum())
    # displacement ≤ path_length always; clip for numerical safety
    straightness = float(np.clip(displacement_m / max(path_length_m, 1e-9),
                                 0.0, 1.0))

    # ── Forward speed (magnitude of displacement / active time) ───────────
    active_time = max(float(t[mask][-1] - t[mask][0]), 1e-3)
    speed_mps   = displacement_m / active_time

    # ── Contact force guard ────────────────────────────────────────────────
    foot_force = np.asarray(recorder.foot_force)[mask]          # (T,19,2,3)
    foot_mag   = np.linalg.norm(foot_force, axis=-1)
    body_weight = recorder.total_mass * recorder.gravity_z
    peak_fw    = float(foot_mag.max()) / max(body_weight, 1e-9)

    # ── Body pitch compliance ──────────────────────────────────────────────
    pitch_q   = np.asarray(getattr(recorder, "pitch_q",   []))
    pitch_tgt = np.asarray(getattr(recorder, "pitch_tgt", []))
    if pitch_q.size > 0 and pitch_q.shape == pitch_tgt.shape and pitch_q.ndim == 2:
        pitch_q   = pitch_q[mask]
        pitch_tgt = pitch_tgt[mask]
        if pitch_q.shape[1] >= 2:
            deflect_per_t = np.std(pitch_q - pitch_tgt, axis=1)
            compliance_rad = float(np.mean(deflect_per_t))
        else:
            compliance_rad = 0.0
    else:
        compliance_rad = 0.0
    compliance_deg      = float(np.degrees(compliance_rad))
    compliance_reward_d = min(compliance_deg, limits["compliance_cap_deg"])

    # ── BIO METRIC 1: vertical body bounce ─────────────────────────────────
    # RMS of CoM z deviation from its mean over the active gait window.
    # Real centipedes glide at ~constant height — this directly penalises
    # visible body bouncing. Units: metres.
    com_z = np.asarray(recorder.com_pos)[mask, 2]
    body_z_rms_m = float(np.std(com_z))

    # ── BIO METRIC 2: body roll RMS ────────────────────────────────────────
    # Convert root quat (w,x,y,z) → roll (deg). Real centipedes are planar.
    rq = np.asarray(getattr(recorder, "root_quat", []))
    if rq.ndim == 2 and rq.shape[1] == 4:
        rq = rq[mask]
        w_, x_, y_, z_ = rq[:, 0], rq[:, 1], rq[:, 2], rq[:, 3]
        sinr = 2.0 * (w_ * x_ + y_ * z_)
        cosr = 1.0 - 2.0 * (x_ * x_ + y_ * y_)
        roll_rad = np.arctan2(sinr, cosr)
        body_roll_rms_deg = float(np.degrees(np.sqrt(np.mean(roll_rad ** 2))))
    else:
        body_roll_rms_deg = 0.0

    # ── BIO METRIC 3: torque jerk (smoothness) ─────────────────────────────
    # Penalise mean( (dτ/dt)² ) on actuator-applied torques. Biological
    # neural activation produces SMOOTH muscle commands; twitchy/oscillating
    # torques are unbiological. Concatenates body_yaw + pitch + leg torques
    # so all actuated DOFs contribute equally.
    torque_arrays = []
    for attr_name, expected_dim in (("body_yaw_act", 2), ("pitch_act", 2),
                                    ("leg_act",      4)):
        arr = np.asarray(getattr(recorder, attr_name, []))
        if arr.ndim == expected_dim and arr.shape[0] == len(t):
            arr = arr[mask]
            if arr.size > 0:
                # Flatten any extra dims to (T_post_settle, n_dof)
                arr = arr.reshape(arr.shape[0], -1)
                torque_arrays.append(arr)
    if torque_arrays and any(a.shape[0] >= 2 for a in torque_arrays):
        # Concatenate along DOF axis. They all share the same time axis
        # (sampled at recorder dt_record).
        tau = np.concatenate(torque_arrays, axis=1)
        dt_record = max(float(np.median(np.diff(t))), 1e-4)
        torque_diff = np.diff(tau, axis=0) / dt_record
        torque_jerk = float(np.mean(torque_diff ** 2))
    else:
        torque_jerk = 0.0

    # ── BIO METRIC 4: cost of transport (work / distance) ──────────────────
    # Total positive mechanical work / displacement. Lower = more efficient,
    # which is what biological systems optimise.
    qd_arrays = []
    for attr_name in ("body_yaw_qd", "pitch_qd", "leg_qd"):
        arr = np.asarray(getattr(recorder, attr_name, []))
        if arr.ndim >= 2 and arr.shape[0] == len(t):
            arr = arr[mask]
            if arr.size > 0:
                arr = arr.reshape(arr.shape[0], -1)
                qd_arrays.append(arr)
    if (torque_arrays and qd_arrays
            and torque_arrays[0].shape[0] == qd_arrays[0].shape[0]):
        tau = np.concatenate(torque_arrays, axis=1)
        qd  = np.concatenate(qd_arrays,     axis=1)
        # Match shapes (in case torque + qd dims differ — they should match)
        m   = min(tau.shape[1], qd.shape[1])
        joint_power = np.abs(tau[:, :m] * qd[:, :m])    # |τ · q̇|
        dt_record   = max(float(np.median(np.diff(t))), 1e-4)
        total_work_J = float(np.sum(joint_power) * dt_record)
        cost_of_transport = total_work_J / max(displacement_m, 1e-3)
    else:
        total_work_J = 0.0
        cost_of_transport = 0.0

    # ── Cost: weighted-sum (LEGACY — preserved for diagnostic) ─────────────
    body_excess  = max(0.0, rmse_body_deg - limits["track_tol_body_deg"])
    leg_excess   = max(0.0, rmse_leg_deg  - limits["track_tol_leg_deg"])
    wobble       = 1.0 - straightness
    force_excess = max(0.0, peak_fw - limits["force_limit"])

    # SPEED CAP — legacy weighted-sum only
    speed_cap = float(limits.get("speed_cap_mps", 1.0))
    speed_capped = min(speed_mps, speed_cap)

    cost_legacy = (
        weights["w_body_track"]   * body_excess
      + weights["w_leg_track"]    * leg_excess
      + weights["w_wobble"]       * wobble
      + weights["w_force"]        * force_excess
      + weights["w_body_z"]       * body_z_rms_m
      + weights["w_body_roll"]    * body_roll_rms_deg
      + weights["w_torque_jerk"]  * torque_jerk
      + weights["w_cot"]          * cost_of_transport
      - weights["w_speed"]        * speed_capped
      - weights.get("w_displacement", 0.0) * displacement_m
      - weights["w_compliance"]   * compliance_reward_d
    )

    # ── Cost: CONSTRAINT-BASED (PRIMARY — matches Isaac Lab §8.2) ──────────
    # minimize CoT (J/m) subject to:
    #   velocity_quality            ≥ 0.30   [§1.11 Pierce 2023]
    #   body_z_rms / nom_clearance  ≤ 0.10   [§1.11 Pierce 2023]
    #   body_roll_rms_deg           ≤ 5.0    [§1.12 Pierce 2026]
    #   peak_F / body_weight        ≤ 8.0    [§1.5 Cocci 2024]
    # Soft regularizer: + w_torque_jerk · torque_jerk (NOT a hard ceiling —
    # see BIO_REFERENCES.md §9.1).
    # If any constraint is violated, cost = CoT + soft + 100·Σ(violation_excess)
    # so TPE still gets a gradient back to feasibility.
    commanded_velocity = float(limits.get("commanded_velocity_mps", 0.025))
    nominal_clearance  = float(limits.get("nominal_clearance_m", 0.0258))
    min_vel_quality    = float(limits.get("min_velocity_quality", 0.3))
    max_z_rms_ratio    = float(limits.get("max_z_rms_ratio", 0.10))
    max_roll_deg       = float(limits.get("max_roll_deg", 5.0))
    force_limit        = float(limits.get("force_limit", 8.0))

    velocity_quality   = speed_mps / max(commanded_velocity, 1e-9)
    z_rms_ratio        = body_z_rms_m / max(nominal_clearance, 1e-9)

    vel_violation      = max(0.0, min_vel_quality - velocity_quality)
    z_violation        = max(0.0, z_rms_ratio - max_z_rms_ratio)
    roll_violation     = max(0.0, body_roll_rms_deg - max_roll_deg)
    force_violation    = max(0.0, peak_fw - force_limit)
    total_violation    = (vel_violation + z_violation
                          + roll_violation + force_violation)

    # CoT alone if feasible; CoT + 100·violations if not.
    # Soft torque-jerk regularizer is always added (NOT a hard constraint —
    # we don't have a centipede-leg muscle bandwidth measurement; see
    # BIO_REFERENCES.md §9.1).
    cost_constraint = (cost_of_transport
                       + weights["w_torque_jerk"] * torque_jerk)
    if total_violation > 0.0:
        cost_constraint += 100.0 * total_violation

    # Choose primary cost based on --legacy-cost flag
    use_legacy = bool(limits.get("use_legacy_cost", False))
    cost_primary = cost_legacy if use_legacy else cost_constraint

    parts = dict(
        buckled=False,
        rmse_body_yaw_deg=rmse_body_deg,
        rmse_leg_deg=rmse_leg_deg,
        body_excess=body_excess,
        leg_excess=leg_excess,
        straightness=straightness,
        wobble=wobble,
        displacement_m=displacement_m,
        path_length_m=path_length_m,
        speed_mps=speed_mps,
        speed_capped_mps=speed_capped,
        peak_force_over_weight=peak_fw,
        force_excess=force_excess,
        pitch_compliance_deg=compliance_deg,
        pitch_compliance_reward_deg=compliance_reward_d,
        body_z_rms_m=body_z_rms_m,
        body_roll_rms_deg=body_roll_rms_deg,
        torque_jerk=torque_jerk,
        total_work_J=total_work_J,
        cost_of_transport=cost_of_transport,
        # ── Constraint-based diagnostics (NEW) ──
        velocity_quality=velocity_quality,
        z_rms_ratio=z_rms_ratio,
        vel_violation=vel_violation,
        z_violation=z_violation,
        roll_violation=roll_violation,
        force_violation=force_violation,
        total_violation=total_violation,
        cost_legacy=cost_legacy,
        cost_constraint=cost_constraint,
        cost=cost_primary,
    )
    return float(cost_primary), parts


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(args, base_cfg, tmp_dir, csv_path, weights, limits,
                   xml_paths):
    """Closure-capturing objective.

    `xml_paths` is a list of (label, xml_path) tuples — one per terrain
    in the multi-terrain evaluation. Each Optuna trial runs the gain
    set on EVERY terrain, computes the per-terrain cost, and returns
    the AVERAGE cost. This selects for gains that walk well across
    the whole roughness range, not just one wavelength.
    """
    trial_rows = []
    tune_distal = args.tune_distal
    tune_pitch  = args.tune_pitch

    def obj(trial):
        params = dict(
            # Bounds aligned to Isaac Lab's optimize_impedance_biological.py
            # — all kp ≤ 0.5 ceiling, kv ranges match. (Earlier MuJoCo
            # bounds went up to 2.0 on body/hip_yaw kp; tightened to 0.5
            # for cross-simulator comparability.)
            body_kp      = trial.suggest_float("body_kp",      1e-4, 0.5,  log=True),
            body_kv      = trial.suggest_float("body_kv",      1e-6, 1.0,  log=True),
            hip_yaw_kp   = trial.suggest_float("hip_yaw_kp",   1e-4, 0.5,  log=True),
            hip_yaw_kv   = trial.suggest_float("hip_yaw_kv",   1e-7, 5e-1, log=True),
            hip_pitch_kp = trial.suggest_float("hip_pitch_kp", 1e-5, 0.5,  log=True),
            hip_pitch_kv = trial.suggest_float("hip_pitch_kv", 1e-8, 5e-2, log=True),
        )
        if tune_distal:
            # Narrow bounds — these joints hold posture, they don't drive gait.
            # Bounds aligned to Isaac Lab bio: tibia/tarsus floors lowered
            # from 0.02 to 1e-3 so the optimizer can find soft-tibia gaits
            # like Isaac Lab's trial #141 (tibia_kp = 0.038); kv ceilings
            # widened to 5e-2.
            params["tibia_kp"]  = trial.suggest_float("tibia_kp",  1e-3, 0.5,  log=True)
            params["tibia_kv"]  = trial.suggest_float("tibia_kv",  1e-7, 5e-2, log=True)
            params["tarsus_kp"] = trial.suggest_float("tarsus_kp", 1e-3, 0.5,  log=True)
            params["tarsus_kv"] = trial.suggest_float("tarsus_kv", 1e-6, 5e-2, log=True)
        if tune_pitch:
            # Body pitch gains — direct knob for compliance.  Range allows
            # very soft (1e-3) up to moderate stiffness (2e-1).  Head pitch
            # gains are intentionally NOT tuned (they need to hold the head
            # against gravity).
            # Bounds aligned to Isaac Lab bio: pitch_kp ceiling raised
            # from 0.2 to 0.5; pitch_kv ceiling raised from 0.1 to 0.5.
            params["pitch_kp"]  = trial.suggest_float("pitch_kp",  1e-4, 0.5,  log=True)
            params["pitch_kv"]  = trial.suggest_float("pitch_kv",  1e-6, 5e-1, log=True)

        cfg = patch_config(base_cfg, params)
        cfg_path = write_tmp_yaml(cfg, tmp_dir, f"trial_{trial.number:05d}")

        # ── Multi-terrain evaluation: run on EACH XML, average cost ──────
        t0 = time.time()
        per_terrain_costs = []
        per_terrain_parts = []
        any_buckled = False
        for label, xml_path in xml_paths:
            try:
                metrics, recorder = run_one(
                    xml_path, cfg_path, args.duration,
                    record=True, record_hz=args.record_hz)
            except Exception as e:
                tdt = time.time() - t0
                print(f"  [trial {trial.number:4d}] CRASH on terrain "
                      f"{label}: {e}  ({tdt:.1f}s)")
                # One bad terrain → trial is invalid for selection
                return BUCKLED_COST

            settle_s = 2.5
            cost_t, parts_t = compute_cost(metrics, recorder, weights,
                                            limits, settle_s=settle_s)
            parts_t["terrain"] = label
            per_terrain_costs.append(cost_t)
            per_terrain_parts.append(parts_t)
            if parts_t.get("buckled", False):
                any_buckled = True

        # Combined cost = mean across terrains. Generalist gains win.
        cost = float(np.mean(per_terrain_costs))
        # Aggregate per-terrain part stats — show min/mean/max for the
        # main metrics in the trial log so we can see how much the
        # cost varies across terrains.
        def _agg(key):
            vals = [p.get(key) for p in per_terrain_parts
                    if p.get(key) is not None and not p.get("buckled", False)]
            return (None if not vals
                    else (float(np.mean(vals)),
                          float(np.min(vals)),
                          float(np.max(vals))))

        speed_agg  = _agg("speed_mps")
        strt_agg   = _agg("straightness")
        z_rms_agg  = _agg("body_z_rms_m")
        roll_agg   = _agg("body_roll_rms_deg")
        cot_agg    = _agg("cost_of_transport")

        dt = time.time() - t0
        if any_buckled:
            n_buck = sum(1 for p in per_terrain_parts
                         if p.get("buckled", False))
            print(f"  [trial {trial.number:4d}] {n_buck}/"
                  f"{len(xml_paths)} BUCKLED  "
                  f"per-terrain costs={[f'{c:.2f}' for c in per_terrain_costs]}"
                  f"  combined={cost:.2f}  ({dt:.1f}s)")
        else:
            sp = speed_agg[0]*1000 if speed_agg else 0.0
            st = strt_agg[0] if strt_agg else 0.0
            zr = z_rms_agg[0]*1000 if z_rms_agg else 0.0
            ro = roll_agg[0] if roll_agg else 0.0
            ct = cot_agg[0] if cot_agg else 0.0
            print(f"  [trial {trial.number:4d}] cost={cost:7.3f}  "
                  f"speed={sp:5.1f}mm/s  strt={st:.3f}  "
                  f"z_rms={zr:4.1f}mm  roll={ro:4.1f}°  "
                  f"CoT={ct:.3f}  ({dt:.1f}s)")

        try:
            os.remove(cfg_path)
        except Exception:
            pass

        # CSV row carries the COMBINED metrics + per-terrain breakdown
        # (one row per trial). The per-terrain costs are folded in as
        # cost_<label>, speed_<label>, etc. for downstream filtering.
        row = {**params, "trial": trial.number, "sim_time_s": dt,
               "cost": cost, "n_buckled_terrains":
                   sum(1 for p in per_terrain_parts if p.get("buckled", False))}
        # Combined averages of common metrics
        for key in ("speed_mps", "straightness", "body_z_rms_m",
                    "body_roll_rms_deg", "torque_jerk", "cost_of_transport",
                    "pitch_compliance_deg"):
            agg = _agg(key)
            if agg is not None:
                row[f"avg_{key}"] = agg[0]
        # Per-terrain breakdown
        for parts_t, c in zip(per_terrain_parts, per_terrain_costs):
            lab = parts_t.get("terrain", "?")
            row[f"cost_{lab}"]    = c
            row[f"buckled_{lab}"] = bool(parts_t.get("buckled", False))
            row[f"speed_{lab}"]   = parts_t.get("speed_mps", 0.0)
            row[f"strt_{lab}"]    = parts_t.get("straightness", 0.0)
            row[f"z_rms_{lab}"]   = parts_t.get("body_z_rms_m", 0.0)
        trial_rows.append(row)
        _dump_rows(csv_path, trial_rows)
        return cost

    return obj, trial_rows


def _dump_rows(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ══════════════════════════════════════════════════════════════════════════════
# Post-optimization: re-run winner with full video + sensor npz
# ══════════════════════════════════════════════════════════════════════════════

def rerun_best(best_params, base_cfg, xml_path, run_dir, duration, record_hz):
    cfg = patch_config(base_cfg, best_params)
    best_cfg_path = os.path.join(run_dir, "best_params.yaml")
    with open(best_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    sensor_npz = os.path.join(run_dir, "best_trial.npz")
    video      = os.path.join(run_dir, "best_trial.mp4")

    print(f"\n[replay] Running winner with video + full sensor capture...")
    metrics, _ = run_one(
        xml_path, best_cfg_path, duration,
        record=True, video_path=video,
        sensor_npz_path=sensor_npz,
        record_hz=record_hz)

    print(f"[replay] buckled={metrics['buckled']}  "
          f"speed={metrics['forward_speed_mps']*1000:.1f}mm/s  "
          f"distance={metrics['distance_m']*1000:.1f}mm  "
          f"wrote:\n"
          f"   - {best_cfg_path}\n"
          f"   - {sensor_npz}\n"
          f"   - {video}")

    return sensor_npz, video, best_cfg_path


# ══════════════════════════════════════════════════════════════════════════════
# Seeding helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clip_to_search_bounds(seed_params, tune_distal, tune_pitch):
    """Clip seed values to the log-uniform search bounds so enqueue_trial
    doesn't reject them."""
    bounds = dict(
        body_kp      = (1e-3, 2.0),
        body_kv      = (1e-5, 0.5),
        hip_yaw_kp   = (1e-3, 2.0),
        hip_yaw_kv   = (1e-6, 1e-1),
        hip_pitch_kp = (1e-4, 1.0),
        hip_pitch_kv = (1e-7, 1e-2),
    )
    if tune_distal:
        bounds.update(dict(
            tibia_kp  = (0.02, 0.5),
            tibia_kv  = (1e-6, 5e-3),
            tarsus_kp = (0.02, 0.5),
            tarsus_kv = (1e-5, 1e-2),
        ))
    if tune_pitch:
        bounds.update(dict(
            pitch_kp  = (1e-3, 2e-1),
            pitch_kv  = (1e-5, 1e-1),
        ))
    clipped = {}
    for k, v in seed_params.items():
        if k in bounds:
            lo, hi = bounds[k]
            clipped[k] = float(np.clip(v, lo, hi))
        else:
            clipped[k] = v
    return clipped


def _seed_from_config(imp_block, tune_distal, tune_pitch):
    """Turn an 'impedance:' YAML dict into a seed_params dict."""
    leg_kp = imp_block.get("leg", {}).get("kp", DEFAULT_LEG_KP)
    leg_kv = imp_block.get("leg", {}).get("kv", DEFAULT_LEG_KV)
    seed_params = dict(
        body_kp      = float(imp_block.get("body_kp", 0.05)),
        body_kv      = float(imp_block.get("body_kv", 0.01)),
        hip_yaw_kp   = float(leg_kp[0]),
        hip_yaw_kv   = float(leg_kv[0]),
        hip_pitch_kp = float(leg_kp[1]),
        hip_pitch_kv = float(leg_kv[1]),
    )
    if tune_distal:
        seed_params["tibia_kp"]  = float(leg_kp[2])
        seed_params["tibia_kv"]  = float(leg_kv[2])
        seed_params["tarsus_kp"] = float(leg_kp[3])
        seed_params["tarsus_kv"] = float(leg_kv[3])
    if tune_pitch:
        seed_params["pitch_kp"]  = float(imp_block.get("pitch_kp", 0.04))
        seed_params["pitch_kv"]  = float(imp_block.get("pitch_kv", 0.01))
    return _clip_to_search_bounds(seed_params, tune_distal, tune_pitch)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",   default=DEFAULT_XML)
    p.add_argument("--config",  default=DEFAULT_CFG)
    p.add_argument("--n-trials", type=int, default=250,
                   help="Total trials in the study (default 250 for 10-D search).")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Seconds per trial (includes 1 s settle + 1 s ramp).")
    p.add_argument("--seed",     type=int, default=42,
                   help="TPE sampler seed (different from --terrain-seed).")
    p.add_argument("--n-jobs",   type=int, default=1,
                   help="Parallel trials. Keep 1 unless your MuJoCo build is "
                        "thread-safe — the physics integration generally is not.")
    p.add_argument("--record-hz", type=float, default=200.0,
                   help="Sensor capture rate during each trial (Hz).")
    p.add_argument("--resume", default=None,
                   help="Path to an existing study.db to resume.")

    # ── Terrain (MULTI-TERRAIN) ──────────────────────────────────────────
    # Each trial evaluates the same gains on EACH wavelength in this list
    # and averages the cost. Default 10mm/18mm/50mm spans the full
    # roughness range the centipede might encounter.
    p.add_argument("--terrain-wavelengths-mm", type=float, nargs="+",
                   default=[10.0, 18.0, 50.0],
                   help="List of wavelengths (mm) to evaluate each trial "
                        "on. Default [10, 18, 50] gives a fine/medium/"
                        "coarse triplet. Single value behaves like the "
                        "old single-terrain optimizer.")
    p.add_argument("--terrain-amplitude",  type=float, default=0.01,
                   help="Peak amplitude (m) shared across all terrains.")
    p.add_argument("--terrain-seed",       type=int, default=42,
                   help="Seed for terrain generation (per-wavelength "
                        "seeds derived from this and the wavelength).")
    p.add_argument("--flat",               action="store_true",
                   help="Use flat ground only (skip multi-terrain).")

    # ── Search space ─────────────────────────────────────────────────────
    p.add_argument("--tune-distal",    dest="tune_distal",
                   action="store_true", default=True,
                   help="Also tune tibia/tarsus kp/kv. ON by default.")
    p.add_argument("--no-tune-distal", dest="tune_distal",
                   action="store_false",
                   help="Disable tibia/tarsus tuning.")
    p.add_argument("--tune-pitch",     dest="tune_pitch",
                   action="store_true", default=True,
                   help="Also tune body pitch_kp/pitch_kv — direct knob for "
                        "body compliance. ON by default.")
    p.add_argument("--no-tune-pitch",  dest="tune_pitch",
                   action="store_false",
                   help="Disable body pitch gain tuning.")

    # ── Seeding ──────────────────────────────────────────────────────────
    p.add_argument("--seed-from", default=None,
                   help="Path to a best_params.yaml from a prior run; used as "
                        "the first enqueued trial so TPE starts from a known-"
                        "reasonable point.")

    # ── Objective weights — match Isaac Lab bio cost defaults ────────────
    p.add_argument("--w-wobble",     type=float, default=6.0,
                   help="Penalty per unit of (1 - straightness)")
    p.add_argument("--w-body-track", type=float, default=0.10)
    p.add_argument("--w-leg-track",  type=float, default=0.20)
    p.add_argument("--w-force",      type=float, default=0.5)
    p.add_argument("--w-speed",      type=float, default=80.0,
                   help="Reward per m/s forward speed (CAPPED at "
                        "--speed-cap-mps).")
    p.add_argument("--w-displacement", type=float, default=40.0,
                   help="Reward per metre of net forward displacement "
                        "(generalist trials need both speed and progress).")
    p.add_argument("--w-compliance", type=float, default=0.05)
    # ── Bio-cost penalties (NEW — match Isaac Lab) ───────────────────────
    p.add_argument("--w-body-z",     type=float, default=400.0,
                   help="Penalty per metre of vertical body-z RMS — "
                        "directly attacks visible body bounce.")
    p.add_argument("--w-body-roll",  type=float, default=2.0,
                   help="Penalty per degree of body-roll RMS.")
    p.add_argument("--w-torque-jerk", type=float, default=1e-3,
                   help="Soft cost-shaping penalty on mean (dτ/dt)² of "
                        "applied joint torques. NOT a bio-anchored hard "
                        "constraint — see BIO_REFERENCES.md §8.2 / §9.1. "
                        "We don't have a direct centipede-leg muscle "
                        "bandwidth measurement, so this stays soft.")
    p.add_argument("--w-cot",        type=float, default=0.5,
                   help="(legacy weighted-sum only) Penalty on cost-of-"
                        "transport. Constraint-based cost uses CoT directly "
                        "as the primary objective.")
    p.add_argument("--speed-cap-mps", type=float, default=0.025,
                   help="(legacy weighted-sum only) Cap on speed reward.")

    # ── Objective limits / dead-zones ────────────────────────────────────
    # Force-limit raised 4.0 → 8.0 to match BIO_REFERENCES.md §1.5 (Cocci 2024:
    # real centipedes register peaks ~10× body weight per leg). 8× admits all
    # biologically observed forces while still excluding numerical contact-
    # spike pathologies.
    p.add_argument("--force-limit",  type=float, default=8.0,
                   help="Peak F/W ratio. BIO_REFERENCES.md §1.5: real "
                        "centipedes peak at ~10× body weight per leg "
                        "(Cocci 2024). 8× is conservative.")
    p.add_argument("--body-tol-deg", type=float, default=5.0,
                   help="Body-yaw tracking RMSE dead-zone (deg)")
    p.add_argument("--leg-tol-deg",  type=float, default=3.0,
                   help="Leg tracking RMSE dead-zone (deg)")
    p.add_argument("--compliance-cap-deg", type=float, default=10.0,
                   help="Saturation cap on the pitch-compliance reward (deg). "
                        "Above this, extra compliance gets no extra reward — "
                        "prevents the optimizer from seeking pathological "
                        "body-folding solutions.")

    # ── Constraint-based cost (NEW — matches Isaac Lab §8.2) ─────────────
    # The constraint-based cost is now the PRIMARY cost (returned to TPE).
    # The previous weighted-sum cost is preserved as cost_legacy in the
    # per-trial CSV for diagnostic comparison. To opt back to the legacy
    # cost as primary, pass --legacy-cost.
    p.add_argument("--legacy-cost", action="store_true",
                   help="Use the legacy weighted-sum cost as primary "
                        "(for reproducing pre-2026-05-04 results). "
                        "Default is the constraint-based cost.")
    p.add_argument("--leg-proj-length-m", type=float, default=0.035,
                   help="Projected leg length on ground (m), used to "
                        "compute commanded velocity from YAML wave params. "
                        "Default 0.035 m matches our 102.5 mm centipede.")
    p.add_argument("--nominal-clearance-m", type=float, default=0.0258,
                   help="Nominal body standing height above ground (m). "
                        "Used to normalize body_z RMS into a dimensionless "
                        "ratio. Default 0.0258 m = NOMINAL_CLEARANCE.")
    p.add_argument("--min-velocity-quality", type=float, default=0.3,
                   help="Hard constraint floor on v_actual / v_commanded. "
                        "BIO_REFERENCES.md §1.11: real centipedes slow on "
                        "rough terrain but never stop walking. 0.3 = "
                        "'still walking, not stuck'.")
    p.add_argument("--max-z-rms-ratio", type=float, default=0.10,
                   help="Hard constraint ceiling on body_z_rms / "
                        "nominal_clearance. BIO_REFERENCES.md §1.11: "
                        "centipedes drape over rough terrain via passive "
                        "limb compliance, not bounce. 0.10 = 10%% of "
                        "standing clearance.")
    p.add_argument("--max-roll-deg", type=float, default=5.0,
                   help="Hard constraint ceiling on body_roll RMS (deg). "
                        "BIO_REFERENCES.md §1.12: level walking is planar; "
                        "body twisting only seen on narrow obstacles.")

    # ── Replay ───────────────────────────────────────────────────────────
    p.add_argument("--rerun-duration",  type=float, default=10.0,
                   help="Duration for the final replay of the winner")
    p.add_argument("--rerun-record-hz", type=float, default=500.0,
                   help="Sensor capture rate for the winner replay")

    args = p.parse_args()

    weights = dict(
        w_wobble       = args.w_wobble,
        w_body_track   = args.w_body_track,
        w_leg_track    = args.w_leg_track,
        w_force        = args.w_force,
        w_speed        = args.w_speed,
        w_displacement = args.w_displacement,
        w_compliance   = args.w_compliance,
        # Bio-cost (NEW)
        w_body_z       = args.w_body_z,
        w_body_roll    = args.w_body_roll,
        w_torque_jerk  = args.w_torque_jerk,
        w_cot          = args.w_cot,
    )

    # ── Load base YAML config (moved earlier so commanded_velocity below
    # can read body_wave / leg_wave; the second `with open` further down
    # has been removed to avoid re-loading) ──
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    # ── Compute commanded velocity from YAML wave params ──
    # Matches Isaac Lab's analytical commanded velocity:
    #   v_cmd = 2 · L_proj · sin(A_yaw) · f
    # The optimizer uses this as a normaliser for velocity_quality (the
    # hard biological constraint from BIO_REFERENCES.md §1.11).
    bw = base_cfg.get("body_wave", {})
    lw = base_cfg.get("leg_wave", {})
    yaml_freq = float(bw.get("frequency", 1.0))
    leg_amps = lw.get("amplitudes", [0.6, 0.2, 0.0, 0.0])
    A_yaw = float(leg_amps[0]) if len(leg_amps) > 0 else 0.6
    commanded_velocity_mps = (2.0 * args.leg_proj_length_m
                              * math.sin(A_yaw) * yaml_freq)
    print(f"[bio-opt] commanded velocity (analytical): "
          f"{commanded_velocity_mps*1000:.1f} mm/s = "
          f"{commanded_velocity_mps*1000/102.5:.2f} BL/s "
          f"(at A_yaw={A_yaw:.2f}rad, f={yaml_freq:.2f}Hz, "
          f"L_proj={args.leg_proj_length_m*1000:.0f}mm)")

    limits = dict(
        track_tol_body_deg = args.body_tol_deg,
        track_tol_leg_deg  = args.leg_tol_deg,
        force_limit        = args.force_limit,
        compliance_cap_deg = args.compliance_cap_deg,
        speed_cap_mps      = args.speed_cap_mps,           # legacy
        # ── Constraint-based cost (NEW — matches Isaac Lab §8.2) ──
        commanded_velocity_mps    = commanded_velocity_mps,
        nominal_clearance_m       = args.nominal_clearance_m,
        min_velocity_quality      = args.min_velocity_quality,
        max_z_rms_ratio           = args.max_z_rms_ratio,
        max_roll_deg              = args.max_roll_deg,
        use_legacy_cost           = bool(args.legacy_cost),
    )

    # ── Run dir / study storage ─────────────────────────────────────────
    if args.resume:
        storage_path = args.resume
        run_dir = os.path.dirname(os.path.abspath(storage_path))
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.flat:
            tag = "flat"
        else:
            wl_str = "_".join(f"{int(round(w))}"
                              for w in args.terrain_wavelengths_mm)
            tag = f"multi_wl{wl_str}"
        run_dir = os.path.join(OUTPUT_ROOT, f"impedance_bio_{tag}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        storage_path = os.path.join(run_dir, "study.db")

    tmp_dir  = os.path.join(run_dir, "tmp_cfgs")
    csv_path = os.path.join(run_dir, "all_trials.csv")

    print("=" * 72)
    print("Bayesian impedance optimization (straightness-based, v3)")
    print("=" * 72)
    print(f"  model            : {args.model}")
    print(f"  base config      : {args.config}")
    print(f"  n_trials         : {args.n_trials}  (resume={bool(args.resume)})")
    print(f"  duration/trial   : {args.duration}s")
    n_dim = 6 + (4 if args.tune_distal else 0) + (2 if args.tune_pitch else 0)
    print(f"  tune_distal      : {args.tune_distal}")
    print(f"  tune_pitch       : {args.tune_pitch}")
    print(f"  search_dim       : {n_dim}-D")
    print(f"  terrain          : " +
          ("flat ground" if args.flat else
           f"λ={args.terrain_wavelengths_mm}  A={args.terrain_amplitude*1000:.1f}mm  "
           f"seed={args.terrain_seed}"))
    print(f"  run_dir          : {run_dir}")
    print(f"  weights          : {weights}")
    print(f"  limits           : {limits}")
    print()

    # base_cfg already loaded above (moved earlier so commanded_velocity
    # could read body_wave / leg_wave from it).

    # ── Terrain setup (happens once per study) ──────────────────────────
    # The patched XML lives next to the original model (relative meshdir
    # reference).  On both fresh runs and resumes we regenerate it from
    # the seed — generation is deterministic so this is a no-op in effect
    # but keeps the XML fresh after any repo refresh on the Lab PC.
    if args.flat:
        xml_paths = [("flat", args.model)]
        print("[terrain] --flat: using base XML (no heightfield)")
    else:
        # ── Multi-terrain XML pre-generation ──────────────────────────────
        # CRITICAL: patch_xml_terrain() inside setup_terrain() always
        # writes to the SAME path: `<base_xml>.sweep_tmp.xml`. Calling
        # setup_terrain() three times in a row therefore OVERWRITES the
        # previous terrain — every call produces the same path, and
        # only the LAST wavelength actually gets used at runtime.
        #
        # Fix: after each setup_terrain() call, COPY the patched XML
        # to a unique per-wavelength path (still next to the original
        # model so the relative meshdir="meshes" reference resolves).
        # We use base_xml.bio_wl{N}.xml as the unique name.
        import shutil as _shutil
        xml_paths = []
        for wl_mm in args.terrain_wavelengths_mm:
            terrain_seed_for_wl = (args.terrain_seed * 1000
                                   + int(round(wl_mm)))
            wl_run_dir = os.path.join(run_dir, f"wl{int(round(wl_mm))}mm")
            os.makedirs(wl_run_dir, exist_ok=True)
            shared_xml = setup_terrain(wl_run_dir,
                                       wl_mm,
                                       args.terrain_amplitude,
                                       terrain_seed_for_wl,
                                       args.model)
            # Copy to a unique path so the next iteration doesn't trample it.
            unique_xml = (args.model.replace('.xml', '')
                          + f'.bio_wl{int(round(wl_mm))}.xml')
            _shutil.copy(shared_xml, unique_xml)
            xml_paths.append((f"wl{int(round(wl_mm))}", unique_xml))
            print(f"[terrain] λ={wl_mm:.0f}mm  amp="
                  f"{args.terrain_amplitude*1000:.1f}mm  "
                  f"seed={terrain_seed_for_wl}  → {unique_xml}")
        # Sanity check: all xml_paths must be DIFFERENT files.
        if len(set(p for _, p in xml_paths)) != len(xml_paths):
            raise RuntimeError(
                "Multi-terrain bug: xml_paths contains duplicate paths "
                "after setup. Aborting before trials begin.")
    xml_to_use = xml_paths[0][1]

    # ── Study creation / resume ──────────────────────────────────────────
    storage = f"sqlite:///{storage_path}"
    sampler = TPESampler(seed=args.seed, multivariate=True, constant_liar=True)
    study_name = (
        f"impedance_bio_" + ('flat' if args.flat else 'multi') + f"_{n_dim}d"
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    # ── Seeding (only on a fresh study) ──────────────────────────────────
    if not args.resume and len(study.trials) == 0:
        if args.seed_from:
            with open(args.seed_from, "r") as f:
                seed_cfg = yaml.safe_load(f)
            seed_params = _seed_from_config(seed_cfg.get("impedance", {}),
                                            args.tune_distal, args.tune_pitch)
            print(f"[seed] from {args.seed_from}:")
        else:
            seed_params = _seed_from_config(base_cfg.get("impedance", {}),
                                            args.tune_distal, args.tune_pitch)
            print(f"[seed] from base config ({args.config}):")
        for k, v in seed_params.items():
            print(f"    {k:14s}= {v:.6g}")
        study.enqueue_trial(seed_params)

    # ── Objective + optimize ─────────────────────────────────────────────
    obj, trial_rows = make_objective(args, base_cfg, tmp_dir, csv_path,
                                     weights, limits, xml_paths)

    t_start = time.time()
    try:
        study.optimize(obj, n_trials=args.n_trials,
                       n_jobs=max(1, args.n_jobs),
                       show_progress_bar=False)
    except KeyboardInterrupt:
        print("\n[optimize] Interrupted — writing partial results.")

    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"Optimization done in {elapsed/60:.1f} min, "
          f"{len(study.trials)} trials total.")
    print(f"{'='*72}")

    _dump_rows(csv_path, trial_rows)

    # ── Pick the best non-buckled trial ──────────────────────────────────
    ok_trials = [t for t in study.trials if t.value is not None
                 and t.value < BUCKLED_COST * 0.99]

    if not ok_trials:
        print("No non-buckled trials! Relax the search ranges or weights.")
        return 1

    best = min(ok_trials, key=lambda t: t.value)
    print(f"\nBest trial: #{best.number}  cost={best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:14s}= {v:.6g}")

    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump({"trial_number":    best.number,
                   "cost":            best.value,
                   "params":          best.params,
                   "n_total_trials":  len(study.trials),
                   "n_non_buckled":   len(ok_trials),
                   "elapsed_min":     elapsed / 60.0,
                   "tune_distal":     args.tune_distal,
                   "terrain_flat":    args.flat,
                   "terrain_wavelength_mm": args.terrain_wavelengths_mm[0],
                   "terrain_amplitude_m":   args.terrain_amplitude,
                   "terrain_seed":          args.terrain_seed}, f, indent=2)

    sensor_npz, video, best_cfg = rerun_best(
        best.params, base_cfg, xml_to_use, run_dir,
        duration=args.rerun_duration,
        record_hz=args.rerun_record_hz,
    )

    # ── Auto-run the analysis pipeline on the winner ────────────────────
    analysis_dir = os.path.join(run_dir, "best_analysis")
    print(f"\n[analysis] Running analyze_sensor_data.py on winner...")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))
    try:
        import analyze_sensor_data as asd
        asd._analyse_one(sensor_npz, analysis_dir, settle=2.5, plot=True)
        print(f"[analysis] → {analysis_dir}")
    except Exception as e:
        print(f"[analysis] FAILED: {e} — you can run it manually:")
        print(f"  python analysis/analyze_sensor_data.py {sensor_npz} "
              f"--out {analysis_dir}/")

    # ── Clean up temp cfgs ───────────────────────────────────────────────
    try:
        for f in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, f))
            except Exception:
                pass
        os.rmdir(tmp_dir)
    except Exception:
        pass

    print(f"\nAll artifacts under: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
