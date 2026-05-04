#!/usr/bin/env python3
"""
optimize_impedance_bayesian.py  —  v3: straightness-based rough-terrain tuning
==============================================================================
Tunes impedance gains on a fixed heightfield terrain using Bayesian (TPE)
optimisation.  Cost is dominated by a geometric *straightness* metric computed
from the COM trajectory, not by explicit jerk / rpy-rate terms.

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
    # We measure "how differently are the body pitch joints deflected from
    # their commanded targets at each moment".  A rigid body sits flat
    # (all deflections ≈ 0, std ≈ 0).  A compliant body drapes over the
    # terrain (each segment on a different slope → varied deflections, std
    # grows).  We use std across joints (not RMS) so a uniform sag/tilt is
    # excluded — draping is the thing we want to reward.
    #
    # Pure reward, saturating at compliance_cap_deg, so the optimizer can't
    # exploit it by folding the body in half (per-joint buckle check still
    # guards the catastrophic case at |q| > 35°).
    pitch_q   = np.asarray(getattr(recorder, "pitch_q",   []))
    pitch_tgt = np.asarray(getattr(recorder, "pitch_tgt", []))
    if pitch_q.size > 0 and pitch_q.shape == pitch_tgt.shape and pitch_q.ndim == 2:
        pitch_q   = pitch_q[mask]
        pitch_tgt = pitch_tgt[mask]
        if pitch_q.shape[1] >= 2:
            deflect_per_t = np.std(pitch_q - pitch_tgt, axis=1)      # (T,)
            compliance_rad = float(np.mean(deflect_per_t))
        else:
            compliance_rad = 0.0
    else:
        compliance_rad = 0.0
    compliance_deg      = float(np.degrees(compliance_rad))
    compliance_reward_d = min(compliance_deg, limits["compliance_cap_deg"])

    # ── Cost ───────────────────────────────────────────────────────────────
    body_excess  = max(0.0, rmse_body_deg - limits["track_tol_body_deg"])
    leg_excess   = max(0.0, rmse_leg_deg  - limits["track_tol_leg_deg"])
    wobble       = 1.0 - straightness
    force_excess = max(0.0, peak_fw - limits["force_limit"])

    cost = (
        weights["w_body_track"] * body_excess
      + weights["w_leg_track"]  * leg_excess
      + weights["w_wobble"]     * wobble
      + weights["w_force"]      * force_excess
      - weights["w_speed"]      * speed_mps
      - weights["w_compliance"] * compliance_reward_d
    )

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
        peak_force_over_weight=peak_fw,
        force_excess=force_excess,
        pitch_compliance_deg=compliance_deg,
        pitch_compliance_reward_deg=compliance_reward_d,
        cost=cost,
    )
    return float(cost), parts


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(args, base_cfg, tmp_dir, csv_path, weights, limits, xml_path):
    """Closure-capturing objective. `xml_path` is the pre-patched terrain XML."""
    trial_rows = []
    tune_distal = args.tune_distal
    tune_pitch  = args.tune_pitch

    def obj(trial):
        params = dict(
            body_kp      = trial.suggest_float("body_kp",      1e-3, 2.0,  log=True),
            body_kv      = trial.suggest_float("body_kv",      1e-5, 0.5,  log=True),
            hip_yaw_kp   = trial.suggest_float("hip_yaw_kp",   1e-3, 2.0,  log=True),
            hip_yaw_kv   = trial.suggest_float("hip_yaw_kv",   1e-6, 1e-1, log=True),
            hip_pitch_kp = trial.suggest_float("hip_pitch_kp", 1e-4, 1.0,  log=True),
            hip_pitch_kv = trial.suggest_float("hip_pitch_kv", 1e-7, 1e-2, log=True),
        )
        if tune_distal:
            # Narrow bounds — these joints hold posture, they don't drive gait.
            params["tibia_kp"]  = trial.suggest_float("tibia_kp",  0.02, 0.5,  log=True)
            params["tibia_kv"]  = trial.suggest_float("tibia_kv",  1e-6, 5e-3, log=True)
            params["tarsus_kp"] = trial.suggest_float("tarsus_kp", 0.02, 0.5,  log=True)
            params["tarsus_kv"] = trial.suggest_float("tarsus_kv", 1e-5, 1e-2, log=True)
        if tune_pitch:
            # Body pitch gains — direct knob for compliance.  Range allows
            # very soft (1e-3) up to moderate stiffness (2e-1).  Head pitch
            # gains are intentionally NOT tuned (they need to hold the head
            # against gravity).
            params["pitch_kp"]  = trial.suggest_float("pitch_kp",  1e-3, 2e-1, log=True)
            params["pitch_kv"]  = trial.suggest_float("pitch_kv",  1e-5, 1e-1, log=True)

        cfg = patch_config(base_cfg, params)
        cfg_path = write_tmp_yaml(cfg, tmp_dir, f"trial_{trial.number:05d}")

        t0 = time.time()
        try:
            metrics, recorder = run_one(
                xml_path, cfg_path, args.duration,
                record=True, record_hz=args.record_hz)
        except Exception as e:
            dt = time.time() - t0
            print(f"  [trial {trial.number:4d}] CRASH: {e}  ({dt:.1f}s)")
            return BUCKLED_COST

        settle_s = 2.5  # 1s settle + 1s ramp + 0.5s slack
        cost, parts = compute_cost(metrics, recorder, weights, limits,
                                   settle_s=settle_s)

        dt = time.time() - t0
        if parts.get("buckled", False):
            print(f"  [trial {trial.number:4d}] BUCKLE  {parts.get('reason','?')}  "
                  f"({dt:.1f}s)  cost={cost:.1f}")
        else:
            print(f"  [trial {trial.number:4d}] cost={cost:7.3f}  "
                  f"speed={parts.get('speed_mps',0.0)*1000:5.1f}mm/s  "
                  f"strt={parts.get('straightness',0.0):.3f}  "
                  f"cmp={parts.get('pitch_compliance_deg',0.0):4.1f}°  "
                  f"F/W={parts.get('peak_force_over_weight',0.0):4.1f}  "
                  f"rmse_b={parts.get('rmse_body_yaw_deg',0.0):4.1f}° "
                  f"rmse_l={parts.get('rmse_leg_deg',0.0):4.1f}°  "
                  f"({dt:.1f}s)")

        try:
            os.remove(cfg_path)
        except Exception:
            pass

        row = {**params, **parts, "trial": trial.number, "sim_time_s": dt}
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

    # ── Terrain ──────────────────────────────────────────────────────────
    p.add_argument("--terrain-wavelength", type=float, default=18.0,
                   help="Terrain wavelength in mm (default 18).")
    p.add_argument("--terrain-amplitude",  type=float, default=0.01,
                   help="Terrain peak amplitude in meters (default 0.01 = 10 mm).")
    p.add_argument("--terrain-seed",       type=int, default=42,
                   help="Seed for terrain generation (fixed across the study).")
    p.add_argument("--flat",               action="store_true",
                   help="Use flat ground (skip terrain generation).")

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

    # ── Objective weights ────────────────────────────────────────────────
    p.add_argument("--w-wobble",     type=float, default=5.0,
                   help="Penalty per unit of (1 - straightness)")
    p.add_argument("--w-body-track", type=float, default=0.05,
                   help="Penalty per deg of body-yaw tracking RMSE above --body-tol-deg")
    p.add_argument("--w-leg-track",  type=float, default=0.10,
                   help="Penalty per deg of leg tracking RMSE above --leg-tol-deg")
    p.add_argument("--w-force",      type=float, default=0.5,
                   help="Penalty per unit of F/W above --force-limit (safety guard)")
    p.add_argument("--w-speed",      type=float, default=50.0,
                   help="Reward per m/s of forward speed")
    p.add_argument("--w-compliance", type=float, default=0.05,
                   help="Reward per degree of pitch-joint-deflection std, "
                        "saturating at --compliance-cap-deg. Larger values "
                        "push the optimizer toward body shapes that drape "
                        "over terrain (soft pitch compliance).")

    # ── Objective limits / dead-zones ────────────────────────────────────
    p.add_argument("--force-limit",  type=float, default=4.0,
                   help="Peak F/W ratio considered 'soft'")
    p.add_argument("--body-tol-deg", type=float, default=5.0,
                   help="Body-yaw tracking RMSE dead-zone (deg)")
    p.add_argument("--leg-tol-deg",  type=float, default=3.0,
                   help="Leg tracking RMSE dead-zone (deg)")
    p.add_argument("--compliance-cap-deg", type=float, default=10.0,
                   help="Saturation cap on the pitch-compliance reward (deg). "
                        "Above this, extra compliance gets no extra reward — "
                        "prevents the optimizer from seeking pathological "
                        "body-folding solutions.")

    # ── Replay ───────────────────────────────────────────────────────────
    p.add_argument("--rerun-duration",  type=float, default=10.0,
                   help="Duration for the final replay of the winner")
    p.add_argument("--rerun-record-hz", type=float, default=500.0,
                   help="Sensor capture rate for the winner replay")

    args = p.parse_args()

    weights = dict(
        w_wobble     = args.w_wobble,
        w_body_track = args.w_body_track,
        w_leg_track  = args.w_leg_track,
        w_force      = args.w_force,
        w_speed      = args.w_speed,
        w_compliance = args.w_compliance,
    )
    limits = dict(
        track_tol_body_deg = args.body_tol_deg,
        track_tol_leg_deg  = args.leg_tol_deg,
        force_limit        = args.force_limit,
        compliance_cap_deg = args.compliance_cap_deg,
    )

    # ── Run dir / study storage ─────────────────────────────────────────
    if args.resume:
        storage_path = args.resume
        run_dir = os.path.dirname(os.path.abspath(storage_path))
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "flat" if args.flat else f"rough_wl{int(round(args.terrain_wavelength))}"
        run_dir = os.path.join(OUTPUT_ROOT, f"impedance_bay_{tag}_{ts}")
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
           f"λ={args.terrain_wavelength}mm  A={args.terrain_amplitude*1000:.1f}mm  "
           f"seed={args.terrain_seed}"))
    print(f"  run_dir          : {run_dir}")
    print(f"  weights          : {weights}")
    print(f"  limits           : {limits}")
    print()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    # ── Terrain setup (happens once per study) ──────────────────────────
    # The patched XML lives next to the original model (relative meshdir
    # reference).  On both fresh runs and resumes we regenerate it from
    # the seed — generation is deterministic so this is a no-op in effect
    # but keeps the XML fresh after any repo refresh on the Lab PC.
    if args.flat:
        xml_to_use = args.model
        print("[terrain] --flat: using base XML (no heightfield)")
    else:
        xml_to_use = setup_terrain(run_dir,
                                   args.terrain_wavelength,
                                   args.terrain_amplitude,
                                   args.terrain_seed,
                                   args.model)

    # ── Study creation / resume ──────────────────────────────────────────
    storage = f"sqlite:///{storage_path}"
    sampler = TPESampler(seed=args.seed, multivariate=True, constant_liar=True)
    study_name = (
        f"impedance_v3_{'flat' if args.flat else 'rough'}_{n_dim}d"
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
                                     weights, limits, xml_to_use)

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
                   "terrain_wavelength_mm": args.terrain_wavelength,
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
