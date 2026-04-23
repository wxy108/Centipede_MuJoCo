#!/usr/bin/env python3
"""
optimize_impedance_bayesian.py — Bayesian (TPE) tuning of 6 impedance gains
============================================================================
Tunes three (kp, kv) pairs:
    body_kp,     body_kv       — body yaw joints 1..18 (head stays fixed)
    hip_yaw_kp,  hip_yaw_kv    — leg DOF 0  (hip yaw)
    hip_pitch_kp,hip_pitch_kv  — leg DOF 1  (hip pitch)

Everything else in the controller config is left untouched.

Goals (implemented in the cost function)
----------------------------------------
    1. STABILITY        drop any trial that buckles (pitch/roll exceed threshold)
    2. SOFT CONTACT     penalize peak foot contact force relative to body weight,
                        and penalize high rms of root-body RPY rate — a spike
                        there is "the body getting snapped by a leg torque".
    3. TRAJECTORY TRACK tracking RMSE between commanded q_target and actual q
                        for body yaw + leg hip, with a dead-zone tolerance
                        (the commanded is a sine wave; we want roughly-follow,
                        not exact).
    4. SPEED BONUS      negative term proportional to forward speed, so TPE
                        doesn't learn the trivial "stand still" solution.

    Cost = INF if buckled
         + w_force      * max(0, peak_force/body_weight   - force_limit)
         + w_rpy_rate   * rms(root_rpy_rate)
         + w_body_track * max(0, rmse_body_yaw_deg         - track_tol_body_deg)
         + w_leg_track  * max(0, rmse_leg_deg              - track_tol_leg_deg)
         + w_cot        * min(cot, cot_cap)
         - w_speed      * forward_speed_mps

Artifacts produced
------------------
  outputs/optimization/impedance_bay_{timestamp}/
    study.db                    SQLite Optuna storage (resumable)
    all_trials.csv              one row per trial, all params + cost breakdown
    best_params.yaml            a valid farms_controller.yaml with the winner
    best_trial.npz              rich SensorRecorder dump of a replay of the winner
    best_trial.mp4              video of that replay (if mediapy installed)
    best_analysis/              analyze_sensor_data.py output for the winner
    opt.log                     console log (tee-d)

Usage
-----
  # quick smoke run
  python scripts/optimization/farms/optimize_impedance_bayesian.py \
      --n-trials 20 --duration 6

  # full tuning run (recommended on the lab PC)
  python scripts/optimization/farms/optimize_impedance_bayesian.py \
      --n-trials 150 --duration 10 --n-jobs 1

  # resume an interrupted study
  python scripts/optimization/farms/optimize_impedance_bayesian.py \
      --resume outputs/optimization/impedance_bay_20260422_180000/study.db \
      --n-trials 50
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import yaml

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
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
from sensor_recorder import SensorRecorder, quat_to_euler  # noqa: E402

DEFAULT_XML = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
DEFAULT_CFG = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "optimization")

# ── Safety thresholds (buckling detection) ──────────────────────────────────
# Root RPY limits (deg) — past these we declare the trial failed.
MAX_ROOT_PITCH_DEG = 45.0
MAX_ROOT_ROLL_DEG  = 45.0
# Per-joint pitch / roll limits (deg) — past these is catastrophic folding.
MAX_JOINT_PITCH_DEG = 35.0
MAX_JOINT_ROLL_DEG  = 35.0

# A very large but finite cost so Optuna records it instead of dropping.
BUCKLED_COST = 1e5


# ══════════════════════════════════════════════════════════════════════════════
# Config patching
# ══════════════════════════════════════════════════════════════════════════════

def patch_config(base_cfg, params):
    """Deep-copy base_cfg and overwrite the 6 tuned parameters."""
    cfg = copy.deepcopy(base_cfg)
    imp = cfg.setdefault("impedance", {})
    imp["body_kp"] = float(params["body_kp"])
    imp["body_kv"] = float(params["body_kv"])
    leg = imp.setdefault("leg", {})
    kp = list(leg.get("kp", [0.127, 0.0147, 0.127, 0.127]))
    kv = list(leg.get("kv", [0.00056, 0.000114, 0.0000436, 0.000910]))
    kp[0] = float(params["hip_yaw_kp"])
    kp[1] = float(params["hip_pitch_kp"])
    kv[0] = float(params["hip_yaw_kv"])
    kv[1] = float(params["hip_pitch_kv"])
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
# Simulation + metrics
# ══════════════════════════════════════════════════════════════════════════════

def run_one(xml_path, cfg_path, duration,
            record=False, video_path=None,
            sensor_npz_path=None, record_hz=200.0):
    """Run one simulation. Returns a metrics dict (plus the recorder if
    record=True)."""
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

    # Optional video renderer
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

    # Root body id
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

    # Pitch/roll joint qpos addresses (for buckle detection)
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
    pitch_rad_hist = []
    roll_rad_hist  = []

    # Integrate
    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if recorder is not None:
            recorder.maybe_record(model, data, ctrl)

        if step_i == int(settle / dt):
            start_pos = data.xpos[root_body].copy()

        # Track root attitude for later rms(rate) metric
        if step_i % 10 == 0:
            R = data.xmat[root_body].reshape(3, 3)
            pitch_rad_hist.append(math.asin(-R[2, 0]))
            roll_rad_hist.append(math.atan2(R[2, 1], R[2, 2]))

        # Buckle check every 200 steps (~0.1 s)
        if step_i > 0 and step_i % 200 == 0:
            # Root attitude
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
            # Per-joint folding
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

        # Render frame (video only — much slower than recording)
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

    # Finish video + recorder outputs
    if video_path and frames:
        try:
            import mediapy
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            mediapy.write_video(video_path, frames, fps=30)
        except Exception as e:
            print(f"  [video] save failed: {e}")

    if recorder is not None and sensor_npz_path is not None:
        recorder.save(sensor_npz_path)

    # Return both recorder (for cost eval) and metrics
    return metrics, recorder


# ══════════════════════════════════════════════════════════════════════════════
# Cost function from SensorRecorder data
# ══════════════════════════════════════════════════════════════════════════════

def compute_cost(metrics, recorder, weights, limits, settle_s):
    """Score a completed trial."""
    if metrics["buckled"]:
        return BUCKLED_COST, {"buckled": True,
                              "reason": metrics["buckle_reason"]}

    if recorder is None or len(recorder.times) < 10:
        return BUCKLED_COST, {"buckled": True,
                              "reason": "no recorder data"}

    # Assemble arrays from the recorder lists (we don't save yet)
    t   = np.asarray(recorder.times, dtype=float)
    mask = t >= settle_s
    if mask.sum() < 5:
        return BUCKLED_COST, {"buckled": True,
                              "reason": "trial too short post-settle"}

    body_q   = np.asarray(recorder.body_yaw_q)[mask]
    body_tgt = np.asarray(recorder.body_yaw_tgt)[mask]
    leg_q    = np.asarray(recorder.leg_q)[mask]
    leg_tgt  = np.asarray(recorder.leg_tgt)[mask]

    # Tracking RMSE (deg) — with dead-zone tolerance
    def rmse_deg(q, g):
        return float(np.degrees(np.sqrt(np.mean((q - g) ** 2))))
    rmse_body = rmse_deg(body_q, body_tgt)
    rmse_leg  = rmse_deg(leg_q,  leg_tgt)

    # Foot contact — has to be accumulated here since we haven't call save()
    foot_force = np.asarray(recorder.foot_force)[mask]      # (T,19,2,3)
    foot_mag   = np.linalg.norm(foot_force, axis=-1)
    body_weight = recorder.total_mass * recorder.gravity_z
    peak_fw = float(foot_mag.max()) / max(body_weight, 1e-9)

    # Root RPY rate — "sudden twist" diagnostic
    root_quat = np.asarray(recorder.root_quat)[mask]
    rpy = quat_to_euler(root_quat)
    dt_rec = max(recorder.dt_record, 1e-6)
    rpy_rate = np.diff(rpy, axis=0) / dt_rec           # rad/s
    rpy_rate_rms = float(np.sqrt(np.mean(rpy_rate ** 2)))

    # Cost of transport
    mech_power = (np.asarray(recorder.body_yaw_act)[mask]
                  * np.asarray(recorder.body_yaw_qd)[mask]).sum(axis=1)
    leg_power  = (np.asarray(recorder.leg_act)[mask]
                  * np.asarray(recorder.leg_qd)[mask]).sum(axis=(1, 2, 3))
    power = mech_power + leg_power
    energy = float(np.sum(np.abs(power) * np.diff(t[mask], prepend=t[mask][0])))
    dist  = float(metrics["distance_m"])
    cot   = energy / max(body_weight * max(dist, 1e-6), 1e-12)
    cot   = min(cot, limits["cot_cap"])

    # ── Cost ──────────────────────────────────────────────────────────────
    force_excess = max(0.0, peak_fw - limits["peak_force_over_weight_ok"])
    body_excess  = max(0.0, rmse_body - limits["track_tol_body_deg"])
    leg_excess   = max(0.0, rmse_leg  - limits["track_tol_leg_deg"])

    cost = (
        weights["w_force"]      * force_excess
      + weights["w_rpy_rate"]   * rpy_rate_rms
      + weights["w_body_track"] * body_excess
      + weights["w_leg_track"]  * leg_excess
      + weights["w_cot"]        * cot
      - weights["w_speed"]      * metrics["forward_speed_mps"]
    )

    parts = dict(
        buckled=False,
        peak_force_over_weight=peak_fw,
        force_excess=force_excess,
        rmse_body_yaw_deg=rmse_body,
        rmse_leg_deg=rmse_leg,
        body_excess=body_excess,
        leg_excess=leg_excess,
        rpy_rate_rms_rad_s=rpy_rate_rms,
        cot=cot,
        speed_mps=metrics["forward_speed_mps"],
        distance_m=dist,
        cost=cost,
    )
    return float(cost), parts


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(args, base_cfg, tmp_dir, csv_path, weights, limits):
    xml_path = args.model
    trial_rows = []

    def obj(trial):
        params = dict(
            body_kp      = trial.suggest_float("body_kp",      1e-3, 2.0,  log=True),
            body_kv      = trial.suggest_float("body_kv",      1e-5, 0.5,  log=True),
            hip_yaw_kp   = trial.suggest_float("hip_yaw_kp",   1e-3, 2.0,  log=True),
            hip_yaw_kv   = trial.suggest_float("hip_yaw_kv",   1e-6, 1e-1, log=True),
            hip_pitch_kp = trial.suggest_float("hip_pitch_kp", 1e-4, 1.0,  log=True),
            hip_pitch_kv = trial.suggest_float("hip_pitch_kv", 1e-7, 1e-2, log=True),
        )
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

        settle_s = 0.0
        # Settle + ramp is 2s in default; use 2.5s to be safe
        settle_s = 2.5
        cost, parts = compute_cost(metrics, recorder, weights, limits,
                                   settle_s=settle_s)

        dt = time.time() - t0
        if parts.get("buckled", False):
            print(f"  [trial {trial.number:4d}] BUCKLE  {parts.get('reason','?')}  "
                  f"({dt:.1f}s)  cost={cost:.1f}")
        else:
            print(f"  [trial {trial.number:4d}] cost={cost:8.3f}  "
                  f"speed={parts.get('speed_mps',0.0)*1000:5.1f}mm/s  "
                  f"F/W={parts.get('peak_force_over_weight',0.0):4.1f}  "
                  f"rmse_b={parts.get('rmse_body_yaw_deg',0.0):4.1f}° "
                  f"rmse_l={parts.get('rmse_leg_deg',0.0):4.1f}° "
                  f"rpy_rate={parts.get('rpy_rate_rms_rad_s',0.0):4.2f}  "
                  f"({dt:.1f}s)")

        # Drop temp yaml to keep things tidy
        try:
            os.remove(cfg_path)
        except Exception:
            pass

        # Append CSV row
        row = {**params, **parts, "trial": trial.number, "sim_time_s": dt}
        trial_rows.append(row)
        # Write once per trial to survive mid-run interruptions
        _dump_rows(csv_path, trial_rows)

        # Use cost directly; Optuna minimises
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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",   default=DEFAULT_XML)
    p.add_argument("--config",  default=DEFAULT_CFG)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--duration", type=float, default=8.0,
                   help="Seconds per trial (incl. 1s settle + 1s ramp).")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--n-jobs",   type=int, default=1,
                   help="Parallel trials. Keep at 1 unless MuJoCo is thread-safe "
                        "in your build — the physics integration is not.")
    p.add_argument("--record-hz", type=float, default=200.0,
                   help="Sensor capture rate during each trial (default 200 Hz). "
                        "Higher = more accurate tracking/peak-force metrics, "
                        "more memory.")
    p.add_argument("--resume", default=None,
                   help="Path to an existing study.db to resume.")

    # Objective weights
    p.add_argument("--w-force",      type=float, default=1.0,
                   help="Penalty per unit of peak-force-over-body-weight above "
                        "--force-limit")
    p.add_argument("--w-rpy-rate",   type=float, default=0.5,
                   help="Penalty per rad/s of root RPY rate RMS")
    p.add_argument("--w-body-track", type=float, default=0.05,
                   help="Penalty per deg of body-yaw tracking RMSE above "
                        "--body-tol-deg")
    p.add_argument("--w-leg-track",  type=float, default=0.10,
                   help="Penalty per deg of leg tracking RMSE above "
                        "--leg-tol-deg")
    p.add_argument("--w-cot",        type=float, default=0.0005,
                   help="Penalty per unit of (capped) CoT")
    p.add_argument("--w-speed",      type=float, default=50.0,
                   help="Reward per m/s of forward speed")
    p.add_argument("--force-limit",  type=float, default=3.0,
                   help="Peak-force/body-weight ratio considered 'soft'")
    p.add_argument("--body-tol-deg", type=float, default=5.0,
                   help="Body-yaw tracking RMSE dead-zone (deg)")
    p.add_argument("--leg-tol-deg",  type=float, default=3.0,
                   help="Leg tracking RMSE dead-zone (deg)")
    p.add_argument("--cot-cap",      type=float, default=500.0,
                   help="Clamp CoT below this before weighting")

    p.add_argument("--rerun-duration", type=float, default=10.0,
                   help="Duration for the final replay of the winner")
    p.add_argument("--rerun-record-hz", type=float, default=500.0,
                   help="Sensor capture rate for the winner replay")

    args = p.parse_args()

    weights = dict(
        w_force      = args.w_force,
        w_rpy_rate   = args.w_rpy_rate,
        w_body_track = args.w_body_track,
        w_leg_track  = args.w_leg_track,
        w_cot        = args.w_cot,
        w_speed      = args.w_speed,
    )
    limits = dict(
        peak_force_over_weight_ok = args.force_limit,
        track_tol_body_deg        = args.body_tol_deg,
        track_tol_leg_deg         = args.leg_tol_deg,
        cot_cap                   = args.cot_cap,
    )

    # ── Run dir / study storage ──────────────────────────────────────────
    if args.resume:
        storage_path = args.resume
        run_dir = os.path.dirname(os.path.abspath(storage_path))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(OUTPUT_ROOT, f"impedance_bay_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        storage_path = os.path.join(run_dir, "study.db")

    tmp_dir  = os.path.join(run_dir, "tmp_cfgs")
    csv_path = os.path.join(run_dir, "all_trials.csv")
    log_path = os.path.join(run_dir, "opt.log")

    print(f"{'='*72}")
    print(f"Bayesian impedance optimization")
    print(f"{'='*72}")
    print(f"  model          : {args.model}")
    print(f"  base config    : {args.config}")
    print(f"  n_trials       : {args.n_trials}  (resume={bool(args.resume)})")
    print(f"  duration/trial : {args.duration}s")
    print(f"  run_dir        : {run_dir}")
    print(f"  weights        : {weights}")
    print(f"  limits         : {limits}")
    print()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Create / resume study
    storage = f"sqlite:///{storage_path}"
    sampler = TPESampler(seed=args.seed, multivariate=True, constant_liar=True)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study   = optuna.create_study(
        study_name="impedance_6p",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    # Seed with the current controller's defaults on the first run so the TPE
    # posterior is anchored near a known-stable point.
    if not args.resume and len(study.trials) == 0:
        imp = base_cfg["impedance"]
        leg_kp = imp.get("leg", {}).get("kp", [0.127, 0.0147, 0.127, 0.127])
        leg_kv = imp.get("leg", {}).get("kv", [0.00056, 0.000114, 0.0000436, 0.000910])
        seed_params = dict(
            body_kp      = float(imp.get("body_kp", 0.05)),
            body_kv      = float(imp.get("body_kv", 0.01)),
            hip_yaw_kp   = float(leg_kp[0]),
            hip_yaw_kv   = float(leg_kv[0]),
            hip_pitch_kp = float(leg_kp[1]),
            hip_pitch_kv = float(leg_kv[1]),
        )
        study.enqueue_trial(seed_params)
        print(f"[seed] Enqueued current-config trial: {seed_params}")

    obj, trial_rows = make_objective(args, base_cfg, tmp_dir, csv_path,
                                     weights, limits)

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

    # Pick the best non-buckled trial
    ok_trials = [t for t in study.trials if t.value is not None
                 and t.value < BUCKLED_COST * 0.99]
    if not ok_trials:
        print("No non-buckled trials! Tighten the search ranges or weights.")
        return 1

    best = min(ok_trials, key=lambda t: t.value)
    print(f"\nBest trial: #{best.number}  cost={best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:14s}= {v:.6g}")

    # Write best_params.json (summary) + re-run to dump video + rich npz
    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump({"trial_number": best.number,
                   "cost": best.value,
                   "params": best.params,
                   "n_total_trials": len(study.trials),
                   "n_non_buckled":  len(ok_trials),
                   "elapsed_min":    elapsed / 60.0}, f, indent=2)

    sensor_npz, video, best_cfg = rerun_best(
        best.params, base_cfg, args.model, run_dir,
        duration=args.rerun_duration,
        record_hz=args.rerun_record_hz,
    )

    # Run the analysis pipeline on the winner automatically
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

    # Clean up temp cfgs
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
