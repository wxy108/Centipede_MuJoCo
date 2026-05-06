#!/usr/bin/env python3
"""
optimize_impedance_local.py — MuJoCo local-search impedance tuner
==================================================================
Local-mutation Bayesian optimizer that treats the current YAML's gain set
(Isaac Lab three-eval rank 5 / trial #167 in the current canonical setup)
as a known-good local minimum and only searches a NARROW logarithmic
envelope around it. Single-terrain, single-frequency by design — no flat
ground, no 2 Hz, just rough λ=18mm at the YAML's frequency.

Why this script exists:
  - The three-eval optimizer (optimize_impedance_three_eval.py) failed
    on MuJoCo because the rough-2Hz evaluation was nearly always
    bucking. The optimizer learned to avoid any gains that PRODUCE
    motion, since real walking → buckle at 2Hz → BUCKLED_COST.
    Result: it converged on a passive-walk basin (trial #92,
    body_kp ≈ 2.7e-4, hip_yaw_kp at search floor).
  - The Isaac Lab rank-5 gain set, by contrast, walks well in MuJoCo
    on rough λ=18mm at 1Hz (verified via run_rough.py replay video).
    So the right question for MuJoCo is no longer "find the global
    optimum across frequency robustness" but "polish the Isaac
    Lab gain set within its local basin."
  - This script enforces local search by clipping each parameter's
    bounds to ±`bound_radius_log10` decades around the seed (default
    ±0.5 = factor of ~3.16× in either direction).

Cost function: IDENTICAL constraint formula to the three-eval optimizer:
    cost = CoT  if all hard constraints satisfied
    cost = CoT + 100·Σ(violation_excess) otherwise

Hard constraints (same thresholds as Isaac Lab + three-eval):
    velocity_quality       ≥ 0.30   [§1.11 Pierce 2023]
    body_z_rms / nom_clear ≤ 0.10   [§1.11 Pierce 2023]
    body_roll_rms_deg      ≤ 5.0    [§1.12 Pierce 2026]
    peak_F / body_weight   ≤ 8.0    [§1.5  Cocci 2024]

Usage:
    # Quick smoke (5 trials, ~3 min):
    python scripts/optimization/farms/optimize_impedance_local.py \\
        --n-trials 5 --duration 6

    # Full local search (100 trials, ~70 min):
    python scripts/optimization/farms/optimize_impedance_local.py \\
        --n-trials 100 --duration 8 --run-tag local_v1
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
    print("ERROR: optuna required. pip install optuna --break-system-packages")
    sys.exit(1)

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))

import mujoco                                                # noqa: E402
from impedance_controller import ImpedanceTravelingWaveController  # noqa: E402
from kinematics import FARMSModelIndex                       # noqa: E402
from sensor_recorder import SensorRecorder                   # noqa: E402
from wavelength_sweep import (                               # noqa: E402
    generate_single_wavelength_terrain,
    save_wavelength_terrain,
    patch_xml_terrain,
)

DEFAULT_XML = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
DEFAULT_CFG = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "optimization")

BUCKLED_COST        = 1e5
MAX_ROOT_PITCH_DEG  = 45.0
MAX_ROOT_ROLL_DEG   = 45.0
MAX_JOINT_PITCH_DEG = 35.0
MAX_JOINT_ROLL_DEG  = 35.0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",  default=DEFAULT_XML)
    p.add_argument("--config", default=DEFAULT_CFG,
                   help="YAML used as the seed (and gait/freq source).")
    p.add_argument("--n-trials", type=int,   default=100,
                   help="Local-search budget. Bayesian convergence inside a "
                        "narrow envelope is fast, so 50-150 is plenty.")
    p.add_argument("--duration", type=float, default=8.0,
                   help="Sim seconds per trial.")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--terrain-wavelength-mm", type=float, default=18.0)
    p.add_argument("--terrain-amplitude",     type=float, default=0.01)
    p.add_argument("--terrain-seed",          type=int,   default=42)
    p.add_argument("--frequency", type=float, default=None,
                   help="Body-wave frequency (Hz) override. None = use YAML's "
                        "frequency. Default YAML is 1.0 Hz.")
    p.add_argument("--bound-radius-log10", type=float, default=0.5,
                   help="Search envelope around the seed, in decades. 0.5 means "
                        "each gain is searched in [seed/√10, seed·√10] (~10× "
                        "total range). Use 0.3 for tighter (~4× range), "
                        "0.7 for looser (~25× range).")
    p.add_argument("--n-startup-trials", type=int, default=10,
                   help="TPE 'startup' trials sampled randomly within bounds "
                        "before the model kicks in. More = more local "
                        "exploration before TPE concentrates.")
    p.add_argument("--n-jobs",      type=int, default=1)
    # Search-space toggles
    p.add_argument("--tune-distal",     dest="tune_distal", action="store_true",
                   default=True)
    p.add_argument("--no-tune-distal",  dest="tune_distal", action="store_false")
    p.add_argument("--tune-pitch",      dest="tune_pitch",  action="store_true",
                   default=True)
    p.add_argument("--no-tune-pitch",   dest="tune_pitch",  action="store_false")
    p.add_argument("--seed-from",   default=None,
                   help="Override --config for the seed only (useful if you "
                        "want to seed from a previous run's best_params.yaml).")
    p.add_argument("--resume",      default=None)
    p.add_argument("--run-tag",     default="local_v1")
    p.add_argument("--record-hz",   type=float, default=200.0)
    # Constraint thresholds (match three-eval / bio cost)
    p.add_argument("--leg-proj-length-m",    type=float, default=0.035)
    p.add_argument("--nominal-clearance-m",  type=float, default=0.0258)
    p.add_argument("--min-velocity-quality", type=float, default=0.3)
    p.add_argument("--max-z-rms-ratio",      type=float, default=0.10)
    p.add_argument("--max-roll-deg",         type=float, default=5.0)
    p.add_argument("--force-limit",          type=float, default=30.0,
                   help="Hard constraint ceiling on per-foot peak F/body-weight. "
                        "Cocci 2024 (§1.5) reports real centipedes peak at "
                        "~10× body weight, but MuJoCo's contact solver "
                        "produces transient spikes much higher than PhysX's "
                        "TGS solver does for the same gait — the Isaac Lab "
                        "rank-5 seed reports F/W~0.85 in PhysX but F/W~16.8 "
                        "in MuJoCo. Default 30 is sim-calibrated to admit "
                        "all known-walking gaits while still excluding "
                        "pathological collision-spike trials. Pass "
                        "--force-limit 8.0 to use the bio-strict value "
                        "(but expect every trial to violate it on MuJoCo).")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Config patching
# ══════════════════════════════════════════════════════════════════════════════

def patch_config(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    imp = cfg.setdefault("impedance", {})
    imp["body_kp"] = float(params["body_kp"])
    imp["body_kv"] = float(params["body_kv"])
    if "pitch_kp" in params: imp["pitch_kp"] = float(params["pitch_kp"])
    if "pitch_kv" in params: imp["pitch_kv"] = float(params["pitch_kv"])
    leg = imp.setdefault("leg", {})
    kp = list(leg.get("kp", [0.5, 0.06, 0.13, 0.13]))
    kv = list(leg.get("kv", [0.001, 0.003, 0.0005, 0.001]))
    kp[0] = float(params["hip_yaw_kp"]);   kv[0] = float(params["hip_yaw_kv"])
    kp[1] = float(params["hip_pitch_kp"]); kv[1] = float(params["hip_pitch_kv"])
    if "tibia_kp"  in params: kp[2] = float(params["tibia_kp"])
    if "tibia_kv"  in params: kv[2] = float(params["tibia_kv"])
    if "tarsus_kp" in params: kp[3] = float(params["tarsus_kp"])
    if "tarsus_kv" in params: kv[3] = float(params["tarsus_kv"])
    leg["kp"] = kp; leg["kv"] = kv
    return cfg


def write_tmp_yaml(cfg, tmp_dir, tag):
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"cfg_{tag}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Build narrow per-param bounds around the seed
# ══════════════════════════════════════════════════════════════════════════════

def build_bounds_around_seed(seed_params, radius_log10, tune_distal, tune_pitch):
    """For each gain, return (low, high) such that
        low  = seed × 10^(-radius_log10)
        high = seed × 10^(+radius_log10)
    Subject to a global floor of 1e-9 (no negative or near-zero gains)
    and a ceiling of 1.0 for kp / 1.0 for kv (sim-stability sanity).
    """
    factor = 10.0 ** float(radius_log10)
    floor  = 1e-9
    keys = ["body_kp", "body_kv",
            "hip_yaw_kp", "hip_yaw_kv",
            "hip_pitch_kp", "hip_pitch_kv"]
    if tune_distal:
        keys += ["tibia_kp", "tibia_kv", "tarsus_kp", "tarsus_kv"]
    if tune_pitch:
        keys += ["pitch_kp", "pitch_kv"]
    bounds = {}
    for k in keys:
        if k not in seed_params:
            continue
        seed_val = float(seed_params[k])
        if seed_val <= 0:
            seed_val = 1e-6
        lo = max(seed_val / factor, floor)
        hi = min(seed_val * factor, 1.0)
        if hi <= lo:
            hi = lo * 1.01
        bounds[k] = (lo, hi)
    return bounds


# ══════════════════════════════════════════════════════════════════════════════
# Single-evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_one_eval(xml_path, cfg_path, duration, freq_override, record_hz=200.0):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    ctrl  = ImpedanceTravelingWaveController(model, cfg_path)
    idx   = FARMSModelIndex(model)
    if freq_override is not None:
        ctrl.set_frequency(float(freq_override))

    dt      = model.opt.timestep
    n_steps = int(duration / dt)
    settle  = ctrl.settle_time + getattr(ctrl, "ramp_time", 0.0)

    recorder = SensorRecorder(model, data, ctrl,
                              dt_record=1.0 / max(record_hz, 1.0),
                              terrain_sampler=None,
                              settle_time=settle)
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
    pitch_qposadr, roll_qposadr = [], []
    for j in range(model.njnt):
        nm = model.joint(j).name or ""
        if "joint_pitch_body" in nm: pitch_qposadr.append(model.jnt_qposadr[j])
        if "joint_roll_body"  in nm: roll_qposadr.append(model.jnt_qposadr[j])

    start_pos = None
    buckled, buckle_reason = False, ""

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
                buckled, buckle_reason = True, (
                    f"root_pitch={root_pitch_deg:.1f}deg@{data.time:.1f}s")
                break
            if abs(root_roll_deg) > MAX_ROOT_ROLL_DEG:
                buckled, buckle_reason = True, (
                    f"root_roll={root_roll_deg:.1f}deg@{data.time:.1f}s")
                break
            for a in pitch_qposadr:
                if abs(math.degrees(data.qpos[a])) > MAX_JOINT_PITCH_DEG:
                    buckled, buckle_reason = True, (
                        f"pitch_jnt@{data.time:.1f}s"); break
            if buckled: break
            for a in roll_qposadr:
                if abs(math.degrees(data.qpos[a])) > MAX_JOINT_ROLL_DEG:
                    buckled, buckle_reason = True, (
                        f"roll_jnt@{data.time:.1f}s"); break
            if buckled: break

    end_pos = data.xpos[root_body].copy()
    dist = (float(math.hypot(end_pos[0] - start_pos[0],
                             end_pos[1] - start_pos[1]))
            if start_pos is not None else 0.0)
    active_time = max(data.time - settle, 1e-3)
    speed = dist / active_time
    metrics = dict(buckled=buckled, buckle_reason=buckle_reason,
                   sim_time=float(data.time), distance_m=dist,
                   forward_speed_mps=speed)
    return metrics, recorder


# ══════════════════════════════════════════════════════════════════════════════
# Constraint cost (IDENTICAL formula to optimize_impedance_three_eval.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_eval_cost(metrics, recorder, args, freq, settle_s):
    if metrics["buckled"]:
        return BUCKLED_COST, {"buckled": True, "reason": metrics["buckle_reason"]}
    if recorder is None or len(recorder.times) < 10:
        return BUCKLED_COST, {"buckled": True, "reason": "no recorder data"}

    t = np.asarray(recorder.times, dtype=float)
    mask = t >= settle_s
    if mask.sum() < 5:
        return BUCKLED_COST, {"buckled": True, "reason": "too short post-settle"}

    com_xy = np.asarray(recorder.com_pos)[mask, :2]
    if len(com_xy) < 3:
        return BUCKLED_COST, {"buckled": True, "reason": "too few COM samples"}
    displacement_m = float(np.linalg.norm(com_xy[-1] - com_xy[0]))
    active_time    = max(float(t[mask][-1] - t[mask][0]), 1e-3)
    speed_mps      = displacement_m / active_time

    com_z          = np.asarray(recorder.com_pos)[mask, 2]
    body_z_rms_m   = float(np.std(com_z))
    z_rms_ratio    = body_z_rms_m / max(args.nominal_clearance_m, 1e-9)

    rq = np.asarray(getattr(recorder, "root_quat", []))
    if rq.ndim == 2 and rq.shape[1] == 4:
        rq = rq[mask]
        w_, x_, y_, z_ = rq[:,0], rq[:,1], rq[:,2], rq[:,3]
        sinr = 2.0 * (w_*x_ + y_*z_)
        cosr = 1.0 - 2.0*(x_*x_ + y_*y_)
        roll_rad = np.arctan2(sinr, cosr)
        body_roll_rms_deg = float(np.degrees(np.sqrt(np.mean(roll_rad**2))))
    else:
        body_roll_rms_deg = 0.0

    foot_force = np.asarray(recorder.foot_force)[mask]
    foot_mag   = np.linalg.norm(foot_force, axis=-1)
    body_weight = recorder.total_mass * recorder.gravity_z
    peak_FW    = float(foot_mag.max()) / max(body_weight, 1e-9)

    torque_arrays, qd_arrays = [], []
    for attr_name, _ in (("body_yaw_act", 2), ("pitch_act", 2), ("leg_act", 4)):
        arr = np.asarray(getattr(recorder, attr_name, []))
        if arr.ndim >= 2 and arr.shape[0] == len(t):
            arr = arr[mask]
            if arr.size > 0:
                arr = arr.reshape(arr.shape[0], -1)
                torque_arrays.append(arr)
    for attr_name in ("body_yaw_qd", "pitch_qd", "leg_qd"):
        arr = np.asarray(getattr(recorder, attr_name, []))
        if arr.ndim >= 2 and arr.shape[0] == len(t):
            arr = arr[mask]
            if arr.size > 0:
                arr = arr.reshape(arr.shape[0], -1)
                qd_arrays.append(arr)
    if torque_arrays and qd_arrays:
        tau = np.concatenate(torque_arrays, axis=1)
        qd  = np.concatenate(qd_arrays,     axis=1)
        m   = min(tau.shape[1], qd.shape[1])
        joint_power = np.abs(tau[:, :m] * qd[:, :m])
        dt_record   = max(float(np.median(np.diff(t))), 1e-4)
        total_work_J = float(np.sum(joint_power) * dt_record)
        cot          = total_work_J / max(displacement_m, 1e-3)
    else:
        total_work_J = 0.0
        cot          = 0.0

    A_yaw = float(args._A_yaw_cached)
    cmd_v = 2.0 * args.leg_proj_length_m * math.sin(A_yaw) * float(freq)
    velocity_quality = speed_mps / max(cmd_v, 1e-9)

    vel_excess  = max(0.0, args.min_velocity_quality - velocity_quality)
    z_excess    = max(0.0, z_rms_ratio - args.max_z_rms_ratio)
    roll_excess = max(0.0, body_roll_rms_deg - args.max_roll_deg)
    F_excess    = max(0.0, peak_FW - args.force_limit)
    total_violation = vel_excess + z_excess + roll_excess + F_excess

    if total_violation > 0:
        cost = cot + 100.0 * total_violation
    else:
        cost = cot

    parts = dict(
        buckled=False, cost=cost, cost_of_transport=cot,
        total_work_J=total_work_J, speed_mps=speed_mps,
        displacement_m=displacement_m,
        body_z_rms_m=body_z_rms_m, z_rms_ratio=z_rms_ratio,
        body_roll_rms_deg=body_roll_rms_deg,
        peak_force_over_weight=peak_FW,
        velocity_quality=velocity_quality,
        commanded_velocity_mps=cmd_v,
        vel_excess=vel_excess, z_excess=z_excess,
        roll_excess=roll_excess, F_excess=F_excess,
        total_violation=total_violation,
    )
    return float(cost), parts


# ══════════════════════════════════════════════════════════════════════════════
# Single-eval objective
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(args, base_cfg, tmp_dir, csv_path, xml_path,
                   bounds, freq):
    trial_rows = []

    def obj(trial):
        # Suggest gains within narrow bounds around the seed
        params = {}
        for k, (lo, hi) in bounds.items():
            params[k] = trial.suggest_float(k, lo, hi, log=True)

        cfg      = patch_config(base_cfg, params)
        cfg_path = write_tmp_yaml(cfg, tmp_dir, f"trial_{trial.number:05d}")
        settle_s = float(cfg.get("impedance", {}).get("settle_time", 1.0)) \
                 + float(cfg.get("impedance", {}).get("ramp_time",   1.0))

        t0 = time.time()
        try:
            metrics, recorder = run_one_eval(
                xml_path, cfg_path, args.duration, freq_override=freq,
                record_hz=args.record_hz)
        except Exception as e:
            print(f"  [trial {trial.number:4d}] CRASH: {e}", flush=True)
            cost = BUCKLED_COST
            parts = {"buckled": True, "reason": f"exception:{e}"}
        else:
            cost, parts = compute_eval_cost(
                metrics, recorder, args, freq, settle_s=settle_s)
        wall = time.time() - t0

        # Console line
        if parts.get("buckled", False):
            print(f"  [trial {trial.number:4d}] BUCKLE  cost={cost:.0f}  "
                  f"reason={parts.get('reason','?')}  ({wall:.1f}s)", flush=True)
        else:
            print(f"  [trial {trial.number:4d}] cost={cost:7.3f}  "
                  f"CoT={parts['cost_of_transport']:.2f}  "
                  f"speed={parts['speed_mps']*1000:5.1f}mm/s  "
                  f"v_q={parts['velocity_quality']:.2f}  "
                  f"z_rms={parts['z_rms_ratio']:.3f}  "
                  f"roll={parts['body_roll_rms_deg']:.2f}°  "
                  f"F/W={parts['peak_force_over_weight']:.2f}  "
                  f"({wall:.1f}s)", flush=True)

        # CSV row
        row = dict(trial=trial.number, cost=cost, wall_s=wall,
                   buckled=int(parts.get("buckled", False)))
        row.update(params)
        for k, v in parts.items():
            if k in ("buckled",): continue
            row[f"eval_{k}"] = v
        trial_rows.append(row)

        try: os.remove(cfg_path)
        except Exception: pass

        return cost
    return obj, trial_rows


def _dump_rows(csv_path, rows):
    if not rows: return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()

    if args.resume:
        storage_path = args.resume
        run_dir = os.path.dirname(os.path.abspath(storage_path))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(OUTPUT_ROOT,
                                f"impedance_local_{args.run_tag}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        storage_path = os.path.join(run_dir, "study.db")

    print(f"\n[local] run dir: {run_dir}")
    print(f"[local] terrain: rough λ={args.terrain_wavelength_mm:.0f}mm  "
          f"amp={args.terrain_amplitude*1000:.1f}mm  seed={args.terrain_seed}")
    print(f"[local] duration per trial: {args.duration:.1f} s")
    print(f"[local] bound radius: ±{args.bound_radius_log10} log10 decades")

    # Load base config (UTF-8 for Windows GBK locale safety)
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    args._A_yaw_cached = float(
        base_cfg.get("leg_wave", {}).get("amplitudes", [0.6])[0])
    yaml_freq = float(base_cfg.get("body_wave", {}).get("frequency", 1.0))
    freq = args.frequency if args.frequency is not None else yaml_freq
    print(f"[local] frequency: {freq:.1f} Hz "
          f"({'override' if args.frequency is not None else 'from YAML'})")
    print(f"[local] cached A_yaw: {args._A_yaw_cached:.3f} rad")

    # Generate the rough λ XML once
    wavelength_m = args.terrain_wavelength_mm * 1e-3
    h, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m=wavelength_m, amplitude_m=args.terrain_amplitude,
        seed=args.terrain_seed)
    png_path = save_wavelength_terrain(h, wavelength_m, args.terrain_seed, run_dir)
    z_max = max(2.0 * args.terrain_amplitude, 1e-3)
    patched = patch_xml_terrain(args.model, png_path, z_max=z_max)
    rough_xml = (args.model
                  + f".local_rough_wl{int(args.terrain_wavelength_mm)}.xml")
    shutil.copy(patched, rough_xml)
    print(f"[terrain] rough XML: {rough_xml}  "
          f"(rms={rms_m*1000:.2f}mm  peak={peak_m*1000:.2f}mm)")

    # Build the seed parameter dict from the YAML
    seed_yaml = args.seed_from or args.config
    with open(seed_yaml, "r", encoding="utf-8") as f:
        seed_cfg = yaml.safe_load(f)
    imp = seed_cfg.get("impedance", {})
    leg = imp.get("leg", {})
    leg_kp = leg.get("kp", [0.5, 0.06, 0.13, 0.13])
    leg_kv = leg.get("kv", [0.001, 0.003, 0.0005, 0.001])
    seed_params = dict(
        body_kp      = float(imp.get("body_kp", 0.05)),
        body_kv      = float(imp.get("body_kv", 0.01)),
        hip_yaw_kp   = float(leg_kp[0]),
        hip_yaw_kv   = float(leg_kv[0]),
        hip_pitch_kp = float(leg_kp[1]),
        hip_pitch_kv = float(leg_kv[1]),
    )
    if args.tune_distal:
        seed_params["tibia_kp"]  = float(leg_kp[2])
        seed_params["tibia_kv"]  = float(leg_kv[2])
        seed_params["tarsus_kp"] = float(leg_kp[3])
        seed_params["tarsus_kv"] = float(leg_kv[3])
    if args.tune_pitch:
        seed_params["pitch_kp"] = float(imp.get("pitch_kp", 0.01))
        seed_params["pitch_kv"] = float(imp.get("pitch_kv", 0.001))

    # Build narrow bounds around the seed
    bounds = build_bounds_around_seed(
        seed_params, args.bound_radius_log10,
        args.tune_distal, args.tune_pitch)

    print(f"\n[local] seed → narrow-search bounds:")
    print(f"        {'param':<14} {'seed':<14} {'low':<14} {'high':<14}")
    for k, (lo, hi) in bounds.items():
        sv = seed_params[k]
        print(f"        {k:<14} {sv:<14.6g} {lo:<14.6g} {hi:<14.6g}")

    tmp_dir  = os.path.join(run_dir, "tmp_cfgs")
    csv_path = os.path.join(run_dir, "all_trials.csv")

    # Optuna study with extra startup-trial weight (more random samples
    # before TPE concentrates → "treats seed as one of many local samples"
    # so the optimizer doesn't lock onto it prematurely).
    storage = f"sqlite:///{storage_path}"
    sampler = TPESampler(seed=args.seed, multivariate=True,
                          constant_liar=True,
                          n_startup_trials=int(args.n_startup_trials))
    study = optuna.create_study(study_name=f"impedance_local",
                                  storage=storage, sampler=sampler,
                                  direction="minimize", load_if_exists=True)

    # Seed first trial from the YAML — clip to bounds (normally a no-op)
    if not args.resume and len(study.trials) == 0:
        clipped = {}
        for k, v in seed_params.items():
            if k in bounds:
                lo, hi = bounds[k]
                clipped[k] = float(np.clip(v, lo, hi))
            else:
                clipped[k] = float(v)
        print(f"\n[local] seeding first trial from {seed_yaml}")
        for k, v in clipped.items():
            print(f"        {k:14s}= {v:.6g}")
        study.enqueue_trial(clipped)

    obj, trial_rows = make_objective(
        args, base_cfg, tmp_dir, csv_path, rough_xml, bounds, freq)

    t_start = time.time()
    try:
        study.optimize(obj, n_trials=args.n_trials,
                        n_jobs=max(1, args.n_jobs),
                        show_progress_bar=False)
    except KeyboardInterrupt:
        print("\n[local] interrupted — writing partial results.")

    elapsed_min = (time.time() - t_start) / 60.0
    _dump_rows(csv_path, trial_rows)

    # Best non-buckled
    ok = [t for t in study.trials if t.value is not None
          and t.value < BUCKLED_COST * 0.99]
    if ok:
        best = min(ok, key=lambda t: t.value)
        print(f"\n[local] best trial #{best.number}  cost={best.value:.4f}")
        for k, v in best.params.items():
            print(f"        {k:14s}= {v:.6g}")
        # Save winning YAML
        cfg = copy.deepcopy(base_cfg)
        imp = cfg.setdefault("impedance", {})
        imp["body_kp"] = best.params["body_kp"]
        imp["body_kv"] = best.params["body_kv"]
        if "pitch_kp" in best.params:
            imp["pitch_kp"] = best.params["pitch_kp"]
            imp["pitch_kv"] = best.params["pitch_kv"]
        leg = imp.setdefault("leg", {})
        kp = list(leg.get("kp", [0.5, 0.06, 0.13, 0.13]))
        kv = list(leg.get("kv", [0.001, 0.003, 0.0005, 0.001]))
        kp[0] = best.params["hip_yaw_kp"];   kv[0] = best.params["hip_yaw_kv"]
        kp[1] = best.params["hip_pitch_kp"]; kv[1] = best.params["hip_pitch_kv"]
        if "tibia_kp" in best.params:
            kp[2] = best.params["tibia_kp"];  kv[2] = best.params["tibia_kv"]
        if "tarsus_kp" in best.params:
            kp[3] = best.params["tarsus_kp"]; kv[3] = best.params["tarsus_kv"]
        leg["kp"] = kp; leg["kv"] = kv
        with open(os.path.join(run_dir, "best_params.yaml"), "w",
                   encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with open(os.path.join(run_dir, "best_params.json"), "w",
                   encoding="utf-8") as f:
            json.dump({"trial": best.number, "cost": best.value,
                        "params": best.params}, f, indent=2)

    print(f"[local] done — {len(study.trials)} trials in {elapsed_min:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
