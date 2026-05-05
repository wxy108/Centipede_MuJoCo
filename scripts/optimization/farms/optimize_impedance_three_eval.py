#!/usr/bin/env python3
"""
optimize_impedance_three_eval.py — MuJoCo three-evaluation impedance tuner
==========================================================================
Sister of optimize_impedance_biological.py and the Isaac Lab
optimize_impedance_three_eval.py. Each trial tests the SAME gain set on
three sequential evaluations (different XML / different frequency), then
reports the average cost. The goal is to find gain sets that survive
across multiple frequencies — so the same impedance controller doesn't
fail when DRL later sweeps gait timing.

Three evaluations per trial:
  A. Flat ground at 2 Hz       — bio-comparable CoT (target 0.20-0.38 J/m
                                  per Full 1989 inter-arthropod scaling
                                  applied to our 2.5 g body).
  B. Rough λ=18mm at 1 Hz      — slow gait robustness on rough terrain.
  C. Rough λ=18mm at 2 Hz      — fast gait robustness on rough terrain.

Trial cost = mean(eval_A_cost, eval_B_cost, eval_C_cost).
Any eval that buckles / flips / stuck → trial cost = BUCKLED_COST (1e5).

Constraint cost (IDENTICAL formula to optimize_impedance_biological.py
and the Isaac Lab three-eval optimizer):
  cost = CoT  if all hard constraints satisfied
  cost = CoT + 100·Σ(violation_excess)  otherwise

Hard constraints per eval:
  velocity_quality       ≥ 0.30   [§1.11 Pierce 2023]
  body_z_rms / nom_clear ≤ 0.10   [§1.11 Pierce 2023]
  body_roll_rms_deg      ≤ 5.0    [§1.12 Pierce 2026]
  peak_F / body_weight   ≤ 8.0    [§1.5  Cocci 2024]

Usage (smoke test, ~3-5 minutes):
    python scripts/optimization/farms/optimize_impedance_three_eval.py \\
        --n-trials 5 --duration 4

Usage (full overnight, ~6-10 h depending on hardware):
    python scripts/optimization/farms/optimize_impedance_three_eval.py \\
        --n-trials 200 --duration 6 --run-tag three_eval_v1
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
from kinematics import (FARMSModelIndex,                     # noqa: E402
                         N_BODY_JOINTS, N_LEGS, N_LEG_DOF)
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
    p.add_argument("--model",  default=DEFAULT_XML,
                   help="MJCF XML to use as the BASE for terrain patching.")
    p.add_argument("--config", default=DEFAULT_CFG,
                   help="YAML config to seed the first trial / supply gait params.")
    p.add_argument("--n-trials",    type=int,   default=200)
    p.add_argument("--duration",    type=float, default=6.0,
                   help="Sim seconds PER EVALUATION. Each trial does 3 evals → "
                        "trial total ≈ 3 × this.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--terrain-wavelength-mm", type=float, default=18.0)
    p.add_argument("--terrain-amplitude",     type=float, default=0.01)
    p.add_argument("--terrain-seed",          type=int,   default=42)
    p.add_argument("--freq-flat",        type=float, default=2.0,
                   help="Body-wave frequency for flat-ground eval (Hz).")
    p.add_argument("--freq-rough-low",   type=float, default=1.0)
    p.add_argument("--freq-rough-high",  type=float, default=2.0)
    p.add_argument("--record-hz",   type=float, default=200.0)
    p.add_argument("--n-jobs",      type=int,   default=1)
    # Search-space toggles
    p.add_argument("--tune-distal",     dest="tune_distal", action="store_true",
                   default=True)
    p.add_argument("--no-tune-distal",  dest="tune_distal", action="store_false")
    p.add_argument("--tune-pitch",      dest="tune_pitch",  action="store_true",
                   default=True)
    p.add_argument("--no-tune-pitch",   dest="tune_pitch",  action="store_false")
    p.add_argument("--seed-from",   default=None)
    p.add_argument("--resume",      default=None)
    p.add_argument("--run-tag",     default="three_eval")
    # Constraint thresholds (match Isaac Lab + bio cost defaults)
    p.add_argument("--leg-proj-length-m",    type=float, default=0.035)
    p.add_argument("--nominal-clearance-m",  type=float, default=0.0258)
    p.add_argument("--min-velocity-quality", type=float, default=0.3)
    p.add_argument("--max-z-rms-ratio",      type=float, default=0.10)
    p.add_argument("--max-roll-deg",         type=float, default=5.0)
    p.add_argument("--force-limit",          type=float, default=8.0)
    p.add_argument("--w-torque-jerk",        type=float, default=1e-3,
                   help="Soft cost-shaping (NOT folded into cost — diagnostic "
                        "only). See BIO_REFERENCES.md §8.2.")
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
# Terrain XMLs (built once at study startup)
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_xmls(args, run_dir):
    """Generate the two XMLs we need for the 3-eval workflow:
        - Flat: just the base XML (no hfield modification needed; the MJCF
                already has a default ground plane).
        - Rough λ: patch the base XML's hfield to the requested wavelength.

    Returns (flat_xml_path, rough_xml_path).
    """
    # Flat: use the base XML directly. MuJoCo's centipede.xml has a default
    # ground plane that becomes flat-ground walking when no <hfield> patching
    # happens.
    flat_xml_path = args.model

    # Rough λ=18mm: setup_terrain() generates the heightmap, saves PNG, and
    # patches the base XML to reference it. The patched XML is written to
    # `<base_xml>.sweep_tmp.xml`. We immediately copy to a stable per-eval
    # path so subsequent setup_terrain calls don't overwrite us.
    wavelength_m = args.terrain_wavelength_mm * 1e-3
    h, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m = wavelength_m,
        amplitude_m  = args.terrain_amplitude,
        seed         = args.terrain_seed,
    )
    png_path = save_wavelength_terrain(h, wavelength_m, args.terrain_seed, run_dir)
    z_max = max(2.0 * args.terrain_amplitude, 1e-3)
    patched = patch_xml_terrain(args.model, png_path, z_max=z_max)
    rough_xml_path = (args.model
                      + f".three_eval_rough_wl{int(args.terrain_wavelength_mm)}.xml")
    shutil.copy(patched, rough_xml_path)

    print(f"[terrain] flat  XML: {flat_xml_path}")
    print(f"[terrain] rough XML: {rough_xml_path}  "
          f"(λ={args.terrain_wavelength_mm:.1f}mm  amp={args.terrain_amplitude*1000:.1f}mm  "
          f"rms={rms_m*1000:.2f}mm  peak={peak_m*1000:.2f}mm)")
    return flat_xml_path, rough_xml_path


# ══════════════════════════════════════════════════════════════════════════════
# Single-evaluation runner with frequency override
# ══════════════════════════════════════════════════════════════════════════════

def run_one_eval(xml_path, cfg_path, duration, freq_override,
                 record_hz=200.0):
    """Run one evaluation. Loads model, builds controller, overrides
    frequency, runs for `duration` sec, returns (metrics, recorder).

    `freq_override` is the frequency in Hz to set via controller.set_frequency()
    after construction. Set to None to use the YAML's frequency as-is.
    """
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
# Constraint-based cost (IDENTICAL formula to optimize_impedance_biological.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_eval_cost(metrics, recorder, args, freq, settle_s):
    if metrics["buckled"]:
        return BUCKLED_COST, {"buckled": True, "reason": metrics["buckle_reason"]}
    if recorder is None or len(recorder.times) < 10:
        return BUCKLED_COST, {"buckled": True, "reason": "no recorder data"}

    t    = np.asarray(recorder.times, dtype=float)
    mask = t >= settle_s
    if mask.sum() < 5:
        return BUCKLED_COST, {"buckled": True, "reason": "too short post-settle"}

    com_xy = np.asarray(recorder.com_pos)[mask, :2]
    if len(com_xy) < 3:
        return BUCKLED_COST, {"buckled": True, "reason": "too few COM samples"}
    displacement_m = float(np.linalg.norm(com_xy[-1] - com_xy[0]))
    active_time    = max(float(t[mask][-1] - t[mask][0]), 1e-3)
    speed_mps      = displacement_m / active_time

    # body z RMS / clearance ratio
    com_z          = np.asarray(recorder.com_pos)[mask, 2]
    body_z_rms_m   = float(np.std(com_z))
    z_rms_ratio    = body_z_rms_m / max(args.nominal_clearance_m, 1e-9)

    # body roll RMS (deg) — derive from root_quat if recorder has it
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

    # peak per-foot force / body weight
    foot_force = np.asarray(recorder.foot_force)[mask]      # (T,19,2,3)
    foot_mag   = np.linalg.norm(foot_force, axis=-1)
    body_weight = recorder.total_mass * recorder.gravity_z
    peak_FW    = float(foot_mag.max()) / max(body_weight, 1e-9)

    # CoT = work / displacement
    torque_arrays = []
    qd_arrays     = []
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

    # Torque jerk (diagnostic only — NOT folded into cost; see BIO_REFERENCES §8.2)
    torque_jerk = 0.0
    if torque_arrays and any(a.shape[0] >= 2 for a in torque_arrays):
        tau_all = np.concatenate(torque_arrays, axis=1)
        dt_record = max(float(np.median(np.diff(t))), 1e-4)
        torque_diff = np.diff(tau_all, axis=0) / dt_record
        torque_jerk = float(np.mean(torque_diff ** 2))

    # Commanded velocity at this eval's frequency
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
        torque_jerk=torque_jerk,
    )
    return float(cost), parts


# ══════════════════════════════════════════════════════════════════════════════
# Three-evaluation objective
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(args, base_cfg, tmp_dir, csv_path, flat_xml, rough_xml):
    trial_rows = []

    def obj(trial):
        # 1. Suggest gains
        params = dict(
            body_kp      = trial.suggest_float("body_kp",      1e-4, 0.5, log=True),
            body_kv      = trial.suggest_float("body_kv",      1e-6, 1.0, log=True),
            hip_yaw_kp   = trial.suggest_float("hip_yaw_kp",   1e-4, 0.5, log=True),
            hip_yaw_kv   = trial.suggest_float("hip_yaw_kv",   1e-7, 5e-1, log=True),
            hip_pitch_kp = trial.suggest_float("hip_pitch_kp", 1e-5, 0.5, log=True),
            hip_pitch_kv = trial.suggest_float("hip_pitch_kv", 1e-8, 5e-2, log=True),
        )
        if args.tune_distal:
            params["tibia_kp"]  = trial.suggest_float("tibia_kp",  1e-3, 0.5,  log=True)
            params["tibia_kv"]  = trial.suggest_float("tibia_kv",  1e-7, 5e-2, log=True)
            params["tarsus_kp"] = trial.suggest_float("tarsus_kp", 1e-3, 0.5,  log=True)
            params["tarsus_kv"] = trial.suggest_float("tarsus_kv", 1e-6, 5e-2, log=True)
        if args.tune_pitch:
            params["pitch_kp"] = trial.suggest_float("pitch_kp", 1e-4, 0.5, log=True)
            params["pitch_kv"] = trial.suggest_float("pitch_kv", 1e-6, 5e-1, log=True)

        # 2. Patch config + write tmp YAML once (gains don't change between evals)
        cfg     = patch_config(base_cfg, params)
        cfg_path = write_tmp_yaml(cfg, tmp_dir, f"trial_{trial.number:05d}")
        settle_s = float(cfg.get("impedance", {}).get("settle_time", 1.0)) \
                 + float(cfg.get("impedance", {}).get("ramp_time",   1.0))

        # 3. Three evaluations
        eval_specs = [
            ("flat_2hz",   flat_xml,  args.freq_flat),
            ("rough_1hz",  rough_xml, args.freq_rough_low),
            ("rough_2hz",  rough_xml, args.freq_rough_high),
        ]
        eval_costs, eval_parts = [], []
        t0 = time.time()
        any_buckled = False

        for label, xml_path, freq in eval_specs:
            try:
                metrics, recorder = run_one_eval(
                    xml_path, cfg_path, args.duration, freq_override=freq,
                    record_hz=args.record_hz)
            except Exception as e:
                print(f"  [trial {trial.number:4d}] CRASH on {label}: {e}",
                      flush=True)
                cost = BUCKLED_COST
                parts = {"buckled": True, "reason": f"exception:{e}",
                         "cost": cost}
            else:
                cost, parts = compute_eval_cost(
                    metrics, recorder, args, freq, settle_s=settle_s)
            parts["eval_label"] = label
            parts["frequency"]  = freq
            eval_costs.append(cost)
            eval_parts.append(parts)
            if parts.get("buckled", False) or cost >= BUCKLED_COST * 0.99:
                any_buckled = True
                # Fill remaining evals with BUCKLED to keep CSV columns aligned
                for remaining in eval_specs[len(eval_costs):]:
                    eval_costs.append(BUCKLED_COST)
                    eval_parts.append({
                        "eval_label": remaining[0], "frequency": remaining[2],
                        "buckled": True, "cost": BUCKLED_COST,
                        "reason": "abort_propagated"})
                break

        trial_cost = float(np.mean(eval_costs))
        wall = time.time() - t0

        # Console line
        if any_buckled:
            n_buck = sum(1 for p in eval_parts if p.get("buckled", False))
            per_eval = [f"{c:.0f}" if c >= BUCKLED_COST*0.99 else f"{c:.2f}"
                        for c in eval_costs]
            print(f"  [trial {trial.number:4d}] BUCKLE in {n_buck}/3 evals  "
                  f"per_eval={per_eval}  combined={trial_cost:.1f}  ({wall:.1f}s)",
                  flush=True)
        else:
            cot_flat = eval_parts[0].get("cost_of_transport", float("nan"))
            cot_r1   = eval_parts[1].get("cost_of_transport", float("nan"))
            cot_r2   = eval_parts[2].get("cost_of_transport", float("nan"))
            speeds   = [eval_parts[i].get("speed_mps", 0.0) * 1000 for i in range(3)]
            print(f"  [trial {trial.number:4d}] cost={trial_cost:7.3f}  "
                  f"CoT(flat2/r1/r2)=({cot_flat:.2f},{cot_r1:.2f},{cot_r2:.2f})  "
                  f"speed_mm/s=({speeds[0]:.1f},{speeds[1]:.1f},{speeds[2]:.1f})  "
                  f"({wall:.1f}s)", flush=True)

        # CSV row
        row = dict(trial=trial.number, cost=trial_cost, wall_s=wall,
                   buckled=int(any_buckled))
        for k, v in params.items():
            row[k] = v
        for parts in eval_parts:
            lbl = parts.get("eval_label", "?")
            for k, v in parts.items():
                if k == "eval_label": continue
                row[f"{lbl}__{k}"] = v
        trial_rows.append(row)

        try:
            os.remove(cfg_path)
        except Exception:
            pass

        return trial_cost

    return obj, trial_rows


def _dump_rows(csv_path, rows):
    if not rows: return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
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
                                f"impedance_three_eval_{args.run_tag}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        storage_path = os.path.join(run_dir, "study.db")

    print(f"\n[3eval] run dir: {run_dir}")
    print(f"[3eval] eval A: flat ground, freq={args.freq_flat:.1f} Hz "
          f"(CoT bio reference)")
    print(f"[3eval] eval B: rough λ={args.terrain_wavelength_mm:.0f}mm, "
          f"freq={args.freq_rough_low:.1f} Hz")
    print(f"[3eval] eval C: rough λ={args.terrain_wavelength_mm:.0f}mm, "
          f"freq={args.freq_rough_high:.1f} Hz")
    print(f"[3eval] duration per eval: {args.duration:.1f} s   "
          f"→ trial total ≈ {3*args.duration:.0f} s sim")

    # Load base config (UTF-8 explicit so Windows GBK locales don't crash on → ≈ ·)
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    args._A_yaw_cached = float(
        base_cfg.get("leg_wave", {}).get("amplitudes", [0.6])[0])
    print(f"[3eval] cached A_yaw = {args._A_yaw_cached:.3f} rad")

    # Generate the two evaluation XMLs once
    flat_xml, rough_xml = build_eval_xmls(args, run_dir)

    tmp_dir  = os.path.join(run_dir, "tmp_cfgs")
    csv_path = os.path.join(run_dir, "all_trials.csv")

    # Optuna study
    storage = f"sqlite:///{storage_path}"
    sampler = TPESampler(seed=args.seed, multivariate=True, constant_liar=True)
    study = optuna.create_study(study_name=f"impedance_three_eval",
                                  storage=storage, sampler=sampler,
                                  direction="minimize", load_if_exists=True)

    # Seed first trial from current YAML (or --seed-from)
    if not args.resume and len(study.trials) == 0:
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
        bounds = dict(
            body_kp=(1e-4,0.5), body_kv=(1e-6,1.0),
            hip_yaw_kp=(1e-4,0.5), hip_yaw_kv=(1e-7,5e-1),
            hip_pitch_kp=(1e-5,0.5), hip_pitch_kv=(1e-8,5e-2),
            tibia_kp=(1e-3,0.5), tibia_kv=(1e-7,5e-2),
            tarsus_kp=(1e-3,0.5), tarsus_kv=(1e-6,5e-2),
            pitch_kp=(1e-4,0.5), pitch_kv=(1e-6,5e-1),
        )
        for k in list(seed_params.keys()):
            if k in bounds:
                lo, hi = bounds[k]
                seed_params[k] = float(np.clip(seed_params[k], lo, hi))
        print(f"[3eval] seeding first trial from {seed_yaml}")
        for k, v in seed_params.items():
            print(f"        {k:14s}= {v:.6g}")
        study.enqueue_trial(seed_params)

    obj, trial_rows = make_objective(
        args, base_cfg, tmp_dir, csv_path, flat_xml, rough_xml)

    t_start = time.time()
    try:
        study.optimize(obj, n_trials=args.n_trials,
                        n_jobs=max(1, args.n_jobs),
                        show_progress_bar=False)
    except KeyboardInterrupt:
        print("\n[3eval] interrupted — writing partial results.")

    elapsed_min = (time.time() - t_start) / 60.0
    _dump_rows(csv_path, trial_rows)

    # Best
    ok = [t for t in study.trials if t.value is not None
          and t.value < BUCKLED_COST * 0.99]
    if ok:
        best = min(ok, key=lambda t: t.value)
        print(f"\n[3eval] best trial #{best.number}  cost={best.value:.4f}")
        for k, v in best.params.items():
            print(f"        {k:14s}= {v:.6g}")
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
        with open(os.path.join(run_dir, "best_params.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(os.path.join(run_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump({"trial": best.number, "cost": best.value,
                        "params": best.params}, f, indent=2)

    print(f"[3eval] done — {len(study.trials)} trials in {elapsed_min:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
