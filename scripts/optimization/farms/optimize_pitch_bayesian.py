#!/usr/bin/env python3
"""
optimize_pitch_bayesian.py — Bayesian optimization for pitch compliance
========================================================================
Uses Optuna (Tree-structured Parzen Estimator) to find the softest pitch
gains (kp, kv independently) that maximize terrain conformity without folding.

Cost metric: terrain conformity error
  For each body segment i at each timestep:
    terrain_z_i = heightfield(body_x_i, body_y_i)
    body_z_i    = world z position of link_body_i
    clearance_i = body_z_i - terrain_z_i
  Ideal clearance = nominal standing height (~0.0283m from XML)
  Error = mean over time and segments of (clearance_i - nominal)²

  A perfectly compliant body tracks the terrain surface exactly →  low error.
  A stiff body stays flat regardless of terrain → high error.
  A folded body collapses to the ground → very high error (+ penalty).

Optimization objective (minimize):
  cost = terrain_conformity_error  (if survived)
  cost = 1e6                       (if buckled/folded)

Optuna handles kp and kv independently (not locked by ratio), exploring
the continuous parameter space efficiently.

Requirements:
    pip install optuna

Usage:
    python scripts/optimization/farms/optimize_pitch_bayesian.py
    python scripts/optimization/farms/optimize_pitch_bayesian.py --n-trials 100 --duration 10
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import mujoco

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed. Run:")
    print("  pip install optuna")
    sys.exit(1)

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex, N_BODY_JOINTS

XML_PATH    = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "optimization", "pitch_bayesian")

# ═══════════════════════════════════════════════════════════════════
# Fixed parameters
# ═══════════════════════════════════════════════════════════════════

FIXED_BODY_KP = 0.5
FIXED_BODY_KV = 0.1
FIXED_ROLL_KP = 0.005
FIXED_ROLL_KV = 0.002

# Number of body segments to track (link_body_0 .. link_body_20)
N_BODY_LINKS = 21

# Failure thresholds
MAX_PITCH_DEG = 35.0
MAX_ROLL_DEG  = 60.0
MIN_HEIGHT_M  = -0.01

# ═══════════════════════════════════════════════════════════════════
# Terrain height sampling from MuJoCo heightfield
# ═══════════════════════════════════════════════════════════════════

class TerrainSampler:
    """
    Samples terrain height at arbitrary (x, y) world coordinates
    by reading the MuJoCo heightfield data.
    """

    def __init__(self, model):
        # Find the heightfield
        hf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hf_id < 0:
            raise ValueError("Heightfield 'terrain' not found in model")

        self.nrow = model.hfield_nrow[hf_id]
        self.ncol = model.hfield_ncol[hf_id]

        # hfield_size: [x_half, y_half, z_top, z_bottom]
        self.x_half   = model.hfield_size[hf_id, 0]
        self.y_half   = model.hfield_size[hf_id, 1]
        self.z_top    = model.hfield_size[hf_id, 2]
        self.z_bottom = model.hfield_size[hf_id, 3]

        # Heightfield data is stored as float array, row-major
        # Values in [0, 1], scaled by z_top (max elevation)
        n_data = self.nrow * self.ncol
        start = model.hfield_adr[hf_id]
        self.data = model.hfield_data[start:start + n_data].reshape(
            self.nrow, self.ncol).copy()

        # Get the geom position offset (terrain_geom pos in XML)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
        if geom_id >= 0:
            self.offset = model.geom_pos[geom_id].copy()
        else:
            self.offset = np.zeros(3)

        print(f"[TerrainSampler] {self.nrow}x{self.ncol} heightfield, "
              f"size={self.x_half*2:.3f}x{self.y_half*2:.3f}m, "
              f"z_range=[{self.z_bottom:.4f}, {self.z_top:.4f}]m")

    def get_height(self, x, y):
        """
        Get terrain height at world coordinates (x, y).
        Returns z (world frame).
        """
        # Transform to heightfield local coordinates
        lx = x - self.offset[0]
        ly = y - self.offset[1]

        # Normalize to [0, 1] range within the heightfield
        # MuJoCo hfield: x spans [-x_half, x_half], y spans [-y_half, y_half]
        u = (lx + self.x_half) / (2.0 * self.x_half)  # [0, 1]
        v = (ly + self.y_half) / (2.0 * self.y_half)  # [0, 1]

        # Clamp to valid range
        u = max(0.0, min(1.0 - 1e-10, u))
        v = max(0.0, min(1.0 - 1e-10, v))

        # Map to grid indices (row = y direction, col = x direction in MuJoCo)
        col = u * (self.ncol - 1)
        row = v * (self.nrow - 1)

        # Bilinear interpolation
        c0, r0 = int(col), int(row)
        c1, r1 = min(c0 + 1, self.ncol - 1), min(r0 + 1, self.nrow - 1)
        fc, fr = col - c0, row - r0

        h00 = self.data[r0, c0]
        h01 = self.data[r0, c1]
        h10 = self.data[r1, c0]
        h11 = self.data[r1, c1]

        h = (h00 * (1 - fc) * (1 - fr) +
             h01 * fc * (1 - fr) +
             h10 * (1 - fc) * fr +
             h11 * fc * fr)

        # Scale: hfield data is [0,1], maps to [z_bottom, z_top] relative to geom pos
        z = self.offset[2] + h * self.z_top
        return z

    def get_heights_batch(self, xy_array):
        """Get terrain heights for array of (N, 2) xy positions."""
        return np.array([self.get_height(xy[0], xy[1]) for xy in xy_array])


# ═══════════════════════════════════════════════════════════════════
# Simulation + terrain conformity evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate(pitch_kp, pitch_kv, duration, wave_params=None, verbose=False):
    """
    Run simulation, compute terrain conformity error.
    Returns dict with metrics.
    """
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return {'status': 'xml_error', 'cost': 1e6, 'error': str(e)}

    data = mujoco.MjData(model)

    try:
        ctrl = ImpedanceTravelingWaveController(
            model, CONFIG_PATH,
            body_kp=FIXED_BODY_KP, body_kv=FIXED_BODY_KV,
            pitch_kp=pitch_kp, pitch_kv=pitch_kv,
            roll_kp=FIXED_ROLL_KP, roll_kv=FIXED_ROLL_KV,
        )
    except Exception as e:
        return {'status': 'ctrl_error', 'cost': 1e6, 'error': str(e)}

    # Override wave params
    if wave_params:
        if 'amplitude' in wave_params:
            ctrl.body_amp = wave_params['amplitude']
        if 'frequency' in wave_params:
            ctrl.freq = wave_params['frequency']
            ctrl.omega = 2.0 * math.pi * ctrl.freq

    # Terrain sampler
    try:
        terrain = TerrainSampler(model)
    except ValueError:
        terrain = None

    # Resolve body segment IDs (link_body_0 .. link_body_20)
    body_ids = []
    for i in range(N_BODY_LINKS):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{i}")
        if bid >= 0:
            body_ids.append(bid)

    # Resolve pitch/roll joint IDs for failure detection
    pitch_ids, roll_ids = [], []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_ids.append(j)
        if nm and 'joint_roll_body' in nm:
            roll_ids.append(j)

    n_steps  = int(duration / model.opt.timestep)
    rec_dt   = 0.01
    last_rec = -np.inf

    # Storage
    clearance_errors = []   # per-timestep: MSE of (clearance - nominal) over segments
    max_pitch_history = []
    max_roll_history  = []

    # Nominal clearance: body standing height above terrain (from XML: z=0.0283m)
    NOMINAL_CLEARANCE = 0.0283

    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Failure check every 100 steps
        if step_i % 100 == 0:
            for jid in pitch_ids:
                q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                if q_deg > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch_fold({q_deg:.1f}° at t={data.time:.2f}s)"
                    break
            if not buckled:
                for jid in roll_ids:
                    q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                    if q_deg > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll_flip({q_deg:.1f}° at t={data.time:.2f}s)"
                        break
            if not buckled:
                root_id = body_ids[0] if body_ids else 0
                cz = data.subtree_com[root_id][2]
                if cz < MIN_HEIGHT_M:
                    buckled = True
                    buckle_reason = f"collapsed(z={cz:.4f}m at t={data.time:.2f}s)"
            if buckled:
                break

        # Record at lower rate
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time

            # Pitch/roll monitoring
            pitch_vals = [abs(data.qpos[model.jnt_qposadr[j]]) for j in pitch_ids]
            roll_vals  = [abs(data.qpos[model.jnt_qposadr[j]]) for j in roll_ids]
            max_pitch_history.append(max(pitch_vals) if pitch_vals else 0.0)
            max_roll_history.append(max(roll_vals) if roll_vals else 0.0)

            # Terrain conformity: for each body segment, measure clearance error
            if terrain and body_ids:
                seg_errors = []
                for bid in body_ids:
                    # Body segment world position
                    body_pos = data.xpos[bid]  # (3,) — world xyz
                    body_z = body_pos[2]

                    # Terrain height at this (x, y)
                    terrain_z = terrain.get_height(body_pos[0], body_pos[1])

                    # Clearance = how high body is above terrain
                    clearance = body_z - terrain_z

                    # Error: deviation from nominal clearance
                    # If body perfectly conforms to terrain, clearance ≈ NOMINAL
                    # everywhere, regardless of terrain shape → error ≈ 0
                    err = (clearance - NOMINAL_CLEARANCE) ** 2
                    seg_errors.append(err)

                clearance_errors.append(np.mean(seg_errors))

    # ── Compute results ──
    result = {
        'pitch_kp': pitch_kp,
        'pitch_kv': pitch_kv,
        'kv_ratio': pitch_kv / pitch_kp if pitch_kp > 0 else 0,
        'duration': duration,
        'survived': not buckled,
        'sim_time': data.time,
    }

    if buckled:
        result['status'] = 'buckled'
        result['buckle_reason'] = buckle_reason
        result['cost'] = 1e6
        return result

    result['status'] = 'ok'

    # Terrain conformity MSE (main cost)
    if clearance_errors:
        conformity_mse = float(np.mean(clearance_errors))
        conformity_rmse = float(np.sqrt(conformity_mse))
    else:
        conformity_mse = 1e6
        conformity_rmse = 1e3

    result['conformity_mse'] = conformity_mse
    result['conformity_rmse_mm'] = conformity_rmse * 1000  # in mm

    # Pitch stats
    max_pitch_deg = math.degrees(max(max_pitch_history)) if max_pitch_history else 0
    result['max_pitch_deg'] = max_pitch_deg

    # Pitch trend
    if len(max_pitch_history) > 100:
        first_half = np.mean(max_pitch_history[:len(max_pitch_history)//2])
        second_half = np.mean(max_pitch_history[len(max_pitch_history)//2:])
        result['pitch_trend'] = float(second_half / max(first_half, 1e-10))
    else:
        result['pitch_trend'] = 1.0

    # Roll stats
    result['max_roll_deg'] = math.degrees(max(max_roll_history)) if max_roll_history else 0

    # Cost = terrain conformity error
    # Lower = better terrain tracking = more compliant body
    result['cost'] = conformity_mse

    return result


# ═══════════════════════════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════════════════════════

# Global storage for trial history
trial_history = []


def make_objective(duration, wave_challenges):
    """Create Optuna objective function."""

    def objective(trial):
        # Sample pitch_kp and pitch_kv INDEPENDENTLY (log-uniform)
        pitch_kp = trial.suggest_float("pitch_kp", 0.001, 0.1, log=True)
        pitch_kv = trial.suggest_float("pitch_kv", 0.0005, 0.05, log=True)

        # Run against all wave challenges, take worst-case cost
        costs = []
        details = []

        for wc in wave_challenges:
            r = evaluate(pitch_kp, pitch_kv, duration, wave_params=wc)
            details.append({
                'wave': wc.get('label', ''),
                'survived': r['survived'],
                'cost': r['cost'],
                'conformity_rmse_mm': r.get('conformity_rmse_mm', 0),
                'max_pitch_deg': r.get('max_pitch_deg', 0),
            })

            if not r['survived']:
                # Failed — return huge cost immediately
                entry = {
                    'trial': trial.number,
                    'pitch_kp': pitch_kp,
                    'pitch_kv': pitch_kv,
                    'cost': 1e6,
                    'survived': False,
                    'reason': r.get('buckle_reason', ''),
                    'details': details,
                }
                trial_history.append(entry)
                return 1e6

            costs.append(r['cost'])

        # Use MEAN cost across challenges (we want good conformity everywhere)
        mean_cost = float(np.mean(costs))
        max_cost  = float(np.max(costs))

        # Combined: weighted mean + penalty for worst case
        final_cost = 0.7 * mean_cost + 0.3 * max_cost

        entry = {
            'trial': trial.number,
            'pitch_kp': pitch_kp,
            'pitch_kv': pitch_kv,
            'kv_ratio': pitch_kv / pitch_kp,
            'cost': final_cost,
            'mean_cost': mean_cost,
            'max_cost': max_cost,
            'survived': True,
            'details': details,
        }
        trial_history.append(entry)

        # Print progress
        best_rmse = max(d['conformity_rmse_mm'] for d in details)
        best_pitch = max(d['max_pitch_deg'] for d in details)
        print(f"  Trial {trial.number:3d}: kp={pitch_kp:.5f} kv={pitch_kv:.5f} "
              f"(ratio={pitch_kv/pitch_kp:.2f}) "
              f"→ cost={final_cost:.6f} "
              f"rmse={best_rmse:.2f}mm pitch={best_pitch:.1f}°")

        return final_cost

    return objective


# ═══════════════════════════════════════════════════════════════════
# Wave challenges
# ═══════════════════════════════════════════════════════════════════

WAVE_CHALLENGES = [
    {'label': 'default',       'amplitude': 0.6, 'frequency': 1.0},
    {'label': 'low_amp',       'amplitude': 0.2, 'frequency': 1.0},
    {'label': 'high_amp_fast', 'amplitude': 0.6, 'frequency': 1.5},
]


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bayesian pitch compliance optimizer")
    parser.add_argument("--n-trials", type=int, default=60,
                        help="Number of Optuna trials (default: 60)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration per trial (default: 10s)")
    parser.add_argument("--challenges", type=int, default=3,
                        help="Number of wave challenges per trial (1=fast, 3=robust)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    challenges = WAVE_CHALLENGES[:args.challenges]

    print("=" * 70)
    print("Bayesian Pitch Compliance Optimization (Optuna TPE)")
    print("=" * 70)
    print(f"Objective: minimize terrain conformity error (MSE of clearance)")
    print(f"  body conforms to terrain → low error")
    print(f"  body stays stiff/flat   → high error")
    print(f"  body folds              → infinite cost")
    print(f"")
    print(f"Search space:")
    print(f"  pitch_kp: [0.001, 0.1]  (log-uniform, independent)")
    print(f"  pitch_kv: [0.0005, 0.05] (log-uniform, independent)")
    print(f"")
    print(f"Fixed: body_kp={FIXED_BODY_KP}, roll_kp={FIXED_ROLL_KP}")
    print(f"Trials: {args.n_trials}")
    print(f"Duration: {args.duration}s per simulation")
    print(f"Wave challenges: {len(challenges)} per trial")
    print(f"  → Total simulations: ~{args.n_trials * len(challenges)}")
    print(f"Fold threshold: {MAX_PITCH_DEG}°")
    print("=" * 70)
    print()

    # ── Create Optuna study ──
    sampler = TPESampler(seed=args.seed, n_startup_trials=15)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="pitch_compliance",
    )

    # Seed with known-good points to warm-start the optimizer
    study.enqueue_trial({"pitch_kp": 0.05,  "pitch_kv": 0.02})    # previous winner
    study.enqueue_trial({"pitch_kp": 0.003, "pitch_kv": 0.0012})  # grid search softest
    study.enqueue_trial({"pitch_kp": 0.005, "pitch_kv": 0.0025})  # grid search balanced
    study.enqueue_trial({"pitch_kp": 0.01,  "pitch_kv": 0.005})   # moderate
    study.enqueue_trial({"pitch_kp": 0.002, "pitch_kv": 0.001})   # edge of fold

    objective = make_objective(args.duration, challenges)

    t0 = time.time()

    # Suppress Optuna INFO logging (we have our own prints)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    elapsed = time.time() - t0

    # ── Results ──
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE ({elapsed:.0f}s, {args.n_trials} trials)")
    print(f"{'=' * 70}")

    # Best trial
    best = study.best_trial
    print(f"\n  BEST TRIAL #{best.number}:")
    print(f"    pitch_kp = {best.params['pitch_kp']:.6f}")
    print(f"    pitch_kv = {best.params['pitch_kv']:.6f}")
    print(f"    kv_ratio = {best.params['pitch_kv']/best.params['pitch_kp']:.3f}")
    print(f"    cost     = {best.value:.6f}")

    # Find the corresponding trial details
    best_detail = None
    for t in trial_history:
        if t['trial'] == best.number:
            best_detail = t
            break

    if best_detail and 'details' in best_detail:
        print(f"\n    Per-challenge breakdown:")
        for d in best_detail['details']:
            print(f"      {d['wave']:<16s}: rmse={d['conformity_rmse_mm']:.2f}mm  "
                  f"pitch={d['max_pitch_deg']:.1f}°")

    # Top 10 trials
    print(f"\n  Top 10 trials:")
    print(f"  {'#':<5} {'kp':<10} {'kv':<10} {'ratio':<7} {'cost':<12} {'survived'}")
    sorted_trials = sorted(trial_history, key=lambda t: t['cost'])
    for t in sorted_trials[:10]:
        print(f"  {t['trial']:<5} {t['pitch_kp']:<10.5f} {t['pitch_kv']:<10.5f} "
              f"{t.get('kv_ratio',0):<7.3f} {t['cost']:<12.6f} "
              f"{'OK' if t['survived'] else 'FAIL'}")

    # Survival stats
    n_survived = sum(1 for t in trial_history if t['survived'])
    n_failed   = sum(1 for t in trial_history if not t['survived'])
    print(f"\n  Survival: {n_survived}/{len(trial_history)} "
          f"({n_failed} folded/failed)")

    # ── Final recommendation ──
    kp = best.params['pitch_kp']
    kv = best.params['pitch_kv']

    print(f"\n  Update configs/farms_controller.yaml:")
    print(f"    impedance:")
    print(f"      pitch_kp: {kp:.6f}")
    print(f"      pitch_kv: {kv:.6f}")

    # ── Save ──
    out_path = args.output or os.path.join(OUTPUT_DIR, f"bayesian_{timestamp}.json")
    output = {
        'timestamp': timestamp,
        'n_trials': args.n_trials,
        'duration': args.duration,
        'n_challenges': len(challenges),
        'fixed_params': {
            'body_kp': FIXED_BODY_KP, 'body_kv': FIXED_BODY_KV,
            'roll_kp': FIXED_ROLL_KP, 'roll_kv': FIXED_ROLL_KV,
        },
        'best': {
            'pitch_kp': kp,
            'pitch_kv': kv,
            'kv_ratio': kv / kp,
            'cost': best.value,
        },
        'trials': sorted_trials,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=lambda x:
                  None if isinstance(x, float) and (x != x or x == float('inf')) else x)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
