#!/usr/bin/env python3
"""
optimize_pitch_v2.py — Bayesian optimization for pitch compliance (v2)
=======================================================================
Fixed cost function from v1:
  - Asymmetric: penalizes body lifting ABOVE terrain much harder than settling
  - Per-segment worst-case: one bad segment dominates cost (not averaged away)
  - Direct pitch angle penalty: prevents the optimizer from exploiting soft gains
    that let head/tail pitch up wildly

Cost = terrain_conformity + pitch_penalty + softness_penalty
  terrain_conformity: for each segment, measure clearance above terrain.
    If clearance > nominal → body is lifting off → HARD penalty (√10× weight ≈ 3.16×)
    If clearance < nominal → body dipping toward terrain → SOFT penalty (1× weight)
    Take 90th percentile across segments (worst-case, not mean)

  pitch_penalty: max pitch angle across all joints and all time
    Quadratic penalty: (max_pitch_deg / 20)²
    This directly prevents the head/tail lift-off problem.

  softness_penalty: explicitly rewards LOWER kp (softer joints)
    log10(kp / kp_min) → 0 at softest, 2 at stiffest
    This ensures the optimizer seeks the softest gains that still conform.

  FAIL: any pitch > 35° or roll > 60° → cost = 1e6

Speed: 5s simulations, 1 challenge (default wave), 40 trials.
  ~40 simulations total, ~1.5 hours on lab PC.

Usage:
    python scripts/optimization/farms/optimize_pitch_v2.py
    python scripts/optimization/farms/optimize_pitch_v2.py --n-trials 40 --duration 5
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
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex

XML_PATH    = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "optimization", "pitch_v2")

# Fixed params
FIXED_BODY_KP = 0.5
FIXED_BODY_KV = 0.1
FIXED_ROLL_KP = 0.005
FIXED_ROLL_KV = 0.002

N_BODY_LINKS = 21
MAX_PITCH_DEG = 35.0
MAX_ROLL_DEG  = 60.0
MIN_HEIGHT_M  = -0.01
NOMINAL_CLEARANCE = 0.0258  # measured mean body height on flat ground under gait

# ═══════════════════════════════════════════════════════════════════
# Terrain sampler
# ═══════════════════════════════════════════════════════════════════

class TerrainSampler:
    def __init__(self, model):
        hf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if hf_id < 0:
            raise ValueError("Heightfield 'terrain' not found")
        self.nrow = model.hfield_nrow[hf_id]
        self.ncol = model.hfield_ncol[hf_id]
        self.x_half = model.hfield_size[hf_id, 0]
        self.y_half = model.hfield_size[hf_id, 1]
        self.z_top  = model.hfield_size[hf_id, 2]
        n_data = self.nrow * self.ncol
        start = model.hfield_adr[hf_id]
        self.data = model.hfield_data[start:start + n_data].reshape(
            self.nrow, self.ncol).copy()
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
        self.offset = model.geom_pos[geom_id].copy() if geom_id >= 0 else np.zeros(3)

    def get_height(self, x, y):
        lx = x - self.offset[0]
        ly = y - self.offset[1]
        u = np.clip((lx + self.x_half) / (2.0 * self.x_half), 0, 1 - 1e-10)
        v = np.clip((ly + self.y_half) / (2.0 * self.y_half), 0, 1 - 1e-10)
        col = u * (self.ncol - 1)
        row = v * (self.nrow - 1)
        c0, r0 = int(col), int(row)
        c1, r1 = min(c0+1, self.ncol-1), min(r0+1, self.nrow-1)
        fc, fr = col - c0, row - r0
        h = (self.data[r0,c0]*(1-fc)*(1-fr) + self.data[r0,c1]*fc*(1-fr) +
             self.data[r1,c0]*(1-fc)*fr + self.data[r1,c1]*fc*fr)
        return self.offset[2] + h * self.z_top


# ═══════════════════════════════════════════════════════════════════
# Evaluate one trial
# ═══════════════════════════════════════════════════════════════════

def evaluate(pitch_kp, pitch_kv, duration):
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return {'survived': False, 'cost': 1e6, 'error': str(e)}

    data = mujoco.MjData(model)

    try:
        ctrl = ImpedanceTravelingWaveController(
            model, CONFIG_PATH,
            body_kp=FIXED_BODY_KP, body_kv=FIXED_BODY_KV,
            pitch_kp=pitch_kp, pitch_kv=pitch_kv,
            roll_kp=FIXED_ROLL_KP, roll_kv=FIXED_ROLL_KV,
        )
    except Exception as e:
        return {'survived': False, 'cost': 1e6, 'error': str(e)}

    try:
        terrain = TerrainSampler(model)
    except ValueError:
        terrain = None

    # Resolve IDs
    body_ids = []
    for i in range(N_BODY_LINKS):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{i}")
        if bid >= 0:
            body_ids.append(bid)

    pitch_ids, roll_ids = [], []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_ids.append(j)
        if nm and 'joint_roll_body' in nm:
            roll_ids.append(j)

    n_steps  = int(duration / model.opt.timestep)
    rec_dt   = 0.02  # record every 20ms (faster than 10ms)
    last_rec = -np.inf

    # Storage: per-timestep, per-segment clearance deviations
    seg_deviations_over_time = []    # list of arrays, each (N_segments,)
    max_pitch_over_time = []
    max_roll_over_time = []

    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Failure check every 200 steps (faster)
        if step_i % 200 == 0:
            for jid in pitch_ids:
                q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                if q_deg > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch({q_deg:.1f}° t={data.time:.1f}s)"
                    break
            if not buckled:
                for jid in roll_ids:
                    q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                    if q_deg > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll({q_deg:.1f}° t={data.time:.1f}s)"
                        break
            if not buckled:
                cz = data.subtree_com[body_ids[0]][2] if body_ids else 0
                if cz < MIN_HEIGHT_M:
                    buckled = True
                    buckle_reason = f"collapse(z={cz:.3f}m)"
            if buckled:
                break

        # Record
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time

            # Per-segment clearance deviation
            if terrain and body_ids:
                devs = np.zeros(len(body_ids))
                for si, bid in enumerate(body_ids):
                    body_z = data.xpos[bid][2]
                    terrain_z = terrain.get_height(data.xpos[bid][0], data.xpos[bid][1])
                    clearance = body_z - terrain_z
                    deviation = clearance - NOMINAL_CLEARANCE

                    # ASYMMETRIC: body ABOVE nominal (lifting off) penalized ~10×
                    # body BELOW nominal (dipping into terrain) penalized 1×
                    # Using sqrt(10)≈3.16 multiplier so after squaring → 10× ratio
                    if deviation > 0:
                        devs[si] = (deviation * 3.16) ** 2   # lifting off → ~10× after sq
                    else:
                        devs[si] = deviation ** 2             # dipping → mild

                seg_deviations_over_time.append(devs)

            # Pitch/roll tracking
            pitch_vals = [abs(data.qpos[model.jnt_qposadr[j]]) for j in pitch_ids]
            roll_vals  = [abs(data.qpos[model.jnt_qposadr[j]]) for j in roll_ids]
            max_pitch_over_time.append(max(pitch_vals) if pitch_vals else 0)
            max_roll_over_time.append(max(roll_vals) if roll_vals else 0)

    # ── Compute cost ──
    result = {
        'pitch_kp': pitch_kp,
        'pitch_kv': pitch_kv,
        'kv_ratio': pitch_kv / pitch_kp if pitch_kp > 0 else 0,
        'survived': not buckled,
        'sim_time': data.time,
    }

    if buckled:
        result['buckle_reason'] = buckle_reason
        result['cost'] = 1e6
        return result

    # Terrain conformity: 90th percentile across segments per timestep, then mean over time
    # This means the WORST segments drive the cost, not the average
    if seg_deviations_over_time:
        all_devs = np.array(seg_deviations_over_time)  # (T, N_seg)
        p90_per_step = np.percentile(all_devs, 90, axis=1)  # (T,)
        conformity_cost = float(np.mean(p90_per_step))
    else:
        conformity_cost = 1e6

    # Pitch penalty: quadratic on max pitch angle
    # 20° reference: anything above 20° gets heavily penalized
    max_pitch_deg = math.degrees(max(max_pitch_over_time)) if max_pitch_over_time else 0
    mean_pitch_deg = math.degrees(np.mean(max_pitch_over_time)) if max_pitch_over_time else 0
    pitch_penalty = (max_pitch_deg / 20.0) ** 2  # 20°→1, 30°→2.25, 10°→0.25

    # Also penalize if pitch is growing (late instability)
    if len(max_pitch_over_time) > 50:
        first_half = np.mean(max_pitch_over_time[:len(max_pitch_over_time)//2])
        second_half = np.mean(max_pitch_over_time[len(max_pitch_over_time)//2:])
        trend = second_half / max(first_half, 1e-10)
        if trend > 1.5:
            pitch_penalty *= trend  # amplify penalty if growing
    else:
        trend = 1.0

    # Softness reward: explicitly prefer lower kp (softer joints)
    # log10(kp/0.001) → 0 at kp=0.001 (softest), 2.0 at kp=0.1 (stiffest)
    softness_penalty = math.log10(pitch_kp / 0.001)  # range [0, 2]

    # Total cost: weighted sum
    # conformity_cost is ~0.0003-0.001 range → ×1000 → 0.3-1.0
    # pitch_penalty is ~0.1-2 range → ×0.5 → 0.05-1.0
    # softness_penalty is 0-2 range → ×0.3 → 0-0.6
    total_cost = conformity_cost * 1000.0 + pitch_penalty * 0.5 + softness_penalty * 0.3

    result['cost'] = float(total_cost)
    result['conformity_cost'] = float(conformity_cost)
    result['conformity_rmse_mm'] = float(np.sqrt(conformity_cost) * 1000)
    result['pitch_penalty'] = float(pitch_penalty)
    result['softness_penalty'] = float(softness_penalty)
    result['max_pitch_deg'] = float(max_pitch_deg)
    result['mean_pitch_deg'] = float(mean_pitch_deg)
    result['max_roll_deg'] = float(math.degrees(max(max_roll_over_time))) if max_roll_over_time else 0
    result['pitch_trend'] = float(trend) if 'trend' in dir() else 1.0

    return result


# ═══════════════════════════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════════════════════════

trial_history = []


def make_objective(duration):
    def objective(trial):
        pitch_kp = trial.suggest_float("pitch_kp", 0.001, 0.1, log=True)
        pitch_kv = trial.suggest_float("pitch_kv", 0.0003, 0.05, log=True)

        r = evaluate(pitch_kp, pitch_kv, duration)

        entry = {
            'trial': trial.number,
            'pitch_kp': pitch_kp,
            'pitch_kv': pitch_kv,
            'kv_ratio': pitch_kv / pitch_kp,
            'cost': r['cost'],
            'survived': r.get('survived', False),
            'conformity_rmse_mm': r.get('conformity_rmse_mm', 0),
            'max_pitch_deg': r.get('max_pitch_deg', 0),
            'max_roll_deg': r.get('max_roll_deg', 0),
            'pitch_penalty': r.get('pitch_penalty', 0),
            'softness_penalty': r.get('softness_penalty', 0),
            'buckle_reason': r.get('buckle_reason', ''),
        }
        trial_history.append(entry)

        if r.get('survived', False):
            print(f"  Trial {trial.number:3d}: kp={pitch_kp:.5f} kv={pitch_kv:.5f} "
                  f"(ratio={pitch_kv/pitch_kp:.2f}) "
                  f"→ cost={r['cost']:.4f} "
                  f"conf={r.get('conformity_rmse_mm',0):.1f}mm "
                  f"pitch={r.get('max_pitch_deg',0):.1f}° "
                  f"soft={r.get('softness_penalty',0):.2f}")
        else:
            print(f"  Trial {trial.number:3d}: kp={pitch_kp:.5f} kv={pitch_kv:.5f} "
                  f"→ FAIL {r.get('buckle_reason','')}")

        return r['cost']

    return objective


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pitch compliance optimizer v2")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Optuna trials (default: 40)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Sim duration (default: 5s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Pitch Compliance Optimization v2 (Bayesian / Optuna)")
    print("=" * 70)
    print(f"FIXED cost function (v2.1):")
    print(f"  - Asymmetric: body lifting OFF terrain penalized ~10×")
    print(f"  - 90th percentile across segments (worst drives cost)")
    print(f"  - Direct pitch angle penalty: (max_pitch/20°)²")
    print(f"  - Trend penalty: growing pitch amplified")
    print(f"  - Softness reward: log10(kp/0.001) penalizes stiff gains")
    print(f"")
    print(f"Search: pitch_kp=[0.001,0.1], pitch_kv=[0.0003,0.05] (log)")
    print(f"Fixed: body_kp={FIXED_BODY_KP}, roll_kp={FIXED_ROLL_KP}")
    print(f"Trials: {args.n_trials}, Duration: {args.duration}s")
    print(f"Estimated time: ~{args.n_trials * 2:.0f} min")
    print("=" * 70)
    print()

    sampler = TPESampler(seed=args.seed, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="pitch_v2")

    # Warm-start with known points
    study.enqueue_trial({"pitch_kp": 0.05,   "pitch_kv": 0.02})
    study.enqueue_trial({"pitch_kp": 0.005,  "pitch_kv": 0.0025})
    study.enqueue_trial({"pitch_kp": 0.01,   "pitch_kv": 0.005})
    study.enqueue_trial({"pitch_kp": 0.003,  "pitch_kv": 0.0012})
    study.enqueue_trial({"pitch_kp": 0.008,  "pitch_kv": 0.004})

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective = make_objective(args.duration)

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    elapsed = time.time() - t0

    # ── Results ──
    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed:.0f}s, {args.n_trials} trials)")
    print(f"{'=' * 70}")

    best = study.best_trial
    kp = best.params['pitch_kp']
    kv = best.params['pitch_kv']

    print(f"\n  BEST TRIAL #{best.number}:")
    print(f"    pitch_kp = {kp:.6f}")
    print(f"    pitch_kv = {kv:.6f}")
    print(f"    kv_ratio = {kv/kp:.3f}")
    print(f"    cost     = {best.value:.4f}")

    # Find best detail
    for t in trial_history:
        if t['trial'] == best.number:
            print(f"    conformity_rmse = {t['conformity_rmse_mm']:.1f}mm")
            print(f"    max_pitch = {t['max_pitch_deg']:.1f}°")
            print(f"    pitch_penalty = {t['pitch_penalty']:.2f}")
            break

    # Top 10
    survived = [t for t in trial_history if t['survived']]
    survived.sort(key=lambda t: t['cost'])

    print(f"\n  Top 10:")
    print(f"  {'#':<5} {'kp':<10} {'kv':<10} {'ratio':<7} {'cost':<10} "
          f"{'rmse_mm':<9} {'pitch°':<8} {'soft':<6}")
    for t in survived[:10]:
        print(f"  {t['trial']:<5} {t['pitch_kp']:<10.5f} {t['pitch_kv']:<10.5f} "
              f"{t['kv_ratio']:<7.2f} {t['cost']:<10.4f} "
              f"{t['conformity_rmse_mm']:<9.1f} {t['max_pitch_deg']:<8.1f} "
              f"{t.get('softness_penalty',0):<6.2f}")

    n_survived = sum(1 for t in trial_history if t['survived'])
    print(f"\n  Survival: {n_survived}/{len(trial_history)}")

    print(f"\n  Update configs/farms_controller.yaml:")
    print(f"    impedance:")
    print(f"      pitch_kp: {kp:.6f}")
    print(f"      pitch_kv: {kv:.6f}")

    # ── Save ──
    out_path = args.output or os.path.join(OUTPUT_DIR, f"pitch_v2_{timestamp}.json")
    output = {
        'timestamp': timestamp,
        'n_trials': args.n_trials,
        'duration': args.duration,
        'best': {'pitch_kp': kp, 'pitch_kv': kv, 'kv_ratio': kv/kp,
                 'cost': best.value},
        'trials': survived[:20],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and x != x else x)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
