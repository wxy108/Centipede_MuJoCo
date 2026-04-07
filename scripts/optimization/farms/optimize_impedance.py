#!/usr/bin/env python3
"""
optimize_impedance.py — Multi-objective grid search for impedance gains
========================================================================
Optimizes 6 parameters (body_kp, body_kv, pitch_kp, pitch_kv, roll_kp, roll_kv)
against 5 objectives:

  1. Torque tracking:  |commanded - actual| should be small
  2. Terrain compliance: gains shouldn't be too large (penalize high kp)
  3. Stability: centipede should not flip (roll angle < threshold)
  4. Anti-folding: pitch angles shouldn't exceed threshold (body folding)
  5. Long-term robustness: winner is verified at 10s after grid search at 3s

Cost function (lower is better):
  cost = w_track * torque_tracking_error
       + w_fold  * pitch_fold_penalty
       + w_flip  * roll_flip_penalty
       + w_soft  * compliance_penalty      (penalizes excessively stiff gains)
       + INFINITY if buckled/flipped

Grid search at 3s, then top-N verified at 10s.

Usage:
    python scripts/optimization/farms/optimize_impedance.py
    python scripts/optimization/farms/optimize_impedance.py --duration 5 --top 5
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import mujoco

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex

XML_PATH    = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "optimization", "impedance")

# ═══════════════════════════════════════════════════════════════════
# Parameter grid
# ═══════════════════════════════════════════════════════════════════

# body_kp: yaw stiffness — controls how tightly yaw tracks the wave
# body_kv: yaw damping — ratio to body_kp (typically 0.1-0.3× body_kp)
# pitch_kp: pitch stiffness — too low = body folds, too high = stiff
# pitch_kv: pitch damping — ratio to pitch_kp (typically 0.3-0.5× pitch_kp)
# roll_kp: roll stiffness — too low = flops sideways, too high = stiff
# roll_kv: roll damping — ratio to roll_kp

PARAM_GRID = {
    'body_kp':  [0.2, 0.5, 1.0, 2.0],
    'pitch_kp': [0.02, 0.05, 0.08, 0.12, 0.2],
    'roll_kp':  [0.002, 0.005, 0.01, 0.02, 0.05],
}

# kv is derived as a ratio of kp (not independently swept — reduces grid size)
KV_RATIOS = {
    'body_kv_ratio':  0.2,   # body_kv = 0.2 * body_kp
    'pitch_kv_ratio': 0.4,   # pitch_kv = 0.4 * pitch_kp
    'roll_kv_ratio':  0.4,   # roll_kv = 0.4 * roll_kp
}

# ═══════════════════════════════════════════════════════════════════
# Failure thresholds
# ═══════════════════════════════════════════════════════════════════

MAX_PITCH_DEG = 45.0   # degrees — if any pitch joint exceeds this → folded
MAX_ROLL_DEG  = 60.0   # degrees — if body roll exceeds this → flipped
MIN_HEIGHT_M  = -0.01  # metres — COM z below this → collapsed

# ═══════════════════════════════════════════════════════════════════
# Cost weights
# ═══════════════════════════════════════════════════════════════════

W_TRACK = 1.0      # torque tracking error (normalized)
W_FOLD  = 5.0      # pitch folding penalty
W_FLIP  = 5.0      # roll flip penalty
W_SOFT  = 0.5      # compliance penalty (prefer softer gains)

# ═══════════════════════════════════════════════════════════════════
# Simulation + evaluation
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(body_kp, pitch_kp, roll_kp, duration, wave_params=None):
    """
    Run a single simulation with given gains, return metrics dict.
    wave_params: optional dict to override body_wave settings for robustness tests.
    """
    body_kv  = body_kp  * KV_RATIOS['body_kv_ratio']
    pitch_kv = pitch_kp * KV_RATIOS['pitch_kv_ratio']
    roll_kv  = roll_kp  * KV_RATIOS['roll_kv_ratio']

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return {'status': 'xml_error', 'error': str(e)}

    data = mujoco.MjData(model)

    try:
        ctrl = ImpedanceTravelingWaveController(
            model, CONFIG_PATH,
            body_kp=body_kp, body_kv=body_kv,
            pitch_kp=pitch_kp, pitch_kv=pitch_kv,
            roll_kp=roll_kp, roll_kv=roll_kv,
        )
    except Exception as e:
        return {'status': 'ctrl_error', 'error': str(e)}

    # Override wave params for robustness testing
    if wave_params:
        if 'amplitude' in wave_params:
            ctrl.body_amp = wave_params['amplitude']
        if 'frequency' in wave_params:
            ctrl.freq = wave_params['frequency']
            ctrl.omega = 2.0 * math.pi * ctrl.freq

    # Resolve joint indices for monitoring
    pitch_ids, roll_ids, yaw_ids = [], [], []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_ids.append(j)
        if nm and 'joint_roll_body' in nm:
            roll_ids.append(j)
        if nm and nm.startswith('joint_body_'):
            yaw_ids.append(j)

    n_steps  = int(duration / model.opt.timestep)
    rec_dt   = 0.01
    last_rec = -np.inf

    # Storage
    times = []
    com_z = []
    max_pitch_per_step = []
    max_roll_per_step  = []
    yaw_cmd_torques  = []
    yaw_act_torques  = []

    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # ── Check for catastrophic failure every 100 steps ──
        if step_i % 100 == 0:
            # Pitch check
            for jid in pitch_ids:
                q = data.qpos[model.jnt_qposadr[jid]]
                if abs(math.degrees(q)) > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch_fold(joint={jid}, q={math.degrees(q):.1f}deg)"
                    break
            # Roll check
            if not buckled:
                for jid in roll_ids:
                    q = data.qpos[model.jnt_qposadr[jid]]
                    if abs(math.degrees(q)) > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll_flip(joint={jid}, q={math.degrees(q):.1f}deg)"
                        break
            # Height check
            if not buckled:
                root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
                cz = data.subtree_com[root_id][2]
                if cz < MIN_HEIGHT_M:
                    buckled = True
                    buckle_reason = f"collapsed(com_z={cz:.4f}m)"

            if buckled:
                break

        # ── Record at lower rate ──
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time
            times.append(data.time)

            # COM height
            root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
            com_z.append(data.subtree_com[root_id][2])

            # Max pitch/roll angles
            pitch_vals = [abs(data.qpos[model.jnt_qposadr[j]]) for j in pitch_ids]
            roll_vals  = [abs(data.qpos[model.jnt_qposadr[j]]) for j in roll_ids]
            max_pitch_per_step.append(max(pitch_vals) if pitch_vals else 0.0)
            max_roll_per_step.append(max(roll_vals) if roll_vals else 0.0)

            # Yaw torque tracking (commanded vs actual)
            if ctrl.idx.has_pitch_actuators:
                cmd = np.array([data.ctrl[aid] for aid in ctrl.idx.body_act_ids])
                act = np.array([data.actuator_force[aid] for aid in ctrl.idx.body_act_ids])
                yaw_cmd_torques.append(cmd)
                yaw_act_torques.append(act)

    # ── Compute metrics ──
    result = {
        'body_kp': body_kp, 'body_kv': body_kv,
        'pitch_kp': pitch_kp, 'pitch_kv': pitch_kv,
        'roll_kp': roll_kp, 'roll_kv': roll_kv,
        'duration': duration,
        'survived': not buckled,
        'sim_time': data.time,
    }

    if buckled:
        result['status'] = 'buckled'
        result['buckle_reason'] = buckle_reason
        result['buckle_time'] = data.time
        result['cost'] = float('inf')
        return result

    result['status'] = 'ok'

    # Metric 1: Torque tracking error (yaw RMS normalized)
    if yaw_cmd_torques:
        cmd_arr = np.array(yaw_cmd_torques)
        act_arr = np.array(yaw_act_torques)
        # For general actuators, ctrl IS the torque, and actuator_force is what MuJoCo applied
        # They should match closely for general actuators; discrepancy = clipping or constraint
        torque_err = np.sqrt(np.mean((cmd_arr - act_arr) ** 2))
        torque_scale = max(np.sqrt(np.mean(cmd_arr ** 2)), 1e-6)
        result['torque_tracking_rms'] = float(torque_err)
        result['torque_tracking_normalized'] = float(torque_err / torque_scale)
    else:
        result['torque_tracking_rms'] = 0.0
        result['torque_tracking_normalized'] = 0.0

    # Metric 2: Max pitch deflection (in degrees)
    max_pitch_deg = math.degrees(max(max_pitch_per_step)) if max_pitch_per_step else 0.0
    mean_pitch_deg = math.degrees(np.mean(max_pitch_per_step)) if max_pitch_per_step else 0.0
    result['max_pitch_deg'] = max_pitch_deg
    result['mean_pitch_deg'] = mean_pitch_deg

    # Metric 3: Max roll deflection (in degrees)
    max_roll_deg = math.degrees(max(max_roll_per_step)) if max_roll_per_step else 0.0
    mean_roll_deg = math.degrees(np.mean(max_roll_per_step)) if max_roll_per_step else 0.0
    result['max_roll_deg'] = max_roll_deg
    result['mean_roll_deg'] = mean_roll_deg

    # Metric 4: COM height stability
    if com_z:
        result['mean_com_z'] = float(np.mean(com_z))
        result['min_com_z'] = float(np.min(com_z))
        result['com_z_std'] = float(np.std(com_z))

    # ── Composite cost ──
    # Torque tracking
    cost_track = result['torque_tracking_normalized']

    # Fold penalty: exponential penalty as max_pitch approaches threshold
    fold_ratio = max_pitch_deg / MAX_PITCH_DEG
    cost_fold = fold_ratio ** 2  # quadratic: gentle near 0, harsh near threshold

    # Flip penalty: same for roll
    flip_ratio = max_roll_deg / MAX_ROLL_DEG
    cost_flip = flip_ratio ** 2

    # Compliance penalty: prefer softer gains (log-scale)
    # Reference gains: body_kp=0.5, pitch_kp=0.05, roll_kp=0.005
    cost_soft = (
        max(0, math.log10(body_kp / 0.5)) +
        max(0, math.log10(pitch_kp / 0.05)) +
        max(0, math.log10(roll_kp / 0.005))
    )

    cost = (W_TRACK * cost_track +
            W_FOLD  * cost_fold +
            W_FLIP  * cost_flip +
            W_SOFT  * cost_soft)

    result['cost'] = float(cost)
    result['cost_breakdown'] = {
        'track': float(W_TRACK * cost_track),
        'fold':  float(W_FOLD * cost_fold),
        'flip':  float(W_FLIP * cost_flip),
        'soft':  float(W_SOFT * cost_soft),
    }

    return result


# ═══════════════════════════════════════════════════════════════════
# Robustness verification
# ═══════════════════════════════════════════════════════════════════

ROBUSTNESS_CHALLENGES = [
    {'amplitude': 0.2, 'frequency': 1.0},   # low amplitude
    {'amplitude': 0.6, 'frequency': 1.0},   # high amplitude (current)
    {'amplitude': 0.4, 'frequency': 0.5},   # slow
    {'amplitude': 0.4, 'frequency': 2.0},   # fast
    {'amplitude': 0.6, 'frequency': 1.5},   # high amp + fast
]


def verify_winner(body_kp, pitch_kp, roll_kp, duration=10.0):
    """Run winner against multiple wave parameter combos at full duration."""
    results = []
    for i, wp in enumerate(ROBUSTNESS_CHALLENGES):
        print(f"    Robustness test {i+1}/{len(ROBUSTNESS_CHALLENGES)}: "
              f"amp={wp['amplitude']}, freq={wp['frequency']}Hz, {duration}s")
        r = run_evaluation(body_kp, pitch_kp, roll_kp, duration, wave_params=wp)
        r['wave_params'] = wp
        results.append(r)
        status = "OK" if r['survived'] else f"FAIL({r.get('buckle_reason','')})"
        print(f"      → {status}  cost={r['cost']:.4f}" if r['survived']
              else f"      → {status} at t={r.get('buckle_time',0):.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Impedance gain optimizer")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Grid search simulation duration (default: 3s)")
    parser.add_argument("--verify-duration", type=float, default=10.0,
                        help="Verification duration for winners (default: 10s)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top candidates to verify (default: 5)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Build parameter combinations ──
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"=" * 70)
    print(f"Impedance Gain Optimization")
    print(f"=" * 70)
    print(f"Parameters: {keys}")
    print(f"Grid sizes: {[len(v) for v in values]}")
    print(f"Total combinations: {total}")
    print(f"Grid search duration: {args.duration}s")
    print(f"Verify duration: {args.verify_duration}s")
    print(f"Top-N to verify: {args.top}")
    print(f"kv ratios: body={KV_RATIOS['body_kv_ratio']}, "
          f"pitch={KV_RATIOS['pitch_kv_ratio']}, "
          f"roll={KV_RATIOS['roll_kv_ratio']}")
    print(f"=" * 70)
    print()

    # ── Phase 1: Grid search ──
    print("Phase 1: Grid Search")
    print("-" * 70)

    all_results = []
    t0 = time.time()

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        body_kp  = params['body_kp']
        pitch_kp = params['pitch_kp']
        roll_kp  = params['roll_kp']

        print(f"  [{idx+1}/{total}] body_kp={body_kp:.3f} "
              f"pitch_kp={pitch_kp:.4f} roll_kp={roll_kp:.4f} ... ", end="", flush=True)

        result = run_evaluation(body_kp, pitch_kp, roll_kp, args.duration)
        all_results.append(result)

        if result['survived']:
            print(f"OK  cost={result['cost']:.4f}  "
                  f"pitch={result['max_pitch_deg']:.1f}° "
                  f"roll={result['max_roll_deg']:.1f}°")
        else:
            print(f"FAIL at t={result.get('buckle_time',0):.1f}s  "
                  f"({result.get('buckle_reason','')})")

    elapsed = time.time() - t0
    survived = [r for r in all_results if r['survived']]
    failed   = [r for r in all_results if not r['survived']]

    print(f"\nPhase 1 complete in {elapsed:.0f}s")
    print(f"  Survived: {len(survived)}/{total}")
    print(f"  Failed:   {len(failed)}/{total}")

    if not survived:
        print("\nERROR: No configuration survived! Try widening the grid or "
              "increasing failure thresholds.")
        # Save results anyway
        out_path = args.output or os.path.join(OUTPUT_DIR, f"grid_{timestamp}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'phase1': all_results, 'phase2': []}, f, indent=2)
        print(f"Results saved to: {out_path}")
        return

    # ── Rank by cost ──
    survived.sort(key=lambda r: r['cost'])
    print(f"\nTop {min(args.top, len(survived))} candidates:")
    print(f"  {'Rank':<5} {'body_kp':<9} {'pitch_kp':<10} {'roll_kp':<10} "
          f"{'cost':<8} {'pitch°':<8} {'roll°':<8} {'trk_err':<8}")
    for i, r in enumerate(survived[:args.top]):
        print(f"  {i+1:<5} {r['body_kp']:<9.3f} {r['pitch_kp']:<10.4f} "
              f"{r['roll_kp']:<10.4f} {r['cost']:<8.4f} "
              f"{r['max_pitch_deg']:<8.1f} {r['max_roll_deg']:<8.1f} "
              f"{r['torque_tracking_normalized']:<8.4f}")

    # ── Phase 2: Verify top-N at longer duration + robustness ──
    print(f"\n{'=' * 70}")
    print(f"Phase 2: Robustness Verification ({args.verify_duration}s)")
    print(f"{'=' * 70}")

    top_candidates = survived[:args.top]
    verify_results = []

    for i, cand in enumerate(top_candidates):
        body_kp  = cand['body_kp']
        pitch_kp = cand['pitch_kp']
        roll_kp  = cand['roll_kp']

        print(f"\n  Candidate {i+1}/{len(top_candidates)}: "
              f"body_kp={body_kp:.3f} pitch_kp={pitch_kp:.4f} roll_kp={roll_kp:.4f}")

        rob_results = verify_winner(body_kp, pitch_kp, roll_kp, args.verify_duration)

        n_pass = sum(1 for r in rob_results if r['survived'])
        avg_cost = np.mean([r['cost'] for r in rob_results if r['survived']]) if n_pass > 0 else float('inf')
        max_pitch = max((r.get('max_pitch_deg', 0) for r in rob_results if r['survived']), default=0)
        max_roll  = max((r.get('max_roll_deg', 0) for r in rob_results if r['survived']), default=0)

        verify_entry = {
            'params': {
                'body_kp': body_kp,
                'body_kv': body_kp * KV_RATIOS['body_kv_ratio'],
                'pitch_kp': pitch_kp,
                'pitch_kv': pitch_kp * KV_RATIOS['pitch_kv_ratio'],
                'roll_kp': roll_kp,
                'roll_kv': roll_kp * KV_RATIOS['roll_kv_ratio'],
            },
            'grid_cost': cand['cost'],
            'robustness_pass': n_pass,
            'robustness_total': len(rob_results),
            'avg_cost_10s': float(avg_cost),
            'max_pitch_deg_10s': float(max_pitch),
            'max_roll_deg_10s': float(max_roll),
            'details': rob_results,
        }
        verify_results.append(verify_entry)

        print(f"    Summary: {n_pass}/{len(rob_results)} passed, "
              f"avg_cost={avg_cost:.4f}, max_pitch={max_pitch:.1f}°, "
              f"max_roll={max_roll:.1f}°")

    # ── Final ranking ──
    # Sort by: most robustness passes, then lowest average cost
    verify_results.sort(key=lambda v: (-v['robustness_pass'], v['avg_cost_10s']))

    print(f"\n{'=' * 70}")
    print(f"FINAL RANKING")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<5} {'body_kp':<9} {'pitch_kp':<10} {'roll_kp':<10} "
          f"{'pass':<6} {'avg_cost':<10} {'max_p°':<8} {'max_r°':<8}")
    for i, v in enumerate(verify_results):
        p = v['params']
        print(f"  {i+1:<5} {p['body_kp']:<9.3f} {p['pitch_kp']:<10.4f} "
              f"{p['roll_kp']:<10.4f} {v['robustness_pass']}/{v['robustness_total']:<4} "
              f"{v['avg_cost_10s']:<10.4f} {v['max_pitch_deg_10s']:<8.1f} "
              f"{v['max_roll_deg_10s']:<8.1f}")

    # ── Recommend winner ──
    winner = verify_results[0]
    wp = winner['params']
    print(f"\n  WINNER: body_kp={wp['body_kp']:.3f} body_kv={wp['body_kv']:.4f}")
    print(f"          pitch_kp={wp['pitch_kp']:.4f} pitch_kv={wp['pitch_kv']:.5f}")
    print(f"          roll_kp={wp['roll_kp']:.4f} roll_kv={wp['roll_kv']:.5f}")
    print(f"          {winner['robustness_pass']}/{winner['robustness_total']} robustness, "
          f"avg_cost={winner['avg_cost_10s']:.4f}")

    print(f"\n  To apply winner, update configs/farms_controller.yaml:")
    print(f"    impedance:")
    print(f"      body_kp:  {wp['body_kp']}")
    print(f"      body_kv:  {wp['body_kv']}")
    print(f"      pitch_kp: {wp['pitch_kp']}")
    print(f"      pitch_kv: {wp['pitch_kv']}")
    print(f"      roll_kp:  {wp['roll_kp']}")
    print(f"      roll_kv:  {wp['roll_kv']}")

    # ── Save ──
    out_path = args.output or os.path.join(OUTPUT_DIR, f"optimize_{timestamp}.json")
    output = {
        'timestamp': timestamp,
        'grid_duration': args.duration,
        'verify_duration': args.verify_duration,
        'param_grid': {k: [float(x) for x in v] for k, v in PARAM_GRID.items()},
        'kv_ratios': KV_RATIOS,
        'cost_weights': {'W_TRACK': W_TRACK, 'W_FOLD': W_FOLD,
                         'W_FLIP': W_FLIP, 'W_SOFT': W_SOFT},
        'thresholds': {'MAX_PITCH_DEG': MAX_PITCH_DEG, 'MAX_ROLL_DEG': MAX_ROLL_DEG,
                       'MIN_HEIGHT_M': MIN_HEIGHT_M},
        'total_combos': total,
        'survived_phase1': len(survived),
        'phase1_results': all_results,
        'phase2_results': [
            {k: v for k, v in vr.items() if k != 'details'}
            for vr in verify_results
        ],
        'winner': wp,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
