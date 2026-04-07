#!/usr/bin/env python3
"""
optimize_pitch_softness.py — Find the SOFTEST pitch gains that don't fold
==========================================================================
Holds body_kp and roll_kp at their optimized values (from optimize_impedance.py)
and sweeps pitch_kp downward to find the minimum stable pitch stiffness.

Goal: maximize terrain compliance (softest possible pitch) while the body
does NOT fold/overbend during 10s simulation across multiple wave challenges.

Sweep strategy:
  - Fine-grained pitch_kp sweep from very soft to moderately stiff
  - pitch_kv = ratio × pitch_kp (sweep the ratio too for damping tuning)
  - Full 10s simulation for each combo (no shortcut — user wants 10s stability)
  - Test across 5 wave parameter challenges for robustness

Metrics:
  - max_pitch_deg: peak pitch deflection across all joints over 10s
  - mean_pitch_deg: average peak pitch over time
  - survived: no joint exceeded fold threshold
  - softness_score: 1/pitch_kp (higher = softer = better, if survived)

Usage:
    python scripts/optimization/farms/optimize_pitch_softness.py
    python scripts/optimization/farms/optimize_pitch_softness.py --duration 10
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
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "optimization", "pitch_softness")

# ═══════════════════════════════════════════════════════════════════
# Fixed parameters (from previous optimization winner)
# ═══════════════════════════════════════════════════════════════════

FIXED_BODY_KP = 0.5
FIXED_BODY_KV = 0.1
FIXED_ROLL_KP = 0.005
FIXED_ROLL_KV = 0.002

# ═══════════════════════════════════════════════════════════════════
# Pitch parameter grid — fine-grained sweep toward softness
# ═══════════════════════════════════════════════════════════════════

PITCH_KP_VALUES = [
    0.001, 0.002, 0.003, 0.005,
    0.008, 0.01,  0.015, 0.02,
    0.03,  0.04,  0.05,  0.08,
]

# kv/kp ratio: controls damping behavior
# Lower ratio = more oscillatory but softer feel
# Higher ratio = more damped but stiffer feel
KV_RATIO_VALUES = [0.2, 0.3, 0.4, 0.5]

# ═══════════════════════════════════════════════════════════════════
# Failure thresholds
# ═══════════════════════════════════════════════════════════════════

MAX_PITCH_DEG    = 35.0    # hard fail: any pitch joint exceeds this
MAX_ROLL_DEG     = 60.0    # hard fail: roll flip
MIN_HEIGHT_M     = -0.01   # hard fail: COM collapsed
WARN_PITCH_DEG   = 15.0    # soft warning: getting close to fold

# ═══════════════════════════════════════════════════════════════════
# Wave parameter challenges for robustness
# ═══════════════════════════════════════════════════════════════════

WAVE_CHALLENGES = [
    {'label': 'default',        'amplitude': 0.6, 'frequency': 1.0},
    {'label': 'low_amp',        'amplitude': 0.2, 'frequency': 1.0},
    {'label': 'high_amp_fast',  'amplitude': 0.6, 'frequency': 1.5},
    {'label': 'slow',           'amplitude': 0.4, 'frequency': 0.5},
    {'label': 'fast',           'amplitude': 0.4, 'frequency': 2.0},
]

# ═══════════════════════════════════════════════════════════════════
# Single simulation run
# ═══════════════════════════════════════════════════════════════════

def run_single(pitch_kp, pitch_kv, duration, wave_params=None):
    """
    Run one simulation. Returns metrics dict.
    """
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return {'status': 'xml_error', 'error': str(e)}

    data = mujoco.MjData(model)

    try:
        ctrl = ImpedanceTravelingWaveController(
            model, CONFIG_PATH,
            body_kp=FIXED_BODY_KP, body_kv=FIXED_BODY_KV,
            pitch_kp=pitch_kp, pitch_kv=pitch_kv,
            roll_kp=FIXED_ROLL_KP, roll_kv=FIXED_ROLL_KV,
        )
    except Exception as e:
        return {'status': 'ctrl_error', 'error': str(e)}

    # Override wave params
    if wave_params:
        if 'amplitude' in wave_params:
            ctrl.body_amp = wave_params['amplitude']
        if 'frequency' in wave_params:
            ctrl.freq = wave_params['frequency']
            ctrl.omega = 2.0 * math.pi * ctrl.freq

    # Resolve pitch/roll joint indices
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

    # Per-timestep tracking
    max_pitch_history = []    # max |pitch| across all joints at each record step
    max_roll_history  = []
    com_z_history     = []
    pitch_all_history = []    # all pitch joint values at each record step

    buckled = False
    buckle_reason = ""

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Check failure every 100 steps
        if step_i % 100 == 0:
            for jid in pitch_ids:
                q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                if q_deg > MAX_PITCH_DEG:
                    buckled = True
                    buckle_reason = f"pitch_fold({q_deg:.1f}deg at t={data.time:.2f}s)"
                    break
            if not buckled:
                for jid in roll_ids:
                    q_deg = abs(math.degrees(data.qpos[model.jnt_qposadr[jid]]))
                    if q_deg > MAX_ROLL_DEG:
                        buckled = True
                        buckle_reason = f"roll_flip({q_deg:.1f}deg at t={data.time:.2f}s)"
                        break
            if not buckled:
                root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
                cz = data.subtree_com[root_id][2]
                if cz < MIN_HEIGHT_M:
                    buckled = True
                    buckle_reason = f"collapsed(z={cz:.4f}m at t={data.time:.2f}s)"
            if buckled:
                break

        # Record at lower rate
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time

            pitch_vals = [abs(data.qpos[model.jnt_qposadr[j]]) for j in pitch_ids]
            roll_vals  = [abs(data.qpos[model.jnt_qposadr[j]]) for j in roll_ids]
            max_pitch_history.append(max(pitch_vals) if pitch_vals else 0.0)
            max_roll_history.append(max(roll_vals) if roll_vals else 0.0)
            pitch_all_history.append(pitch_vals)

            root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
            com_z_history.append(data.subtree_com[root_id][2])

    # ── Compute result ──
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
        return result

    result['status'] = 'ok'

    # Peak pitch over entire simulation
    max_pitch_rad = max(max_pitch_history) if max_pitch_history else 0
    result['max_pitch_deg'] = math.degrees(max_pitch_rad)
    result['mean_max_pitch_deg'] = math.degrees(np.mean(max_pitch_history))

    # Peak pitch in last 3 seconds (check for late-onset instability)
    if len(max_pitch_history) > 300:  # last 3s at 0.01s recording
        late_pitch = max_pitch_history[-300:]
        result['late_max_pitch_deg'] = math.degrees(max(late_pitch))
        result['late_mean_pitch_deg'] = math.degrees(np.mean(late_pitch))
    else:
        result['late_max_pitch_deg'] = result['max_pitch_deg']
        result['late_mean_pitch_deg'] = result['mean_max_pitch_deg']

    # Pitch trend: is it growing over time? (sign of instability)
    if len(max_pitch_history) > 100:
        first_half = np.mean(max_pitch_history[:len(max_pitch_history)//2])
        second_half = np.mean(max_pitch_history[len(max_pitch_history)//2:])
        result['pitch_trend_ratio'] = float(second_half / max(first_half, 1e-10))
    else:
        result['pitch_trend_ratio'] = 1.0

    # Roll stats
    max_roll_rad = max(max_roll_history) if max_roll_history else 0
    result['max_roll_deg'] = math.degrees(max_roll_rad)

    # COM height
    result['mean_com_z'] = float(np.mean(com_z_history))
    result['min_com_z']  = float(np.min(com_z_history))

    return result


# ═══════════════════════════════════════════════════════════════════
# Main optimization
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pitch softness optimizer")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration (default: 10s)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build combos
    combos = list(itertools.product(PITCH_KP_VALUES, KV_RATIO_VALUES))
    total = len(combos)

    print("=" * 70)
    print("Pitch Softness Optimization")
    print("=" * 70)
    print(f"Goal: find SOFTEST pitch_kp that survives {args.duration}s")
    print(f"Fixed: body_kp={FIXED_BODY_KP}, roll_kp={FIXED_ROLL_KP}")
    print(f"Sweep: pitch_kp={PITCH_KP_VALUES}")
    print(f"       kv_ratio={KV_RATIO_VALUES}")
    print(f"Total combos: {total}")
    print(f"Duration: {args.duration}s per run")
    print(f"Wave challenges: {len(WAVE_CHALLENGES)}")
    print(f"Fold threshold: {MAX_PITCH_DEG}°")
    print("=" * 70)
    print()

    # ── Phase 1: Default wave params (amp=0.6, freq=1.0) ──
    print("Phase 1: Sweep with default wave params (amp=0.6, freq=1.0)")
    print("-" * 70)

    phase1_results = []
    t0 = time.time()

    for idx, (kp, ratio) in enumerate(combos):
        kv = kp * ratio
        print(f"  [{idx+1}/{total}] pitch_kp={kp:.4f} kv_ratio={ratio:.1f} "
              f"(kv={kv:.5f}) ... ", end="", flush=True)

        r = run_single(kp, kv, args.duration,
                       wave_params={'amplitude': 0.6, 'frequency': 1.0})
        phase1_results.append(r)

        if r['survived']:
            trend_flag = " ⚠GROWING" if r.get('pitch_trend_ratio', 1) > 1.5 else ""
            print(f"OK  max_p={r['max_pitch_deg']:.1f}° "
                  f"late_p={r['late_max_pitch_deg']:.1f}° "
                  f"trend={r.get('pitch_trend_ratio',0):.2f}"
                  f"{trend_flag}")
        else:
            print(f"FAIL  {r.get('buckle_reason','')}")

    elapsed = time.time() - t0
    survived_p1 = [r for r in phase1_results if r['survived']]
    print(f"\nPhase 1 done in {elapsed:.0f}s  "
          f"({len(survived_p1)}/{total} survived)")

    if not survived_p1:
        print("ERROR: Nothing survived! The pitch gains might all be too soft "
              "or the thresholds too tight.")
        out_path = args.output or os.path.join(OUTPUT_DIR, f"pitch_soft_{timestamp}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'phase1': phase1_results}, f, indent=2)
        print(f"Results: {out_path}")
        return

    # ── Filter: remove configs where pitch is still growing at the end ──
    stable_survivors = [r for r in survived_p1
                        if r.get('pitch_trend_ratio', 1) < 2.0]
    print(f"  Stable (trend < 2.0×): {len(stable_survivors)}/{len(survived_p1)}")

    if not stable_survivors:
        stable_survivors = survived_p1  # fallback

    # ── Rank by softness: lowest pitch_kp wins (that's the goal) ──
    stable_survivors.sort(key=lambda r: (r['pitch_kp'], r.get('max_pitch_deg', 999)))

    print(f"\n  {'Rank':<5} {'kp':<8} {'ratio':<7} {'kv':<8} "
          f"{'max_p°':<8} {'late_p°':<8} {'trend':<7} {'roll°':<7}")
    for i, r in enumerate(stable_survivors[:20]):
        print(f"  {i+1:<5} {r['pitch_kp']:<8.4f} {r['kv_ratio']:<7.1f} "
              f"{r['pitch_kv']:<8.5f} {r['max_pitch_deg']:<8.1f} "
              f"{r['late_max_pitch_deg']:<8.1f} "
              f"{r.get('pitch_trend_ratio',0):<7.2f} "
              f"{r.get('max_roll_deg',0):<7.1f}")

    # ── Phase 2: Robustness test top candidates ──
    # Pick softest 5 unique pitch_kp values that survived
    seen_kp = set()
    top_candidates = []
    for r in stable_survivors:
        if r['pitch_kp'] not in seen_kp and len(top_candidates) < 5:
            seen_kp.add(r['pitch_kp'])
            top_candidates.append(r)

    print(f"\n{'=' * 70}")
    print(f"Phase 2: Robustness verification for top {len(top_candidates)} softest configs")
    print(f"{'=' * 70}")

    phase2_results = []

    for ci, cand in enumerate(top_candidates):
        kp = cand['pitch_kp']
        kv = cand['pitch_kv']
        ratio = cand['kv_ratio']

        print(f"\n  Candidate {ci+1}: pitch_kp={kp:.4f} kv_ratio={ratio:.1f} (kv={kv:.5f})")

        challenge_results = []
        for wi, wc in enumerate(WAVE_CHALLENGES):
            print(f"    [{wi+1}/{len(WAVE_CHALLENGES)}] {wc['label']}: "
                  f"amp={wc['amplitude']}, freq={wc['frequency']}Hz ... ",
                  end="", flush=True)

            r = run_single(kp, kv, args.duration, wave_params=wc)
            r['wave_label'] = wc['label']
            r['wave_params'] = wc
            challenge_results.append(r)

            if r['survived']:
                print(f"OK  max_p={r['max_pitch_deg']:.1f}° "
                      f"trend={r.get('pitch_trend_ratio',0):.2f}")
            else:
                print(f"FAIL  {r.get('buckle_reason','')}")

        n_pass = sum(1 for r in challenge_results if r['survived'])
        survived_costs = [r for r in challenge_results if r['survived']]
        worst_pitch = max((r.get('max_pitch_deg', 0) for r in survived_costs), default=0)
        worst_trend = max((r.get('pitch_trend_ratio', 0) for r in survived_costs), default=0)

        entry = {
            'pitch_kp': kp,
            'pitch_kv': kv,
            'kv_ratio': ratio,
            'robustness_pass': n_pass,
            'robustness_total': len(WAVE_CHALLENGES),
            'worst_pitch_deg': float(worst_pitch),
            'worst_trend': float(worst_trend),
            'details': challenge_results,
        }
        phase2_results.append(entry)

        print(f"    → {n_pass}/{len(WAVE_CHALLENGES)} passed, "
              f"worst_pitch={worst_pitch:.1f}°, worst_trend={worst_trend:.2f}")

    # ── Final ranking: most robust, then softest ──
    phase2_results.sort(key=lambda v: (-v['robustness_pass'],
                                        v['pitch_kp'],
                                        v['worst_pitch_deg']))

    print(f"\n{'=' * 70}")
    print("FINAL RANKING (sorted: most robust → softest)")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<5} {'kp':<8} {'ratio':<7} {'kv':<8} "
          f"{'pass':<7} {'worst_p°':<10} {'trend':<7}")
    for i, v in enumerate(phase2_results):
        print(f"  {i+1:<5} {v['pitch_kp']:<8.4f} {v['kv_ratio']:<7.1f} "
              f"{v['pitch_kv']:<8.5f} "
              f"{v['robustness_pass']}/{v['robustness_total']:<5} "
              f"{v['worst_pitch_deg']:<10.1f} {v['worst_trend']:<7.2f}")

    # ── Find the winner: softest that passed all challenges ──
    full_pass = [v for v in phase2_results if v['robustness_pass'] == len(WAVE_CHALLENGES)]

    if full_pass:
        # Among full-pass, pick softest (lowest kp)
        winner = min(full_pass, key=lambda v: v['pitch_kp'])
        print(f"\n  WINNER (softest with full robustness):")
    else:
        # Fallback: most passes, then softest
        winner = phase2_results[0]
        print(f"\n  WINNER (best available, {winner['robustness_pass']}/{winner['robustness_total']} robust):")

    print(f"    pitch_kp = {winner['pitch_kp']}")
    print(f"    pitch_kv = {winner['pitch_kv']}")
    print(f"    kv_ratio = {winner['kv_ratio']}")
    print(f"    worst pitch deflection: {winner['worst_pitch_deg']:.1f}°")
    print(f"    worst trend: {winner['worst_trend']:.2f}")

    print(f"\n  Update configs/farms_controller.yaml:")
    print(f"    impedance:")
    print(f"      pitch_kp: {winner['pitch_kp']}")
    print(f"      pitch_kv: {winner['pitch_kv']}")

    # Also identify the fold boundary
    failed_kps = set()
    for r in phase1_results:
        if not r['survived']:
            failed_kps.add(r['pitch_kp'])
    survived_kps = set(r['pitch_kp'] for r in survived_p1)
    boundary_kp = min(survived_kps) if survived_kps else None

    print(f"\n  Fold boundary analysis:")
    print(f"    Survived pitch_kp values: {sorted(survived_kps)}")
    print(f"    Failed pitch_kp values:   {sorted(failed_kps - survived_kps)}")
    if boundary_kp:
        print(f"    Minimum surviving kp: {boundary_kp}")
        print(f"    Safety margin: winner kp / boundary kp = "
              f"{winner['pitch_kp']/boundary_kp:.1f}×")

    # ── Save ──
    out_path = args.output or os.path.join(OUTPUT_DIR, f"pitch_soft_{timestamp}.json")
    output = {
        'timestamp': timestamp,
        'duration': args.duration,
        'fixed_params': {
            'body_kp': FIXED_BODY_KP, 'body_kv': FIXED_BODY_KV,
            'roll_kp': FIXED_ROLL_KP, 'roll_kv': FIXED_ROLL_KV,
        },
        'pitch_kp_values': PITCH_KP_VALUES,
        'kv_ratio_values': KV_RATIO_VALUES,
        'thresholds': {
            'MAX_PITCH_DEG': MAX_PITCH_DEG,
            'MAX_ROLL_DEG': MAX_ROLL_DEG,
        },
        'phase1_survived': len(survived_p1),
        'phase1_total': total,
        'phase1_results': phase1_results,
        'phase2_results': [
            {k: v for k, v in vr.items() if k != 'details'}
            for vr in phase2_results
        ],
        'winner': {
            'pitch_kp': winner['pitch_kp'],
            'pitch_kv': winner['pitch_kv'],
            'kv_ratio': winner['kv_ratio'],
        },
        'fold_boundary_kp': boundary_kp,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
