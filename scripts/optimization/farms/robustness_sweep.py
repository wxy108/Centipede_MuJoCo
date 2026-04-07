"""
robustness_sweep.py — Find impedance gains robust across wave parameter ranges
================================================================================
Problem: current gains (body_kp=0.5, pitch_kp=0.01) work for body_amp=0.2 but
the centipede flips when amplitude increases to 0.6. We need gains that are
stable across realistic wave parameter ranges.

Approach:
  1. Define a grid of "challenge" wave configs (amplitude, frequency, wave_number,
     leg amplitude) that the controller must survive.
  2. For each candidate impedance gain set, run ALL challenge configs.
  3. Score = fraction of configs that survive 10s without flipping.
  4. Among surviving sets, pick the softest (lowest torque).

The key insight: higher body_amp creates larger dynamic forces → pitch/roll need
more damping. But we want to keep them as soft as possible for compliance.
The body_kp may also need to scale with amplitude.

Usage:
    python robustness_sweep.py                     # full sweep
    python robustness_sweep.py --duration 5        # quicker test
    python robustness_sweep.py --quick             # reduced grid (fast screening)
"""

import sys, os, math, json, time, argparse, copy, itertools
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "..", "..", "..")
XML_PATH   = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONFIG     = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "optimization", "compliance")

sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))
import mujoco
from impedance_controller import ImpedanceTravelingWaveController


# ═══════════════════════════════════════════════════════════════════════
# Challenge wave configurations the controller must survive
# ═══════════════════════════════════════════════════════════════════════

def get_challenge_configs(quick=False):
    """
    Returns list of dicts with wave parameters to test.
    Each config overrides fields in the YAML.
    """
    if quick:
        # Minimal set: just the extremes
        body_amps  = [0.2, 0.4, 0.6]
        freqs      = [1.0, 2.0]
        wave_nums  = [3.0]
        leg_amps_0 = [0.6]  # leg yaw amplitude
    else:
        # Full grid of realistic wave parameters
        body_amps  = [0.2, 0.3, 0.4, 0.5, 0.6]
        freqs      = [1.0, 1.5, 2.0]
        wave_nums  = [2.0, 3.0, 4.0]
        leg_amps_0 = [0.3, 0.6]

    configs = []
    for ba, f, wn, la in itertools.product(body_amps, freqs, wave_nums, leg_amps_0):
        configs.append({
            "body_amp": ba,
            "freq": f,
            "wave_num": wn,
            "leg_amp_0": la,
            "tag": f"ba{ba}_f{f}_wn{wn}_la{la}",
        })
    return configs


# ═══════════════════════════════════════════════════════════════════════
# Impedance gain candidates
# ═══════════════════════════════════════════════════════════════════════

def get_gain_candidates():
    """
    Returns list of dicts with impedance gains to test.
    We sweep body_kp, pitch_kp, and their damping ratios.
    """
    candidates = []

    # body_kp range: 0.5 (current soft) to 2.0 (stiffer for high amp)
    # pitch_kp range: 0.01 (current) to 0.1 (much stiffer)
    # damping ratio for kv = ratio * kp
    body_kps   = [0.5, 1.0, 1.5, 2.0]
    pitch_kps  = [0.01, 0.02, 0.05, 0.1]
    body_damp  = 0.2   # kv = 0.2 * kp
    pitch_damp = 0.4   # pitch_kv = 0.4 * pitch_kp

    for bkp, pkp in itertools.product(body_kps, pitch_kps):
        bkv = body_damp * bkp
        pkv = pitch_damp * pkp
        candidates.append({
            "body_kp":  bkp,
            "body_kv":  round(bkv, 4),
            "pitch_kp": pkp,
            "pitch_kv": round(pkv, 4),
            "tag": f"bkp{bkp}_pkp{pkp}",
        })
    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════

def make_config(base_cfg, wave_cfg):
    """Create a modified config dict with the given wave parameters."""
    cfg = copy.deepcopy(base_cfg)
    cfg["body_wave"]["amplitude"]   = wave_cfg["body_amp"]
    cfg["body_wave"]["frequency"]   = wave_cfg["freq"]
    cfg["body_wave"]["wave_number"] = wave_cfg["wave_num"]
    cfg["leg_wave"]["amplitudes"][0] = wave_cfg["leg_amp_0"]
    return cfg


def simulate_one(model_path, cfg, gains, duration):
    """
    Run one simulation. Returns dict with success/failure and metrics.
    """
    # Write temp config
    tmp_config = os.path.join(BASE_DIR, "configs", "_robustness_test.yaml")
    with open(tmp_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data  = mujoco.MjData(model)

        ctrl = ImpedanceTravelingWaveController(
            model, config_path=tmp_config,
            body_kp=gains["body_kp"], body_kv=gains["body_kv"],
            pitch_kp=gains["pitch_kp"], pitch_kv=gains["pitch_kv"])

        dt = model.opt.timestep
        n_steps = int(duration / dt)
        rec_interval = max(1, int(0.05 / dt))  # every 50ms (coarse, fast)

        # Track flip detection and basic metrics
        heights = []
        pitch_maxes = []
        roll_maxes = []
        com_x = []
        body_torques = []

        pitch_qpos_adr = ctrl.pitch_qpos_adr if ctrl.has_pitch else []

        # Find roll joint IDs
        roll_ids = []
        for j in range(model.njnt):
            if "roll" in model.joint(j).name and "body" in model.joint(j).name:
                roll_ids.append(j)

        flipped = False
        flip_time = None

        for step in range(n_steps):
            ctrl.step(model, data)
            mujoco.mj_step(model, data)

            if not np.isfinite(data.qpos).all():
                flipped = True
                flip_time = data.time
                break

            if step % rec_interval == 0:
                # COM height — if it drops below half initial or goes very high → flip
                h = data.subtree_com[0, 2]
                heights.append(h)
                com_x.append(data.subtree_com[0, 0])

                # Pitch angles
                if len(pitch_qpos_adr) > 0:
                    pvals = np.array([data.qpos[a] for a in pitch_qpos_adr])
                    pitch_maxes.append(np.max(np.abs(pvals)))
                else:
                    pitch_maxes.append(0.0)

                # Roll angles
                if roll_ids:
                    rvals = np.array([data.qpos[model.jnt_qposadr[j]] for j in roll_ids])
                    roll_maxes.append(np.max(np.abs(rvals)))
                else:
                    roll_maxes.append(0.0)

                # Body yaw torque
                bt = np.array([data.ctrl[ctrl.idx.body_act_ids[i]]
                               for i in range(19)])
                body_torques.append(np.sqrt(np.mean(bt ** 2)))

                # Flip detection: if body rolls > 90° or pitch > 90°
                if pitch_maxes[-1] > math.radians(90) or roll_maxes[-1] > math.radians(90):
                    flipped = True
                    flip_time = data.time
                    break

                # Height-based flip detection
                if len(heights) > 20:
                    if h < 0 or h > heights[0] * 5:
                        flipped = True
                        flip_time = data.time
                        break

        heights = np.array(heights)
        pitch_maxes = np.array(pitch_maxes)
        roll_maxes = np.array(roll_maxes)
        body_torques = np.array(body_torques)
        com_x = np.array(com_x)

        if flipped:
            return {
                "survived": False,
                "flip_time": round(flip_time, 2) if flip_time else -1,
                "pitch_max_deg": round(float(np.degrees(max(pitch_maxes))) if len(pitch_maxes) > 0 else 999, 1),
                "roll_max_deg": round(float(np.degrees(max(roll_maxes))) if len(roll_maxes) > 0 else 999, 1),
            }
        else:
            # After warmup
            warmup_idx = max(1, int(2.0 / (dt * rec_interval)))
            trq_rms = float(np.mean(body_torques[warmup_idx:])) if len(body_torques) > warmup_idx else 0
            fwd = float((com_x[-1] - com_x[0]) * 1000) if len(com_x) > 1 else 0
            return {
                "survived": True,
                "pitch_max_deg": round(float(np.degrees(np.max(pitch_maxes))), 1),
                "pitch_end_deg": round(float(np.degrees(pitch_maxes[-1])), 1),
                "roll_max_deg":  round(float(np.degrees(np.max(roll_maxes))), 1),
                "trq_rms": round(trq_rms, 6),
                "fwd_mm": round(fwd, 1),
                "height_std_mm": round(float(np.std(heights[warmup_idx:]) * 1000), 2) if len(heights) > warmup_idx else 0,
            }

    except Exception as e:
        return {"survived": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Robustness sweep: find impedance gains stable across wave parameter ranges")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration per test (default: 10s)")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced grid for fast screening")
    args = parser.parse_args()

    with open(CONFIG, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    challenges = get_challenge_configs(quick=args.quick)
    candidates = get_gain_candidates()

    print(f"Challenge configs: {len(challenges)}")
    print(f"Gain candidates:   {len(candidates)}")
    print(f"Total simulations: {len(challenges) * len(candidates)}")
    print(f"Duration per sim:  {args.duration}s")
    print(f"Estimated time:    ~{len(challenges) * len(candidates) * 150 / 60:.0f} min")
    print()

    # ── Run sweep ─────────────────────────────────────────────────────
    all_results = {}
    for gi, gains in enumerate(candidates):
        tag_g = gains["tag"]
        survive_count = 0
        total_trq = 0
        total_fwd = 0
        worst_pitch = 0
        worst_roll = 0
        details = []

        print(f"\n{'='*70}")
        print(f"  [{gi+1}/{len(candidates)}] {tag_g}  "
              f"(body_kp={gains['body_kp']}, body_kv={gains['body_kv']}, "
              f"pitch_kp={gains['pitch_kp']}, pitch_kv={gains['pitch_kv']})")
        print(f"{'='*70}")

        for ci, wave_cfg in enumerate(challenges):
            cfg = make_config(base_cfg, wave_cfg)
            t0 = time.time()
            r = simulate_one(XML_PATH, cfg, gains, args.duration)
            wall = time.time() - t0

            r["wave"] = wave_cfg["tag"]
            details.append(r)

            if r["survived"]:
                survive_count += 1
                total_trq += r.get("trq_rms", 0)
                total_fwd += r.get("fwd_mm", 0)
                worst_pitch = max(worst_pitch, r.get("pitch_max_deg", 0))
                worst_roll = max(worst_roll, r.get("roll_max_deg", 0))
                status = f"OK  pitch={r['pitch_max_deg']:5.1f}° roll={r['roll_max_deg']:5.1f}°"
            else:
                ft = r.get("flip_time", "?")
                status = f"FLIP t={ft}s pitch={r.get('pitch_max_deg','?')}° roll={r.get('roll_max_deg','?')}°"

            print(f"  [{ci+1}/{len(challenges)}] {wave_cfg['tag']:30s} → {status}  ({wall:.0f}s)")

        survive_rate = survive_count / len(challenges)
        avg_trq = total_trq / max(survive_count, 1)
        avg_fwd = total_fwd / max(survive_count, 1)

        all_results[tag_g] = {
            "gains": gains,
            "survive_count": survive_count,
            "survive_rate": round(survive_rate, 3),
            "avg_trq_rms": round(avg_trq, 6),
            "avg_fwd_mm": round(avg_fwd, 1),
            "worst_pitch_deg": round(worst_pitch, 1),
            "worst_roll_deg": round(worst_roll, 1),
            "details": details,
        }

        print(f"\n  Summary: {survive_count}/{len(challenges)} survived "
              f"({survive_rate*100:.0f}%)  "
              f"avg_trq={avg_trq:.5f}  avg_fwd={avg_fwd:.0f}mm  "
              f"worst_pitch={worst_pitch:.1f}°  worst_roll={worst_roll:.1f}°")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "robustness_sweep.json")

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = to_serializable(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nFull results saved → {out_path}")

    # ── Final ranking ─────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"  RANKING (sorted by survive_rate, then by softness = lowest avg_trq)")
    print(f"{'='*90}")
    print(f"{'body_kp':>8} {'body_kv':>8} {'pitch_kp':>9} {'pitch_kv':>9} "
          f"{'survive':>8} {'avg_trq':>10} {'avg_fwd':>8} {'w_pitch':>8} {'w_roll':>8}")
    print(f"{'-'*90}")

    ranked = sorted(all_results.values(),
                    key=lambda x: (-x["survive_rate"], x["avg_trq_rms"]))

    for r in ranked:
        g = r["gains"]
        print(f"{g['body_kp']:8.2f} {g['body_kv']:8.3f} "
              f"{g['pitch_kp']:9.3f} {g['pitch_kv']:9.4f} "
              f"{r['survive_count']:>3}/{len(challenges):>2} "
              f"{r['avg_trq_rms']:10.5f} {r['avg_fwd_mm']:8.1f} "
              f"{r['worst_pitch_deg']:8.1f} {r['worst_roll_deg']:8.1f}")

    # Highlight best
    best = ranked[0]
    bg = best["gains"]
    print(f"\n  BEST: body_kp={bg['body_kp']}, body_kv={bg['body_kv']}, "
          f"pitch_kp={bg['pitch_kp']}, pitch_kv={bg['pitch_kv']}")
    print(f"        survive={best['survive_count']}/{len(challenges)}, "
          f"avg_trq={best['avg_trq_rms']:.5f}, avg_fwd={best['avg_fwd_mm']:.0f}mm")


if __name__ == "__main__":
    main()
