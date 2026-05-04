"""
tune_pitch_impedance.py — Sweep pitch impedance kp/kv with gravity compensation
=================================================================================
Tests different pitch_kp values with online gravity compensation enabled.
The gravity comp term (from data.qfrc_bias) cancels static gravitational load,
so pitch_kp only needs to handle dynamic disturbances and terrain conformity.

This means pitch_kp can be MUCH softer than passive spring stiffness
while still maintaining stability.

Usage:
    python tune_pitch_impedance.py                    # run full sweep at 10s
    python tune_pitch_impedance.py --duration 5       # shorter test
    python tune_pitch_impedance.py --kp 0.01 0.02     # specific values only
"""

import sys, os, math, json, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "..", "..", "..")
CONFIG     = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "optimization", "compliance")

sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))
import mujoco
from impedance_controller import ImpedanceTravelingWaveController


def simulate(model_path, duration, config_path, pitch_kp, pitch_kv):
    """Run simulation with given pitch impedance params, return metrics."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    ctrl = ImpedanceTravelingWaveController(
        model, config_path=config_path,
        pitch_kp=pitch_kp, pitch_kv=pitch_kv)

    dt = model.opt.timestep
    n_steps = int(duration / dt)

    # Find pitch joint IDs
    pitch_jnt_ids = ctrl.idx.pitch_jnt_ids
    pitch_qpos_adr = ctrl.pitch_qpos_adr

    # Recording
    rec_interval = max(1, int(0.01 / dt))
    times, pitch_maxes, pitch_all_list = [], [], []
    heights, com_x = [], []
    pitch_torques = []  # commanded pitch torques

    t0 = time.time()
    diverged = False
    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all():
            diverged = True
            break

        if step % rec_interval == 0:
            times.append(data.time)
            pvals = np.array([data.qpos[a] for a in pitch_qpos_adr])
            pitch_maxes.append(np.max(np.abs(pvals)))
            pitch_all_list.append(pvals.copy())
            heights.append(data.subtree_com[0, 2])
            com_x.append(data.subtree_com[0, 0])
            # Record pitch torques
            pt = np.array([data.ctrl[a] for a in ctrl.idx.pitch_act_ids])
            pitch_torques.append(pt.copy())

    wall = time.time() - t0

    if diverged:
        return {"diverged": True, "wall_time": round(wall, 1)}

    times = np.array(times)
    pitch_maxes = np.array(pitch_maxes)
    pitch_all = np.array(pitch_all_list)
    heights = np.array(heights)
    com_x = np.array(com_x)
    pitch_torques = np.array(pitch_torques)

    # After warmup (2s)
    warmup_mask = times >= 2.0
    if warmup_mask.sum() < 10:
        warmup_mask = np.ones(len(times), dtype=bool)

    # Check if pitch is still growing at end
    last_quarter = times >= (times[-1] * 0.75)
    first_quarter = (times >= 2.0) & (times <= times[-1] * 0.25 + 2.0)
    if first_quarter.sum() > 5 and last_quarter.sum() > 5:
        early_max = np.max(np.abs(pitch_all[first_quarter]))
        late_max  = np.max(np.abs(pitch_all[last_quarter]))
        growing = bool(late_max > early_max * 1.5)
    else:
        growing = False

    return {
        "diverged":         False,
        "wall_time":        round(wall, 1),
        "pitch_max_deg":    round(float(np.degrees(np.max(pitch_maxes))), 3),
        "pitch_rms_deg":    round(float(np.degrees(np.sqrt(np.mean(pitch_all[warmup_mask] ** 2)))), 3),
        "pitch_end_max_deg": round(float(np.degrees(np.max(np.abs(pitch_all[-1])))), 3),
        "pitch_growing":    growing,
        "height_init_mm":   round(float(heights[0]) * 1000, 2),
        "height_final_mm":  round(float(heights[-1]) * 1000, 2),
        "height_drop_mm":   round(float((heights[0] - heights[-1]) * 1000), 3),
        "height_std_mm":    round(float(np.std(heights[warmup_mask]) * 1000), 3),
        "fwd_mm":           round(float((com_x[-1] - com_x[0]) * 1000), 2),
        "pitch_trq_rms":    round(float(np.sqrt(np.mean(pitch_torques[warmup_mask] ** 2))), 6),
        "pitch_trq_max":    round(float(np.max(np.abs(pitch_torques[warmup_mask]))), 6),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep pitch impedance kp with gravity compensation")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--kp", nargs="+", type=float, default=None,
                        help="Specific kp values to test")
    parser.add_argument("--damping-ratio", type=float, default=0.4,
                        help="kv = ratio * kp (default: 0.4)")
    args = parser.parse_args()

    model_path = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")

    # Default sweep: very soft to moderate
    # With gravity comp, even very low kp should be stable
    kp_values = args.kp or [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    results = []
    for kp in kp_values:
        kv = args.damping_ratio * kp
        print(f"\n{'='*60}")
        print(f"  pitch_kp={kp}, pitch_kv={kv:.4f}, duration={args.duration}s")
        print(f"{'='*60}")

        r = simulate(model_path, args.duration, CONFIG, kp, kv)
        r["pitch_kp"] = kp
        r["pitch_kv"] = round(kv, 6)
        results.append(r)

        if r["diverged"]:
            print(f"  DIVERGED after {r['wall_time']}s")
        else:
            growing_str = " ⚠ GROWING" if r["pitch_growing"] else ""
            print(f"  pitch max:    {r['pitch_max_deg']:.2f} deg{growing_str}")
            print(f"  pitch end:    {r['pitch_end_max_deg']:.2f} deg")
            print(f"  pitch RMS:    {r['pitch_rms_deg']:.2f} deg")
            print(f"  pitch trq:    RMS={r['pitch_trq_rms']:.6f}  peak={r['pitch_trq_max']:.6f}")
            print(f"  height drop:  {r['height_drop_mm']:.3f} mm")
            print(f"  forward:      {r['fwd_mm']:.1f} mm")
            print(f"  wall time:    {r['wall_time']}s")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "pitch_impedance_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'kp':>8} {'kv':>8} {'max°':>8} {'end°':>8} {'rms°':>8} "
          f"{'trq_rms':>10} {'h_drop':>8} {'fwd_mm':>8} {'grow':>6} {'status':>8}")
    print(f"{'='*100}")
    for r in results:
        if r["diverged"]:
            print(f"{r['pitch_kp']:8.4f} {r['pitch_kv']:8.4f} {'--':>8} {'--':>8} "
                  f"{'--':>8} {'--':>10} {'--':>8} {'--':>8} {'--':>6} {'DIV':>8}")
        else:
            g = "YES" if r["pitch_growing"] else "no"
            print(f"{r['pitch_kp']:8.4f} {r['pitch_kv']:8.4f} "
                  f"{r['pitch_max_deg']:8.2f} {r['pitch_end_max_deg']:8.2f} "
                  f"{r['pitch_rms_deg']:8.2f} {r['pitch_trq_rms']:10.6f} "
                  f"{r['height_drop_mm']:8.3f} {r['fwd_mm']:8.1f} "
                  f"{g:>6} {'OK':>8}")

    print(f"\nLook for: low pitch angles + pitch NOT growing + stable height + good forward progress")
    print(f"The softest kp where pitch_growing=no is your answer.")


if __name__ == "__main__":
    main()
