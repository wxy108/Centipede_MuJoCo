"""
tune_pitch.py — Find stable pitch stiffness/damping for 10s simulations
========================================================================
Sweeps pitch stiffness values and measures max pitch angle, pitch std,
body height stability, and forward locomotion over 10s.

Physics: pitch spring must resist gravity torque on downstream segments.
Estimated minimum stiffness for <5° sag: ~0.004, for <2° sag: ~0.01.
"""

import sys, os, math, json, time, copy
import numpy as np
import xml.etree.ElementTree as ET

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "..", "..", "..")
XML_PATH   = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONFIG     = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "optimization", "compliance")

sys.path.insert(0, os.path.join(BASE_DIR, "controllers", "farms"))
import mujoco
from impedance_controller import ImpedanceTravelingWaveController
from kinematics import FARMSModelIndex, N_BODY_JOINTS

DURATION = 10.0  # seconds — full verification


def patch_pitch(xml_str, stiffness, damping):
    """Set all pitch joint stiffness/damping in XML string."""
    root = ET.fromstring(xml_str)
    count = 0
    for jnt in root.iter("joint"):
        name = jnt.get("name", "")
        if "pitch" in name:
            jnt.set("stiffness", f"{stiffness:.6e}")
            jnt.set("damping",   f"{damping:.6e}")
            count += 1
    return ET.tostring(root, encoding="unicode"), count


def simulate(xml_str, duration, config_path):
    """Run simulation, return metrics dict."""
    # Must provide assets dict or load from file path for meshdir resolution
    xml_path_temp = os.path.join(BASE_DIR, "models", "farms", "_pitch_test.xml")
    with open(xml_path_temp, "w") as fw:
        fw.write(xml_str)
    model = mujoco.MjModel.from_xml_path(xml_path_temp)
    data  = mujoco.MjData(model)

    ctrl = ImpedanceTravelingWaveController(model, config_path=config_path)
    idx  = ctrl.idx

    dt = model.opt.timestep
    n_steps = int(duration / dt)

    # Find pitch joint IDs
    pitch_ids = []
    for j in range(model.njnt):
        if "pitch" in model.joint(j).name:
            pitch_ids.append(j)

    # Recording arrays
    rec_interval = max(1, int(0.01 / dt))  # every 10ms
    times = []
    pitch_angles = []  # max |pitch| across all joints at each step
    pitch_all    = []  # all pitch angles (T, n_pitch)
    heights      = []  # COM z
    com_x        = []  # COM x for forward progress

    t0 = time.time()
    diverged = False
    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Check divergence
        if not np.isfinite(data.qpos).all():
            diverged = True
            break

        if step % rec_interval == 0:
            times.append(data.time)
            pvals = np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids])
            pitch_angles.append(np.max(np.abs(pvals)))
            pitch_all.append(pvals.copy())
            heights.append(data.subtree_com[0, 2])  # root body COM z
            com_x.append(data.subtree_com[0, 0])

    wall = time.time() - t0

    if diverged:
        return {"diverged": True, "wall_time": wall}

    times = np.array(times)
    pitch_angles = np.array(pitch_angles)
    pitch_all = np.array(pitch_all)
    heights = np.array(heights)
    com_x = np.array(com_x)

    # After warmup (2s)
    warmup_mask = times >= 2.0
    if warmup_mask.sum() < 10:
        warmup_mask = np.ones(len(times), dtype=bool)

    return {
        "diverged":       False,
        "wall_time":      round(wall, 1),
        "pitch_max_deg":  round(float(np.degrees(np.max(pitch_angles))), 3),
        "pitch_rms_deg":  round(float(np.degrees(np.sqrt(np.mean(pitch_all[warmup_mask] ** 2)))), 3),
        "pitch_end_max_deg": round(float(np.degrees(np.max(np.abs(pitch_all[-1])))), 3),
        "height_init":    round(float(heights[0]) * 1000, 2),  # mm
        "height_final":   round(float(heights[-1]) * 1000, 2),
        "height_drop_mm": round(float((heights[0] - heights[-1]) * 1000), 3),
        "height_std_mm":  round(float(np.std(heights[warmup_mask]) * 1000), 3),
        "fwd_mm":         round(float((com_x[-1] - com_x[0]) * 1000), 2),
    }


def main():
    with open(XML_PATH, "r") as f:
        base_xml = f.read()

    # Sweep pitch stiffness with damping ratio = 0.4 * stiffness
    # (same ratio as roll joints)
    # Full sweep: find the softest stable pitch stiffness at 10s
    # Old value (1e-4) collapsed; gravity analysis says minimum ~0.004 for <5°
    stiffness_values = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05]
    damping_ratio = 0.4

    results = []
    for k in stiffness_values:
        d_val = damping_ratio * k
        xml_str, n_patched = patch_pitch(base_xml, k, d_val)
        print(f"\n{'='*60}")
        print(f"  pitch_k={k}, pitch_d={d_val:.4f}  ({n_patched} joints patched)")
        print(f"{'='*60}")

        r = simulate(xml_str, DURATION, CONFIG)
        r["pitch_k"] = k
        r["pitch_d"] = round(d_val, 6)
        results.append(r)

        if r["diverged"]:
            print(f"  DIVERGED after {r['wall_time']}s")
        else:
            print(f"  pitch max:    {r['pitch_max_deg']:.2f} deg")
            print(f"  pitch end:    {r['pitch_end_max_deg']:.2f} deg")
            print(f"  pitch RMS:    {r['pitch_rms_deg']:.2f} deg")
            print(f"  height drop:  {r['height_drop_mm']:.3f} mm")
            print(f"  height std:   {r['height_std_mm']:.3f} mm")
            print(f"  forward:      {r['fwd_mm']:.1f} mm")
            print(f"  wall time:    {r['wall_time']}s")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "pitch_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'pitch_k':>10} {'pitch_d':>10} {'max_deg':>10} {'end_deg':>10} "
          f"{'rms_deg':>10} {'h_drop_mm':>10} {'fwd_mm':>10} {'status':>10}")
    print(f"{'='*80}")
    for r in results:
        if r["diverged"]:
            print(f"{r['pitch_k']:10.4f} {r['pitch_d']:10.4f} {'--':>10} {'--':>10} "
                  f"{'--':>10} {'--':>10} {'--':>10} {'DIVERGED':>10}")
        else:
            print(f"{r['pitch_k']:10.4f} {r['pitch_d']:10.4f} "
                  f"{r['pitch_max_deg']:10.2f} {r['pitch_end_max_deg']:10.2f} "
                  f"{r['pitch_rms_deg']:10.2f} {r['height_drop_mm']:10.3f} "
                  f"{r['fwd_mm']:10.1f} {'OK':>10}")


if __name__ == "__main__":
    main()
