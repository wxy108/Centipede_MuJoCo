#!/usr/bin/env python3
"""
sweep_compliance.py — Sweep impedance + roll stiffness for max compliance
==========================================================================
Criteria:
  1. Forward distance traveled (more = better locomotion)
  2. Torque RMS on body yaw actuators (lower = more compliant)
  3. Stability (no divergence, no buckling)

Score = forward_distance_mm - lambda_torque * torque_rms
        (maximize forward progress, penalize high effort)

Sweep parameters:
  - body_kp: softer range (0.3 .. 2.0)
  - body_kv: (0.01 .. 0.1)
  - body_roll_k: stiffer range (5e-3 .. 5e-2)
  - leg_roll_k:  stiffer range (1e-2 .. 1e-1)
  - damping ratio fixed at 0.4 for all roll joints
"""

import os, sys, re, math, time, json, copy
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF

XML_PATH = os.path.join(BASE, "models", "farms", "centipede.xml")
CFG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
OUT_DIR  = os.path.join(BASE, "outputs", "optimization", "compliance")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Terrain paths for testing ──
TERRAIN_ROUGH = os.path.join(BASE, "terrain", "output",
                             "low0.0060_mid0.0030_high0.0020_s0", "1.png")
TERRAIN_FLAT  = os.path.join(BASE, "terrain", "output", "flat_terrain.png")

# ── Sweep grid ──
# Softer kp values (current winner = 2.0, try much softer)
KP_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
# kv values
KV_VALUES = [0.01, 0.03, 0.05, 0.1]
# Body roll stiffness (current = 1e-3, make MUCH stiffer)
BODY_ROLL_K = [5e-3, 1e-2, 2e-2, 5e-2]
# Leg roll stiffness (current = 5e-3, make stiffer)
LEG_ROLL_K  = [1e-2, 2e-2, 5e-2, 1e-1]
DAMP_RATIO = 0.4  # damping = ratio * stiffness


def patch_xml_roll(xml_str, body_rk, leg_rk):
    """Patch roll stiffness/damping in XML string."""
    body_rd = body_rk * DAMP_RATIO
    leg_rd  = leg_rk * DAMP_RATIO

    # Body roll joints (joint_roll_body_N and joint_roll_passive_N)
    def fix_body_roll(m):
        s = m.group(0)
        s = re.sub(r'stiffness="[^"]*"', f'stiffness="{body_rk:.6e}"', s)
        s = re.sub(r'damping="[^"]*"',   f'damping="{body_rd:.6e}"', s)
        return s
    xml_str = re.sub(r'<joint\s+name="joint_roll_body_\d+"[^/]*/>', fix_body_roll, xml_str)
    xml_str = re.sub(r'<joint\s+name="joint_roll_passive_\d+"[^/]*/>', fix_body_roll, xml_str)

    # Leg roll joints
    def fix_leg_roll(m):
        s = m.group(0)
        s = re.sub(r'stiffness="[^"]*"', f'stiffness="{leg_rk:.6e}"', s)
        s = re.sub(r'damping="[^"]*"',   f'damping="{leg_rd:.6e}"', s)
        return s
    xml_str = re.sub(r'<joint\s+name="joint_roll_leg_\d+_[LR]"[^/]*/>', fix_leg_roll, xml_str)

    return xml_str


def patch_xml_terrain(xml_str, terrain_png, z_max=0.04):
    """Patch terrain path and z_max."""
    m = re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', xml_str)
    if m:
        xml_str = xml_str.replace(f'file="{m.group(1)}"', f'file="{terrain_png}"')
    # z_max
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3:
            parts[2] = f"{z_max:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    xml_str = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, xml_str)
    return xml_str


def run_sim(xml_str, body_kp, body_kv, duration=2.0):
    """
    Run simulation, return metrics dict or None if unstable.
    """
    # Write patched XML
    with open(XML_PATH, 'w') as f:
        f.write(xml_str)
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return {"status": f"xml_error: {e}"}

    data = mujoco.MjData(model)
    ctrl = ImpedanceTravelingWaveController(model, CFG_PATH,
                                            body_kp=body_kp, body_kv=body_kv)
    cfg = load_config(CFG_PATH)
    bw = cfg['body_wave']
    amp = bw['amplitude']; om = 2*math.pi*bw['frequency']
    nw = bw['wave_number']; sp = bw['speed']

    # Find pitch joint IDs for buckling check
    pitch_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)

    n_steps = int(duration / model.opt.timestep)
    rec_dt = 0.01; last_rec = -np.inf
    T, FWD_X, TORQUES, TRK_ERR = [], [], [], []
    buckled = False

    for s in range(n_steps):
        ctrl.step(model, data)

        # Record BEFORE physics step
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time
            t = data.time
            T.append(t)
            FWD_X.append(ctrl.idx.com_pos(data)[0])

            # Commanded torques
            trqs = np.array([data.ctrl[ctrl.idx.body_act_ids[i]]
                             for i in range(N_BODY_JOINTS)])
            TORQUES.append(trqs)

            # Tracking error (commanded position vs actual)
            for i in range(N_BODY_JOINTS):
                phase = om * t - 2*math.pi*nw*sp*i / max(N_BODY_JOINTS-1, 1)
                target = amp * math.sin(phase)
                actual = ctrl.idx.body_joint_pos(data, i+1)
                TRK_ERR.append(abs(target - actual))

        mujoco.mj_step(model, data)

        # Stability checks
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return {"status": "diverged"}
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True

    T = np.array(T); FWD_X = np.array(FWD_X); TORQUES = np.array(TORQUES)

    # Metrics after warmup (t > 0.5s)
    m = T > 0.5
    if m.sum() < 10:
        return {"status": "too_short"}

    fwd_dist_mm = (FWD_X[m][-1] - FWD_X[m][0]) * 1000
    torque_rms  = float(np.sqrt(np.mean(TORQUES[m]**2)))
    torque_max  = float(np.max(np.abs(TORQUES[m])))
    trk_rms_deg = float(np.degrees(np.sqrt(np.mean(np.array(TRK_ERR)**2))))

    return {
        "status":       "buckled" if buckled else "ok",
        "fwd_mm":       round(fwd_dist_mm, 2),
        "torque_rms":   round(torque_rms, 6),
        "torque_max":   round(torque_max, 6),
        "trk_rms_deg":  round(trk_rms_deg, 3),
    }


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1: kp × kv sweep (fixed roll at stiffer defaults)
# ═══════════════════════════════════════════════════════════════════════

def phase1():
    """Sweep kp × kv with stiffer roll defaults on rough terrain."""
    print("\n" + "="*72)
    print("  PHASE 1: kp × kv sweep (body_roll=1e-2, leg_roll=2e-2)")
    print("="*72)

    with open(XML_PATH, 'r') as f:
        base_xml = f.read()

    # Use stiffer roll defaults for this phase
    xml = patch_xml_roll(base_xml, body_rk=1e-2, leg_rk=2e-2)
    xml = patch_xml_terrain(xml, TERRAIN_ROUGH, z_max=0.04)

    results = []
    for kp in KP_VALUES:
        for kv in KV_VALUES:
            t0 = time.time()
            m = run_sim(xml, kp, kv, duration=2.0)
            wall = time.time() - t0
            m['kp'] = kp; m['kv'] = kv
            m['wall'] = round(wall, 1)

            if m['status'] == 'ok':
                # Score: forward distance - torque penalty
                # Normalize: torque_rms typically 0.01-0.5 Nm, fwd typically 5-50 mm
                m['score'] = round(m['fwd_mm'] - 50 * m['torque_rms'], 2)
            else:
                m['score'] = -999

            results.append(m)
            status_str = m['status']
            if status_str == 'ok':
                print(f"  kp={kp:4.1f} kv={kv:4.2f} | fwd={m['fwd_mm']:7.1f}mm "
                      f"trq={m['torque_rms']:.4f}Nm trk={m['trk_rms_deg']:.1f}° "
                      f"score={m['score']:6.1f} | {wall:.0f}s")
            else:
                print(f"  kp={kp:4.1f} kv={kv:4.2f} | {status_str:>8s} | {wall:.0f}s")

    # Sort by score
    results.sort(key=lambda r: r['score'], reverse=True)

    with open(os.path.join(OUT_DIR, "phase1.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Top 5:")
    for r in results[:5]:
        print(f"    kp={r['kp']:4.1f} kv={r['kv']:4.2f} → "
              f"fwd={r.get('fwd_mm','?'):>7}mm trq={r.get('torque_rms','?')} "
              f"score={r['score']}")

    return results[:5]  # top 5


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2: Roll stiffness sweep for top kp/kv combos
# ═══════════════════════════════════════════════════════════════════════

def phase2(top_kp_kv):
    """Sweep roll stiffness for top kp/kv combos on rough terrain."""
    print("\n" + "="*72)
    print("  PHASE 2: Roll stiffness sweep for top kp/kv combos")
    print("="*72)

    with open(XML_PATH, 'r') as f:
        base_xml = f.read()

    results = []
    # Only use top 3 from phase1
    for combo in top_kp_kv[:3]:
        kp = combo['kp']; kv = combo['kv']
        for brk in BODY_ROLL_K:
            for lrk in LEG_ROLL_K:
                # Leg roll should be >= body roll
                if lrk < brk:
                    continue
                xml = patch_xml_roll(base_xml, body_rk=brk, leg_rk=lrk)
                xml = patch_xml_terrain(xml, TERRAIN_ROUGH, z_max=0.04)

                t0 = time.time()
                m = run_sim(xml, kp, kv, duration=2.0)
                wall = time.time() - t0
                m['kp'] = kp; m['kv'] = kv
                m['body_roll_k'] = brk; m['leg_roll_k'] = lrk
                m['wall'] = round(wall, 1)

                if m['status'] == 'ok':
                    m['score'] = round(m['fwd_mm'] - 50 * m['torque_rms'], 2)
                else:
                    m['score'] = -999

                results.append(m)
                if m['status'] == 'ok':
                    print(f"  kp={kp:4.1f} kv={kv:4.2f} brk={brk:.0e} lrk={lrk:.0e} | "
                          f"fwd={m['fwd_mm']:7.1f}mm trq={m['torque_rms']:.4f} "
                          f"trk={m['trk_rms_deg']:.1f}° score={m['score']:6.1f} | {wall:.0f}s")
                else:
                    print(f"  kp={kp:4.1f} kv={kv:4.2f} brk={brk:.0e} lrk={lrk:.0e} | "
                          f"{m['status']:>8s} | {wall:.0f}s")

    results.sort(key=lambda r: r['score'], reverse=True)
    with open(os.path.join(OUT_DIR, "phase2.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Top 5:")
    for r in results[:5]:
        print(f"    kp={r['kp']:4.1f} kv={r['kv']:4.2f} brk={r['body_roll_k']:.0e} "
              f"lrk={r['leg_roll_k']:.0e} → fwd={r['fwd_mm']}mm trq={r['torque_rms']} "
              f"score={r['score']}")

    return results[:3]


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3: Extended eval on flat + rough for top combos
# ═══════════════════════════════════════════════════════════════════════

def phase3(top_combos):
    """Extended 3s eval on flat and rough terrain."""
    print("\n" + "="*72)
    print("  PHASE 3: Extended eval (3s) on flat + rough for top combos")
    print("="*72)

    with open(XML_PATH, 'r') as f:
        base_xml = f.read()

    results = []
    for combo in top_combos:
        kp = combo['kp']; kv = combo['kv']
        brk = combo['body_roll_k']; lrk = combo['leg_roll_k']
        xml_base = patch_xml_roll(base_xml, body_rk=brk, leg_rk=lrk)

        for terrain_name, terrain_png, z_max in [
            ("flat", TERRAIN_FLAT, 0.001),
            ("rough", TERRAIN_ROUGH, 0.04),
        ]:
            xml = patch_xml_terrain(xml_base, terrain_png, z_max)
            t0 = time.time()
            m = run_sim(xml, kp, kv, duration=3.0)
            wall = time.time() - t0
            m.update({
                'kp': kp, 'kv': kv,
                'body_roll_k': brk, 'leg_roll_k': lrk,
                'terrain': terrain_name, 'wall': round(wall, 1),
            })
            if m['status'] == 'ok':
                m['score'] = round(m['fwd_mm'] - 50 * m['torque_rms'], 2)
            else:
                m['score'] = -999
            results.append(m)
            print(f"  kp={kp:4.1f} kv={kv:4.2f} brk={brk:.0e} lrk={lrk:.0e} "
                  f"{terrain_name:>6s} | fwd={m.get('fwd_mm','?'):>7}mm "
                  f"trq={m.get('torque_rms','?')} trk={m.get('trk_rms_deg','?')}° "
                  f"score={m.get('score','?')} | {wall:.0f}s")

    with open(os.path.join(OUT_DIR, "phase3.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    top_kp_kv = phase1()
    top_combos = phase2(top_kp_kv)
    phase3(top_combos)
    print("\n  All results saved to:", OUT_DIR)
