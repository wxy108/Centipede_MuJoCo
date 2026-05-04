#!/usr/bin/env python3
"""
verify_winner.py — 5s verification of winning Option B parameters
Tests on flat, mild rough, and aggressive rough terrain.
"""
import os, sys, re, time, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS
from controller import FARMSTravelingWaveController, load_config

XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP  = XML_PATH + ".optionb_backup"
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")

# Winner params
PK, PD = 1e-3, 5e-4
STC, SDR = 0.01, 1.5

def patch_xml(base, tpng, zmax):
    x = base
    x = re.sub(r'solref="[\d.\-e]+ [\d.\-e]+"', f'solref="{STC} {SDR}"', x)
    for pat in [r'joint_pitch_body_\d+', r'joint_pitch_body_\d+']:
        x = re.sub(rf'(<joint\s[^>]*name="{pat}"[^>]*?)stiffness="[^"]*"',
                    rf'\g<1>stiffness="{PK:.6e}"', x)
        x = re.sub(rf'(<joint\s[^>]*name="{pat}"[^>]*?)damping="[^"]*"',
                    rf'\g<1>damping="{PD:.6e}"', x)
    x = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{tpng}\2', x)
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{zmax:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    x = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, x)
    return x

def run_sim(xml_text, duration):
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    idx = FARMSModelIndex(model)
    cfg = load_config(CONFIG_PATH)
    ctrl = FARMSTravelingWaveController.__new__(FARMSTravelingWaveController)
    bw, lw = cfg['body_wave'], cfg['leg_wave']
    ctrl.body_amp = float(bw['amplitude']); ctrl.freq = float(bw['frequency'])
    ctrl.n_wave = float(bw['wave_number']); ctrl.speed = float(bw['speed'])
    ctrl.omega = 2*np.pi*ctrl.freq
    ctrl.leg_amps = np.array(lw['amplitudes'], dtype=float)
    ctrl.leg_phase_offsets = np.array(lw['phase_offsets'], dtype=float)
    ctrl.leg_dc_offsets = np.array(lw['dc_offsets'], dtype=float)
    ctrl.active_dofs = set(lw['active_dofs']); ctrl.idx = idx

    pitch_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)

    n_steps = int(duration / model.opt.timestep)
    rec_dt = 0.01; last_rec = -np.inf
    T, CZ, BJ, PJ = [], [], [], []
    max_pitch = 0

    for s in range(n_steps):
        ctrl.step(model, data); mujoco.mj_step(model, data)
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "DIVERGED"
            for jid in pitch_ids:
                p = abs(data.qpos[model.jnt_qposadr[jid]])
                if p > max_pitch: max_pitch = p
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; T.append(data.time)
            CZ.append(idx.com_pos(data)[2])
            BJ.append(np.array([idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))

    return {'time': np.array(T), 'com_z': np.array(CZ),
            'body_jnt': np.array(BJ), 'pitch_jnt': np.array(PJ),
            'max_pitch_deg': np.degrees(max_pitch)}, "OK"

def analyze(d, label):
    t = d['time']; m = t > 1.0
    cz = d['com_z'][m]; pp = d['pitch_jnt'][m]; ba = d['body_jnt'][m]
    cfg = load_config(CONFIG_PATH); bw = cfg['body_wave']
    om = 2*np.pi*bw['frequency']; nw = bw['wave_number']; sp = bw['speed']; N = 18
    bc = np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:, i] = bw['amplitude'] * np.sin(om * d['time'][m] - 2*np.pi*nw*sp*i/N)
    trk = np.degrees(np.sqrt(np.mean((ba - bc)**2)))

    print(f"\n  {label}:")
    print(f"    Tracking RMS:    {trk:.4f}° {'PASS' if trk < 1.0 else 'FAIL'}")
    print(f"    COM Z min:       {np.min(cz)*1000:.2f} mm {'PASS' if np.min(cz) > 0 else 'FAIL'}")
    print(f"    COM Z std:       {np.std(cz)*1000:.3f} mm")
    print(f"    Pitch max:       {d['max_pitch_deg']:.1f}° {'PASS' if d['max_pitch_deg'] < 45 else 'WARN'}")
    print(f"    Pitch std:       {np.degrees(np.std(pp)):.2f}°")
    print(f"    Pitch mean|abs|: {np.degrees(np.mean(np.abs(pp))):.2f}°")
    return trk < 1.0 and np.min(cz) > 0 and d['max_pitch_deg'] < 45

def main():
    with open(XML_BACKUP, 'r') as f: base_xml = f.read()

    print(f"\n{'='*72}")
    print(f"  OPTION B VERIFICATION — pk={PK:.0e}, pd={PD:.0e}, solref={STC} {SDR}")
    print(f"{'='*72}")

    # Build terrain configs
    terrains = [('Flat (z_max=0.001)', FLAT_TERRAIN, 0.001)]

    # Find rough terrains
    for d in sorted(os.listdir(TERRAIN_DIR)):
        dp = os.path.join(TERRAIN_DIR, d)
        if os.path.isdir(dp) and 'low0.0060' in d:
            terrains.append((f'Rough ({d}, z=0.04)', os.path.join(dp, "1.png"), 0.04))
            # Also test with higher z_max
            terrains.append((f'Rough ({d}, z=0.06)', os.path.join(dp, "1.png"), 0.06))
            break

    all_pass = True
    for label, tpng, tz in terrains:
        print(f"\n  Testing: {label}")
        xml = patch_xml(base_xml, tpng, tz)
        t0 = time.time()
        d, status = run_sim(xml, 5.0)
        wall = time.time() - t0
        print(f"    Status: {status} ({wall:.0f}s)")
        if d is None:
            print(f"    FAILED: {status}")
            all_pass = False
            continue
        ok = analyze(d, label)
        if not ok: all_pass = False

    print(f"\n{'='*72}")
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'='*72}\n")

if __name__ == "__main__":
    main()
