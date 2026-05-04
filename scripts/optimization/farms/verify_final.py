#!/usr/bin/env python3
"""Verify winner parameters on all terrain levels."""
import os, sys, re, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'controllers', 'farms'))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS
from controller import FARMSTravelingWaveController, load_config

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
XML = os.path.join(BASE, 'models', 'farms', 'centipede.xml')
BACKUP = XML + '.compliance_backup'
CFG = os.path.join(BASE, 'configs', 'farms_controller.yaml')
TDIR = os.path.join(BASE, 'terrain', 'output')

TERRAINS = {
    'L0_flat':  (os.path.join(TDIR, 'flat_terrain.png'), 0.001),
    'L1_mild':  (os.path.join(TDIR, 'low0.0040_mid0.0020_high0.0010_s0', '1.png'), 0.010),
    'L2_mod':   (os.path.join(TDIR, 'low0.0060_mid0.0030_high0.0020_s0', '1.png'), 0.040),
    'L3_rough': (os.path.join(TDIR, 'low0.0100_mid0.0050_high0.0030_s0', '1.png'), 0.070),
}

def _pj(xml, pat, attr, val):
    def r(m):
        l = m.group(0)
        if f'{attr}="' in l:
            return re.sub(rf'{attr}="[^"]*"', f'{attr}="{val}"', l)
        return l.replace('/>', f' {attr}="{val}"/>')
    return re.sub(rf'<joint\s[^>]*name="{pat}"[^>]*/>', r, xml)

def patch(xml, tpng, zmax):
    # Solver
    xml = re.sub(r'(solref=")[^"]*(")', r'\g<1>0.01 1.5\2', xml)
    xml = re.sub(r'(solimp=")[^"]*(")', r'\g<1>0.9 0.95 0.001\2', xml)
    # Pitch joints
    xml = _pj(xml, r'joint_pitch_body_\d+', 'stiffness', '0.0001')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'damping', '0.001')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'stiffness', '0.0001')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'damping', '0.001')
    # Terrain file
    xml = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{tpng}\2', xml)
    # Terrain z_max (3rd component of size="x y z r")
    def fix_size(m):
        parts = m.group(1).split()
        if len(parts) >= 3:
            parts[2] = f"{zmax:.6g}"
        return f'size="{" ".join(parts)}"'
    xml = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', lambda m: f'{m.group(1)}{fix_size_str(m.group(2), zmax)}"', xml)
    return xml

def fix_size_str(s, zmax):
    parts = s.split()
    if len(parts) >= 3:
        parts[2] = f"{zmax:.6g}"
    return " ".join(parts)

with open(BACKUP if os.path.exists(BACKUP) else XML, 'r') as f:
    base_xml = f.read()

print(f"{'Terrain':12s}  {'Status':8s}  {'Wall':5s}  {'COM Z range':18s}  {'Z std':8s}  {'Pitch std':10s}  {'Pitch max':10s}")
print('-' * 85)

for tname, (tpng, tz) in TERRAINS.items():
    if not os.path.exists(tpng):
        print(f"{tname:12s}  {'MISSING':8s}  terrain file not found")
        continue

    xml = patch(base_xml, tpng, tz)
    with open(XML, 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    idx = FARMSModelIndex(model)
    ctrl = FARMSTravelingWaveController(model, config_path=CFG)

    pids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pids.append(i)

    dur = 5.0
    n_steps = int(dur / model.opt.timestep)
    T, CZ, PP = [], [], []
    rec_dt = 0.01
    last = -np.inf
    t0 = time.time()
    stable = True

    for s in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)
        if s % 500 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                stable = False
                break
        if data.time - last >= rec_dt - 1e-10:
            last = data.time
            T.append(data.time)
            CZ.append(idx.com_pos(data)[2])
            PP.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pids]))

    wall = time.time() - t0

    if not stable:
        print(f"{tname:12s}  {'UNSTABLE':8s}")
        continue

    T = np.array(T)
    CZ = np.array(CZ)
    PP = np.array(PP)
    m = T > 1.0

    z_min = np.min(CZ[m]) * 1000
    z_max = np.max(CZ[m]) * 1000
    z_std = np.std(CZ[m]) * 1000
    p_std = np.degrees(np.std(PP[m]))
    p_max = np.degrees(np.max(np.abs(PP[m])))

    print(f"{tname:12s}  {'OK':8s}  {wall:5.0f}s  [{z_min:6.1f},{z_max:6.1f}]mm     "
          f"{z_std:6.2f}mm  {p_std:8.2f} deg  {p_max:8.2f} deg")

print("\nDone. Winner params: pitch_k=1e-4, pitch_d=1e-3, solref=(0.01, 1.5)")
