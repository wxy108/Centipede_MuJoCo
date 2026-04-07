#!/usr/bin/env python3
"""Phase 1: Sweep body_kp × body_kv on rough terrain (1s sims)"""
import os, sys, re, time, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from impedance_controller import ImpedanceTravelingWaveController, load_config

XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR  = os.path.join(BASE, "outputs", "optimization", "impedance")

ROUGH_PNG = None
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and 'low0.0060' in d:
        ROUGH_PNG = os.path.abspath(os.path.join(dp, "1.png"))
        break


def patch_terrain(xml, tpng, zmax):
    m = re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', xml)
    if m: xml = xml.replace(f'file="{m.group(1)}"', f'file="{tpng}"')
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{zmax:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    return re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, xml)


def run_sim(xml_text, duration, kp, kv):
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    try: model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e: return None, str(e)
    data = mujoco.MjData(model)
    ctrl = ImpedanceTravelingWaveController(model, CONFIG_PATH, body_kp=kp, body_kv=kv)

    pitch_ids, roll_ids = [], []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm): pitch_ids.append(i)
        if nm and 'joint_roll_' in nm: roll_ids.append(i)

    n_steps = int(duration / model.opt.timestep)
    rec_dt = 0.01; last_rec = -np.inf
    T, CZ, BJ, PJ, RJ = [], [], [], [], []
    buckled = False

    for s in range(n_steps):
        ctrl.step(model, data); mujoco.mj_step(model, data)
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "diverge"
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55): buckled = True
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; T.append(data.time)
            CZ.append(ctrl.idx.com_pos(data)[2])
            BJ.append(np.array([ctrl.idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))
            RJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in roll_ids]))

    return {'time': np.array(T), 'com_z': np.array(CZ), 'body_jnt': np.array(BJ),
            'pitch_jnt': np.array(PJ), 'roll_jnt': np.array(RJ)}, \
           ("buckled" if buckled else "ok")


def metrics(d, warmup=0.3):
    t = d['time']; m = t > warmup
    if m.sum() < 5: return None
    cz = d['com_z'][m]; ba = d['body_jnt'][m]; pp = d['pitch_jnt'][m]; rr = d['roll_jnt'][m]
    cfg = load_config(CONFIG_PATH); bw = cfg['body_wave']
    om = 2*np.pi*bw['frequency']; nw = bw['wave_number']; sp = bw['speed']; N = 18
    bc = np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:, i] = bw['amplitude'] * np.sin(om * d['time'][m] - 2*np.pi*nw*sp*i/N)
    trk = float(np.sqrt(np.mean((ba - bc)**2)))
    return {
        'trk_deg': np.degrees(trk), 'z_min': float(np.min(cz))*1000,
        'z_std': float(np.std(cz))*1000,
        'p_max': float(np.degrees(np.max(np.abs(pp)))),
        'p_std': float(np.degrees(np.std(pp))),
        'r_max': float(np.degrees(np.max(np.abs(rr)))),
        'r_std': float(np.degrees(np.std(rr))),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(XML_PATH, 'r') as f: base_xml = f.read()
    xml_rough = patch_terrain(base_xml, ROUGH_PNG, 0.04)

    # Reduced sweep: focus on the interesting range
    KP_VALS = [1.0, 2.0, 5.0, 10.0, 20.0]
    KV_VALS = [0.01, 0.05, 0.1, 0.2]
    configs = [{'kp': kp, 'kv': kv} for kp in KP_VALS for kv in KV_VALS]

    print(f"\n{'='*72}")
    print(f"PHASE 1: Impedance gain sweep — {len(configs)} configs (1s rough terrain)")
    print(f"{'='*72}\n")

    results = []
    for i, c in enumerate(configs):
        t0 = time.time()
        d, status = run_sim(xml_rough, 1.0, c['kp'], c['kv'])
        wall = time.time() - t0
        entry = {'kp': c['kp'], 'kv': c['kv'], 'status': status}
        if d is not None:
            met = metrics(d)
            if met: entry.update(met)
        results.append(entry)
        flag = "✗" if status != "ok" else ("⚠" if entry.get('trk_deg',99) > 5 else "✓")
        print(f"  [{i+1:2d}/{len(configs)}] kp={c['kp']:5.1f} kv={c['kv']:5.3f} → "
              f"{status:8s} {flag}  trk={entry.get('trk_deg',99):.2f}° "
              f"pMax={entry.get('p_max',99):.1f}° rMax={entry.get('r_max',99):.1f}° ({wall:.0f}s)")

    # Score and save
    ok = [r for r in results if r['status'] == 'ok' and 'trk_deg' in r]
    for r in ok:
        trk_s  = max(0, 10 - r['trk_deg'] * 2)
        soft_s = min(5, max(0, r['trk_deg'] - 0.1) * 2)
        buck_s = max(0, 10 - r['p_max'] / 5)
        pen_s  = 10 if r['z_min'] > 0.5 else 0
        comp_s = min(5, r['p_std'] / 2)
        r['score'] = trk_s + soft_s + buck_s + pen_s + comp_s
    ok.sort(key=lambda r: -r['score'])

    print(f"\n  OK: {len(ok)}/{len(results)}")
    print(f"\n  Top 10:")
    print(f"  {'#':>3} {'kp':>6} {'kv':>6} {'trk°':>6} {'pMax':>6} {'rMax':>6} {'rStd':>6} {'Score':>6}")
    for i, r in enumerate(ok[:10]):
        print(f"  {i+1:3d} {r['kp']:6.1f} {r['kv']:6.3f} {r['trk_deg']:6.2f} "
              f"{r['p_max']:6.1f} {r['r_max']:6.1f} {r['r_std']:6.2f} {r['score']:6.1f}")

    with open(os.path.join(OUTPUT_DIR, 'phase1.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x,'item') else str(x))
    print(f"\n  Saved: {OUTPUT_DIR}/phase1.json")


if __name__ == "__main__":
    main()
