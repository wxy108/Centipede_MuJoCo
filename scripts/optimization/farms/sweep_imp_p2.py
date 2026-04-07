#!/usr/bin/env python3
"""Phase 2+3: Roll stiffness sweep + extended eval for top kp/kv combos."""
import os, sys, re, time, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS
from impedance_controller import ImpedanceTravelingWaveController, load_config

XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR  = os.path.join(BASE, "outputs", "optimization", "impedance")
FLAT_PNG    = os.path.abspath(os.path.join(TERRAIN_DIR, "flat_terrain.png"))

ROUGH_PNG = None
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and 'low0.0060' in d:
        ROUGH_PNG = os.path.abspath(os.path.join(dp, "1.png")); break


def patch_terrain(xml, tpng, zmax):
    m = re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', xml)
    if m: xml = xml.replace(f'file="{m.group(1)}"', f'file="{tpng}"')
    def fix_size(m):
        p = m.group(2).split()
        if len(p) >= 3: p[2] = f"{zmax:.6g}"
        return f'{m.group(1)}{" ".join(p)}"'
    return re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, xml)

def patch_roll(xml, brk, brd, lrk, lrd):
    for pat in [r'joint_roll_body_\d+', r'joint_roll_body_\d+']:
        xml = re.sub(rf'(<joint\s[^>]*name="{pat}"[^>]*?)stiffness="[^"]*"',
                      rf'\g<1>stiffness="{brk:.6e}"', xml)
        xml = re.sub(rf'(<joint\s[^>]*name="{pat}"[^>]*?)damping="[^"]*"',
                      rf'\g<1>damping="{brd:.6e}"', xml)
    xml = re.sub(rf'(<joint\s[^>]*name="joint_roll_leg_\d+_[LR]"[^>]*?)stiffness="[^"]*"',
                  rf'\g<1>stiffness="{lrk:.6e}"', xml)
    xml = re.sub(rf'(<joint\s[^>]*name="joint_roll_leg_\d+_[LR]"[^>]*?)damping="[^"]*"',
                  rf'\g<1>damping="{lrd:.6e}"', xml)
    return xml

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
            'pitch_jnt': np.array(PJ), 'roll_jnt': np.array(RJ)}, ("buckled" if buckled else "ok")

def metrics(d, warmup=0.5):
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

    # Top 3 from Phase 1
    TOP3 = [
        {'kp': 5.0,  'kv': 0.2},    # trk=1.98°, nice compliance
        {'kp': 2.0,  'kv': 0.05},   # trk=1.26°
        {'kp': 1.0,  'kv': 0.05},   # trk=2.44°, softest
    ]

    # ── PHASE 2: Roll stiffness sweep ──
    print(f"\n{'='*72}")
    print("PHASE 2: Roll stiffness sweep (1s rough terrain)")
    print(f"{'='*72}")

    BODY_ROLL_K = [1e-3, 5e-3, 1e-2]
    LEG_ROLL_K  = [5e-3, 1e-2, 5e-2]
    DR = 0.4  # damping ratio

    xml_rough = patch_terrain(base_xml, ROUGH_PNG, 0.04)
    results = []
    total = len(TOP3) * len(BODY_ROLL_K) * len(LEG_ROLL_K)
    idx = 0
    for kpkv in TOP3:
        for brk in BODY_ROLL_K:
            for lrk in LEG_ROLL_K:
                idx += 1
                xml = patch_roll(xml_rough, brk, brk*DR, lrk, lrk*DR)
                t0 = time.time()
                d, status = run_sim(xml, 1.0, kpkv['kp'], kpkv['kv'])
                wall = time.time() - t0
                entry = {'kp': kpkv['kp'], 'kv': kpkv['kv'], 'brk': brk, 'lrk': lrk, 'status': status}
                if d is not None:
                    met = metrics(d, 0.3)
                    if met: entry.update(met)
                results.append(entry)
                print(f"  [{idx:2d}/{total}] kp={kpkv['kp']:4.1f} brk={brk:.0e} lrk={lrk:.0e} → "
                      f"trk={entry.get('trk_deg',99):.2f}° r={entry.get('r_std',0):.2f}° ({wall:.0f}s)")

    # Score
    ok = [r for r in results if r['status'] == 'ok' and 'trk_deg' in r]
    for r in ok:
        trk_s = max(0, 10 - r['trk_deg']*2)
        soft_s = min(5, max(0, r['trk_deg']-0.1)*2)
        buck_s = max(0, 10 - r['p_max']/5)
        pen_s = 10 if r['z_min'] > 0.5 else 0
        comp_s = min(5, r['p_std']/2) + min(3, r['r_std']/1)
        r['score'] = trk_s + soft_s + buck_s + pen_s + comp_s
    ok.sort(key=lambda r: -r['score'])

    print(f"\n  Top 10:")
    print(f"  {'#':>3} {'kp':>5} {'kv':>5} {'brk':>7} {'lrk':>7} {'trk':>6} {'rStd':>6} {'Score':>6}")
    for i, r in enumerate(ok[:10]):
        print(f"  {i+1:3d} {r['kp']:5.1f} {r['kv']:5.3f} {r['brk']:7.0e} {r['lrk']:7.0e} "
              f"{r['trk_deg']:6.2f} {r['r_std']:6.2f} {r['score']:6.1f}")

    # ── PHASE 3: Extended eval top 3 ──
    print(f"\n{'='*72}")
    print("PHASE 3: Extended eval (3s flat + rough)")
    print(f"{'='*72}")

    top3 = ok[:3]
    ext = []
    for rank, r in enumerate(top3):
        print(f"\n  Cand {rank+1}: kp={r['kp']:.1f} kv={r['kv']:.3f} brk={r['brk']:.0e} lrk={r['lrk']:.0e}")
        for tname, tpng, tz in [('flat', FLAT_PNG, 0.001), ('rough', ROUGH_PNG, 0.04)]:
            xml = patch_terrain(base_xml, tpng, tz)
            xml = patch_roll(xml, r['brk'], r['brk']*DR, r['lrk'], r['lrk']*DR)
            t0 = time.time()
            d, status = run_sim(xml, 3.0, r['kp'], r['kv'])
            wall = time.time() - t0
            entry = {'rank': rank+1, 'terrain': tname, 'status': status,
                     'kp': r['kp'], 'kv': r['kv'], 'brk': r['brk'], 'lrk': r['lrk']}
            if d is not None:
                met = metrics(d, 1.0)
                if met: entry.update(met)
            ext.append(entry)
            print(f"    {tname:6s}: trk={entry.get('trk_deg',99):.2f}° rStd={entry.get('r_std',0):.2f}° "
                  f"zStd={entry.get('z_std',0):.2f}mm ({wall:.0f}s)")

    # Pick winner
    print(f"\n{'='*72}")
    print("WINNER")
    print(f"{'='*72}")
    cands = {}
    for e in ext:
        k = e['rank']
        if k not in cands: cands[k] = {}
        cands[k][e['terrain']] = e

    best_key, best_sc = None, -999
    for k, cs in cands.items():
        fl, ro = cs.get('flat',{}), cs.get('rough',{})
        if not fl.get('trk_deg') or not ro.get('trk_deg'): continue
        trk = max(0, 10 - fl['trk_deg']*2)
        soft = min(5, max(0, fl['trk_deg']-0.1)*2)
        nobuk = max(0, 10 - max(fl.get('p_max',0), ro.get('p_max',0))/5)
        nopen = 10 if min(fl.get('z_min',0), ro.get('z_min',0)) > 0.5 else 0
        zcomp = min(5, max(0, ro.get('z_std',0) - fl.get('z_std',0)))
        rcomp = min(5, max(0, ro.get('r_std',0) - fl.get('r_std',0)))
        sc = trk + soft + nobuk + nopen + zcomp + rcomp
        print(f"  Cand {k}: score={sc:.1f} (trk_flat={fl['trk_deg']:.2f}° trk_rough={ro['trk_deg']:.2f}°)")
        if sc > best_sc: best_sc = sc; best_key = k

    w = cands[best_key]['rough']
    print(f"\n  ★ WINNER: Candidate {best_key}")
    print(f"    body_kp:     {w['kp']:.2f}")
    print(f"    body_kv:     {w['kv']:.4f}")
    print(f"    body_roll_k: {w['brk']:.2e}")
    print(f"    body_roll_d: {w['brk']*DR:.2e}")
    print(f"    leg_roll_k:  {w['lrk']:.2e}")
    print(f"    leg_roll_d:  {w['lrk']*DR:.2e}")

    out = {'winner': {'body_kp': w['kp'], 'body_kv': w['kv'],
                       'body_roll_k': w['brk'], 'body_roll_d': w['brk']*DR,
                       'leg_roll_k': w['lrk'], 'leg_roll_d': w['lrk']*DR},
           'phase2_top10': [{kk:(float(v) if hasattr(v,'item') else v) for kk,v in r.items()} for r in ok[:10]],
           'phase3': [{kk:(float(v) if hasattr(v,'item') else v) for kk,v in e.items()} for e in ext]}
    with open(os.path.join(OUTPUT_DIR, 'final.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR}/final.json")

if __name__ == "__main__":
    main()
