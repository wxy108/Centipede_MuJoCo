#!/usr/bin/env python3
"""
sweep_impedance.py — Optimize impedance gains + roll joint stiffness
=====================================================================
Sweeps:
  body_kp:    impedance proportional gain (how stiff yaw tracking is)
  body_kv:    impedance velocity damping
  body_roll_k: body roll stiffness
  leg_roll_k:  leg roll stiffness

Criteria (same as pitch tuning):
  1. Stability (no divergence, no buckling)
  2. Tracking accuracy (yaw wave should still be followed, but softer)
  3. No terrain penetration
  4. Terrain compliance (body conforms to rough terrain)

Phase 1: Sweep body_kp × body_kv on rough terrain (1s) — find stable tracking
Phase 2: Sweep roll stiffness for top kp/kv combos (1s)
Phase 3: Extended eval top 5 on flat+rough (3s)
"""
import os, sys, re, time, json, shutil
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
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")

# Sandbox absolute path for terrain
SANDBOX_TERRAIN_DIR = os.path.abspath(TERRAIN_DIR)

ROUGH_PNG = None
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and 'low0.0060' in d:
        ROUGH_PNG = os.path.join(dp, "1.png")
        break


def patch_terrain(xml, terrain_png, z_max):
    """Patch terrain path and z_max."""
    m = re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', xml)
    if m:
        xml = xml.replace(f'file="{m.group(1)}"', f'file="{terrain_png}"')
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{z_max:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    xml = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, xml)
    return xml


def patch_roll_stiffness(xml, body_roll_k, body_roll_d, leg_roll_k, leg_roll_d):
    """Patch roll joint stiffness/damping."""
    for pat in [r'joint_roll_body_\d+', r'joint_roll_body_\d+']:
        xml = re.sub(
            rf'(<joint\s[^>]*name="{pat}"[^>]*?)stiffness="[^"]*"',
            rf'\g<1>stiffness="{body_roll_k:.6e}"', xml)
        xml = re.sub(
            rf'(<joint\s[^>]*name="{pat}"[^>]*?)damping="[^"]*"',
            rf'\g<1>damping="{body_roll_d:.6e}"', xml)
    xml = re.sub(
        rf'(<joint\s[^>]*name="joint_roll_leg_\d+_[LR]"[^>]*?)stiffness="[^"]*"',
        rf'\g<1>stiffness="{leg_roll_k:.6e}"', xml)
    xml = re.sub(
        rf'(<joint\s[^>]*name="joint_roll_leg_\d+_[LR]"[^>]*?)damping="[^"]*"',
        rf'\g<1>damping="{leg_roll_d:.6e}"', xml)
    return xml


def run_sim(xml_text, duration, body_kp, body_kv):
    """Run simulation with impedance controller, return data + status."""
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return None, f"xml_error: {e}"

    data = mujoco.MjData(model)
    ctrl = ImpedanceTravelingWaveController(
        model, CONFIG_PATH, body_kp=body_kp, body_kv=body_kv)

    # Find pitch + roll joint IDs
    pitch_ids, roll_ids = [], []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)
        if nm and 'joint_roll_' in nm:
            roll_ids.append(i)

    n_steps = int(duration / model.opt.timestep)
    rec_dt = 0.01; last_rec = -np.inf
    T, CZ, BJ, PJ, RJ = [], [], [], [], []
    buckled = False

    for s in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "diverge"
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; T.append(data.time)
            CZ.append(ctrl.idx.com_pos(data)[2])
            BJ.append(np.array([ctrl.idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))
            RJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in roll_ids]))

    return {
        'time': np.array(T), 'com_z': np.array(CZ),
        'body_jnt': np.array(BJ), 'pitch_jnt': np.array(PJ),
        'roll_jnt': np.array(RJ),
    }, ("buckled" if buckled else "ok")


def metrics(d, warmup=0.3):
    t = d['time']; m = t > warmup
    if m.sum() < 5: return None
    cz = d['com_z'][m]; ba = d['body_jnt'][m]
    pp = d['pitch_jnt'][m]; rr = d['roll_jnt'][m]

    cfg = load_config(CONFIG_PATH); bw = cfg['body_wave']
    om = 2*np.pi*bw['frequency']; nw = bw['wave_number']; sp = bw['speed']; N = 18
    bc = np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:, i] = bw['amplitude'] * np.sin(om * d['time'][m] - 2*np.pi*nw*sp*i/N)
    trk = float(np.sqrt(np.mean((ba - bc)**2)))

    return {
        'trk_deg': np.degrees(trk),
        'z_min': float(np.min(cz)) * 1000,
        'z_std': float(np.std(cz)) * 1000,
        'p_max': float(np.degrees(np.max(np.abs(pp)))),
        'p_std': float(np.degrees(np.std(pp))),
        'r_max': float(np.degrees(np.max(np.abs(rr)))),
        'r_std': float(np.degrees(np.std(rr))),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(XML_PATH, 'r') as f: base_xml = f.read()

    # Use sandbox absolute paths for terrain
    rough_png = os.path.abspath(ROUGH_PNG) if ROUGH_PNG else None
    flat_png  = os.path.abspath(FLAT_TERRAIN)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: Sweep body_kp × body_kv on rough terrain
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("PHASE 1: Impedance gain sweep (1s rough terrain)")
    print(f"{'='*72}")

    KP_VALS = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    KV_VALS = [0.01, 0.02, 0.05, 0.1, 0.2]
    configs = [{'kp': kp, 'kv': kv} for kp in KP_VALS for kv in KV_VALS]
    print(f"  {len(configs)} kp×kv configs\n")

    xml_rough = patch_terrain(base_xml, rough_png, 0.04)
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

    # Score phase 1
    ok = [r for r in results if r['status'] == 'ok' and 'trk_deg' in r]
    print(f"\n  OK: {len(ok)} / {len(results)}")

    for r in ok:
        # Want: good tracking (but not too stiff), no buckling, compliance
        trk_s  = max(0, 10 - r['trk_deg'] * 2)     # sweet spot ~1-3 deg
        too_stiff = max(0, r['trk_deg'] - 0.1) * 2  # bonus for being softer
        too_stiff = min(5, too_stiff)
        buck_s = max(0, 10 - r['p_max'] / 5)
        pen_s  = 10 if r['z_min'] > 0.5 else 0
        comp_s = min(5, r['p_std'] / 2)
        r['score'] = trk_s + too_stiff + buck_s + pen_s + comp_s

    ok.sort(key=lambda r: -r['score'])
    print(f"\n  Top 10 kp×kv:")
    print(f"  {'#':>3} {'kp':>6} {'kv':>6} {'trk°':>6} {'pMax':>6} {'rMax':>6} {'Score':>6}")
    for i, r in enumerate(ok[:10]):
        print(f"  {i+1:3d} {r['kp']:6.1f} {r['kv']:6.3f} {r['trk_deg']:6.2f} "
              f"{r['p_max']:6.1f} {r['r_max']:6.1f} {r['score']:6.1f}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: Sweep roll stiffness for top 3 kp/kv combos
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("PHASE 2: Roll stiffness sweep (1s rough terrain)")
    print(f"{'='*72}")

    top3_kpkv = ok[:3]
    BODY_ROLL_K = [1e-3, 3e-3, 5e-3, 1e-2]
    LEG_ROLL_K  = [5e-3, 1e-2, 2e-2, 5e-2]
    # Damping = 0.4× stiffness (moderate)
    DAMP_RATIO = 0.4

    roll_results = []
    total = len(top3_kpkv) * len(BODY_ROLL_K) * len(LEG_ROLL_K)
    print(f"  {total} configs (3 kp/kv × 4 body_roll × 4 leg_roll)\n")

    idx = 0
    for kpkv in top3_kpkv:
        for brk in BODY_ROLL_K:
            for lrk in LEG_ROLL_K:
                idx += 1
                brd = brk * DAMP_RATIO
                lrd = lrk * DAMP_RATIO
                xml = patch_roll_stiffness(xml_rough, brk, brd, lrk, lrd)

                t0 = time.time()
                d, status = run_sim(xml, 1.0, kpkv['kp'], kpkv['kv'])
                wall = time.time() - t0

                entry = {
                    'kp': kpkv['kp'], 'kv': kpkv['kv'],
                    'brk': brk, 'lrk': lrk, 'status': status
                }
                if d is not None:
                    met = metrics(d)
                    if met: entry.update(met)
                roll_results.append(entry)

                if idx % 8 == 0:
                    print(f"  [{idx:3d}/{total}] kp={kpkv['kp']:5.1f} brk={brk:.0e} lrk={lrk:.0e} → "
                          f"{status:8s} trk={entry.get('trk_deg',99):.2f}° "
                          f"rMax={entry.get('r_max',99):.1f}° ({wall:.0f}s)")

    # Score phase 2
    ok2 = [r for r in roll_results if r['status'] == 'ok' and 'trk_deg' in r]
    for r in ok2:
        trk_s  = max(0, 10 - r['trk_deg'] * 2)
        soft_s = min(5, max(0, r['trk_deg'] - 0.1) * 2)
        buck_s = max(0, 10 - r['p_max'] / 5)
        pen_s  = 10 if r['z_min'] > 0.5 else 0
        comp_s = min(5, r['p_std'] / 2) + min(3, r['r_std'] / 1)  # roll compliance too
        r['score'] = trk_s + soft_s + buck_s + pen_s + comp_s

    ok2.sort(key=lambda r: -r['score'])
    print(f"\n  Top 10:")
    print(f"  {'#':>3} {'kp':>5} {'kv':>5} {'brk':>7} {'lrk':>7} {'trk°':>6} "
          f"{'pMax':>6} {'rMax':>6} {'rStd':>6} {'Score':>6}")
    for i, r in enumerate(ok2[:10]):
        print(f"  {i+1:3d} {r['kp']:5.1f} {r['kv']:5.3f} {r['brk']:7.0e} {r['lrk']:7.0e} "
              f"{r['trk_deg']:6.2f} {r['p_max']:6.1f} {r['r_max']:6.1f} "
              f"{r['r_std']:6.2f} {r['score']:6.1f}")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3: Extended eval top 5 on flat + rough (3s)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("PHASE 3: Extended eval (3s flat + rough)")
    print(f"{'='*72}")

    top5 = ok2[:5]
    ext = []

    for rank, r in enumerate(top5):
        print(f"\n  Candidate {rank+1}: kp={r['kp']:.1f} kv={r['kv']:.3f} "
              f"brk={r['brk']:.0e} lrk={r['lrk']:.0e}")

        for tname, tpng, tz in [('flat', flat_png, 0.001), ('rough', rough_png, 0.04)]:
            xml = patch_terrain(base_xml, tpng, tz)
            xml = patch_roll_stiffness(xml, r['brk'], r['brk']*DAMP_RATIO,
                                       r['lrk'], r['lrk']*DAMP_RATIO)
            t0 = time.time()
            d, status = run_sim(xml, 3.0, r['kp'], r['kv'])
            wall = time.time() - t0

            entry = {'rank': rank+1, 'terrain': tname, 'status': status,
                     'kp': r['kp'], 'kv': r['kv'], 'brk': r['brk'], 'lrk': r['lrk']}
            if d is not None:
                met = metrics(d, warmup=1.0)
                if met: entry.update(met)
            ext.append(entry)
            print(f"    {tname:6s}: {status:8s} trk={entry.get('trk_deg',99):.2f}° "
                  f"pMax={entry.get('p_max',99):.1f}° rStd={entry.get('r_std',0):.2f}° "
                  f"zStd={entry.get('z_std',0):.2f}mm ({wall:.0f}s)")

    # ══════════════════════════════════════════════════════════════════
    #  FINAL: pick winner
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("FINAL SELECTION")
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
        if fl.get('status')=='buckled' or ro.get('status')=='buckled':
            sc = -100
        else:
            trk   = max(0, 10 - fl['trk_deg']*2)
            soft  = min(5, max(0, fl['trk_deg']-0.1)*2)
            nobuk = max(0, 10 - max(fl.get('p_max',0), ro.get('p_max',0))/5)
            nopen = 10 if min(fl.get('z_min',0), ro.get('z_min',0)) > 0.5 else 0
            zcomp = min(5, max(0, ro.get('z_std',0) - fl.get('z_std',0)))
            rcomp = min(5, max(0, ro.get('r_std',0) - fl.get('r_std',0)))
            sc = trk + soft + nobuk + nopen + zcomp + rcomp
        print(f"  Candidate {k}: score={sc:.1f}")
        if sc > best_sc: best_sc = sc; best_key = k

    if best_key is None:
        print("  No winner!"); return

    w = cands[best_key]['rough']
    print(f"\n  ★ WINNER: Candidate {best_key}")
    print(f"    body_kp:    {w['kp']:.2f}")
    print(f"    body_kv:    {w['kv']:.4f}")
    print(f"    body_roll_k: {w['brk']:.2e}")
    print(f"    leg_roll_k:  {w['lrk']:.2e}")

    # Save results
    out = {
        'winner': {
            'body_kp': w['kp'], 'body_kv': w['kv'],
            'body_roll_k': w['brk'], 'body_roll_d': w['brk']*DAMP_RATIO,
            'leg_roll_k': w['lrk'], 'leg_roll_d': w['lrk']*DAMP_RATIO,
        },
        'phase1_top10': [{kk: (float(v) if isinstance(v, (np.floating,np.integer)) else v)
                          for kk,v in r.items()} for r in ok[:10]],
        'phase3': [{kk: (float(v) if isinstance(v, (np.floating,np.integer)) else v)
                    for kk,v in e.items()} for e in ext],
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results: {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
