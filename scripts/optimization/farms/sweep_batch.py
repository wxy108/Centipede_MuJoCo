#!/usr/bin/env python3
"""
sweep_batch.py — Run a batch of pitch_k × d_ratio configs
Usage: python sweep_batch.py --batch 1   (configs 0-17)
       python sweep_batch.py --batch 2   (configs 18-34)
       python sweep_batch.py --batch ext  (extended eval of top candidates)
"""
import os, sys, re, time, json, shutil
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP  = XML_PATH + ".optionb_backup"
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR  = os.path.join(BASE, "outputs", "optimization", "option_b")
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")

ROUGH_PNG, ROUGH_ZMAX = None, 0.04
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and 'low0.0060' in d:
        ROUGH_PNG = os.path.join(dp, "1.png")
        break

STC, SDR = 0.01, 1.5  # fixed solref

def patch_xml(base, pk, pd, stc, sdr, tpng, zmax):
    x = base
    x = re.sub(r'solref="[\d.\-e]+ [\d.\-e]+"', f'solref="{stc} {sdr}"', x)
    for pat in [r'joint_pitch_body_\d+', r'joint_pitch_body_\d+']:
        x = re.sub(rf'(<joint\s[^>]*name="{pat}"[^>]*?)stiffness="[^"]*"',
                    rf'\g<1>stiffness="{pk:.6e}"', x)
        x = re.sub(rf'(<joint\s[^>]*name="{pat}"[^>]*?)damping="[^"]*"',
                    rf'\g<1>damping="{pd:.6e}"', x)
    x = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{tpng}\2', x)
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{zmax:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    x = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, x)
    return x

def run_sim(xml_text, duration):
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    try: model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e: return None, str(e)
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
    buckled = False

    for s in range(n_steps):
        ctrl.step(model, data); mujoco.mj_step(model, data)
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "diverge"
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; T.append(data.time)
            CZ.append(idx.com_pos(data)[2])
            BJ.append(np.array([idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))

    return {'time': np.array(T), 'com_z': np.array(CZ),
            'body_jnt': np.array(BJ), 'pitch_jnt': np.array(PJ)}, \
           ("buckled" if buckled else "ok")

def metrics(d, warmup=0.5):
    t = d['time']; m = t > warmup
    if m.sum() < 5: return None
    cz = d['com_z'][m]; pp = d['pitch_jnt'][m]; ba = d['body_jnt'][m]
    cfg = load_config(CONFIG_PATH); bw = cfg['body_wave']
    om = 2*np.pi*bw['frequency']; nw = bw['wave_number']; sp = bw['speed']; N = 18
    bc = np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:, i] = bw['amplitude'] * np.sin(om * d['time'][m] - 2*np.pi*nw*sp*i/N)
    trk = float(np.sqrt(np.mean((ba - bc)**2)))
    return {
        'trk_deg': np.degrees(trk), 'z_min': float(np.min(cz))*1000,
        'z_max': float(np.max(cz))*1000, 'z_std': float(np.std(cz))*1000,
        'p_std': float(np.degrees(np.std(pp))),
        'p_max': float(np.degrees(np.max(np.abs(pp)))),
        'p_mean': float(np.degrees(np.mean(np.abs(pp)))),
    }

PK_VALS = [3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]
DR_VALS = [0.5, 1.0, 2.0, 5.0, 10.0]

def all_configs():
    configs = []
    for pk in PK_VALS:
        for dr in DR_VALS:
            configs.append({'pk': pk, 'pd': pk*dr, 'dr': dr})
    return configs

def run_batch(batch_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(XML_BACKUP): shutil.copy2(XML_PATH, XML_BACKUP)
    with open(XML_BACKUP, 'r') as f: base_xml = f.read()

    configs = all_configs()
    if batch_id == 1:
        subset = configs[:18]
        label = "Batch 1 (configs 0-17)"
    elif batch_id == 2:
        subset = configs[18:]
        label = "Batch 2 (configs 18-34)"
    else:
        print("Invalid batch"); return

    print(f"\n{'='*72}")
    print(f"  {label}: {len(subset)} configs, 1s rough terrain sims")
    print(f"{'='*72}\n")

    results = []
    for i, c in enumerate(subset):
        xml = patch_xml(base_xml, c['pk'], c['pd'], STC, SDR, ROUGH_PNG, ROUGH_ZMAX)
        t0 = time.time()
        d, status = run_sim(xml, 1.0)
        wall = time.time() - t0

        entry = {'pk': c['pk'], 'pd': c['pd'], 'dr': c['dr'], 'status': status}
        if d is not None:
            met = metrics(d, warmup=0.3)
            if met: entry.update(met)

        results.append(entry)
        flag = "✗" if status != "ok" else ("⚠" if entry.get('p_max',0) > 40 else "✓")
        print(f"  [{i+1:2d}/{len(subset)}] pk={c['pk']:.0e} d/k={c['dr']:4.1f} → "
              f"{status:8s} {flag}  trk={entry.get('trk_deg',99):.3f}° "
              f"pMax={entry.get('p_max',99):.1f}° zMin={entry.get('z_min',-99):.1f}mm ({wall:.0f}s)")

    outfile = os.path.join(OUTPUT_DIR, f'batch{batch_id}.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\n  Saved: {outfile}")
    return results

def run_extended():
    """Load batch results, pick top 5, run 3s on flat+rough."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(XML_BACKUP, 'r') as f: base_xml = f.read()

    # Load all batch results
    all_results = []
    for b in [1, 2]:
        fp = os.path.join(OUTPUT_DIR, f'batch{b}.json')
        if os.path.exists(fp):
            with open(fp) as f: all_results.extend(json.load(f))

    ok = [r for r in all_results if r['status'] == 'ok' and 'trk_deg' in r]
    print(f"\nLoaded {len(all_results)} results, {len(ok)} non-buckling\n")

    if not ok:
        ok = [r for r in all_results if 'trk_deg' in r]
        ok.sort(key=lambda r: r.get('p_max', 999))
        ok = ok[:5]
        print("No non-buckling; using lowest p_max configs")

    # Score
    for r in ok:
        trk_s  = max(0, 10 - r['trk_deg'] * 10)
        buck_s = max(0, 10 - r['p_max'] / 5)
        pen_s  = 10 if r['z_min'] > 0.5 else 0
        comp_s = min(5, r['p_std'] / 2)
        z_s    = min(5, r['z_std'])
        r['score'] = trk_s + buck_s + pen_s + comp_s + z_s

    ok.sort(key=lambda r: -r['score'])

    print("Top candidates:")
    print(f"  {'#':>3} {'pk':>8} {'d/k':>5} {'trk°':>6} {'pMax':>6} {'pStd':>6} {'zMin':>6} {'zStd':>6} {'Score':>6}")
    for i, r in enumerate(ok[:10]):
        print(f"  {i+1:3d} {r['pk']:8.1e} {r['dr']:5.1f} {r['trk_deg']:6.3f} "
              f"{r['p_max']:6.1f} {r['p_std']:6.2f} {r['z_min']:6.1f} {r['z_std']:6.2f} {r['score']:6.1f}")

    # Extended eval top 5
    top5 = ok[:5]
    ext = []
    print(f"\n{'='*72}")
    print("Extended evaluation (3s) — flat + rough")
    print(f"{'='*72}")

    for rank, r in enumerate(top5):
        print(f"\n  Candidate {rank+1}: pk={r['pk']:.1e} d/k={r['dr']:.1f}")
        for tname, tpng, tz in [('flat', FLAT_TERRAIN, 0.001), ('rough', ROUGH_PNG, ROUGH_ZMAX)]:
            xml = patch_xml(base_xml, r['pk'], r['pk']*r['dr'], STC, SDR, tpng, tz)
            t0 = time.time()
            d, status = run_sim(xml, 3.0)
            wall = time.time() - t0
            entry = {'rank': rank+1, 'terrain': tname, 'status': status,
                     'pk': r['pk'], 'pd': r['pk']*r['dr'], 'dr': r['dr']}
            if d is not None:
                met = metrics(d, warmup=1.0)
                if met: entry.update(met)
            ext.append(entry)
            print(f"    {tname:6s}: {status:8s} trk={entry.get('trk_deg',99):.3f}° "
                  f"pMax={entry.get('p_max',99):.1f}° zStd={entry.get('z_std',0):.2f}mm ({wall:.0f}s)")

    # Pick winner
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
            trk   = max(0, 10 - fl['trk_deg']*10)
            nobuk = max(0, 10 - max(fl.get('p_max',0), ro.get('p_max',0))/5)
            nopen = 10 if min(fl.get('z_min',0), ro.get('z_min',0)) > 0.5 else 0
            zcomp = min(5, max(0, ro.get('z_std',0) - fl.get('z_std',0)))
            pcomp = min(5, max(0, ro.get('p_std',0) - fl.get('p_std',0)))
            sc = trk + nobuk + nopen + zcomp + pcomp
        print(f"  Candidate {k}: score={sc:.1f}")
        if sc > best_sc: best_sc = sc; best_key = k

    if best_key is None: print("  No winner!"); return

    w = cands[best_key]['rough']
    print(f"\n  ★ WINNER: Candidate {best_key}")
    print(f"    pitch_k:   {w['pk']:.2e}")
    print(f"    pitch_d:   {w['pd']:.2e}")
    print(f"    d/k ratio: {w['dr']:.1f}")
    print(f"    solref:    {STC} {SDR}")

    # Apply to XML with Windows terrain path
    win_xml = patch_xml(
        base_xml, w['pk'], w['pd'], STC, SDR,
        "C:/Users/wxy22/Documents/Centipede_MUJOCO-main/terrain/output/low0.0060_mid0.0030_high0.0020_s0/1.png",
        0.04)
    with open(XML_PATH, 'w') as f: f.write(win_xml)
    print(f"\n  Applied to {XML_PATH}")

    # Save
    out = {'winner': {'pitch_k': w['pk'], 'pitch_d': w['pd'], 'd_ratio': w['dr'],
                       'solref_tc': STC, 'solref_dr': SDR},
           'extended': [{kk: (float(v) if isinstance(v,(np.floating,np.integer)) else v)
                         for kk,v in e.items()} for e in ext]}
    with open(os.path.join(OUTPUT_DIR, 'final_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results: {OUTPUT_DIR}/final_results.json")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, help="1, 2, or ext")
    args = p.parse_args()

    if args.batch == "ext":
        run_extended()
    else:
        run_batch(int(args.batch))
