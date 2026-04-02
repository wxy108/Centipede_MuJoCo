#!/usr/bin/env python3
"""
sweep_fast.py — Fast Option B sweep: pitch_k × d_ratio with fixed solref
=========================================================================
Phase 1: 35 configs on rough terrain (1.5s each) → find non-buckling sweet spot
Phase 2: Top 5 on flat+rough (3s each) → pick winner
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

# Find rough terrain
ROUGH_PNG, ROUGH_ZMAX = None, 0.04
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and 'low0.0060' in d:
        ROUGH_PNG = os.path.join(dp, "1.png")
        break

# ── XML patching ──────────────────────────────────────────────────────
def patch_xml(base, pk, pd, stc, sdr, tpng, zmax):
    x = base
    # Solref — replace any existing pair
    x = re.sub(r'solref="[\d.\-e]+ [\d.\-e]+"', f'solref="{stc} {sdr}"', x)
    # Pitch joints stiffness + damping
    for pat in [r'joint_pitch_body_\d+', r'joint_passive_\d+']:
        x = re.sub(
            rf'(<joint\s[^>]*name="{pat}"[^>]*?)stiffness="[^"]*"',
            rf'\g<1>stiffness="{pk:.6e}"', x)
        x = re.sub(
            rf'(<joint\s[^>]*name="{pat}"[^>]*?)damping="[^"]*"',
            rf'\g<1>damping="{pd:.6e}"', x)
    # Terrain file
    x = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{tpng}\2', x)
    # z_max in size
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{zmax:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    x = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, x)
    return x

# ── Simulation ────────────────────────────────────────────────────────
def run_sim(xml_text, duration):
    with open(XML_PATH, 'w') as f:
        f.write(xml_text)
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return None, str(e)

    data = mujoco.MjData(model)
    idx  = FARMSModelIndex(model)
    cfg  = load_config(CONFIG_PATH)

    # Build controller inline (faster than constructor that re-reads files)
    ctrl = FARMSTravelingWaveController.__new__(FARMSTravelingWaveController)
    bw, lw = cfg['body_wave'], cfg['leg_wave']
    ctrl.body_amp = float(bw['amplitude'])
    ctrl.freq     = float(bw['frequency'])
    ctrl.n_wave   = float(bw['wave_number'])
    ctrl.speed    = float(bw['speed'])
    ctrl.omega    = 2 * np.pi * ctrl.freq
    ctrl.leg_amps          = np.array(lw['amplitudes'], dtype=float)
    ctrl.leg_phase_offsets = np.array(lw['phase_offsets'], dtype=float)
    ctrl.leg_dc_offsets    = np.array(lw['dc_offsets'], dtype=float)
    ctrl.active_dofs = set(lw['active_dofs'])
    ctrl.idx = idx

    # Pitch joint IDs
    pitch_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm or 'joint_passive' in nm):
            pitch_ids.append(i)

    n_steps = int(duration / model.opt.timestep)
    rec_dt  = 0.01
    last_rec = -np.inf
    T, CZ, BJ, PJ = [], [], [], []
    buckled = False

    for s in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Stability + buckling check every 200 steps
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "diverge"
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True

        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time
            T.append(data.time)
            CZ.append(idx.com_pos(data)[2])
            BJ.append(np.array([idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))

    return {
        'time': np.array(T), 'com_z': np.array(CZ),
        'body_jnt': np.array(BJ), 'pitch_jnt': np.array(PJ),
    }, ("buckled" if buckled else "ok")

# ── Metrics ───────────────────────────────────────────────────────────
def metrics(d):
    t = d['time']; m = t > 0.5  # 0.5s warmup for short sims
    if m.sum() < 5: return None
    cz = d['com_z'][m]; pp = d['pitch_jnt'][m]; ba = d['body_jnt'][m]

    cfg = load_config(CONFIG_PATH); bw = cfg['body_wave']
    om = 2*np.pi*bw['frequency']; nw = bw['wave_number']; sp = bw['speed']; N = 18
    bc = np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:, i] = bw['amplitude'] * np.sin(om * d['time'][m] - 2*np.pi*nw*sp*i/N)
    trk = float(np.sqrt(np.mean((ba - bc)**2)))

    return {
        'trk_deg':  np.degrees(trk),
        'z_min':    float(np.min(cz)) * 1000,
        'z_max':    float(np.max(cz)) * 1000,
        'z_std':    float(np.std(cz)) * 1000,
        'p_std':    float(np.degrees(np.std(pp))),
        'p_max':    float(np.degrees(np.max(np.abs(pp)))),
        'p_mean':   float(np.degrees(np.mean(np.abs(pp)))),
    }

# ── Main ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(XML_BACKUP):
        shutil.copy2(XML_PATH, XML_BACKUP)
    with open(XML_BACKUP, 'r') as f:
        base_xml = f.read()

    # Fixed solref (known good from previous tuning)
    STC, SDR = 0.01, 1.5

    # ── PHASE 1: rough terrain quick sweep ────────────────────────────
    print(f"\n{'='*72}")
    print("PHASE 1: Quick rough-terrain sweep (1.5s sims)")
    print(f"{'='*72}")

    if not ROUGH_PNG:
        print("ERROR: No rough terrain found!"); return
    print(f"  Terrain: {ROUGH_PNG}")

    PK_VALS = [3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]
    DR_VALS = [0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    total = len(PK_VALS) * len(DR_VALS)
    print(f"  {total} configurations\n")

    for idx_i, pk in enumerate(PK_VALS):
        for idx_j, dr in enumerate(DR_VALS):
            pd = pk * dr
            n = idx_i * len(DR_VALS) + idx_j + 1
            xml = patch_xml(base_xml, pk, pd, STC, SDR, ROUGH_PNG, ROUGH_ZMAX)
            t0 = time.time()
            d, status = run_sim(xml, 1.5)
            wall = time.time() - t0

            entry = {'pk': pk, 'pd': pd, 'dr': dr, 'status': status}
            if d is not None:
                met = metrics(d)
                if met: entry.update(met)

            results.append(entry)
            flag = "✗" if status != "ok" else ("⚠" if entry.get('p_max', 0) > 40 else "✓")
            print(f"  [{n:2d}/{total}] pk={pk:.0e} d/k={dr:4.1f} pd={pd:.0e} → "
                  f"{status:8s} {flag}  "
                  f"trk={entry.get('trk_deg',99):.3f}° "
                  f"pMax={entry.get('p_max',99):.1f}° "
                  f"zMin={entry.get('z_min',-99):.1f}mm "
                  f"({wall:.0f}s)")

    # ── PHASE 2: score non-buckling configs ───────────────────────────
    print(f"\n{'='*72}")
    print("PHASE 2: Scoring")
    print(f"{'='*72}")

    ok = [r for r in results if r['status'] == 'ok' and 'trk_deg' in r]
    buckled = [r for r in results if r['status'] == 'buckled']
    failed  = [r for r in results if r['status'] not in ('ok', 'buckled')]
    print(f"  OK={len(ok)}  Buckled={len(buckled)}  Failed={len(failed)}")

    if not ok:
        print("  No non-buckling configs! Using buckled with lowest p_max.")
        ok = [r for r in results if 'trk_deg' in r]
        ok.sort(key=lambda r: r.get('p_max', 999))
        ok = ok[:10]

    for r in ok:
        trk_s   = max(0, 10 - r['trk_deg'] * 10)       # <1° good
        buck_s  = max(0, 10 - r['p_max'] / 5)           # <50° good
        pen_s   = 10 if r['z_min'] > 0.5 else 0         # above ground
        comp_s  = min(5, r['p_std'] / 2)                 # some compliance
        z_s     = min(5, r['z_std'])                     # z variation
        r['score'] = trk_s + buck_s + pen_s + comp_s + z_s

    ok.sort(key=lambda r: -r['score'])

    print(f"\n  Top 10:")
    print(f"  {'#':>3} {'pk':>8} {'d/k':>5} {'trk°':>6} {'pMax':>6} {'pStd':>6} "
          f"{'zMin':>6} {'zStd':>6} {'Score':>6}")
    for i, r in enumerate(ok[:10]):
        print(f"  {i+1:3d} {r['pk']:8.1e} {r['dr']:5.1f} {r['trk_deg']:6.3f} "
              f"{r['p_max']:6.1f} {r['p_std']:6.2f} {r['z_min']:6.1f} "
              f"{r['z_std']:6.2f} {r['score']:6.1f}")

    # ── PHASE 3: extended eval top 5 on flat+rough ────────────────────
    print(f"\n{'='*72}")
    print("PHASE 3: Extended evaluation (3s) — flat + rough")
    print(f"{'='*72}")

    top5 = ok[:5]
    ext = []

    for rank, r in enumerate(top5):
        print(f"\n  Candidate {rank+1}: pk={r['pk']:.1e} d/k={r['dr']:.1f}")

        for tname, tpng, tz in [('flat', FLAT_TERRAIN, 0.001),
                                  ('rough', ROUGH_PNG, ROUGH_ZMAX)]:
            xml = patch_xml(base_xml, r['pk'], r['pd'], STC, SDR, tpng, tz)
            t0 = time.time()
            d, status = run_sim(xml, 3.0)
            wall = time.time() - t0

            entry = {'rank': rank+1, 'terrain': tname, 'status': status,
                     'pk': r['pk'], 'pd': r['pd'], 'dr': r['dr']}
            if d is not None:
                met = metrics(d)
                if met: entry.update(met)

            ext.append(entry)
            print(f"    {tname:6s}: {status:8s} trk={entry.get('trk_deg',99):.3f}° "
                  f"pMax={entry.get('p_max',99):.1f}° "
                  f"z=[{entry.get('z_min',-99):.1f},{entry.get('z_max',99):.1f}]mm "
                  f"zStd={entry.get('z_std',0):.2f}mm ({wall:.0f}s)")

    # ── PHASE 4: pick winner ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("FINAL SELECTION")
    print(f"{'='*72}")

    # Group by candidate
    cands = {}
    for e in ext:
        k = e['rank']
        if k not in cands: cands[k] = {}
        cands[k][e['terrain']] = e

    best_key, best_score = None, -999
    for k, cs in cands.items():
        flat  = cs.get('flat', {})
        rough = cs.get('rough', {})
        if not flat.get('trk_deg') or not rough.get('trk_deg'):
            continue
        if flat.get('status') == 'buckled' or rough.get('status') == 'buckled':
            sc = -100
        else:
            # Combined score
            trk   = max(0, 10 - flat['trk_deg'] * 10)
            nobuk = max(0, 10 - max(flat.get('p_max',0), rough.get('p_max',0)) / 5)
            nopen = 10 if min(flat.get('z_min',0), rough.get('z_min',0)) > 0.5 else 0
            # Compliance = rough shows more z variation than flat
            zcomp = min(5, max(0, rough.get('z_std',0) - flat.get('z_std',0)))
            # Pitch compliance: rough shows more pitch than flat
            pcomp = min(5, max(0, rough.get('p_std',0) - flat.get('p_std',0)))
            sc = trk + nobuk + nopen + zcomp + pcomp

        print(f"  Candidate {k}: score={sc:.1f}")
        if sc > best_score:
            best_score = sc; best_key = k

    if best_key is None:
        print("  No valid winner!"); return

    w = cands[best_key]['rough']  # use rough terrain entry for params
    print(f"\n  ★ WINNER: Candidate {best_key}")
    print(f"    pitch_k:   {w['pk']:.2e}")
    print(f"    pitch_d:   {w['pd']:.2e}")
    print(f"    d/k ratio: {w['dr']:.1f}")
    print(f"    solref:    {STC} {SDR}")

    # Apply winning params to XML with WINDOWS terrain path
    win_xml = patch_xml(
        base_xml, w['pk'], w['pd'], STC, SDR,
        "C:/Users/wxy22/Documents/Centipede_MUJOCO-main/terrain/output/low0.0060_mid0.0030_high0.0020_s0/1.png",
        0.04
    )
    with open(XML_PATH, 'w') as f:
        f.write(win_xml)
    print(f"\n  Applied to {XML_PATH}")

    # Save results
    out = {
        'winner': {'pitch_k': w['pk'], 'pitch_d': w['pd'], 'd_ratio': w['dr'],
                    'solref_tc': STC, 'solref_dr': SDR},
        'phase1_results': [{kk: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                            for kk, v in r.items()} for r in results],
        'phase3_results': [{kk: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                            for kk, v in e.items()} for e in ext],
    }
    with open(os.path.join(OUTPUT_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results: {OUTPUT_DIR}/sweep_results.json")

if __name__ == "__main__":
    main()
