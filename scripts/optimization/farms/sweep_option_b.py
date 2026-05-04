#!/usr/bin/env python3
"""
sweep_option_b.py — Find optimal pitch stiffness/damping that prevents buckling
while maintaining terrain compliance and tracking accuracy.

The buckling failure: when pitch stiffness is too low, the front body segments
sag under gravity until they contact terrain directly. Contact forces then
fold the body in the vertical plane.

Detection: monitor max pitch joint angle. If any pitch joint exceeds ~60 deg,
the body is buckling. Good compliance shows 5-20 deg pitch on rough terrain.

Sweep ranges (informed by torque analysis):
  pitch_k: 3e-4 to 5e-3 (must resist gravity on 3 segments < 15 deg)
  pitch_d: scale relative to k for target damping behavior
  solref_tc: 0.005 to 0.02
  solref_dr: 1.0 to 2.0
"""
import os, sys, re, time, json, shutil
import numpy as np, yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

XML_PATH = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP = XML_PATH + ".optionb_backup"
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR = os.path.join(BASE, "outputs", "optimization", "option_b")
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")

# Rough terrain (L2 moderate)
ROUGH_TERRAIN = None
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and 'low0.0060' in d:
        ROUGH_TERRAIN = (os.path.join(dp, "1.png"), 0.040)
        break

def _pj(xml, pat, attr, val):
    def r(m):
        l = m.group(0)
        if f'{attr}="' in l:
            return re.sub(rf'{attr}="[^"]*"', f'{attr}="{val}"', l)
        return l.replace('/>', f' {attr}="{val}"/>')
    return re.sub(rf'<joint\s[^>]*name="{pat}"[^>]*/>', r, xml)

def patch_xml(xml, pitch_k, pitch_d, solref_tc, solref_dr, terrain_png, z_max):
    xml = xml.replace('solref="0.005 1"', f'solref="{solref_tc} {solref_dr}"')
    xml = xml.replace('solref="0.01 1.5"', f'solref="{solref_tc} {solref_dr}"')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'stiffness', f'{pitch_k:.6e}')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'damping', f'{pitch_d:.6e}')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'stiffness', f'{pitch_k:.6e}')
    xml = _pj(xml, r'joint_pitch_body_\d+', 'damping', f'{pitch_d:.6e}')
    xml = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{terrain_png}\2', xml)
    # Patch z_max
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{z_max:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    xml = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, xml)
    return xml

def run_sim(xml_text, duration):
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    try: model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e: return None, str(e)

    data = mujoco.MjData(model)
    idx = FARMSModelIndex(model)

    ctrl = FARMSTravelingWaveController.__new__(FARMSTravelingWaveController)
    cfg = load_config(CONFIG_PATH)
    bw = cfg['body_wave']; lw = cfg['leg_wave']
    ctrl.body_amp = float(bw['amplitude']); ctrl.freq = float(bw['frequency'])
    ctrl.n_wave = float(bw['wave_number']); ctrl.speed = float(bw['speed'])
    ctrl.omega = 2 * np.pi * ctrl.freq
    ctrl.leg_amps = np.array(lw['amplitudes'], dtype=float)
    ctrl.leg_phase_offsets = np.array(lw['phase_offsets'], dtype=float)
    ctrl.leg_dc_offsets = np.array(lw['dc_offsets'], dtype=float)
    ctrl.active_dofs = set(lw['active_dofs']); ctrl.idx = idx

    pitch_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)

    dt = model.opt.timestep; n_steps = int(duration / dt)
    rec_dt = 0.01; last_rec = -np.inf
    T, CZ, BJ, PJ = [], [], [], []
    buckling_detected = False

    for s in range(n_steps):
        ctrl.step(model, data); mujoco.mj_step(model, data)
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "nan/diverge"
            # Check for buckling: any pitch > 60 deg
            for jid in pitch_ids:
                pval = abs(data.qpos[model.jnt_qposadr[jid]])
                if pval > np.radians(60):
                    buckling_detected = True

        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; T.append(data.time)
            CZ.append(idx.com_pos(data)[2])
            BJ.append(np.array([idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))

    d = {'time': np.array(T), 'com_z': np.array(CZ),
         'body_jnt_pos': np.array(BJ), 'pitch_jnt_pos': np.array(PJ)}
    status = "buckled" if buckling_detected else "ok"
    return d, status

def compute_metrics(d, label=""):
    t = d['time']; m = t > 1.0
    if m.sum() < 5: return None

    cz = d['com_z'][m]
    pp = d['pitch_jnt_pos'][m]
    ba = d['body_jnt_pos'][m]

    # Tracking error
    cfg = load_config(CONFIG_PATH); bw = cfg['body_wave']
    om = 2 * np.pi * bw['frequency']; nw = bw['wave_number']; sp = bw['speed']; N = 18
    bc = np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:, i] = bw['amplitude'] * np.sin(om * d['time'][m] - 2*np.pi*nw*sp*i/N)
    trk = float(np.sqrt(np.mean((ba - bc)**2)))

    return {
        'tracking_deg': np.degrees(trk),
        'z_min_mm': float(np.min(cz)) * 1000,
        'z_max_mm': float(np.max(cz)) * 1000,
        'z_std_mm': float(np.std(cz)) * 1000,
        'pitch_std_deg': float(np.degrees(np.std(pp))),
        'pitch_max_deg': float(np.degrees(np.max(np.abs(pp)))),
        'pitch_mean_abs_deg': float(np.degrees(np.mean(np.abs(pp)))),
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(XML_BACKUP): shutil.copy2(XML_PATH, XML_BACKUP)
    with open(XML_BACKUP, 'r') as f: base_xml = f.read()

    # ================================================================
    # STAGE 1: Quick sweep on ROUGH terrain (2s) to find non-buckling
    # ================================================================
    print(f"\n{'='*70}")
    print("STAGE 1: Rough terrain sweep (2s) — find stable non-buckling params")
    print(f"{'='*70}")

    if not ROUGH_TERRAIN:
        print("No rough terrain found, using flat only")
        terrain_png, z_max = FLAT_TERRAIN, 0.001
    else:
        terrain_png, z_max = ROUGH_TERRAIN
        print(f"  Terrain: {terrain_png} (z_max={z_max})")

    # Sweep: pitch_k from 3e-4 to 5e-3, pitch_d as multiples of k
    results = []
    configs = []
    for pk in [3e-4, 5e-4, 8e-4, 1e-3, 2e-3, 3e-3, 5e-3]:
        for d_ratio in [0.5, 1.0, 2.0, 5.0, 10.0]:
            # d_ratio = pitch_d / pitch_k
            pd = pk * d_ratio
            for stc in [0.005, 0.01]:
                for sdr in [1.0, 1.5]:
                    configs.append({
                        'pitch_k': pk, 'pitch_d': pd,
                        'd_ratio': d_ratio,
                        'solref_tc': stc, 'solref_dr': sdr,
                    })

    print(f"  {len(configs)} configurations to test\n")

    for i, c in enumerate(configs):
        xml = patch_xml(base_xml, c['pitch_k'], c['pitch_d'],
                        c['solref_tc'], c['solref_dr'], terrain_png, z_max)
        t0 = time.time()
        d, status = run_sim(xml, 2.0)
        wall = time.time() - t0

        if d is None:
            if i % 20 == 0:
                print(f"  [{i+1:3d}] pk={c['pitch_k']:.0e} d/k={c['d_ratio']:.0f} "
                      f"stc={c['solref_tc']:.3f} dr={c['solref_dr']:.1f} -> {status}")
            results.append({**c, 'status': status})
            continue

        met = compute_metrics(d)
        if met is None:
            results.append({**c, 'status': 'no_metrics'})
            continue

        results.append({**c, 'status': status, **met})

        if (i+1) % 10 == 0:
            print(f"  [{i+1:3d}] pk={c['pitch_k']:.0e} d/k={c['d_ratio']:.0f} "
                  f"stc={c['solref_tc']:.3f} dr={c['solref_dr']:.1f} "
                  f"-> {status:7s} trk={met['tracking_deg']:.3f}° "
                  f"pitch_max={met['pitch_max_deg']:.1f}° "
                  f"z=[{met['z_min_mm']:.1f},{met['z_max_mm']:.1f}]mm ({wall:.1f}s)")

    # ================================================================
    # STAGE 2: Score and rank
    # ================================================================
    print(f"\n{'='*70}")
    print("STAGE 2: Scoring")
    print(f"{'='*70}")

    ok_results = [r for r in results if r['status'] == 'ok' and 'tracking_deg' in r]
    buckled = [r for r in results if r['status'] == 'buckled']
    failed = [r for r in results if r['status'] not in ('ok', 'buckled') or 'tracking_deg' not in r]

    print(f"  OK: {len(ok_results)}  Buckled: {len(buckled)}  Failed: {len(failed)}")

    if not ok_results:
        print("  No non-buckling configs found!")
        # Fall back to buckled configs with lowest pitch_max
        ok_results = [r for r in results if 'tracking_deg' in r]
        ok_results.sort(key=lambda r: r.get('pitch_max_deg', 999))
        ok_results = ok_results[:10]
        print(f"  Using top 10 lowest-pitch-max configs instead")

    # Score: want low tracking, low pitch_max, positive z_min, but nonzero pitch_std
    for r in ok_results:
        trk_score = max(0, 10 - r['tracking_deg'] * 10)  # <1 deg = good
        buckle_score = max(0, 10 - r['pitch_max_deg'] / 5)  # <50 deg = good
        penetration_score = 10 if r['z_min_mm'] > 1 else 0
        # Compliance: we want SOME pitch variation (not rigid rod)
        compliance_score = min(5, r['pitch_std_deg'] / 2)  # reward up to 5 deg std
        z_follow = min(5, r['z_std_mm'])  # reward COM height variation

        r['score'] = trk_score + buckle_score + penetration_score + compliance_score + z_follow

    ok_results.sort(key=lambda r: -r['score'])

    print(f"\n  Top 15 candidates:")
    print(f"  {'Rank':>4s}  {'pk':>8s} {'d/k':>5s} {'stc':>6s} {'dr':>4s}  "
          f"{'trk°':>6s} {'pMax°':>6s} {'pStd°':>6s} {'Zmin':>6s} {'Zstd':>6s} {'Score':>6s}")
    print(f"  {'-'*80}")
    for rank, r in enumerate(ok_results[:15]):
        print(f"  {rank+1:4d}  {r['pitch_k']:8.1e} {r['d_ratio']:5.1f} "
              f"{r['solref_tc']:6.3f} {r['solref_dr']:4.1f}  "
              f"{r['tracking_deg']:6.3f} {r['pitch_max_deg']:6.1f} "
              f"{r['pitch_std_deg']:6.2f} {r['z_min_mm']:6.1f} "
              f"{r['z_std_mm']:6.2f} {r['score']:6.1f}")

    # ================================================================
    # STAGE 3: Extended eval of top 3 on BOTH flat and rough
    # ================================================================
    print(f"\n{'='*70}")
    print("STAGE 3: Extended evaluation (5s) — flat + rough terrain")
    print(f"{'='*70}")

    top3 = ok_results[:3]
    ext_results = []

    for rank, r in enumerate(top3):
        print(f"\n  Candidate {rank+1}: pk={r['pitch_k']:.1e} d/k={r['d_ratio']:.0f} "
              f"solref=({r['solref_tc']},{r['solref_dr']})")

        for tname, tpng, tz in [('flat', FLAT_TERRAIN, 0.001),
                                  ('rough', terrain_png, z_max)]:
            xml = patch_xml(base_xml, r['pitch_k'], r['pitch_d'],
                            r['solref_tc'], r['solref_dr'], tpng, tz)
            t0 = time.time()
            d, status = run_sim(xml, 5.0)
            wall = time.time() - t0

            if d is None:
                print(f"    {tname:6s}: {status} ({wall:.0f}s)")
                continue

            met = compute_metrics(d, f"{rank}_{tname}")
            if met is None:
                print(f"    {tname:6s}: no metrics ({wall:.0f}s)")
                continue

            print(f"    {tname:6s}: {status:7s} trk={met['tracking_deg']:.3f}° "
                  f"pitch=[{met['pitch_std_deg']:.2f}°,{met['pitch_max_deg']:.1f}°] "
                  f"z=[{met['z_min_mm']:.1f},{met['z_max_mm']:.1f}]mm "
                  f"z_std={met['z_std_mm']:.2f}mm ({wall:.0f}s)")

            ext_results.append({
                'rank': rank+1, 'terrain': tname, 'status': status,
                'params': {k: r[k] for k in ['pitch_k','pitch_d','d_ratio','solref_tc','solref_dr']},
                **met,
            })

    # ================================================================
    # STAGE 4: Pick winner
    # ================================================================
    print(f"\n{'='*70}")
    print("FINAL SELECTION")
    print(f"{'='*70}")

    # Group by candidate, score on combined flat+rough performance
    cand_scores = {}
    for er in ext_results:
        key = er['rank']
        if key not in cand_scores:
            cand_scores[key] = {'flat': None, 'rough': None, 'params': er['params']}
        cand_scores[key][er['terrain']] = er

    best_key = None
    best_total = -999
    for key, cs in cand_scores.items():
        flat = cs.get('flat')
        rough = cs.get('rough')
        if not flat or not rough:
            continue
        if flat.get('status') == 'buckled' or rough.get('status') == 'buckled':
            total = -100
        else:
            # Composite: tracking + controlled pitch + no penetration + compliance difference
            trk = max(0, 10 - flat['tracking_deg'] * 10)
            no_buckle = max(0, 10 - max(flat['pitch_max_deg'], rough['pitch_max_deg']) / 5)
            no_pen = 10 if min(flat['z_min_mm'], rough['z_min_mm']) > 0.5 else 0
            # Compliance: rough terrain should show MORE z variation than flat
            z_compliance = min(5, max(0, rough['z_std_mm'] - flat['z_std_mm']))
            total = trk + no_buckle + no_pen + z_compliance

        print(f"  Candidate {key}: total={total:.1f}")
        if total > best_total:
            best_total = total
            best_key = key

    if best_key is None:
        print("  No valid candidates!")
        return

    winner = cand_scores[best_key]
    wp = winner['params']
    print(f"\n  WINNER: Candidate {best_key}")
    print(f"    pitch_k:    {wp['pitch_k']:.2e}")
    print(f"    pitch_d:    {wp['pitch_d']:.2e}")
    print(f"    d/k ratio:  {wp['d_ratio']:.1f}")
    print(f"    solref_tc:  {wp['solref_tc']}")
    print(f"    solref_dr:  {wp['solref_dr']}")

    # Apply to XML (restore Windows terrain path)
    win_xml = patch_xml(base_xml, wp['pitch_k'], wp['pitch_d'],
                        wp['solref_tc'], wp['solref_dr'],
                        "C:/Users/wxy22/Documents/Centipede_MUJOCO-main/terrain/output/low0.0060_mid0.0030_high0.0020_s0/1.png",
                        0.04)
    with open(XML_PATH, 'w') as f:
        f.write(win_xml)
    print(f"\n  Applied to {XML_PATH}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({
            'winner_params': wp,
            'ext_results': [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                             for k, v in er.items()} for er in ext_results],
            'all_ok_count': len([r for r in results if r['status']=='ok']),
            'all_buckled_count': len(buckled),
        }, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()
