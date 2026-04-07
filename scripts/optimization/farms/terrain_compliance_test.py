#!/usr/bin/env python3
"""
terrain_compliance_test.py — Test pitch compliance on flat vs rough terrain
===========================================================================
Tests a few carefully chosen parameter sets and compares pitch behavior on
flat vs rough terrain. Picks the set that shows meaningful terrain-following
without instability.
"""
import os, sys, re, json, time, shutil
import numpy as np, yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

XML_PATH = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP = XML_PATH + ".compliance_backup"
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR = os.path.join(BASE, "outputs", "optimization", "compliance_test")
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")

# Find rough terrain
ROUGH_TERRAINS = {}
for d in sorted(os.listdir(TERRAIN_DIR)):
    dp = os.path.join(TERRAIN_DIR, d)
    if os.path.isdir(dp) and os.path.exists(os.path.join(dp, "1.png")):
        if 'low0.0040' in d: ROUGH_TERRAINS['L1_mild'] = (os.path.join(dp,"1.png"), 0.010)
        elif 'low0.0060' in d: ROUGH_TERRAINS['L2_mod'] = (os.path.join(dp,"1.png"), 0.040)
        elif 'low0.0100' in d: ROUGH_TERRAINS['L3_rough'] = (os.path.join(dp,"1.png"), 0.070)

def _pa(xml, np_, attr, val):
    def r(m): return re.sub(rf'{attr}="[^"]*"', f'{attr}="{val:.8g}"', m.group(0))
    return re.sub(rf'<position\s[^>]*name="{np_}"[^>]*/>', r, xml)

def _pj(xml, np_, attr, val):
    def r(m):
        l=m.group(0)
        if f'{attr}="' in l: return re.sub(rf'{attr}="[^"]*"', f'{attr}="{val:.8g}"', l)
        return l.replace('/>', f' {attr}="{val:.8g}"/>')
    return re.sub(rf'<joint\s[^>]*name="{np_}"[^>]*/>', r, xml)

def apply_params(xml, p, terrain_png, z_max=0.001):
    xml = re.sub(r'(<option\s[^>]*timestep=")[^"]*(")', rf'\g<1>{p["timestep"]:.6g}\2', xml)
    sr = f'{p["solref_tc"]:.6g} {p["solref_dr"]:.6g}'
    xml = re.sub(r'(solref=")[^"]*(")', rf'\g<1>{sr}\2', xml)
    si = f'{p["solimp_dmin"]:.6g} {p["solimp_dmax"]:.6g} 0.001'
    xml = re.sub(r'(solimp=")[^"]*(")', rf'\g<1>{si}\2', xml)

    xml = _pa(xml, r'act_joint_body_\d+', 'kp', p['body_kp'])
    xml = _pa(xml, r'act_joint_body_\d+', 'kv', p['body_kv'])
    for d in range(4):
        xml = _pa(xml, rf'act_joint_leg_\d+_[LR]_{d}', 'kp', p[f'leg_dof{d}_kp'])
        xml = _pa(xml, rf'act_joint_leg_\d+_[LR]_{d}', 'kv', p[f'leg_dof{d}_kv'])
    xml = _pa(xml, r'act_joint_foot_\d+_[01]', 'kp', p['leg_dof3_kp'])
    xml = _pa(xml, r'act_joint_foot_\d+_[01]', 'kv', p['leg_dof3_kv'])

    xml = _pj(xml, r'joint_body_\d+', 'damping', p['body_yaw_damping'])
    xml = _pj(xml, r'joint_leg_\d+_[LR]_0', 'damping', p['leg_dof01_damping'])
    xml = _pj(xml, r'joint_leg_\d+_[LR]_1', 'damping', p['leg_dof01_damping'])
    xml = _pj(xml, r'joint_leg_\d+_[LR]_2', 'damping', p['leg_dof2_damping'])
    xml = _pj(xml, r'joint_leg_\d+_[LR]_3', 'damping', p['leg_dof3_damping'])
    xml = _pj(xml, r'joint_foot_\d+_[01]', 'damping', p['leg_dof3_damping'])

    xml = _pj(xml, r'joint_pitch_body_\d+', 'stiffness', p['pitch_k'])
    xml = _pj(xml, r'joint_pitch_body_\d+', 'damping', p['pitch_d'])
    xml = _pj(xml, r'joint_pitch_body_\d+', 'stiffness', p['pitch_k'])
    xml = _pj(xml, r'joint_pitch_body_\d+', 'damping', p['pitch_d'])

    xml = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{terrain_png}\2', xml)
    # Patch z_max in hfield
    def rep_size(m):
        parts = m.group(1).split(); parts[2] = f"{z_max:.6g}"
        return f'size="{" ".join(parts)}"'
    xml = re.sub(r'(<hfield[^>]*size=")([^"]*)"', lambda m: f'{m.group(1)}{" ".join(m.group(2).split()[:2])} {z_max:.6g} {m.group(2).split()[3]}"', xml)
    return xml

def run_sim(xml_text, duration):
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    try: model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e: return None, str(e)
    data = mujoco.MjData(model)
    idx = FARMSModelIndex(model)

    ctrl = FARMSTravelingWaveController.__new__(FARMSTravelingWaveController)
    cfg = load_config(CONFIG_PATH)
    bw=cfg['body_wave']; lw=cfg['leg_wave']
    ctrl.body_amp=float(bw['amplitude']); ctrl.freq=float(bw['frequency'])
    ctrl.n_wave=float(bw['wave_number']); ctrl.speed=float(bw['speed'])
    ctrl.omega=2*np.pi*ctrl.freq
    ctrl.leg_amps=np.array(lw['amplitudes'],dtype=float)
    ctrl.leg_phase_offsets=np.array(lw['phase_offsets'],dtype=float)
    ctrl.leg_dc_offsets=np.array(lw['dc_offsets'],dtype=float)
    ctrl.active_dofs=set(lw['active_dofs']); ctrl.idx=idx

    pitch_ids=[]
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm): pitch_ids.append(i)

    dt=model.opt.timestep; n_steps=int(duration/dt)
    rec_dt=0.005; last_rec=-np.inf
    T,CP,BJ,PJ=[],[],[],[]

    for s in range(n_steps):
        ctrl.step(model, data); mujoco.mj_step(model, data)
        if s%200==0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10])>50):
                return None, "unstable"
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec=data.time; T.append(data.time)
            CP.append(idx.com_pos(data).copy())
            BJ.append(np.array([idx.body_joint_pos(data,i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))

    return {'time':np.array(T),'com_pos':np.array(CP),'body_jnt_pos':np.array(BJ),'pitch_jnt_pos':np.array(PJ)}, "ok"

def analyze(d, label):
    t=d['time']; m=t>1.0
    if m.sum()<5: return {'label': label, 'status': 'too_short'}
    cz = d['com_pos'][m,2]
    pp = d['pitch_jnt_pos'][m]

    cfg=load_config(CONFIG_PATH); bw=cfg['body_wave']
    om=2*np.pi*bw['frequency']; nw=bw['wave_number']; sp=bw['speed']; N=18
    ba=d['body_jnt_pos'][m]; bc=np.zeros_like(ba)
    for i in range(ba.shape[1]):
        bc[:,i]=bw['amplitude']*np.sin(om*d['time'][m]-2*np.pi*nw*sp*i/N)
    trk = float(np.sqrt(np.mean((ba-bc)**2)))

    return {
        'label': label,
        'tracking_deg': np.degrees(trk),
        'com_z_min_mm': np.min(cz)*1000,
        'com_z_max_mm': np.max(cz)*1000,
        'com_z_mean_mm': np.mean(cz)*1000,
        'com_z_std_mm': np.std(cz)*1000,
        'pitch_std_deg': np.degrees(np.std(pp)),
        'pitch_max_deg': np.degrees(np.max(np.abs(pp))),
        'pitch_mean_deg': np.degrees(np.mean(np.abs(pp))),
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(XML_BACKUP): shutil.copy2(XML_PATH, XML_BACKUP)
    with open(XML_BACKUP if os.path.exists(XML_BACKUP) else XML_PATH, 'r') as f:
        base_xml = f.read()

    # Known-good base (actuator gains from previous Bayesian optimization)
    base = {
        'body_kp':64.945, 'body_kv':0.0552,
        'leg_dof0_kp':1.2695, 'leg_dof0_kv':0.00560,
        'leg_dof1_kp':0.1475, 'leg_dof1_kv':0.00114,
        'leg_dof2_kp':1.2695, 'leg_dof2_kv':0.000436,
        'leg_dof3_kp':1.2695, 'leg_dof3_kv':0.00910,
        'body_yaw_damping':4.075e-6, 'leg_dof01_damping':3e-8,
        'leg_dof2_damping':8.531e-5, 'leg_dof3_damping':6.117e-3,
        'solref_tc':0.005, 'solref_dr':1.0,
        'solimp_dmin':0.9, 'solimp_dmax':0.95,
        'timestep':0.0005,
    }

    # Define candidate parameter sets — varying pitch compliance + contact
    candidates = {
        'A_current':    {'pitch_k': 1e-3,  'pitch_d': 1e-4,  'solref_tc': 0.005, 'solref_dr': 1.0},
        'C_softer':     {'pitch_k': 2e-4,  'pitch_d': 2e-4,  'solref_tc': 0.010, 'solref_dr': 1.0},
        'D_soft_stiff': {'pitch_k': 1e-4,  'pitch_d': 1e-3,  'solref_tc': 0.010, 'solref_dr': 1.5},
        'F_balanced':   {'pitch_k': 3e-4,  'pitch_d': 3e-4,  'solref_tc': 0.010, 'solref_dr': 1.2},
    }

    terrains = {'flat': (FLAT_TERRAIN, 0.001)}
    if 'L2_mod' in ROUGH_TERRAINS:
        terrains['L2_rough'] = ROUGH_TERRAINS['L2_mod']

    all_results = {}

    for cname, overrides in candidates.items():
        p = base.copy()
        p.update(overrides)
        all_results[cname] = {}

        print(f"\n{'='*60}")
        print(f"Candidate {cname}: pitch_k={p['pitch_k']:.1e} pitch_d={p['pitch_d']:.1e} "
              f"solref=({p['solref_tc']},{p['solref_dr']})")

        for tname, (tpng, tz) in terrains.items():
            xml = apply_params(base_xml, p, tpng, tz)
            t0 = time.time()
            d, st = run_sim(xml, 2.0)
            wall = time.time()-t0

            if st != "ok":
                print(f"  {tname:12s}: {st} ({wall:.1f}s)")
                all_results[cname][tname] = {'status': st}
                continue

            res = analyze(d, f"{cname}_{tname}")
            all_results[cname][tname] = res
            print(f"  {tname:12s}: trk={res['tracking_deg']:.3f}° "
                  f"z=[{res['com_z_min_mm']:.1f},{res['com_z_max_mm']:.1f}]mm "
                  f"pitch_std={res['pitch_std_deg']:.2f}° max={res['pitch_max_deg']:.2f}° "
                  f"({wall:.1f}s)")

    # === SCORING ===
    print(f"\n\n{'='*60}")
    print("SCORING")
    print(f"{'='*60}")

    scores = {}
    for cname, tres in all_results.items():
        flat = tres.get('flat', {})
        rough = tres.get('L2_rough', {})

        if flat.get('status') or rough.get('status'):
            scores[cname] = -999
            continue

        if 'tracking_deg' not in flat:
            scores[cname] = -999
            continue

        # Score components:
        # 1. Tracking accuracy (lower = better, target < 1°)
        trk_score = max(0, 5 - flat['tracking_deg'] * 5)

        # 2. Flat terrain pitch control (should be low, < 15° ideal)
        flat_pitch_score = max(0, 5 - flat['pitch_std_deg'] / 3)

        # 3. No penetration (com_z_min should be > 2mm)
        pen_flat = 5 if flat['com_z_min_mm'] > 2 else 0
        pen_rough = 0
        if 'com_z_min_mm' in rough:
            pen_rough = 5 if rough['com_z_min_mm'] > 0 else 0

        # 4. Compliance on rough terrain (pitch should be HIGHER than flat)
        compliance_score = 0
        if 'pitch_std_deg' in rough and 'pitch_std_deg' in flat:
            delta = rough['pitch_std_deg'] - flat['pitch_std_deg']
            if delta > 0:
                compliance_score = min(5, delta / 2)  # reward up to 10° increase

        # 5. COM Z variation on rough terrain (should be nonzero = following terrain)
        z_follow = 0
        if 'com_z_std_mm' in rough:
            z_follow = min(5, rough['com_z_std_mm'] * 2)

        total = trk_score + flat_pitch_score + pen_flat + pen_rough + compliance_score + z_follow
        scores[cname] = total

        print(f"  {cname:20s}: trk={trk_score:.1f} flat_pitch={flat_pitch_score:.1f} "
              f"pen_f={pen_flat:.0f} pen_r={pen_rough:.0f} comply={compliance_score:.1f} "
              f"z_follow={z_follow:.1f} => TOTAL={total:.1f}")

    # Pick winner
    winner_name = max(scores, key=scores.get)
    winner_score = scores[winner_name]
    winner_overrides = candidates[winner_name]

    print(f"\n{'='*60}")
    print(f"WINNER: {winner_name} (score={winner_score:.1f})")
    print(f"{'='*60}")
    for k, v in winner_overrides.items():
        print(f"  {k}: {v}")

    # Apply winner params
    p = base.copy()
    p.update(winner_overrides)
    xml = apply_params(base_xml, p, FLAT_TERRAIN, 0.001)
    with open(XML_PATH, 'w') as f: f.write(xml)

    # Save
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({
            'winner': winner_name, 'score': winner_score,
            'winner_params': {**base, **winner_overrides},
            'all_scores': scores,
            'all_results': {k: {tk: {rk: (float(rv) if isinstance(rv, (np.floating,np.integer)) else rv)
                                      for rk,rv in tv.items()}
                                 for tk,tv in v.items()}
                            for k,v in all_results.items()},
        }, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/results.json")
    print(f"XML updated with winner params (flat terrain hfield)")
    print(f"\nFull winner parameter set:")
    for k, v in sorted({**base, **winner_overrides}.items()):
        print(f"  {k:22s}: {v}")

if __name__ == "__main__":
    main()
