#!/usr/bin/env python3
"""
quick_tune.py — Focused parameter search for pitch compliance + stability
=========================================================================
Small focused sweep of the most critical parameters, then extended evaluation.
"""

import os, sys, re, json, time, shutil
import numpy as np, yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.join(SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

XML_PATH     = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP   = XML_PATH + ".quick_backup"
CONFIG_PATH  = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR  = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR   = os.path.join(BASE, "outputs", "optimization", "quick_tune")
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")

def _pa(xml, np_, attr, val):
    def r(m):
        l=m.group(0); return re.sub(rf'{attr}="[^"]*"', f'{attr}="{val:.8g}"', l)
    return re.sub(rf'<position\s[^>]*name="{np_}"[^>]*/>', r, xml)

def _pj(xml, np_, attr, val):
    def r(m):
        l=m.group(0)
        if f'{attr}="' in l:
            return re.sub(rf'{attr}="[^"]*"', f'{attr}="{val:.8g}"', l)
        return l.replace('/>', f' {attr}="{val:.8g}"/>')
    return re.sub(rf'<joint\s[^>]*name="{np_}"[^>]*/>', r, xml)

def apply_params(xml, p):
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

    xml = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")', rf'\g<1>{FLAT_TERRAIN}\2', xml)
    return xml

def run_sim(xml_text, duration):
    with open(XML_PATH, 'w') as f: f.write(xml_text)
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return None, str(e)
    data = mujoco.MjData(model)
    idx = FARMSModelIndex(model)

    # Init controller manually (no print)
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
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)

    dt=model.opt.timestep; n_steps=int(duration/dt)
    rec_dt=0.01; last_rec=-np.inf
    T,CP,BJ,PJ=[],[],[],[]

    for s in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)
        if s%200==0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10])>50):
                return None, "unstable"
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec=data.time; T.append(data.time)
            CP.append(idx.com_pos(data).copy())
            BJ.append(np.array([idx.body_joint_pos(data,i+1) for i in range(N_BODY_JOINTS)]))
            PJ.append(np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids]))

    return {'time':np.array(T),'com_pos':np.array(CP),'body_jnt_pos':np.array(BJ),'pitch_jnt_pos':np.array(PJ)}, "ok"

def tracking_error(d):
    cfg=load_config(CONFIG_PATH); bw=cfg['body_wave']
    t=d['time']; ba=d['body_jnt_pos']; m=t>=0.5
    if m.sum()<5: return 999.
    tw=t[m]; bw_=ba[m]
    om=2*np.pi*bw['frequency']; nw=bw['wave_number']; sp=bw['speed']; N=18
    bc=np.zeros_like(bw_)
    for i in range(bw_.shape[1]):
        bc[:,i]=bw['amplitude']*np.sin(om*tw-2*np.pi*nw*sp*i/N)
    return float(np.sqrt(np.mean((bw_-bc)**2)))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(XML_BACKUP): shutil.copy2(XML_PATH, XML_BACKUP)
    with open(XML_BACKUP, 'r') as f: base_xml = f.read()

    # Known-good base parameters
    base = {
        'body_kp':64.945, 'body_kv':0.0552,
        'leg_dof0_kp':1.2695, 'leg_dof0_kv':0.00560,
        'leg_dof1_kp':0.1475, 'leg_dof1_kv':0.00114,
        'leg_dof2_kp':1.2695, 'leg_dof2_kv':0.000436,
        'leg_dof3_kp':1.2695, 'leg_dof3_kv':0.00910,
        'body_yaw_damping':4.075e-6, 'leg_dof01_damping':3e-8,
        'leg_dof2_damping':8.531e-5, 'leg_dof3_damping':6.117e-3,
        'pitch_k':1e-3, 'pitch_d':1e-4,
        'solref_tc':0.005, 'solref_dr':1.0,
        'solimp_dmin':0.9, 'solimp_dmax':0.95,
        'timestep':0.0005,
    }

    # SWEEP: pitch_k × pitch_d × solref_tc (keep timestep=0.0005 for accuracy)
    configs = []
    for pk in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]:
        for pd in [1e-4, 5e-5, 1e-5, 5e-6]:
            for stc in [0.005, 0.01]:
                p = base.copy()
                p['pitch_k']=pk; p['pitch_d']=pd; p['solref_tc']=stc
                configs.append(p)

    print(f"Total configs: {len(configs)}")
    print(f"Quick stability check (0.3s each)...\n")

    results = []
    for i, p in enumerate(configs):
        xml = apply_params(base_xml, p)
        t0 = time.time()
        d, st = run_sim(xml, 0.3)
        dt_w = time.time()-t0

        if st != "ok" or d is None:
            if i % 20 == 0:
                print(f"  [{i+1:3d}/{len(configs)}] pk={p['pitch_k']:.0e} pd={p['pitch_d']:.0e} "
                      f"stc={p['solref_tc']:.3f} ts={p['timestep']:.4f} -> {st} ({dt_w:.1f}s)")
            continue

        cz = d['com_pos'][:,2]
        if np.min(cz) < -0.001:
            continue

        trk = tracking_error(d)
        ps = np.std(d['pitch_jnt_pos']) if d['pitch_jnt_pos'].size else 0

        results.append({
            'params': p, 'tracking': trk, 'z_min': float(np.min(cz)),
            'z_mean': float(np.mean(cz)), 'pitch_std': float(ps),
        })
        if len(results) % 5 == 0:
            print(f"  [{i+1:3d}] pk={p['pitch_k']:.0e} pd={p['pitch_d']:.0e} "
                  f"stc={p['solref_tc']:.3f} ts={p['timestep']:.4f} "
                  f"trk={np.degrees(trk):.2f}° z={np.min(cz)*1000:.1f}mm "
                  f"pitch_std={np.degrees(ps):.3f}° ({dt_w:.1f}s)")

    print(f"\n{len(results)} stable configs out of {len(configs)}")

    if not results:
        print("No stable configs! Keeping current params.")
        shutil.copy2(XML_BACKUP, XML_PATH)
        return

    # Sort by tracking, then select those with lowest pitch_k (most compliant)
    results.sort(key=lambda r: r['tracking'])
    top_20 = results[:min(20, len(results))]

    print(f"\nTop 20 by tracking error:")
    for i, r in enumerate(top_20):
        p = r['params']
        print(f"  {i+1:2d}. trk={np.degrees(r['tracking']):.3f}° "
              f"pk={p['pitch_k']:.0e} pd={p['pitch_d']:.0e} "
              f"stc={p['solref_tc']:.3f} ts={p['timestep']:.4f} "
              f"z_min={r['z_min']*1000:.1f}mm pitch_std={np.degrees(r['pitch_std']):.3f}°")

    # Extended eval of top 5
    print(f"\nExtended eval (2s) of top 5...")
    ext_results = []
    for i, r in enumerate(top_20[:5]):
        p = r['params']
        xml = apply_params(base_xml, p)
        t0 = time.time()
        d, st = run_sim(xml, 2.0)
        dt_w = time.time()-t0

        if st != "ok":
            print(f"  {i+1}. UNSTABLE at 2s ({dt_w:.1f}s)")
            continue

        trk = tracking_error(d)
        m = d['time'] > 0.5
        cz = d['com_pos'][m, 2]
        pp = d['pitch_jnt_pos'][m] if d['pitch_jnt_pos'].size else np.array([0])

        print(f"  {i+1}. trk={np.degrees(trk):.3f}° "
              f"z=[{np.min(cz)*1000:.1f},{np.max(cz)*1000:.1f}]mm "
              f"pitch_std={np.degrees(np.std(pp)):.3f}° max={np.degrees(np.max(np.abs(pp))):.3f}° "
              f"({dt_w:.1f}s)")

        ext_results.append({
            'params': p, 'tracking': trk,
            'z_min': float(np.min(cz)), 'z_max': float(np.max(cz)),
            'z_std': float(np.std(cz)),
            'pitch_std': float(np.std(pp)), 'pitch_max': float(np.max(np.abs(pp))),
        })

    if not ext_results:
        print("No extended results, using quick top.")
        ext_results = [{'params': top_20[0]['params'], 'tracking': top_20[0]['tracking']}]

    # Pick the most compliant among those with acceptable tracking
    # (lowest pitch_k with tracking < 2x best tracking)
    best_trk = min(r['tracking'] for r in ext_results)
    acceptable = [r for r in ext_results if r['tracking'] < best_trk * 2.0]
    if not acceptable:
        acceptable = ext_results[:1]
    # Among acceptable, pick lowest pitch_k
    acceptable.sort(key=lambda r: r['params']['pitch_k'])
    winner = acceptable[0]

    print(f"\n{'='*60}")
    print(f"WINNER: Most compliant with acceptable tracking")
    print(f"{'='*60}")
    p = winner['params']
    print(f"  pitch_k:    {p['pitch_k']:.2e}")
    print(f"  pitch_d:    {p['pitch_d']:.2e}")
    print(f"  solref_tc:  {p['solref_tc']:.4f}")
    print(f"  timestep:   {p['timestep']:.4f}")
    print(f"  tracking:   {np.degrees(winner['tracking']):.3f}°")
    if 'z_min' in winner:
        print(f"  COM Z:      [{winner['z_min']*1000:.1f}, {winner.get('z_max',0)*1000:.1f}] mm")
        print(f"  pitch_std:  {np.degrees(winner.get('pitch_std',0)):.3f}°")
        print(f"  pitch_max:  {np.degrees(winner.get('pitch_max',0)):.3f}°")

    # Apply
    xml = apply_params(base_xml, p)
    with open(XML_PATH, 'w') as f: f.write(xml)
    print(f"\nApplied to {XML_PATH}")

    # Save
    with open(os.path.join(OUTPUT_DIR, 'optimal_params.json'), 'w') as f:
        json.dump(p, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}/optimal_params.json")

    # Also save full results summary
    summary = {
        'total_configs': len(configs),
        'stable_configs': len(results),
        'winner': {k: float(v) if isinstance(v, (np.floating,)) else v
                   for k, v in winner.items() if k != 'params'},
        'winner_params': p,
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

if __name__ == "__main__":
    main()
