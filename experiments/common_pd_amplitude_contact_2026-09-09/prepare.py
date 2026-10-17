import argparse, json, math, sys
from pathlib import Path
import core as c

FINE=c.EXP/'amplitude_optimization_recovery_2026-09-08/fine_step'
OLD=c.EXP/'mujoco_wave_amplitude_natural_gait_optimization_2026-08-31'
RL=c.EXP.parent.parent/'Centipede_IsaacLab_RL'
NEW_SELECTION=FINE/'outputs/data/selected_gaits.json'
OLD_SELECTION=OLD/'outputs/data/final_certified_selection.json'
RL_SELECTION=RL/'outputs/report/p8_flat_roll_causal_summary_2026-07-21/data/flat_locked_gait_map_v4_13knot.json'

def anchors():
    current={r['wave']:list(r['amplitudes_deg'].values()) for r in c.read(NEW_SELECTION)['gaits']}
    old={float(w):[s['selected'][k] for k in ('body_yaw_amplitude_deg','leg_yaw_amplitude_deg','leg_pitch_amplitude_deg')]
         for w,s in c.read(OLD_SELECTION)['waves'].items() if s.get('selected')}
    rl={float(r['wave']):[math.degrees(.4*r['body_amp']),math.degrees(.6*r['leg_yaw_amp']),math.degrees(.4*1.15)]
        for r in c.read(RL_SELECTION)['wave_map']}
    return dict(recovery=current,august31=old,rl_v4=rl)

def prepare():
    assert not (c.DATA/'manifest.json').exists()
    manifest=dict(version='common_pd_amplitude_contact_v1',pd=c.CFG,plants={},frozen_inputs={},
        frequency_hz=1.,dt_s=.000125,amplitude_domain_deg=[[0,36],[0,90],[0,60]],waves=c.WAVES,
        no_pd_search=True,one_pd_for_all_waves=True,clean_changes=False)
    geometry={}
    for name,path in c.PLANTS.items():
        m=c.e.runtime_model(path);ix=c.e.fx.opt.tip.LeanIndex(m)
        c.ORIGINAL_APPLY(m,ix,c.CFG)
        pd=dict(names=ix.active_joint_names,kp=ix.kp.tolist(),kv=ix.kv.tolist())
        import hashlib
        digest=hashlib.sha256(json.dumps(pd,sort_keys=True).encode()).hexdigest().upper()
        if 'active_pd_sha256' in manifest:assert manifest['active_pd_sha256']==digest
        manifest['active_pd_sha256']=digest
        manifest['active_pd']=pd
        assert float(m.opt.timestep)==.00025
        manifest['plants'][name]=dict(path=str(path),sha256=c.sha(path),nq=m.nq,nv=m.nv,mass_kg=float(m.body_mass.sum()),
           source_timestep_s=float(m.opt.timestep),tip_axes=m.jnt_axis[ix.tip_joint_ids].tolist(),
           tip_stiffness=m.jnt_stiffness[ix.tip_joint_ids].tolist(),tip_damping=m.dof_damping[ix.tip_dof_addresses].tolist(),
           tip_ranges=m.jnt_range[ix.tip_joint_ids].tolist())
        ev=c.e.BASE_EVALUATOR(m,ix);c.mujoco.mj_resetData(m,ev.data);c.mujoco.mj_forward(m,ev.data)
        init=c.ORIGINAL_NEAR(ev)
        d=ev.data;body_x=d.xanchor[ix.active_joint_ids[ix.body_positions],0]
        leg_x=d.xanchor[ix.active_joint_ids[ix.leg_positions[:,0,0]],0]
        step=float(c.np.median(c.np.diff(body_x)))
        anterior_fraction=(leg_x-body_x)/step
        geometry[name]=dict(body_yaw_anchor_world_x=body_x.tolist(),leg_yaw_anchor_world_x=leg_x.tolist(),
            leg_position_in_joint_sample_steps=anterior_fraction.tolist(),median=float(c.np.median(anterior_fraction)),initialization=init)
    inputs={Path(__file__),c.ROOT/'core.py',c.ROOT/'screening.py',c.ROOT/'PROTOCOL.md',NEW_SELECTION,OLD_SELECTION,RL_SELECTION,
        FINE/'FINAL_RECEIPT.json',*c.PLANTS.values()}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).is_file() and Path(p).resolve().is_relative_to(c.EXP):inputs.add(Path(p).resolve())
    for p in inputs:manifest['frozen_inputs'][str(p)]=c.sha(p)
    c.write(c.DATA/'geometry_phase_audit.json',geometry)
    c.write(c.DATA/'manifest.json',manifest)
    c.write(c.DATA/'historical_amplitude_anchors.json',anchors())
    print(json.dumps(dict(status='PREPARED',active_pd_sha256=manifest['active_pd_sha256'],
        geometry_leg_sample_offsets={k:v['median'] for k,v in geometry.items()},frozen_inputs=len(inputs)),indent=2))

def verify(workers):
    old={r['wave']:r for r in c.read(NEW_SELECTION)['gaits']}
    tasks=[c.task(w,list(old[w]['amplitudes_deg'].values()),'verify',trace=True,prefix=False) for w in (1.5,4.,7.)]
    rows=c.bank(tasks,'verify',workers)
    checks=[]
    for r in rows:
        expected=old[r['wave']]['nominal_D_mm']
        checks.append(dict(wave=r['wave'],old_D=expected,new_D=r['D'],difference=None if r['D'] is None else r['D']-expected,
             match=bool(r['full'] and abs(r['D']-expected)<1e-8),pd_readback_match=(r.get('pd_readback') or {}).get('active_pd_sha256')==c.read(c.DATA/'manifest.json')['active_pd_sha256']))
    passed=all(x['match'] and x['pd_readback_match'] for x in checks)
    c.write(c.DATA/'reproduction_check.json',dict(passed=passed,checks=checks))
    print(json.dumps(dict(passed=passed,checks=checks),indent=2))
    assert passed

def bridge(workers):
    assert c.read(c.DATA/'reproduction_check.json')['passed']
    history=anchors();tasks=[]
    for w in (1.5,2.,3.,4.5,6.,7.):
        for family,mapping in history.items():
            choices=mapping.get(w)
            if choices is None:continue
            for plant in ('current','august31'):
                for phase in ('legacy','midpoint'):
                    tasks.append(c.task(w,choices,'bridge',plant=plant,phase=phase,trace=True,prefix=False))
    c.bank(tasks,'bridge',workers)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','bridge']);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='verify':verify(args.workers)
    else:bridge(args.workers)
