"""Registered broad 3-amplitude flat search, exact per-step rejection only."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import concurrent.futures
import copy
from collections import deque
import itertools
import json
import math
import time
from pathlib import Path
import engine as e

np,mujoco=e.np,e.mujoco
MANIFEST=e.DATA/'amplitude_manifest.json'
LOW=np.array([0.,0.,0.]);HIGH=np.array([36.,90.,60.])
KEYS=('body','leg_yaw','leg_pitch')
_CACHE=None
_MANIFEST_DATA=None
_REF_CACHE={}


def freeze(power=7):
    if MANIFEST.exists():
        old=json.loads(MANIFEST.read_text());assert old['sobol_power']==power
        return old
    from scipy.stats import qmc
    ledger=json.loads(e.fx.PD_LEDGER.read_text())
    amplitude=ledger['amplitudes_frozen_deg']
    profiles={};tasks=[];reference=[]
    for wave in e.WAVES:
        w=f'{wave:g}'
        if w in ledger['selected_profiles']:
            item=ledger['selected_profiles'][w]
            source=e.fx.PD_ROOT/'outputs'/'common'/f"{item['validation_ids'][0]}.json"
            provenance='historical validated PD, old axis'
        else:
            item=ledger['unresolved_wave_numbers'][w]
            source=e.fx.PD_ROOT/'outputs'/'common'/f"{item['best_diagnostic_id']}.json"
            provenance='historical diagnostic PD only, not validated'
        raw=json.loads(source.read_text());assert raw['case']['wave']==wave
        cfg=raw['config']
        profiles[w]=dict(config=cfg,source=str(source),sha256=e.fx.sha256(source),status=provenance)
        seed=92026+int(2*wave)
        samples=qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(power),LOW,HIGH)
        points=[('sobol',list(map(float,x))) for x in samples]
        base=np.array([amplitude[k] for k in KEYS])
        points += [('current_fixed_amplitudes',base.tolist()),('isaac_yaml_anchor',list(np.degrees([.4,.6,.4])))]
        for scale in (.5,.75,1.25):points.append((f'current_scale_{scale}',np.minimum(HIGH,base*scale).tolist()))
        for dim in range(3):
            ablate=base.copy();ablate[dim]=0.;points.append((f'axis_ablation_{dim}',ablate.tolist()))
        points += [('box_corner',(HIGH*np.asarray(x)).tolist()) for x in itertools.product((0.,1.),repeat=3)]
        for dim in range(3):
            for end in (0.,HIGH[dim]):
                face=HIGH/2;face[dim]=end;points.append((f'face_{dim}_{end}',face.tolist()))
        common=dict(topology='flexible',terrain='flat',wave=wave,config=cfg,duration=6.,dt_scale=1.,trace=False)
        reference.append(dict(common,stage='reference',amplitudes=dict(amplitude)))
        seen=set()
        for origin,x in points:
            key=tuple(round(float(v),10) for v in x)
            if key in seen:continue
            seen.add(key)
            tasks.append(dict(common,stage='broad',origin=origin,amplitudes=dict(zip(KEYS,map(float,x)))))
    payload=dict(scope='Centipede Model only',plant=str(e.model_path('flexible','flat')),
        plant_sha256=e.fx.sha256(e.model_path('flexible','flat')),waves=list(e.WAVES),
        topology='flexible primary; rigid remains paired control',
        PD_profiles_frozen=profiles,controller_law_unchanged=True,body_leg_wave_coupled=True,
        frequency_hz=1.,lower_deg=LOW.tolist(),upper_deg=HIGH.tolist(),sobol_power=power,
        candidate_count=len(tasks),references=reference,tasks=tasks,
        objective='maximize displacement after all gates; no scalar weighted reward',
        comparison_reference='same plant/PD/wave at fixed prior amplitudes; old-axis assurance also required at waves 2.5 and 3',
        posture_supplement=dict(coherent_roll_p95_deg='min(10,max(5,1.25*reference))',
            front_roll_p95_deg='min(12,max(6,1.25*reference))',center_height_ptp_mm='min(2,max(.75,1.25*reference))'),
        global_optimum_proved=False)
    e.DATA.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


def load_reference(task):
    global _MANIFEST_DATA
    if _MANIFEST_DATA is None:_MANIFEST_DATA=json.loads(MANIFEST.read_text())
    data=_MANIFEST_DATA
    base=next(t for t in data['references'] if t['wave']==task['wave'])
    base=dict(base,duration=task['duration'],dt_scale=task['dt_scale'])
    tid=e.task_id(base)
    path=e.TRIALS/'reference'/tid/f'{tid}.json'
    if tid not in _REF_CACHE:_REF_CACHE[tid]=json.loads(path.read_text())
    return _REF_CACHE[tid],path


def assess(r):
    if not r.get('metrics'):
        return dict(id=r['id'],wave=r['task']['wave'],feasible=False,failed=list(r['gates']),
            legacy_feasible=False,full=False,record=r['record_path'],D=None,amplitudes=r['task']['amplitudes'])
    task=r['task'];b,path=load_reference(task);a=r['metrics'];base=b['metrics']
    gates=dict(r['gates'])
    references=[('fixed_amp',base)]
    if task['wave'] in (2.5,3.):
        cfg=e.fx.pd_config(task['wave'])
        name=f"{cfg['name']}_w{task['wave']:g}_t{task['duration']:g}_dt{.00025*task['dt_scale']:g}".replace('.','p')
        old=e.fx.PD_ROOT/'outputs'/'common'/f'{name}.json'
        references.append(('legacy_axis',json.loads(old.read_text())['metrics']))
    for prefix,ref in references:
        gates[prefix+'_displacement']=a['displacement_mm']>=.9*max(0.,ref['displacement_mm'])
        gates[prefix+'_work_per_distance']=a['work_per_distance_n']<=1.25*ref['work_per_distance_n']
        for key in ('loaded_foot_slip_p95_mm_s','left_right_force_imbalance_rms_bw','normal_impulse_10ms_bw_s','chassis_impulse_10ms_bw_s'):
            gates[prefix+'_'+key]=a[key]<=1.1*ref[key]+1e-12
    for key,absolute,floor in (('coherent_roll_p95_deg',10.,5.),('front_roll_p95_deg',12.,6.),('center_height_ptp_mm',2.,.75)):
        gates['posture_'+key]=a[key]<=min(absolute,max(floor,1.25*base[key]))
    # Forward distance is the objective, not a percentage-of-unstable-reference
    # safety condition. Preserve the original floor as a separate diagnostic.
    hard={k:v for k,v in gates.items() if not k.endswith('_displacement')}
    return dict(id=r['id'],wave=task['wave'],feasible=all(hard.values()),failed=[k for k,v in hard.items() if not v],
        legacy_feasible=all(gates.values()),legacy_failed=[k for k,v in gates.items() if not v],
        full=True,record=r['record_path'],D=a['displacement_mm'],amplitudes=task['amplitudes'],
        gates=hard,legacy_gates=gates,metrics=a,reference=str(path))


def settled(task):
    global _CACHE
    key=json.dumps(dict(config=task['config'],wave=task['wave'],model=str(e.model_path('flexible','flat')),dt=task['dt_scale']),sort_keys=True)
    if _CACHE and _CACHE['key']==key:return _CACHE
    e.fx.opt.MODEL=e.model_path('flexible','flat')
    ev=e.bounded_evaluator();m,d,ix=ev.model,ev.data,ev.index
    e.fx.pilot.apply_impedance(m,ix,task['config']);m.opt.timestep*=task['dt_scale']
    mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d);init=e.fx.near.near_ground(ev)
    ctl=e.fx.opt.ThreeAmplitudeController(m,ix,task['wave'],*[task['amplitudes'][k] for k in KEYS])
    steps=int(round(1./m.opt.timestep));startup_torque=0.
    for _ in range(steps):
        ctl.step(d);mujoco.mj_step(m,d)
        startup_torque=max(startup_torque,float(np.max(np.abs(d.actuator_force[ix.active_actuator_ids]))))
    assert np.max(np.abs(ctl.previous_target))==0
    _CACHE=dict(key=key,ev=ev,snapshot=copy.copy(d),steps=steps,initialization=init,startup_torque=startup_torque,previous_target=ctl.previous_target.copy())
    return _CACHE


def precheck(task):
    state=settled(task);ev=state['ev'];m,ix=ev.model,ev.index;d=copy.copy(state['snapshot'])
    ctl=e.fx.opt.ThreeAmplitudeController(m,ix,task['wave'],*[task['amplitudes'][k] for k in KEYS])
    ctl.previous_target=state['previous_target'].copy()
    dt=float(m.opt.timestep);n=int(round(task['duration']/dt));stride=int(round(.001/dt))
    cf=np.zeros(6);weight=float(m.body_mass.sum()*9.81)
    ground=ev.ground;chassis=set(map(int,ix.chassis_body_ids));foot=set(map(int,ev.foot_geoms))
    base,_=load_reference(task);bm=base['metrics']
    limits=dict(coherent_roll=min(10.,max(5.,1.25*bm['coherent_roll_p95_deg'])),
        front_roll=min(12.,max(6.,1.25*bm['front_roll_p95_deg'])))
    ns=len(range(int(round(1.5/dt)),n,stride));bad={k:0 for k in ('coherent_roll','front_roll','leg_yaw_speed','leg_pitch_speed','swing_height')}
    bad_chassis=0;max_final_fine_steps=n-int(round(1.5/dt))
    history=deque(maxlen=11);height_low=float('inf');height_high=-float('inf')
    bodyqa=np.stack((ix.qpos_addresses[ix.body_positions],ix.qpos_addresses[ix.pitch_positions],ix.roll_qpos_addresses),axis=-1)
    footsites=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'site_foot_{j}_{side}') for j in range(19) for side in range(2)])
    evidence=[];checked=1.
    # Prefix rejection only: survivors receive one complete cold 6 s evaluation.
    # Quantile count budgets still use the FULL requested horizon, never 2.5 s.
    for k in range(state['steps'],min(n,int(round(2.5/dt)))):
        sampled=k>=int(round(1.5/dt)) and k%stride==0
        if sampled:
            legv=d.qvel[ix.dof_addresses[ix.leg_positions.reshape(-1)]].reshape(19,2,2).copy()
            bodyq=d.qpos[bodyqa].copy()
        ctl.step(d);mujoco.mj_step(m,d);checked=(k+1)*dt
        if np.any(d.warning.number):
            warnings=e.fx.opt.tip.stage1.warning_counts(d)
            if any('CNSTRFULL' in key or 'CONTACTFULL' in key for key in warnings):
                raise RuntimeError('Constraint/contact allocation warning during precheck; evaluation invalid')
            evidence=[dict(gate='finite_warning_free',value=sum(warnings.values()),limit=0,warning_counts=warnings)]
            break
        vals=[('torque',float(np.max(np.abs(d.actuator_force[ix.active_actuator_ids]))),2e-4)]
        if k>=int(round(1.5/dt)):
            vals.extend((name,float(np.max(np.abs(d.qvel[addresses]))),limit) for name,addresses,limit in (
                ('body_yaw_speed',ix.dof_addresses[ix.body_positions],12.),
                ('body_pitch_speed',ix.dof_addresses[ix.pitch_positions],8.),
                ('body_roll_speed',ix.roll_dof_addresses,5.),('active_speed',ix.dof_addresses,100.),
                ('tip_speed',ix.tip_dof_addresses,1000.)))
            vals.append(('tip_peak',float(np.degrees(np.max(np.abs(d.qpos[ix.tip_qpos_addresses])))),40.))
            if k%stride==0:
                rm=d.xmat[int(ix.chassis_body_ids[0])].reshape(3,3)
                vals.append(('root_roll_peak',abs(float(np.degrees(np.arctan2(rm[2,1],rm[2,2])))),20.))
                vals.append(('upright',-float(rm[2,2]),-math.cos(math.radians(50))))
                wr=np.radians(e.fx.opt.tip.signed_world_roll_deg(d,ix.chassis_body_ids))
                coherent=abs(float(np.degrees(np.arctan2(np.sin(wr).mean(),np.cos(wr).mean()))))
                front=abs(float(np.degrees(np.arctan2(np.sin(wr[:7]).mean(),np.cos(wr[:7]).mean()))))
                for name,value in (('coherent_roll',coherent),('front_roll',front)):
                    bad[name]+=int(value>limits[name])
                    vals.append((name+'_quantile_irreversible',bad[name],math.ceil(.05*ns)+1))
                for j,name,limit in ((0,'leg_yaw_speed',8.),(1,'leg_pitch_speed',7.)):
                    bad[name]+=int(np.count_nonzero(np.abs(legv[...,j])>limit))
                    vals.append((name+'_quantile_irreversible',bad[name],math.ceil(.01*ns*38)+1))
                xyz=d.xpos[ix.chassis_body_ids];height=float(xyz[:,2].mean()*1000)
                height_low=min(height_low,height);height_high=max(height_high,height)
                vals.append(('posture_center_height_ptp_mm',height_high-height_low,min(2.,max(.75,1.25*bm['center_height_ptp_mm']))))
                vals.append(('spine',float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()/max(np.linalg.norm(xyz[-1]-xyz[0]),1e-15)),1.2))
                omega=d.cvel[ix.chassis_body_ids,:3].copy();history.append((bodyq,omega))
                if len(history)==11:
                    dq=np.degrees(np.abs(bodyq-history[0][0]))
                    vals.extend([('pitch_10ms',float(dq[:,1].max()),2.),('roll_10ms',float(dq[:,2].max()),2.),
                        ('body_velocity_jump',float(np.linalg.norm(omega-history[0][1],axis=1).max()),10.)])
            if not any(v>limit for _,v,limit in vals):
                total=body=0.;feet={};touching=set();chassis_loaded=False
                for i in range(d.ncon):
                    con=d.contact[i];g1,g2=int(con.geom1),int(con.geom2)
                    other=g2 if g1==ground else (g1 if g2==ground else -1)
                    if other<0:continue
                    touching.add(other)
                    mujoco.mj_contactForce(m,d,i,cf);fn=max(0.,float(cf[0]));total+=fn
                    if int(m.geom_bodyid[other]) in chassis:
                        body+=fn
                        # A single chassis contact > threshold proves the full
                        # nonnegative summed chassis force > threshold. This
                        # lower bound avoids any summation-roundoff ambiguity.
                        if other not in foot and fn>1e-7:chassis_loaded=True
                    if other in foot:feet[other]=feet.get(other,0.)+fn
                vals.extend([('total_force',total/weight,2.5),('chassis_force',body/weight,1.),('foot_force',max(feet.values(),default=0.)/weight,1.)])
                bad_chassis+=int(chassis_loaded)
                vals.append(('chassis_contact_irreversible',bad_chassis,math.ceil(.03*max_final_fine_steps)+1))
                if sampled:
                    swing=np.array([int(g) not in touching for g in ev.foot_geoms])
                    bad['swing_height']+=int(np.count_nonzero(swing&(d.site_xpos[footsites,2]*1000>5.5)))
                    vals.append(('swing_height_quantile_irreversible',bad['swing_height'],math.ceil(.05*ns*38)+1))
        evidence=[dict(gate=name,value=value,limit=limit) for name,value,limit in vals if not np.isfinite(value) or value>limit]
        if evidence:break
    if state['startup_torque']>2e-4:evidence.append(dict(gate='torque',value=state['startup_torque'],limit=2e-4))
    if not evidence:return None
    tid=e.task_id(task);path=e.TRIALS/task['stage']/tid/f'{tid}.json'
    r=dict(id=tid,task=task,model_path=str(e.model_path('flexible','flat')),model_sha256=e.fx.sha256(e.model_path('flexible','flat')),
        record_path=str(path),proven_infeasible=True,metrics=None,gates={x['gate']:False for x in evidence},
        failure_evidence=evidence,checked_until_s=checked,remaining_gates='not evaluated',
        irreversible_quantile_proof=dict(full_pose_samples=ns,strict_over_limit_counts=bad,
            posture_limits=limits,rule='bad count > ceil((1-q)*maximum_final_sample_count)+1'),
        chassis_contact_proof=dict(bad_step_lower_bound=bad_chassis,max_final_fine_steps=max_final_fine_steps,
            fraction_limit=.03,rule='single chassis contact force > 1e-7; count > ceil(0.03*max final steps)+1'),
        objective='not ranked: incomplete rollout',startup_reuse='exact independent MjData copy after common zero-target 1 s settling',
        runtime_arena_bytes=int(m.narena),maxuse_arena_bytes=int(d.maxuse_arena))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r


def screen(task):
    tid=e.task_id(task);path=e.TRIALS/task['stage']/tid/f'{tid}.json'
    if path.exists():
        r=json.loads(path.read_text())
        if r['model_sha256']==e.fx.sha256(e.model_path('flexible','flat')):return r
    base,_=load_reference(task)
    if base['id']==tid:return base
    fail=precheck(task)
    return fail if fail else e.run_one(task)


def summarize(records,name):
    assessed=[assess(r) for r in records]
    waves={}
    for w in e.WAVES:
        rows=[r for r in assessed if r['wave']==w]
        passed=sorted([r for r in rows if r['feasible']],key=lambda r:(-r['D'],sum(r['amplitudes'].values())))
        diagnostic=sorted([r for r in rows if r['full']],key=lambda r:(len(r['failed']),-r['D']))
        waves[f'{w:g}']=dict(count=len(rows),full=sum(r['full'] for r in rows),feasible=len(passed),
            best=passed[0] if passed else None,top=passed[:10],diagnostic=diagnostic[:3])
    payload=dict(stage=name,global_optimum_proved=False,objective_definition='stable first; old 90% distance floor diagnostic only',assessments=assessed,waves=waves)
    (e.DATA/f'{name}_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


def run_bank(tasks,workers,name):
    start=time.monotonic();records=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(screen,t):t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:r=f.result()
            except Exception as exc:
                error=dict(task=futures[f],error=repr(exc));errors.append(error)
                print(json.dumps(error),flush=True)
                (e.DATA/f'{name}_errors.json').write_text(json.dumps(errors,indent=2))
                continue
            records.append(r)
            if i%10==0 or r.get('metrics'):
                a=assess(r)
                print(json.dumps(dict(done=i,total=len(tasks),id=r['id'],D=a['D'],feasible=a['feasible'],
                    failed=a['failed'],elapsed_s=time.monotonic()-start)),flush=True)
    result=summarize(records,name)
    if errors:raise RuntimeError(f'{len(errors)} resource/evaluation errors; rerun to resume')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','reference','broad'))
    p.add_argument('--workers',type=int,default=12);p.add_argument('--power',type=int,default=7)
    args=p.parse_args();manifest=freeze(args.power)
    if args.command=='freeze':print(json.dumps({k:v for k,v in manifest.items() if k not in ('tasks','references')},indent=2))
    elif args.command=='reference':
        rows=e.run_tasks(manifest['references'],args.workers)
        (e.DATA/'amplitude_references.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    else:
        result=run_bank(manifest['tasks'],args.workers,'broad')
        print(json.dumps({w:dict(count=x['count'],full=x['full'],feasible=x['feasible']) for w,x in result['waves'].items()},indent=2))
