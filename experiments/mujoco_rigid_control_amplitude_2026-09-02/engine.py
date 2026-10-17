"""Frozen-plant paired ablation and amplitude-only evaluations. Model project only."""
from __future__ import annotations
import concurrent.futures
import importlib.util
import json
import hashlib
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
PREVIOUS=ROOT.parent/'mujoco_tip_passive_gliding_2026-09-01'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

fx=load(PREVIOUS/'run_tip_flexion_search.py','rigid_control_flexion')
ab=load(ROOT.parent/'mujoco_rigid_tip_ablation_2026-09-01'/'run_ablation.py','rigid_control_reference')
builder=load(ROOT.parent/'mujoco_rigid_tip_ablation_2026-09-01'/'build_rigid_tip_model.py','rigid_control_builder')
np,mujoco=fx.np,fx.mujoco
SOURCE_ID='localz_k2em05_d3p2_r38'
OUT=ROOT/'outputs';MODELS=ROOT/'models';DATA=OUT/'data';TRIALS=OUT/'trials'
ORIGINAL_RIGID_INIT=ab.near.near_ground
WAVES=(.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,7.,8.,9.)
BASE_EVALUATOR=fx.opt.FastEvaluator


def runtime_tree(source):
    tree=ET.parse(source);size=tree.getroot().find('size')
    if size is None:size=ET.SubElement(tree.getroot(),'size')
    size.attrib.pop('nstack',None);size.attrib.pop('njmax',None)
    size.set('memory','64M')
    return tree


def runtime_model(source):
    return mujoco.MjModel.from_xml_string(ET.tostring(runtime_tree(source).getroot(),encoding='unicode'))


def bounded_evaluator(shared_model=None,shared_index=None):
    if shared_model is None or shared_index is None:
        shared_model=runtime_model(fx.opt.MODEL)
        shared_index=fx.opt.tip.LeanIndex(shared_model)
        fx.opt.tip.configure_model(shared_model,shared_index)
    return BASE_EVALUATOR(shared_model,shared_index)


def model_path(topology,terrain):
    if topology=='flexible':return fx.MODELS/f'centipede_{terrain}_{SOURCE_ID}.xml'
    return MODELS/f'centipede_{terrain}_rigid_from_localz.xml'


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    audits=[]
    for terrain in ('flat','rough'):
        builder.SOURCE=model_path('flexible',terrain)
        builder.MODEL=model_path('rigid',terrain)
        builder.OUT=OUT/'plant_audit'/terrain
        result=builder.build()
        # Verify identical active PD arrays despite the removed 38 tip DOFs.
        for wave in (2.5,3.):
            a=mujoco.MjModel.from_xml_path(str(builder.SOURCE));b=mujoco.MjModel.from_xml_path(str(builder.MODEL))
            ai=fx.opt.tip.LeanIndex(a);bi=ab.RigidTipIndex(b)
            fx.pilot.apply_impedance(a,ai,fx.pd_config(wave));ab.pilot.apply_impedance(b,bi,fx.pd_config(wave))
            assert np.array_equal(ai.kp,bi.kp) and np.array_equal(ai.kv,bi.kv)
            assert np.array_equal(a.actuator_gainprm,b.actuator_gainprm)
            assert np.array_equal(a.actuator_biasprm,b.actuator_biasprm)
            assert not np.any(a.dof_armature) and not np.any(b.dof_armature)
        result['active_PD_exactly_equal']=True
        audits.append(result)
    (DATA/'paired_plant_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    return audits


def task_id(task):
    cfg={k:v for k,v in task['config'].items() if k!='name'}
    identity={k:task[k] for k in ('topology','terrain','wave','amplitudes','duration','dt_scale')}
    identity['config']=cfg
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:14]
    return f"{task['topology']}_{task['terrain']}_w{task['wave']:g}_{digest}".replace('.','p')


def applicable_gates(result):
    gates=dict(result['gates'])
    gates.pop('yaw_tracking',None)
    gates['leg_yaw_tracking']=result['metrics']['leg_yaw_tracking_rms_deg']<=10
    return gates


def run_one(task):
    tid=task_id(task);folder=TRIALS/task.get('stage','ablation')/tid
    path=folder/f'{tid}.json';model=model_path(task['topology'],task['terrain'])
    if path.exists() and (not task.get('trace') or path.with_suffix('.npz').exists()):
        record=json.loads(path.read_text())
        if record['model_sha256']==fx.sha256(model) and record.get('metrics'):return record
    cfg=dict(task['config']);wave=task['wave'];case=dict(task['amplitudes'],wave=wave)
    if task['topology']=='flexible':
        fx.opt.MODEL=model
        initializer=fx.near.near_ground if task['terrain']=='flat' else fx.rough_near_ground
        original=fx.opt.FastEvaluator;fx.opt.FastEvaluator=bounded_evaluator
        try:
            result=fx.pilot.run_task((cfg,wave,task['duration'],task['dt_scale'],task.get('trace',False)),
                initialize=initializer,case_override=case,output_dir=folder)
        finally:fx.opt.FastEvaluator=original
        result['runtime_arena_bytes']=64*1024*1024
    else:
        ab.MODEL=model;ab.OUT=folder
        ab.selected_config=lambda w:dict(cfg)
        ab.fixed_case=lambda w:dict(case)
        ab.near.near_ground=ORIGINAL_RIGID_INIT if task['terrain']=='flat' else fx.rough_near_ground
        result=ab.run(wave,task['duration'],task['dt_scale'],task.get('trace',False))
    legacy=result['id']
    if any('CNSTRFULL' in str(k) or 'CONTACTFULL' in str(k) for k in result.get('warnings',{})):
        raise RuntimeError('Constraint/contact allocation warning: evaluation invalid, not a physical gate failure')
    result.update(id=tid,task=task,model_path=str(model),model_sha256=fx.sha256(model),
        terrain=task['terrain'],topology=task['topology'],record_path=str(path),
        controller_law_unchanged=True,contact_erasure=False)
    result['gates']=applicable_gates(result)
    result['absolute_feasible']=all(result['gates'].values())
    result['failed']=[k for k,v in result['gates'].items() if not v]
    if task.get('trace'):
        old=folder/f'{legacy}.npz'
        if old.exists():old.replace(path.with_suffix('.npz'))
    folder.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def run_tasks(tasks,workers=4):
    records=[];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            r=f.result();records.append(r)
            print(json.dumps(dict(done=i,total=len(tasks),id=r['id'],D=r['metrics']['displacement_mm'],
                failed=r['failed'],elapsed_s=time.monotonic()-start)),flush=True)
    return records


def paired_tasks():
    return [dict(stage='ablation',topology=topology,terrain=terrain,wave=wave,
        config=fx.pd_config(wave),amplitudes={k:v for k,v in fx.amplitude_case(wave).items() if k!='wave'},
        duration=6.,dt_scale=1.,trace=True)
        for terrain in ('flat','rough') for wave in (2.5,3.) for topology in ('flexible','rigid')]


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','ablation'))
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(),indent=2))
    else:
        rows=run_tasks(paired_tasks(),args.workers)
        (DATA/'paired_ablation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
