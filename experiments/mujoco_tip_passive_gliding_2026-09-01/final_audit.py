"""Final frozen-plant and inherited non-regression assurance audit."""
from __future__ import annotations
import importlib.util
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_fx',ROOT/'run_tip_flexion_search.py')
fx=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fx;spec.loader.exec_module(fx)
np,mujoco=fx.np,fx.mujoco
SELECTED='localz_k2em05_d3p2_r38'
FIELDS=('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
        'dof_armature','geom_size','geom_pos','geom_quat','geom_friction','geom_solref',
        'geom_solimp','geom_contype','geom_conaffinity','jnt_range','jnt_pos','qpos0',
        'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_trnid')


def canonical(node):
    attrs=dict(node.attrib)
    if node.tag=='mujoco':attrs.pop('model',None)
    if node.tag=='joint' and attrs.get('name','').startswith('joint_tip_'):
        for name in ('axis','stiffness','damping','range'):attrs.pop(name,None)
    return (node.tag,tuple(sorted(attrs.items())),(node.text or '').strip(),tuple(canonical(ch) for ch in node))


def audit_plant():
    xml_checks=[];compiled=[]
    for terrain in ('flat','rough'):
        source=fx.source_model(terrain)
        old_tree=ET.parse(source).getroot()
        old_canonical=canonical(old_tree)
        for path in fx.MODELS.glob(f'centipede_{terrain}_*.xml'):
            assert canonical(ET.parse(path).getroot())==old_canonical,path
            xml_checks.append(dict(path=str(path),hash=fx.sha256(path),only_registered_tip_fields_differ=True))
        target=fx.MODELS/f'centipede_{terrain}_{SELECTED}.xml'
        a=mujoco.MjModel.from_xml_path(str(source));b=mujoco.MjModel.from_xml_path(str(target))
        ia=fx.opt.tip.LeanIndex(a);ib=fx.opt.tip.LeanIndex(b)
        checks={f:bool(np.array_equal(getattr(a,f),getattr(b,f))) for f in FIELDS}
        assert all(checks.values()),checks
        other_joints=np.setdiff1d(np.arange(a.njnt),ia.tip_joint_ids)
        other_dofs=np.setdiff1d(np.arange(a.nv),ia.tip_dof_addresses)
        assert np.array_equal(a.jnt_axis[other_joints],b.jnt_axis[other_joints])
        assert np.array_equal(a.jnt_stiffness[other_joints],b.jnt_stiffness[other_joints])
        assert np.array_equal(a.dof_damping[other_dofs],b.dof_damping[other_dofs])
        assert np.allclose(b.jnt_axis[ib.tip_joint_ids],[0,0,1])
        assert not np.any(b.dof_armature)
        for wave in fx.WAVES:
            aa=mujoco.MjModel.from_xml_path(str(source));bb=mujoco.MjModel.from_xml_path(str(target))
            ixa=fx.opt.tip.LeanIndex(aa);ixb=fx.opt.tip.LeanIndex(bb)
            fx.pilot.apply_impedance(aa,ixa,fx.pd_config(wave));fx.pilot.apply_impedance(bb,ixb,fx.pd_config(wave))
            assert np.array_equal(ixa.kp,ixb.kp) and np.array_equal(ixa.kv,ixb.kv)
            assert np.array_equal(aa.actuator_gainprm,bb.actuator_gainprm)
            assert np.array_equal(aa.actuator_biasprm,bb.actuator_biasprm)
        compiled.append(dict(terrain=terrain,source_hash=fx.sha256(source),target_hash=fx.sha256(target),
            frozen_fields=checks,mass_g=float(b.body_mass.sum()*1000),nq=b.nq,nv=b.nv,nu=b.nu,
            tip_dof_count=38,armature_zero=True,active_pd_arrays_identical=True))
    return dict(xml_variants_checked=len(xml_checks),xml_checks=xml_checks,compiled_selected=compiled,
        old_axis_inertia=fx.opt.tip._tip_inertia(fx.source_model('flat')),
        local_z_inertia=fx.opt.tip._tip_inertia(fx.MODELS/f'centipede_flat_{SELECTED}.xml'))


def baseline(wave,duration,dt):
    cfg=fx.pd_config(wave)
    tag=f"{cfg['name']}_w{wave:g}_t{duration:g}_dt{dt:g}".replace('.','p')
    path=fx.PD_ROOT/'outputs'/'common'/f'{tag}.json'
    return json.loads(path.read_text(encoding='utf-8')),path


def assurance(record):
    old,path=baseline(record['case']['wave'],record['duration'],record['dt'])
    a,b=record['metrics'],old['metrics']
    assert record['config']==old['config'],(record['config'],old['config'])
    assert record['case']==old['case']
    gates=dict(record['gates'])
    gates['leg_yaw_tracking']=a['leg_yaw_tracking_rms_deg']<=10.
    gates['displacement_retention']=a['displacement_mm']>=.9*b['displacement_mm']
    gates['work_per_distance']=a['work_per_distance_n']<=1.25*b['work_per_distance_n']
    for key in ('loaded_foot_slip_p95_mm_s','left_right_force_imbalance_rms_bw','normal_impulse_10ms_bw_s','chassis_impulse_10ms_bw_s'):
        gates['nonregression_'+key]=a[key]<=1.1*b[key]+1e-12
    return dict(id=record['experiment_id'],candidate=record['model_candidate']['name'],wave=record['case']['wave'],
        duration=record['duration'],dt=record['dt'],baseline=str(path),gates=gates,
        feasible=all(gates.values()),failed=[k for k,v in gates.items() if not v],
        displacement_retention=a['displacement_mm']/b['displacement_mm'],
        displacement_mm=a['displacement_mm'],tip_speed_peak=a['tip_speed_peak'],metrics=a)


def main():
    global SELECTED
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',default=SELECTED)
    args=parser.parse_args()
    SELECTED=args.candidate
    plant=audit_plant()
    records=[]
    for path in fx.TRIALS.rglob('*.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('experiment_id')==path.stem:records.append(r)
    checked=[assurance(r) for r in records if r['terrain']=='flat']
    nominal=[r for r in checked if r['duration']==6 and r['dt']==.00025]
    grouped={}
    for r in nominal:grouped.setdefault(r['candidate'],[]).append(r)
    ranking=[]
    for name,rows in grouped.items():
        ranking.append(dict(candidate=name,feasible=len(rows)==2 and all(r['feasible'] for r in rows),
            min_retention=min(r['displacement_retention'] for r in rows),
            max_tip_speed=max(r['tip_speed_peak'] for r in rows),
            failed={str(r['wave']):r['failed'] for r in rows}))
    ranking.sort(key=lambda r:(not r['feasible'],-r['min_retention'],r['max_tip_speed']))
    selected_checks=[r for r in checked if r['candidate']==SELECTED]
    robustness=[]
    for candidate in ranking:
        rows=[r for r in checked if r['candidate']==candidate['candidate']]
        robustness.append(dict(candidate=candidate['candidate'],conditions_completed=len(rows),
            fully_flat_assured=len(rows)==6 and all(r['feasible'] for r in rows),
            failed={r['id']:r['failed'] for r in rows if r['failed']}))
    selected_rough=[r for r in records if r['terrain']=='rough' and r['model_candidate']['name']==SELECTED]
    payload=dict(plant=plant,unique_condition_runs=len(records),flat_nominal_count=len(nominal),
        rough_unique_count=sum(r['terrain']=='rough' for r in records),
        inherited_assurance='same numeric non-regression thresholds as near-ground PD protocol; matched old-axis selected-PD references',
        all_flat_assessments=checked,nominal_ranking=ranking,robustness=robustness,selected=SELECTED,selected_flat=selected_checks,
        selected_flat_all_assurance_pass=len(selected_checks)==6 and all(r['feasible'] for r in selected_checks),
        selected_rough_all_absolute_gates_pass=len(selected_rough)==2 and all(all(r['gates'].values()) for r in selected_rough),
        selected_rough=[dict(wave=r['case']['wave'],metrics=r['metrics'],failed=[k for k,v in r['gates'].items() if not v]) for r in selected_rough],
        default_model_replaced=False,continuous_global_optimum_proved=False)
    (fx.DATA/'contract_and_selection.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('plant','all_flat_assessments','selected_flat','selected_rough')},indent=2))
    print(json.dumps([dict(id=r['id'],D=r['displacement_mm'],failed=r['failed']) for r in selected_checks],indent=2))


if __name__=='__main__':main()
