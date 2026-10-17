"""Compare static MuJoCo joint bounds with the saved native backend readback."""
from pathlib import Path
import datetime,hashlib,json,re
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parent
NATIVE=ROOT.parents[2]/'Centipede_IsaacLab_RL/outputs/analysis/common_pd_amplitude_contact_20260909/physical_contract_readback'
MODEL=ROOT.parent/'mujoco_tip_passive_gliding_2026-09-01/models/centipede_flat_localz_k2em05_d3p2_r38.xml'


def mapping(name):
    match=re.fullmatch(r'body_(yaw|pitch|roll)_(\d+)',name)
    if match:
        family,index=match.groups()
        if int(index)==0:return 'head_'+family,None
        stem={'yaw':'joint_body','pitch':'joint_pitch_body','roll':'joint_roll_body'}[family]
        return 'body_'+family,f'{stem}_{index}'
    match=re.fullmatch(r'legbase_(yaw|pitch)_(\d+)_(L|R)',name)
    if match:
        family,index,side=match.groups()
        return 'leg_'+family,f'joint_leg_{index}_{side}_{0 if family=="yaw" else 1}'
    match=re.fullmatch(r'tarsus_yaw_(\d+)_(L|R)',name)
    assert match,name
    index,side=match.groups()
    return 'tip',f'joint_tip_{index}_{side}_roll'


def main():
    native=json.loads((NATIVE/'physical_contract_readback.json').read_text(encoding='utf-8'))
    with np.load(NATIVE/'backend_arrays.npz',allow_pickle=False) as z:
        array=z['dof_limits'];np.testing.assert_array_equal(array,np.broadcast_to(array[:1],array.shape));limits=array[0].copy()
    model=mujoco.MjModel.from_xml_path(str(MODEL));rows=[];unmatched=[]
    for i,name in enumerate(native['joint_names']):
        family,target=mapping(name)
        if target is None:
            unmatched.append(dict(native_name=name,family=family,native_limit_deg=np.degrees(limits[i]).tolist()));continue
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,target);assert j>=0,target
        rows.append(dict(native_name=name,mujoco_name=target,family=family,
            mujoco_limit_deg=np.degrees(model.jnt_range[j]).tolist(),native_limit_deg=np.degrees(limits[i]).tolist(),
            max_difference_deg=float(np.max(np.abs(np.degrees(model.jnt_range[j]-limits[i])))),
            mujoco_limited=bool(model.jnt_limited[j])))
    assert len(rows)==171 and len(unmatched)==2
    groups=[]
    for family in sorted({r['family'] for r in rows}):
        chosen=[r for r in rows if r['family']==family]
        groups.append(dict(family=family,n=len(chosen),
            mujoco_ranges_deg=np.unique(np.round([r['mujoco_limit_deg'] for r in chosen],5),axis=0).tolist(),
            native_ranges_deg=np.unique(np.round([r['native_limit_deg'] for r in chosen],5),axis=0).tolist(),
            max_difference_deg=max(r['max_difference_deg'] for r in chosen),
            differences_over_001deg=sum(r['max_difference_deg']>.01 for r in chosen)))
    files=[Path(__file__),MODEL,NATIVE/'backend_arrays.npz',NATIVE/'physical_contract_readback.json']
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),physics_steps=0,matched_joints=len(rows),
        family_summary=groups,native_head_joints_absent_from_mujoco=unmatched,rows=rows,
        interpretation='Named functional-joint correspondence; MuJoCo compiled-model position bounds versus saved actual PhysX bounds. This is not an axis, torque-cap, or solver-equivalence test, and does not isolate the performance effect of a bound difference.',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},mujoco_version=mujoco.__version__)
    (ROOT/'outputs/data/joint_limit_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(families=groups,native_only=unmatched),ensure_ascii=False))


if __name__=='__main__':main()
