"""Frozen physical fields, exact gain values, and initialization-only transform."""
import json
from run_search import ROOT,OUT,np,mujoco,opt,pilot,cfg,PREVIOUS,near_ground

FIELDS=('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','dof_armature',
        'geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp',
        'geom_contype','geom_conaffinity','jnt_range','jnt_axis','jnt_pos','qpos0')

def main():
    bank=[cfg('baseline'),PREVIOUS]
    for name in ('broad_bank.json','coordinated_bank.json','basin15_bank.json','basin18_bank.json','basin2906_bank.json','legyaw_boundary_bank.json','legyaw_refine_bank.json'):
        bank+=json.loads((ROOT/name).read_text())
    records=[]
    for c in bank:
        ev=opt.FastEvaluator();m,d,ix=ev.model,ev.data,ev.index
        before={key:getattr(m,key).copy() for key in FIELDS}
        tips=(m.jnt_stiffness[ix.tip_joint_ids].copy(),m.dof_damping[ix.tip_dof_addresses].copy())
        pilot.apply_impedance(m,ix,c)
        mujoco.mj_forward(m,d);init=near_ground(ev)
        assert all(np.array_equal(before[key],getattr(m,key)) for key in FIELDS)
        assert np.array_equal(tips[0],m.jnt_stiffness[ix.tip_joint_ids])
        assert np.array_equal(tips[1],m.dof_damping[ix.tip_dof_addresses])
        assert not np.any(m.dof_armature)
        gain={}
        for name,pos in [('body_yaw',ix.body_positions),('body_pitch',ix.pitch_positions),('leg_yaw',ix.leg_positions[:,:,0].reshape(-1)),('leg_pitch',ix.leg_positions[:,:,1].reshape(-1))]:
            gain[name]=dict(kp_nm_per_rad=float(ix.kp[pos][0]),kd_nms_per_rad=float(ix.kv[pos][0]))
        gain['passive_body_roll']=dict(k_nm_per_rad=float(m.jnt_stiffness[ix.roll_joint_ids][0]),c_nms_per_rad=float(m.dof_damping[ix.roll_dof_addresses][0]))
        records.append(dict(name=c['name'],frozen_fields_equal=True,absolute_gains=gain))
    r=dict(model_sha256=opt.tip.sha256(opt.MODEL),mass_g=float(m.body_mass.sum()*1000),
        checked_candidates=len(records),initialization=init,candidates=records)
    (OUT/'contract_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='candidates'}))

if __name__=='__main__':main()
