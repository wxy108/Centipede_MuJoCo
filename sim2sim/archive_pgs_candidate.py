import json
import shared as s
profile=json.loads((s.ROOT/'OPERATING_PROFILE.json').read_text())
summary=json.loads((s.ROOT/'audit/coupled_candidate_summary.json').read_text())
assert summary['status']=='FAILED_CRITERIA' and not summary['missing']
if profile['name']=='common_physics_coupled_pd_native_gpu':
    profile['status']='rejected_gpu_refinement';profile['rejection_evidence']='audit/coupled_candidate_summary.json';profile['failed_criteria']=summary['failed']
    s.write(s.ROOT/'OPERATING_PROFILE_PGS_NOT_ADOPTED.json',profile)
    s.write(s.ROOT/'OPERATING_PROFILE.json',profile)
for name in ['REPORT.md','AUDIT_AND_STATUS.zh-CN.md','RUNTIME_ALIGNMENT.md']:
    target=s.ROOT/(name.rsplit('.',1)[0]+'_PGS_NOT_ADOPTED.'+name.rsplit('.',1)[1])
    if not target.exists():target.write_bytes((s.ROOT/name).read_bytes())
s.write(s.ROOT/'TGS_CANDIDATE_PROFILE.json',dict(status='validation_pending',name='common_physics_coupled_pd_tgs_external_each_iteration',physical_contract='COMMON_CONTRACT.json',mujoco_runner='run_mujoco_v8.py',native_runner='run_native_v13.py',mujoco_dt_s=.00003125,mujoco_refinement_dt_s=.000015625,native_dt_s=.00025,native_refinement_dt_s=.000125,command_dt_s=.00025,duration_s=6.,native_solver='TGS',native_position_iterations=16,native_velocity_iterations=4,external_forces_every_iteration=True,physical_gain_fitting=False,physical_friction=.5,physical_friction_model='circular Coulomb tangent limit, independently tested on both native backends',scope='Open-loop flat-ground common-physics benchmark; CPU and GPU require independent complete validation. Original RL training is not migrated.',amendment='NUMERICAL_AMENDMENT_13.md'))
print('PGS candidate retained as rejected; TGS candidate remains pending.')
