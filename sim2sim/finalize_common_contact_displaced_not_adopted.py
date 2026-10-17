# Archived displaced-origin candidate; retained for provenance only.
raise RuntimeError('Displaced-origin candidate was superseded by the centered layout. No promotion is allowed.')
"""Promote the shared-contact profile only after complete measured evidence."""
import argparse,json
from datetime import datetime,timezone
import numpy as np
import shared as s
R=s.ROOT;p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');a=p.parse_args()
summary=json.loads((R/'audit/common_contact_summary.json').read_text())
if not a.finalize:
    print(json.dumps(dict(status=summary['status'],missing=len(summary['missing']),failed=summary['failed'],promotion_performed=False)));raise SystemExit(0)
assert summary['status']=='NUMERICAL_GATES_PASSED_QUALITY_REVIEW_REQUIRED' and not summary['missing'] and not summary['failed']
checks=[];compression=[]
def inspect(name,expected_rows,dt,native):
    f=R/'runs'/name;result=json.loads((f/'results.json').read_text());quality=json.loads((f/'quality_audit.json').read_text());tr=np.load(f/'trajectory.npz')
    assert len(result['rows'])==len(expected_rows) and tr['t'].shape==(601,) and np.allclose(tr['t'],np.arange(601)*.01,rtol=0,atol=1e-10)
    assert result['arguments']['dt']==dt and result['arguments']['command_dt']==.00025 and 'COMMON_CONTACT_V2' in result['contact_model']
    assert np.isfinite(tr['q']).all() and np.isfinite(tr['v']).all() and np.isfinite(tr['body']).all()
    ids=[tr['body_names'].tolist().index(f'link_body_{i}') for i in range(21)]
    for e,c in enumerate(expected_rows):
        r=result['rows'][e];assert r['finite'] and r['complete_s']==6 and r['wave']==c['wave'] and r['amplitudes']==c['amplitudes'] and r['D_mm']>=50
        D=-float(tr['body'][600,e,ids,0].astype(float).mean()-tr['body'][200,e,ids,0].astype(float).mean())*1000
        assert abs(D-r['D_mm'])<.001,(name,e,D,r['D_mm'])
    # The frozen compliant model permits finite penetration as deformation proxy.
    # Report it explicitly; do not silently import the old rigid zero-overlap gate.
    for q in quality['rows']:assert q['finite'] and q['inverted_spine_samples']==0
    ca=json.loads((f/'compliance_geometry_audit.json').read_text());compression.append(dict(run=name,quality=quality['rows'],mesh_compression=ca['rows']))
    if native:
        meta=json.loads((f/'readback.json').read_text());done=json.loads((f/'complete.json').read_text());b=json.loads((f/'common_contact_backend_validation.json').read_text())
        assert b['passed'] and done['complete'] and done['physics_steps']==round(6/dt) and not result['stopped'] and abs(result['actual_physics_elapsed_s']-6)<1e-6
        assert max(result['peak_pre_step_estimated_pd_effort_Nm'])<1 and meta['pos_iters']==16 and meta['vel_iters']==4 and meta['solver_type']==0
        hashes=meta['source_hashes']
    else:
        assert result['native_contact_disabled'] and result['physical_armature_kgm2']==0 and result['numerical_effective_inertia_adapter'] and not any(result['stopped']) and not any(x for row in result['warnings'] for x in row)
        assert max(result['peak_numerical_actuator_force_Nm'])<1;hashes=result['source_hashes']
    for path,h in hashes.items():assert s.sha(path)==h,path
    checks.append(dict(run=name,result_sha256=s.sha(f/'results.json'),trajectory_sha256=s.sha(f/'trajectory.npz'),quality_sha256=s.sha(f/'quality_audit.json'),compression_sha256=s.sha(f/'compliance_geometry_audit.json')))
for grid,dt in [(31,.00003125),(15,.000015625)]:
    for i,c in enumerate(s.cases('holdout')):inspect(f'mu_common_contact_dt{grid}_case{i:02d}_000',[c],dt,False)
    for device in ['cpu','gpu']:inspect(f'native_common_contact_validation_{device}_dt{grid}_000',s.cases('calibration')+s.cases('holdout'),dt,True)
for name in ['body_wrench_cpu_000','body_wrench_gpu_001']:
    j=json.loads((R/'runs'/name/'comparison.json').read_text());assert j['passed'];checks.append(dict(probe=name,comparison_sha256=s.sha(R/'runs'/name/'comparison.json')))
for relative in ['audit/common_contact_geometry_preparation.json','audit/common_contact_vectorization.json']:
    assert json.loads((R/relative).read_text())['passed'];checks.append(dict(proof=relative,sha256=s.sha(R/relative)))
gpu=json.loads((R/'runs/native_common_contact_validation_gpu_dt15_000/execution_backend.json').read_text())
assert gpu['torch_device']=='cuda:0' and gpu['scene_attributes']=={'physxScene:enableGPUDynamics':True,'physxScene:broadphaseType':'GPU','physxScene:solverType':'PGS'}
preservation=json.loads((R/'audit/source_preservation.json').read_text());assert preservation['all_preserved']
for row in preservation['rows']:assert s.sha(row['path'])==row['initial_sha256'],row['path']
anchor=json.loads((R/'REPRODUCTION_MANIFEST.json').read_text())
for relative in ['assets/mapping.json']+[str(x.relative_to(R)).replace('\\','/') for x in (R/'assets/meshes').glob('*.obj')]:
    assert s.sha(R/relative)==anchor['required_sha256'][relative],relative
checks.append(dict(proof='Pre-contact-branch frozen mapping and mesh inputs remain unchanged',anchor_manifest_sha256=s.sha(R/'REPRODUCTION_MANIFEST.json')))
profile=json.loads((R/'COMMON_CONTACT_CANDIDATE_PROFILE.json').read_text());profile['status']='validated_common_contact_cpu_gpu';profile['validated_utc']=datetime.now(timezone.utc).isoformat();profile['observed_maxima_pct']={k:max(r[k] for r in summary['rows']) for k in summary['criteria']}
motion=json.loads((R/'audit/common_contact_motion_comparison.json').read_text());assert len(motion['rows'])==32
profile['additional_motion_observations']={key:max(r[key] for r in motion['rows']) for key in ['max_centroid_xy_path_difference_mm','max_heading_difference_deg','max_joint_position_difference_deg']}
profile['known_limits']=['Forward displacement gates do not establish arbitrary world-origin invariance. Origin-dependent lateral drift is measured separately; reducing the step can worsen positional quantization at large global offsets.','The host-computed shared compliant contact is a slow mechanical audit baseline, not a migrated high-throughput RL training pipeline.','Native contact sensors do not measure this externally applied contact law; original RL rewards, observations, resets and terrain are not migrated.']
checks.append(dict(proof='audit/common_contact_motion_comparison.json',sha256=s.sha(R/'audit/common_contact_motion_comparison.json')))
checks.append(dict(proof='audit/common_contact_coordinate_precision.json',sha256=s.sha(R/'audit/common_contact_coordinate_precision.json')))
s.write(R/'COMMON_CONTACT_OPERATING_PROFILE.json',profile)
s.write(R/'COMMON_CONTACT_VALIDATION_RECEIPT.json',dict(profile=profile,checks=checks,compression_report=compression,summary_sha256=s.sha(R/'audit/common_contact_summary.json'),qualification='Validated only for the stated new common compliant-contact physical benchmark; original rigid-contact candidates remain rejected.'))
required=['COMMON_CONTRACT.json','COMMON_CONTACT_V2.json','COMMON_CONTACT_OPERATING_PROFILE.json','COMMON_CONTACT_VALIDATION_RECEIPT.json','shared.py','command_cache.py','common_contact_v2.py','common_contact_fast.py','run_mujoco_common_contact.py','run_native_common_contact_fast_recorded.py','reproduce_common_contact.py','assets/common_contact_geometry.json','assets/common.xml','assets/common.usda','assets/mapping.json']+[str(x.relative_to(R)).replace('\\','/') for x in (R/'assets/meshes').glob('*.obj')]
s.write(R/'COMMON_CONTACT_REPRODUCTION_MANIFEST.json',dict(profile_status=profile['status'],required_sha256={n:s.sha(R/n) for n in required},runtime='C:/Users/wxy22/anaconda3/envs/isaaclab_50/python.exe',runtime_versions_sha256=s.sha(R/'audit/runtime_versions.json')))
print(json.dumps(dict(status=profile['status'],maxima_pct=profile['observed_maxima_pct'],required_files=len(required),promotion_performed=True)))
