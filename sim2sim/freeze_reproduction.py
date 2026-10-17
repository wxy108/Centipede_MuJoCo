import argparse,json
from datetime import datetime,timezone
import shared as s
p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');a=p.parse_args()
profile=json.loads((s.ROOT/'OPERATING_PROFILE.json').read_text());checks=[]
if a.finalize:
    summary=json.loads((s.ROOT/'audit/coupled_candidate_summary.json').read_text())
    assert summary['status']=='NUMERICAL_GATES_PASSED_GPU_SCOPE_QUALITY_REVIEW_REQUIRED',summary['status']
    assert not summary['missing'] and not summary['failed']
    for g,dt in [('dt31',.00003125),('dt15',.000015625)]:
        for i in range(8):
            name=f'mu_coupled_rigid_{g}_case{i:02d}_000';f=s.ROOT/'runs'/name;r=json.loads((f/'results.json').read_text());q=json.loads((f/'quality_audit.json').read_text())
            assert r['arguments']['dt']==dt and r['arguments']['command_dt']==.0000625 and r['numerical_effective_inertia_adapter'] and r['physical_armature_kgm2']==0
            assert not any(r['stopped']) and not any(v for row in r['warnings'] for v in row)
            assert max(r['peak_numerical_actuator_force_Nm'])<1.
            for path,sha in r['source_hashes'].items():assert s.sha(path)==sha,path
            for v in q['rows']:assert v['finite'] and v['inverted_spine_samples']==0 and v['spine_below_plane_samples']==0
            checks.append(dict(name=name,result_sha=s.sha(f/'results.json'),quality_sha=s.sha(f/'quality_audit.json')))
    for name,dt in [('native_pgs_validation_gpu_dt62_000',.0000625),('native_pgs_validation_gpu_dt31_000',.00003125)]:
        f=s.ROOT/'runs'/name;r=json.loads((f/'results.json').read_text());b=json.loads((f/'backend_validation.json').read_text());meta=json.loads((f/'readback.json').read_text());done=json.loads((f/'complete.json').read_text());q=json.loads((f/'quality_audit.json').read_text())
        assert b['passed'] and meta['physics_dt_s']==dt and meta['solver_type']==0 and meta['pos_iters']==16 and meta['vel_iters']==4
        assert r['device']=='cuda:0' and not r['stopped'] and abs(r['actual_physics_elapsed_s']-6)<1e-6 and done['complete'] and done['physics_steps']==round(6/dt)
        assert max(r['peak_pre_step_estimated_pd_effort_Nm'])<1.
        for path,sha in meta['source_hashes'].items():assert s.sha(path)==sha,path
        for v in q['rows']:assert v['finite'] and v['inverted_spine_samples']==0 and v['spine_below_plane_samples']==0
        for i,c in enumerate(s.cases('calibration')+s.cases('holdout')):assert r['rows'][i]['wave']==c['wave'] and r['rows'][i]['amplitudes']==c['amplitudes']
        checks.append(dict(name=name,result_sha=s.sha(f/'results.json'),backend_sha=s.sha(f/'backend_validation.json'),quality_sha=s.sha(f/'quality_audit.json')))
    physical=json.loads((s.ROOT/'audit/source_preservation.json').read_text());assert all(r['preserved'] for r in physical['rows'])
    profile['status']='validated_gpu_scope';profile['validated_utc']=datetime.now(timezone.utc).isoformat();profile['maximum_observed_gap_pct']=max(r['primary_gap_pct'] for r in summary['rows']);profile['maximum_mu_refinement_pct']=max(r['mu_refinement_pct'] for r in summary['rows']);profile['maximum_gpu_refinement_pct']=max(r['gpu_refinement_pct'] for r in summary['rows']);s.write(s.ROOT/'OPERATING_PROFILE.json',profile)
    s.write(s.ROOT/'VALIDATION_RECEIPT.json',dict(profile=profile,checks=checks,summary_sha=s.sha(s.ROOT/'audit/coupled_candidate_summary.json'),scope_limits=profile['scope']))
required=['COMMON_CONTRACT.json','OPERATING_PROFILE.json','shared.py','command_cache.py','run_mujoco_v8.py','run_native_v6.py','assets/common.xml','assets/common.usda','assets/mapping.json']+[str(p.relative_to(s.ROOT)).replace('\\','/') for p in (s.ROOT/'assets/meshes').glob('*.obj')]
s.write(s.ROOT/'REPRODUCTION_MANIFEST.json',dict(profile_status=profile['status'],required_sha256={n:s.sha(s.ROOT/n) for n in required},runtime='C:/Users/wxy22/anaconda3/envs/isaaclab_50/python.exe',runtime_versions_sha=s.sha(s.ROOT/'audit/runtime_versions.json')))
print(json.dumps(dict(status=profile['status'],required_files=len(required),finalized=a.finalize)))
