"""Verify actual input, condition, selection and delivery evidence."""
import argparse,datetime,json,hashlib
from pathlib import Path
import psutil
import core as c
import validate
np=c.np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest().upper()

def inspect(final=False):
    manifest=c.read(c.DATA/'manifest.json');inputs=[]
    for item in manifest['frozen_inputs']:
        assert digest(item['path'])==item['sha256'],item['path'];inputs.append(item)
    runtime=c.read(c.ROOT.parent/'outputs/data/runtime.json')
    for item in runtime['files']:assert digest(item['path'])==item['sha256'],item['path']
    io_check=c.OUT/'audit/file_retry_verification.json'
    if io_check.exists():
        verified=c.read(io_check)
        assert verified['passed'] and verified['core_sha256']==c.sha(c.ROOT/'core.py')
    records=c.records();by_id={r['id']:r for r in records};full=0;prefix=0
    for r in records:
        assert 'error' not in r,r
        assert c.identity(r['task'])==r['id']
        assert r['task']['dt_scale'] in (.5,.25) and r['task']['duration'] in (6,12)
        assert r['task']['wave'] in c.WAVES
        x=np.array([r['amplitudes'][k] for k in c.KEYS]);assert np.all(x>=0) and np.all(x<=c.HIGH)
        path=Path(r['record']);assert digest(path)==r['raw_sha256'],str(path)
        raw=c.read(path);assert raw['model_sha256']==c.history()['plant_sha256']
        if r['full']:
            a=c.policy.assess(raw);assert a['D']==r['D'] and a['feasible']==r['feasible'] and a['failed']==r['failed']
            assert raw['metrics']==r['metrics'];full+=1
        else:
            assert r.get('proven_infeasible') and r['D'] is None and not r['feasible']
            proof=r['proof'];dt=.00025*r['task']['dt_scale']
            assert abs(proof['actual_sim_time_s']-proof['label_time_s']-dt)<1e-7
            assert proof['violations'] and all(v['value']>v['limit'] for v in proof['violations'])
            state=np.load(path.parent/'prefix_state.npz');assert np.isfinite(state['qpos']).all() and np.isfinite(state['qvel']).all();prefix+=1
    banks=[]
    for path in sorted((c.DATA/'banks').glob('*.json')):
        tasks=c.read(path)
        if not isinstance(tasks,list):continue
        receipt=c.DATA/'completed'/path.name
        if not receipt.exists():
            assert not final,path.name
            continue
        done=c.read(receipt);ids=[c.identity(t) for t in tasks]
        assert len(ids)==len(set(ids)) and set(ids)==set(done['ids']) and done['count']==len(ids)
        assert all(key in by_id for key in ids)
        assert sum(by_id[k]['full'] for k in ids)==done['full']
        assert sum(by_id[k]['feasible'] for k in ids)==done['feasible']
        banks.append(dict(stage=path.stem,count=len(ids)))
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),frozen_input_count=len(inputs),assessment_count=len(records),full=full,
        proven_prefix_failures=prefix,completed_banks=banks,nominal_dt_s=.000125,half_dt_s=.0000625,passed=True,final=final)
    if final:
        for name in ['warm','warm_validation','broad','adaptive_00','adaptive_01','adaptive_02']:
            assert any(b['stage']==name for b in banks),name
        assert (c.DATA/'local_complete.json').exists()
        assert c.read(c.DATA/'feasibility_complete.json')['completed']
        selection=c.read(c.DATA/'validated_selection.json');qualified=[]
        if (c.DATA/'polish_plan.json').exists():
            plan=c.read(c.DATA/'polish_plan.json');polish=c.read(c.DATA/'polish_complete.json')
            assert polish['completed'] and selection['qualified_polish_complete']
            assert c.sha(plan['protocol'])==plan['protocol_sha256']
            assert c.sha(c.DATA/'selection_before_polish.json')==plan['initial_selection_sha256']
            for state in polish['states']:assert validate.assess(state['center'],by_id)['qualified']
        known=c.read(c.DATA/'validation_index.json');nominal=c.nominal();remaining_advantages={}
        for w,cell in selection['waves'].items():
            if not cell['qualified']:continue
            sel=cell['selected'];r=sel['nominal'];a=validate.assess(r,by_id)
            assert a['qualified'] and r['task']['dt_scale']==.5 and r['task']['duration']==6
            assert a['half_step_relative_difference']==sel['half_step_relative_difference']
            raw=c.read(r['record']);assert raw['executed_full_horizon'] and raw['dt']==.000125
            z=np.load(Path(r['record']).parent/'motion.npz');good=z['t']>=1.5
            D=float(c.e.fx.opt.base.FORWARD_SIGN*(z['center'][good][-1,0]-z['center'][good][0,0])*1000)
            assert D==r['D'];qualified.append(float(w))
            gap=max([0.,*[candidate['D']-r['D'] for candidate in nominal if candidate['wave']==float(w) and candidate['feasible'] and candidate['id'] not in known]])
            remaining_advantages[w]=gap
            if (c.DATA/'polish_plan.json').exists():assert gap<=.2+1e-9,(w,gap)
        result['qualified_waves']=qualified
        result['max_unchecked_nominal_advantage_mm']=remaining_advantages
        for name in ['selected_motion_verification.json','baseline_motion_verification.json','baseline_diagnostic_motion_verification.json','unresolved_motion_verification.json']:
            assert c.read(c.DATA/name)['passed'],name
        videos=[]
        for wave in [.5,1,1.5,2,4,5,6,7,9]:
            info=c.read(c.OUT/'videos'/f'wave{wave:g}_before_after.json')
            assert Path(info['video']).exists() and info['frames']==180 and info['fps']==30
            assert info['recorded_qpos_replayed_exactly'] and info['no_interpolation'] and not info['physics_rerun']
            assert info['max_nearest_recorded_time_error_s']<=.017
            assert info['progress_aligned_to_recorded_pose'] and info['max_progress_sample_time_error_s']<=.000501
            assert info['render_source_sha256']==c.sha(c.ROOT/'render.py')
            assert selection['waves'][f'{wave:g}']['selected']['nominal']['record'] in info['records']
            videos.append(info)
        result['verified_selected_videos']=videos
        for wave,cell in selection['waves'].items():
            if cell['qualified']:continue
            info=c.read(c.OUT/'videos'/f'unresolved_wave{wave}.json')
            assert Path(info['video']).exists() and info['frames']==180 and info['recorded_qpos_replayed_exactly']
            assert not info['physics_rerun'] and info['progress_aligned_to_recorded_pose']
            assert info['max_progress_sample_time_error_s']<=.000501 and info['render_source_sha256']==c.sha(c.ROOT/'render.py')
        qa=c.read(c.DATA/'visual_qa.json');assert qa['passed']
        for item in qa['files']:assert digest(c.ROOT/item['path'])==item['sha256'],item['path']
        owners=c.read(c.DATA/'background_processes.json')['processes']
        owners += [c.read(c.DATA/name) for name in ['supplement_process.json','delivery_process.json'] if (c.DATA/name).exists()]
        process_checks=[]
        for owner in owners:
            try:
                p=psutil.Process(owner['pid'])
                alive=p.create_time()==owner['create_time'] and p.cmdline()==owner['cmdline'] and p.cwd()==owner['cwd']
            except psutil.NoSuchProcess:alive=False
            assert not alive,owner
            process_checks.append(dict(pid=owner['pid'],owned_alive=alive))
        result['completed_process_checks']=process_checks
        for name in ['RESULTS.md','outputs/data/selected_gaits.json','outputs/data/final_summary.json','outputs/figures/optimization_overview.png','outputs/figures/search_progress.png','outputs/figures/validation_overview.png']:
            assert (c.ROOT/name).exists(),name
        delivery=c.read(c.DATA/'delivery_status.json')
        assert delivery['complete']
        delivery.update(final_audit_pending=False,final_audit_passed=True,final_audit_time=result['time'])
        c.write(c.DATA/'delivery_status.json',delivery)
        c.write(c.DATA/'final_audit.json',result)
        files=[]
        for path in sorted(c.ROOT.rglob('*')):
            if not path.is_file() or path.name=='FINAL_RECEIPT.json' or '__pycache__' in path.parts or path.name.endswith('.tmp'):continue
            files.append(dict(path=str(path.relative_to(c.ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
        c.write(c.ROOT/'FINAL_RECEIPT.json',dict(**result,files=files))
    else:c.write(c.DATA/'current_audit.json',result)
    print({k:v for k,v in result.items() if k not in ['completed_banks','verified_selected_videos']},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args();inspect(a.final)
