"""Independently reconstruct saved mask schedules; no simulator imports or steps."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def transient(task,record,z):
    ch=record['channel'];dt=float(record['dt']);duration=task['duration']
    start=round(1.5/dt);stop=round(duration/dt);period=round(.5/dt)
    windows=int(np.ceil((duration-1.5)/.5))
    uniforms=np.random.default_rng(909202600+task['erasure']['seed']).random((windows,19))
    np.testing.assert_array_equal(uniforms,np.asarray(ch['uniforms']))
    assert hashlib.sha256(uniforms.tobytes()).hexdigest()==ch['uniform_sha256']
    desired=uniforms<task['erasure']['probability']
    assert ch['desired_pairs_by_window']==[np.flatnonzero(a).tolist() for a in desired]
    commands={start+i*period:np.repeat(mask,2) for i,mask in enumerate(desired)}
    assert ch['transitions']==[dict(step=int(k),window=i,desired_pairs=np.flatnonzero(desired[i]).tolist()) for i,k in enumerate(commands)]
    restores={}
    for event in ch['restore_events']:
        k=int(event['step']);leg=int(event['leg'])
        assert start<=k<stop and k%round(.001/dt)==0 and 0<=leg<38
        assert np.isfinite(event['clearance_m']) and event['clearance_m']>=50e-6
        restores.setdefault(k,[]).append(leg)
    assert len(ch['restore_events'])==ch['restorations']
    if ch['restore_events']:
        assert ch['min_restore_clearance_m']==min(v['clearance_m'] for v in ch['restore_events'])
    else:assert ch['min_restore_clearance_m'] is None
    keys=sorted(set(commands)|set(restores));selected=np.zeros(38,bool);command=selected.copy()
    states=[];erased_steps=np.zeros(38,np.int64);desired_steps=erased_steps.copy()
    for i,k in enumerate(keys):
        if k in commands:command=commands[k];selected|=command
        for leg in restores.get(k,[]):
            assert selected[leg] and not command[leg],('Invalid saved restore event',k,leg)
            selected[leg]=False
        states.append(selected.copy())
        interval=(keys[i+1] if i+1<len(keys) else stop)-k
        erased_steps+=interval*selected;desired_steps+=interval*command
    states=np.asarray(states)
    sample_steps=np.rint(z['mask_t']/dt).astype(int)
    expected_samples=np.arange(start,stop,round(.001/dt))
    np.testing.assert_array_equal(sample_steps,expected_samples)
    indexes=np.searchsorted(keys,sample_steps,side='right')-1
    np.testing.assert_array_equal(z['erased_legs'],states[indexes])
    n=stop-start
    np.testing.assert_array_equal(erased_steps/n,ch['actual_erased_fraction_per_leg'])
    assert ch['final_erased_legs']==np.flatnonzero(selected).tolist()
    actual=float(erased_steps.sum()/(n*38));wanted=float(desired_steps.sum()/(n*38))
    assert actual==ch['actual_erased_fraction'] and wanted==ch['desired_erased_fraction']
    assert float((erased_steps-desired_steps).sum()/(n*38))==ch['delayed_extra_erased_fraction']
    assert actual>=wanted
    return dict(restorations=ch['restorations'],sampled_masks=len(sample_steps),actual=actual,desired=wanted,
                minimum_logged_restore_clearance_m=ch['min_restore_clearance_m'])


def main(final=False):
    if final:assert read(ROOT/'outputs/contact_workflow_status.json')['stage']=='mujoco_amplitude_and_two_contact_channels_complete'
    channels=[];hashes={};checks=[];endpoints={};endpoint_parity=[]
    for name in ('contact_erase','contact_erase_transient'):
        out=ROOT/'outputs'/name;rows=sorted((out/'assessments').glob('*.json'))
        checked=0;skipped=0;restorations=0;sampled_masks=0;p1=[];min_gap=None
        for path in rows:
            row=read(path)
            if not row['full']:skipped+=1;continue
            record_path=Path(row['record']);record=read(record_path);ch=record['channel'];task=row['task']
            trajectory=record_path.parent/'contact_trajectory.npz'
            with np.load(trajectory,allow_pickle=False) as z:
                if name.endswith('transient'):
                    result=transient(task,record,z);restorations+=result['restorations'];sampled_masks+=result['sampled_masks']
                    gap=result['minimum_logged_restore_clearance_m']
                    if gap is not None:min_gap=gap if min_gap is None else min(min_gap,gap)
                else:
                    u=np.random.default_rng(909202600+task['erasure']['seed']).random(19)
                    np.testing.assert_array_equal(u,ch['uniforms'])
                    assert hashlib.sha256(u.tobytes()).hexdigest()==ch['uniform_sha256']
                    mask=u<task['erasure']['probability']
                    np.testing.assert_array_equal(mask,z['erased_pairs'])
                    assert ch['erased_pairs']==np.flatnonzero(mask).tolist()
                    assert float(mask.mean())==ch['actual_erased_fraction'] and ch['restorations']==0
            if task['erasure']['probability']==1:
                assert row['metrics']['foot_force_peak_bw']==0 and row['metrics']['mean_contacting_feet']==0
                p1.append(row['wave'])
            if task['erasure']['probability'] in (0,1) and task['duration']==6 and task['dt_scale']==.5:
                key=(row['wave'],task['erasure']['probability'])
                if key in endpoints:
                    with np.load(endpoints[key],allow_pickle=False) as a,np.load(trajectory,allow_pickle=False) as b:
                        for field in ('t','center','upright','pose_t','qpos'):
                            np.testing.assert_array_equal(a[field],b[field])
                    endpoint_parity.append(dict(wave=key[0],probability=key[1],all_saved_state_arrays_exact=True))
                else:endpoints[key]=trajectory
            checked+=1
            for p in (path,record_path,trajectory):hashes[str(p)]=sha(p)
            checks.append(dict(channel=name,id=row['id'],passed=True))
        if final:assert len(set(p1))==14
        channels.append(dict(channel=name,full_schedules_checked=checked,incomplete_schedules_not_certified=skipped,
            restoration_events_checked=restorations,sampled_masks_exact=sampled_masks,
            minimum_logged_restore_clearance_m=min_gap,p1_waves_with_exact_zero_foot_support=sorted(set(p1))))
    if final:assert len(endpoint_parity)==28
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),passed=True,final=final,physics_steps=0,
        uniform_rng_independently_recreated=True,transient_exposure_reconstructed_from_events=True,
        limitation='Restoration clearance is checked from logged values, not independently reconstructed at every event because those event qpos are not saved. This audit does not create missing geometry observations.',
        channels=channels,deterministic_endpoint_cross_channel_parity=endpoint_parity,
        checks=checks,input_hashes=hashes,source_sha256=sha(__file__))
    dest=ROOT/'outputs/data'/('contact_schedule_final_audit.json' if final else 'contact_schedule_progress_audit.json')
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','input_hashes')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
