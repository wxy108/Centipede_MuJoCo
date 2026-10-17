"""Verify saved experiment evidence without importing or running a simulator."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

def main(final=False):
    out=ROOT/'outputs/physical_geometry_v2';manifest=read(out/'data/manifest.json')
    reference=None;count=0;full=0;pd_by_wave={}
    for p in (out/'assessments').glob('*.json'):
        r=read(p);pd=r['pd_readback'];assert pd
        assert pd['active_pd_sha256']==manifest['active_pd_sha256']
        if reference is None:reference=pd
        assert pd==reference,('Complete active, body-roll and tip PD differ',p)
        assert sha(r['record'])==r['raw_sha256'],('Saved record changed',p)
        assert not r.get('error'),('Execution error remains',p)
        if not r['full']:assert r['D'] is None
        count+=1;full+=r['full'];pd_by_wave[str(r['wave'])]=pd_by_wave.get(str(r['wave']),0)+1
    inputs={}
    for p,h in manifest['frozen_inputs'].items():
        assert sha(p)==h.upper(),p;inputs[p]=h
    seal=read(ROOT/'outputs/data/precontact_analysis_seal.json')
    for p,h in seal['source_hashes'].items():assert sha(p)==h.upper(),p;inputs[p]=h
    if 'amendment_receipt' in seal:
        amendment_path=seal['amendment_receipt']
        assert sha(amendment_path)==seal['amendment_sha256'].upper()
        amendment=read(amendment_path)
        assert amendment['before_any_nonzero_contact_dynamics']
        assert not any(amendment[k] for k in ('changed_physics','changed_pd','changed_selection','changed_random_schedule','changed_statistics'))
        for p,h in amendment['archive_file_hashes'].items():assert sha(p)==h.upper(),p
        assert sha(amendment['changed_source'])==amendment['new_source_sha256'].upper()
        inputs[amendment_path]=seal['amendment_sha256']
        parity_path=ROOT/'outputs/data/p0_metadata_repair_dynamic_parity.json'
        if final or parity_path.exists():
            parity=read(parity_path);assert parity['passed'] and len(parity['checks'])==14
            for item in parity['checks']:
                assert item['all_saved_trajectory_arrays_exact']
                for p,h in item['source_hashes'].items():assert sha(p)==h.upper(),p
            inputs[str(parity_path)]=sha(parity_path)
    for item in seal.get('additional_amendments',[]):
        assert sha(item['receipt'])==item['sha256'].upper()
        amendment=read(item['receipt'])
        assert amendment['before_any_heldout_contact_dynamics']
        assert amendment['all_discovery_statistics_and_candidate_order_identical']
        assert not any(amendment[k] for k in ('changed_physics','changed_pd','changed_selection','changed_random_schedule','changed_statistics'))
        for key in ('archive_file_hashes','unchanged_discovery_input_hashes'):
            for p,h in amendment[key].items():assert sha(p)==h.upper(),p
        assert sha(amendment['changed_source'])==amendment['new_source_sha256'].upper()
        inputs[item['receipt']]=item['sha256']
    for name in ('convergence_extension_handover.json','joint_polish_handover.json'):
        path=out/'data'/name
        if final or path.exists():
            receipt=read(path)
            for p,h in receipt['source_hashes'].items():assert sha(p)==h.upper(),p;inputs[p]=h
    declaration=out/'data/joint_finish_declaration.json'
    if final or declaration.exists():
        for p,h in read(declaration)['source_hashes'].items():assert sha(p)==h.upper(),p;inputs[p]=h
    decision=out/'data/joint_finish_decision.json'
    if final or decision.exists():
        value=read(decision)
        assert sha(out/'data/joint_polish_audit.json')==value['original_audit_sha256']
        if value['extension_needed']:
            for name in ('joint_finish_handover.json','joint_finish_audit.json'):
                path=out/'data'/name
                if final or path.exists():
                    for p,h in read(path)['source_hashes'].items():assert sha(p)==h.upper(),p;inputs[p]=h
    channels=[]
    for name in ('contact_erase','contact_erase_transient'):
        dest=ROOT/'outputs'/name;path=dest/'data/manifest.json'
        if not path.exists():
            assert not final,('Contact manifest missing',name)
            channels.append(dict(channel=name,started=False));continue
        m=read(path)
        for p,h in m['frozen_inputs'].items():assert sha(p)==h.upper(),p
        masks={};prefixes={};n=0;incomplete=0;raw_D_checks=0;prefix_checks=0
        for p in (dest/'assessments').glob('*.json'):
            r=read(p);assert r['pd_readback']==reference
            task=r['task'];profile=m['profiles'][str(r['wave'])]
            assert r['amplitudes']==profile['amplitude'] and profile['qualified']
            n+=1
            if not r['full']:
                assert r['D'] is None
                if r.get('record'):
                    assert sha(r['record'])==r['raw_sha256']
                else:
                    assert r.get('numerical_failure')
                    partial=dest/'trials'/task['stage']/r['id']
                    saved=read(partial/'incomplete_contact_outcome.json')
                    assert saved['task']==task and saved['full_window_displacement'] is None
                    assert (partial/'incomplete_contact_state.npz').is_file()
                incomplete+=1;continue
            assert sha(r['record'])==r['raw_sha256']
            record=read(r['record']);channel=record['channel']
            key=(task['duration'],task['erasure']['seed'])
            digest=channel['uniform_sha256']
            if key in masks:assert masks[key]==digest,('Unpaired desired masks',name,key,p)
            else:masks[key]=digest
            with np.load(Path(r['record']).parent/'contact_trajectory.npz',allow_pickle=False) as z:
                good=z['t']>=1.5
                D=float(-(z['center'][good][-1,0]-z['center'][good][0,0])*1000)
                before=z['pose_t']<1.5-1e-10
                prefix=(z['pose_t'][before].copy(),z['qpos'][before].copy())
                assert len(prefix[0])>0
            prefix_key=(r['wave'],task['duration'],task['dt_scale'])
            if prefix_key in prefixes:
                baseline=prefixes[prefix_key]
                np.testing.assert_array_equal(prefix[0],baseline[0],err_msg='Different pre-erasure observation times')
                np.testing.assert_array_equal(prefix[1],baseline[1],err_msg='Different pre-erasure states across probabilities/seeds')
                prefix_checks+=1
            else:prefixes[prefix_key]=prefix
            assert D==r['D']==channel['raw_D_recomputed'];raw_D_checks+=1
        if final:
            assert read(dest/'data/p0_verification.json')['passed']
            assert (dest/'data/completed/discovery.json').exists()
        channels.append(dict(channel=name,started=True,assessments=n,incomplete=incomplete,
                             exact_raw_displacement_checks=raw_D_checks,paired_mask_groups=len(masks),
                             pre_erasure_pose_groups=len(prefixes),exact_pre_erasure_pair_checks=prefix_checks,
                             pre_erasure_check_scope='Saved 30Hz poses strictly before1.5s, same wave/duration/timestep; includes the mandatory6s nominal p0 baseline. This checks the recorded prefix, not unsaved intermediate states.'))
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),passed=True,
                final_stage_check=final,physics_steps=0,complete_pd_arrays_identical=True,
                active_pd_sha256=reference['active_pd_sha256'],amplitude_assessments=count,
                full_amplitude_assessments=full,counts_by_wave=pd_by_wave,
                checked_frozen_inputs=len(inputs),channels=channels,source_sha256=sha(__file__))
    path=ROOT/'outputs/data'/('final_evidence_check.json' if final else 'progress_evidence_check.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');a=parser.parse_args();main(a.final)
