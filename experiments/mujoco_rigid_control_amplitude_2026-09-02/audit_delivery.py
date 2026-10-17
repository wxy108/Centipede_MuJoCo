"""Read-only evidence checks; emit a final audit of the delivered experiment."""
import hashlib
import json
from pathlib import Path
import refine as rf
e, s = rf.e, rf.s


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    manifest = json.loads(s.MANIFEST.read_text())
    validation = json.loads((e.DATA / 'validated_selection.json').read_text())
    snapshot = json.loads((e.DATA / 'runtime_parameter_snapshot.json').read_text())
    videos = json.loads((e.DATA / 'delivery_video_manifest.json').read_text())
    reusable=json.loads((e.DATA/'selected_amplitude_profiles.json').read_text())
    rows = rf.collect(['broad', 'refine', 'polish', 'boundary'])
    assert validation['validation_complete']
    assert not validation['global_optimum_proved']
    assert not validation['baseline_replaced']
    assert set(reusable['selected_profiles']) == {w for w,r in validation['waves'].items() if r['selected']}
    assert sha(manifest['plant']) == manifest['plant_sha256']
    assert sha(snapshot['model']) == snapshot['model_sha256']
    source_checks = []
    for item in snapshot['source_files']:
        assert sha(item['path']) == item['sha256'], item['path']
        source_checks.append(item['path'])
    for item in manifest['PD_profiles_frozen'].values():
        assert sha(item['source']) == item['sha256'], item['source']
    paired=json.loads((e.DATA/'paired_ablation.json').read_text())
    assert len(paired)==8
    for record in paired:
        assert sha(record['model_path'])==record['model_sha256']
    paired_videos=json.loads((e.DATA/'paired_video_manifest.json').read_text())
    assert len(paired_videos)==8
    for video in paired_videos:
        assert Path(video['video']).is_file() and video['frames']==180
        assert video['unsmoothed'] and not video['physics_changes']

    stages = ['broad', 'refine', 'polish', 'boundary']
    task_count = 0
    for stage in stages:
        bank = manifest if stage == 'broad' else json.loads((e.DATA / f'{stage}_manifest.json').read_text())
        for task in bank['tasks']:
            reference = next(t for t in manifest['references'] if t['wave'] == task['wave'])
            for key in ('config', 'duration', 'dt_scale', 'topology', 'terrain'):
                assert task[key] == reference[key], (stage, task['wave'], key)
            assert task['topology'] == 'flexible' and task['terrain'] == 'flat'
            assert set(task['amplitudes']) == set(s.KEYS)
            for key, lower, upper in zip(s.KEYS, s.LOW, s.HIGH):
                assert lower <= task['amplitudes'][key] <= upper
            task_count += 1
    assert len(rows) == task_count, 'Unexpected duplicate or missing amplitude conditions'

    wave_checks = []
    for wave, result in validation['waves'].items():
        nominal = sorted((r for r in rows if r['wave'] == float(wave) and r['feasible']), key=lambda r: -r['D'])
        checked = {c['nominal']['id']: c for c in result['candidates']}
        chosen = result['selected']
        if chosen:
            assert reusable['selected_profiles'][wave]['amplitudes_deg'] == chosen['nominal']['amplitudes']
            assert reusable['selected_profiles'][wave]['frozen_PD'] == manifest['PD_profiles_frozen'][wave]
            assert chosen['qualified']
            assert chosen['nominal']['feasible']
            assert all(c['feasible'] for c in chosen['checks'])
            assert chosen['halfstep_D_relative_difference'] <= .1
            for candidate in nominal:
                if candidate['D'] > chosen['nominal']['D']:
                    assert candidate['id'] in checked, 'Faster nominal candidate was not validated'
                    assert not checked[candidate['id']]['qualified'], 'Qualified faster candidate was not selected'
        else:
            assert all(r['id'] in checked for r in nominal), 'Unresolved wave has unchecked feasible candidates'
            assert not any(c['qualified'] for c in checked.values())
        video = next(v for v in videos if v['wave'] == float(wave))
        assert video['selection_status'] == result['status']
        assert video['frames'] == 180 and video['shape'] == [880, 1920, 3]
        assert video['unsmoothed'] and not video['physics_changes']
        assert Path(video['video']).is_file()
        assert all(max(c['body_error_m'], c['foot_error_m']) < 1e-7 for c in video['pose_checks'])
        wave_checks.append(dict(wave=float(wave), status=result['status'], nominal_feasible=len(nominal),
            validated=len(checked), selected_id=chosen['nominal']['id'] if chosen else None))

    errors = list(e.DATA.glob('*_errors.json'))
    assert not errors, f'Unresolved evaluator errors: {errors}'
    report = dict(passed=True, distinct_amplitude_conditions=task_count, waves=wave_checks,
        original_model_unchanged=True, historical_PD_sources_unchanged=True,
        historical_helper_sources_unchanged=source_checks, only_three_amplitudes_varied=True,
        every_nominally_faster_candidate_checked=True, all_unresolved_nominal_candidates_exhausted=True,
        all_delivery_videos_verified=True, mathematical_global_optimum_proved=False,
        paired_control_records_verified=len(paired),paired_control_videos_verified=len(paired_videos),
        scope='Centipede Model only; no canonical baseline promotion')
    (e.DATA / 'final_delivery_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
