"""Bounded interim validation/demos, independent of the search and final ranking."""
import concurrent.futures
import json
import validate as v
s, e = v.s, v.e


def run(task):
    return e.run_one(task) if task.get('trace') else v.run_task(task)


if __name__ == '__main__':
    manifest_path = e.DATA / 'early_demo_manifest.json'
    if manifest_path.exists():
        manifest = s.read(manifest_path)
    else:
        rows = []
        for path in (e.OUT / 'assessments').glob('*.json'):
            try:
                rows.append(s.read(path))
            except json.JSONDecodeError:
                pass
        candidates, tasks = [], []
        for wave in (2., 3.5):
            row = max((r for r in rows if r['wave'] == wave and r['feasible']), key=lambda r: r['D'])
            base = s.read(row['record'])['task']
            checks = [v.validation_task(base, duration, dt) for duration, dt in ((12., 1.), (6., .5))]
            for task in checks:
                assert s.reference_path(task).exists()
            trace = dict(base, stage='early_delivery', trace=True)
            candidates.append(dict(nominal=row, checks=checks, trace=trace))
            tasks.extend([*checks, trace])
        manifest = dict(status='interim only; does not affect the search or final ranking',
                        policy_sha256=s.gates.POLICY_SHA, candidates=candidates, tasks=tasks)
        s.write(manifest_path, manifest)
    assert manifest['policy_sha256'] == s.gates.POLICY_SHA
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, task) for task in manifest['tasks']]
        for future in concurrent.futures.as_completed(futures):
            raw = future.result()
            print(json.dumps(dict(id=raw['id'], D=raw['metrics']['displacement_mm'])), flush=True)
    previous = s.read(e.PARENT / 'outputs/data/validated_selection.json')
    render = e.load(e.PARENT / 'render.py', 'relaxed_early_renderer')
    results = []
    for item in manifest['candidates']:
        nominal = item['nominal']
        wave = nominal['wave']
        checks = [s.assess(s.read(v.result_path(task))) for task in item['checks']]
        difference = abs(checks[1]['D'] - nominal['D']) / abs(nominal['D'])
        qualified = all(r['feasible'] for r in checks) and difference <= .1
        old_id = previous['waves'][f'{wave:g}']['selected']['nominal']['id']
        before = e.PARENT / 'outputs/trials/delivery' / old_id / f'{old_id}.json'
        after = e.TRIALS / 'early_delivery' / nominal['id'] / f"{nominal['id']}.json"
        raw = s.read(after)
        for key, value in nominal['metrics'].items():
            assert s.np.array_equal(s.np.asarray(value), s.np.asarray(raw['metrics'][key]), equal_nan=True)
        with s.np.load(after.with_suffix('.npz')) as trace:
            assert float(trace['t'][-1]) >= 5.998
        labels = ['PRIOR STRICT OPTIMUM',
                  'INTERIM CANDIDATE / CHECKS PASSED' if qualified else 'INTERIM DIAGNOSTIC / CHECKS FAILED']
        tag = f'{wave:g}'.replace('.', 'p')
        video = render.make([str(before), str(after)],
                            e.OUT / 'videos' / f'interim_w{tag}_three_views.mp4', labels=labels)
        result = dict(wave=wave, nominal=nominal, checks=checks,
                      qualified_candidate=qualified, final_optimum=False,
                      halfstep_D_relative_difference=difference, video=video)
        results.append(result)
        s.write(e.DATA / 'early_demo_results.json', results)
        print(json.dumps(dict(wave=wave, qualified=qualified, D=nominal['D'],
                              amplitudes=nominal['amplitudes'], checks=[r['failed'] for r in checks],
                              video=video['video'])), flush=True)
