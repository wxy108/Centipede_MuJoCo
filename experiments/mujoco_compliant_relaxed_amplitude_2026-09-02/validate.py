"""Descending-displacement independent validation under one frozen policy."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import study as s
e = s.e


def validation_task(base, duration, dt, stage='validation'):
    task = {k: v for k, v in base.items() if k not in
            ('source_record', 'source_sha256', 'reuse_record', 'reuse_sha256')}
    task.update(stage=stage, duration=duration, dt_scale=dt, trace=False)
    tid = e.task_id(task)
    own = e.TRIALS / stage / tid / f'{tid}.json'
    if not own.exists():
        for old_stage in (stage, 'reference'):
            path = e.PARENT / 'outputs/trials' / old_stage / tid / f'{tid}.json'
            if path.exists():
                raw = s.read(path)
                assert e.task_id(raw['task']) == tid
                assert raw['model_sha256'] == s.manifest()['plant_sha256']
                if raw.get('metrics'):
                    task.update(reuse_record=str(path), reuse_sha256=s.sha(path))
                    break
    return task


def result_path(task):
    tid = e.task_id(task)
    own = e.TRIALS / task['stage'] / tid / f'{tid}.json'
    return own if own.exists() else Path(task.get('reuse_record', own))


def run_task(task):
    path = result_path(task)
    if path.exists():
        if str(path) == task.get('reuse_record'):
            assert s.sha(path) == task['reuse_sha256']
        raw = s.read(path)
        assert raw['model_sha256'] == s.manifest()['plant_sha256']
        if raw.get('metrics'):
            return raw
    return e.run_one(task)


def run_tasks(tasks, workers):
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            raw = future.result()
            print(json.dumps(dict(done=i, total=len(tasks), id=raw['id'],
                D=raw['metrics']['displacement_mm'], elapsed_s=time.monotonic()-start)), flush=True)


def freeze():
    path = e.DATA / 'validation_manifest.json'
    if path.exists():
        return s.read(path)
    rows = s.collect()
    candidates, all_candidates, tasks, references = {}, {}, [], []
    for wave in e.WAVES:
        key = f'{wave:g}'
        ranked = sorted((r for r in rows if r['wave'] == wave and r['feasible']), key=lambda r: -r['D'])
        all_candidates[key] = ranked
        candidates[key] = ranked[:3]
        if not ranked:
            continue
        base = next(t for t in s.manifest()['references'] if t['wave'] == wave)
        references.extend(validation_task(base, duration, dt, 'reference')
                          for duration, dt in ((12., 1.), (6., .5)))
        for row in candidates[key]:
            base = s.read(row['record'])['task']
            tasks.extend(validation_task(base, duration, dt) for duration, dt in ((12., 1.), (6., .5)))
    data = dict(policy_sha256=s.gates.POLICY_SHA, batch_per_wave=3,
        selection='Highest nominal D first; exhaust higher-ranked candidates before selecting a lower one',
        all_nominal_candidates=all_candidates, candidates=candidates, tasks=tasks, references=references,
        halfstep_displacement_relative_limit=.10, global_optimum_proved=False)
    s.write(path, data)
    return data


def evaluate(manifest):
    waves = {}
    rows = s.collect()
    task_lookup = {e.task_id(task): task for task in manifest['tasks']}
    traces = []
    for wave in e.WAVES:
        key = f'{wave:g}'
        tested = []
        for nominal in manifest['candidates'][key]:
            base = s.read(nominal['record'])['task']
            checks = []
            for duration, dt in ((12., 1.), (6., .5)):
                identity = dict(base, duration=duration, dt_scale=dt)
                task = task_lookup[e.task_id(identity)]
                checks.append(s.assess(s.read(result_path(task))))
            half = checks[1]
            delta = abs(half['D']-nominal['D']) / max(abs(nominal['D']), 1e-9)
            qualified = all(r['feasible'] for r in checks) and delta <= manifest['halfstep_displacement_relative_limit']
            tested.append(dict(nominal=nominal, checks=checks,
                               halfstep_D_relative_difference=delta, qualified=qualified))
        passing = sorted((r for r in tested if r['qualified']), key=lambda r: -r['nominal']['D'])
        chosen = passing[0] if passing else None
        full = sorted((r for r in rows if r['wave'] == wave and r['full']),
                      key=lambda r: (len(r['failed']), -r['D']))
        diagnostic = None if chosen else full[0] if full else None
        waves[key] = dict(status='qualified' if chosen else 'unresolved', selected=chosen,
                          candidates=tested, diagnostic=diagnostic)
        row = chosen['nominal'] if chosen else diagnostic
        if row:
            base = s.read(row['record'])['task']
            traces.append(dict(base, stage='delivery' if chosen else 'delivery_diagnostic', trace=True))
    result = dict(scope='Compliant-tip relaxed-gate amplitude-only study; Model only',
        policy_sha256=s.gates.POLICY_SHA, waves=waves, traces=traces, validation_complete=False,
        global_optimum_proved=False, canonical_baseline_replaced=False)
    s.write(e.DATA / 'validated_selection.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    m = freeze()
    print(json.dumps(dict(reference_tasks=len(m['references']), initial_validation_tasks=len(m['tasks']))), flush=True)
    run_tasks(m['references'], args.workers)
    run_tasks(m['tasks'], args.workers)
    result = evaluate(m)
    round_number = 1
    while True:
        tasks, additions = [], {}
        for wave in e.WAVES:
            key = f'{wave:g}'
            if result['waves'][key]['selected']:
                continue
            cursor = len(m['candidates'][key])
            batch = m['all_nominal_candidates'][key][cursor:cursor+m['batch_per_wave']]
            if not batch:
                continue
            additions[key] = [r['id'] for r in batch]
            m['candidates'][key].extend(batch)
            for row in batch:
                base = s.read(row['record'])['task']
                tasks.extend(validation_task(base, duration, dt) for duration, dt in ((12., 1.), (6., .5)))
        if not tasks:
            break
        round_number += 1
        m['tasks'].extend(tasks)
        s.write(e.DATA / f'validation_round_{round_number:02d}.json',
                dict(round=round_number, additions=additions, tasks=tasks))
        print(json.dumps(dict(round=round_number, additions=additions, tasks=len(tasks))), flush=True)
        run_tasks(tasks, args.workers)
        result = evaluate(m)
    s.write(e.DATA / 'validation_effective_manifest.json', m)
    result['validation_complete'] = True
    s.write(e.DATA / 'validated_selection.json', result)
    e.run_tasks(result['traces'], args.workers)
    print(json.dumps({wave: row['status'] for wave, row in result['waves'].items()}), flush=True)
