"""Resumable, provenance-checked relaxed-gate amplitude study, Model only."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import itertools
import json
from pathlib import Path
import time
import engine as e
import gates

np = e.np
KEYS = ('body', 'leg_yaw', 'leg_pitch')
LOW = np.array([0., 0., 0.])
HIGH = np.array([36., 90., 60.])
MANIFEST = e.DATA / 'amplitude_manifest.json'
_manifest = None
_references = {}
_prefix = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def manifest():
    global _manifest
    if _manifest is None:
        _manifest = read(MANIFEST)
        assert _manifest['gate_policy_sha256'] == gates.POLICY_SHA
    return _manifest


def freeze():
    if MANIFEST.exists():
        return manifest()
    prior = read(e.PARENT / 'outputs/data/amplitude_manifest.json')
    audit = read(e.PARENT / 'outputs/data/final_delivery_audit.json')
    assert audit['passed'] and audit['distinct_amplitude_conditions'] == 3298
    assert sha(prior['plant']) == prior['plant_sha256']
    snapshot = read(e.PARENT / 'outputs/data/runtime_parameter_snapshot.json')
    sources = list(snapshot['source_files'])
    sources += [dict(path=str(e.PARENT / name), sha256=sha(e.PARENT / name))
                for name in ('engine.py', 'search.py', 'render.py')]
    sources += [dict(path=str(e.PARENT / 'outputs/data/validated_selection.json'),
                    sha256=sha(e.PARENT / 'outputs/data/validated_selection.json'))]
    for source in sources:
        assert sha(source['path']) == source['sha256']
    for profile in prior['PD_profiles_frozen'].values():
        assert sha(profile['source']) == profile['sha256']
    tasks = []
    seen = set()
    for stage in ('broad', 'refine', 'polish', 'boundary'):
        p = e.PARENT / 'outputs/data' / f'{stage}_summary.json'
        if stage == 'broad':
            p = p.with_name('broad_objective_summary.json')
        for row in read(p)['assessments']:
            raw = read(row['record'])
            task = dict(raw['task'], stage='search', trace=False,
                        origin='prior_condition_reassessment',
                        source_record=row['record'], source_sha256=sha(row['record']))
            tid = e.task_id(task)
            assert tid not in seen and tid == row['id']
            seen.add(tid)
            tasks.append(task)
    assert len(tasks) == 3298
    from scipy.stats import qmc
    for wave in e.WAVES:
        base = next(t for t in prior['references'] if t['wave'] == wave)
        seed = 4922026 + int(2 * wave)
        points = qmc.scale(qmc.Sobol(3, scramble=True, seed=seed).random_base2(6), LOW, HIGH)
        for i, x in enumerate(points):
            task = dict(base, stage='search', trace=False,
                        amplitudes=dict(zip(KEYS, map(float, x))),
                        origin=dict(kind='fresh_scrambled_sobol', seed=seed, index=i))
            if e.task_id(task) in seen:
                continue
            seen.add(e.task_id(task))
            tasks.append(task)
    data = dict(scope='Centipede Model only; compliant local-Z tip; flat; erasure OFF',
        plant=prior['plant'], plant_sha256=prior['plant_sha256'],
        PD_profiles_frozen=prior['PD_profiles_frozen'], references=prior['references'],
        source_files=sources, gate_policy=gates.SPEC, gate_policy_sha256=gates.POLICY_SHA,
        source_physics_audits=[str(e.PARENT / 'outputs/data' / name) for name in
            ('runtime_arena_equivalence.json', 'settled_state_replay_verification.json')],
        frequency_hz=1., body_leg_wave_coupled=True, controller_unchanged=True,
        lower_deg=LOW.tolist(), upper_deg=HIGH.tolist(), waves=list(e.WAVES),
        prior_condition_count=3298, fresh_sobol_power=6, count=len(tasks), tasks=tasks,
        global_optimum_proved=False, canonical_baseline_replaced=False)
    write(MANIFEST, data)
    return data


def reference_path(task):
    base = next(t for t in manifest()['references'] if t['wave'] == task['wave'])
    base = dict(base, duration=task['duration'], dt_scale=task['dt_scale'])
    tid = e.task_id(base)
    for folder in (e.TRIALS, e.PARENT / 'outputs/trials'):
        path = folder / 'reference' / tid / f'{tid}.json'
        if path.exists():
            return path
    return e.TRIALS / 'reference' / tid / f'{tid}.json'


def load_reference(task):
    path = reference_path(task)
    if str(path) not in _references:
        _references[str(path)] = read(path)
    return _references[str(path)], path


def assess(record):
    reference, path = load_reference(record['task'])
    result = gates.assess(record, reference)
    result.update(id=record['id'], wave=record['task']['wave'],
        amplitudes=record['task']['amplitudes'], record=record['record_path'],
        reference=str(path), gate_policy_sha256=gates.POLICY_SHA)
    return result


def prefix(task):
    global _prefix
    if _prefix is None:
        _prefix = e.load(e.PARENT / 'search.py', 'relaxed_prefix_backend')
        _prefix.load_reference = load_reference
    record = _prefix.precheck(task)
    if record is None:
        return None
    retained = gates.applicable_prefix_evidence(record)
    if not retained:
        # Its old, tighter failure is not a rejection under the new policy.
        # Full cold evaluation replaces the incomplete record in our folder.
        return None
    record['old_prefix_gates'] = dict(record['gates'])
    record['gates'] = {p['gate']: False for p in retained}
    record['relaxed_retained_prefix_evidence'] = retained
    record['gate_policy_sha256'] = gates.POLICY_SHA
    write(record['record_path'], record)
    return record


def screen(task):
    tid = e.task_id(task)
    cache = e.OUT / 'assessments' / f'{tid}.json'
    if cache.exists():
        row = read(cache)
        assert row['gate_policy_sha256'] == gates.POLICY_SHA
        return row
    own = e.TRIALS / task['stage'] / tid / f'{tid}.json'
    source = task.get('source_record')
    raw = None
    if own.exists():
        r = read(own)
        if r.get('metrics') or gates.applicable_prefix_evidence(r):
            raw = r
    elif source:
        assert sha(source) == task['source_sha256'], source
        r = read(source)
        assert r['model_sha256'] == manifest()['plant_sha256']
        if r.get('metrics') or gates.applicable_prefix_evidence(r):
            raw = r
    if raw is None:
        raw = prefix(task)
        if raw is None:
            raw = e.run_one(task)
    row = assess(raw)
    row['evaluation_source'] = 'prior_evidence_reused' if Path(raw['record_path']).is_relative_to(e.PARENT) else 'new_evaluation'
    row['raw_record_sha256'] = sha(raw['record_path'])
    write(cache, row)
    return row


def summarize(rows, name):
    waves = {}
    for wave in e.WAVES:
        group = [r for r in rows if r['wave'] == wave]
        feasible = sorted((r for r in group if r['feasible']), key=lambda r: -r['D'])
        full = sorted((r for r in group if r['full']), key=lambda r: (len(r['failed']), -r['D']))
        waves[f'{wave:g}'] = dict(count=len(group), feasible=len(feasible),
            best=feasible[0] if feasible else None, diagnostic=full[0] if full else None)
    result = dict(stage=name, assessments=rows, waves=waves, gate_policy_sha256=gates.POLICY_SHA,
                  global_optimum_proved=False)
    write(e.DATA / f'{name}_summary.json', result)
    return result


def run_bank(tasks, name, workers):
    rows = []
    errors = []
    start = time.monotonic()
    best = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(screen, t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:
                errors.append(dict(task=futures[future], error=repr(exc)))
                write(e.DATA / f'{name}_errors.json', dict(status='unresolved', errors=errors))
                print(json.dumps(errors[-1]), flush=True)
                continue
            rows.append(row)
            improved = row['feasible'] and row['D'] > best.get(row['wave'], -float('inf'))
            if improved:
                best[row['wave']] = row['D']
            if improved or i % 50 == 0 or i == len(tasks):
                print(json.dumps(dict(stage=name, done=i, total=len(tasks), wave=row['wave'],
                    D=row['D'], feasible=row['feasible'], failed=row['failed'],
                    source=row['evaluation_source'], elapsed_s=time.monotonic()-start)), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} evaluation errors; completed evidence retained for resume')
    assert len(rows) == len(tasks)
    old_errors = e.DATA / f'{name}_errors.json'
    if old_errors.exists():
        previous = read(old_errors)
        write(old_errors, dict(status='resolved_by_complete_rerun', prior=previous))
    return summarize(rows, name)


def collect(stages=('initial', 'coarse', 'polish')):
    rows = []
    seen = set()
    for stage in stages:
        path = e.DATA / f'{stage}_summary.json'
        if not path.exists():
            continue
        for row in read(path)['assessments']:
            assert row['id'] not in seen
            seen.add(row['id'])
            rows.append(row)
    return rows


def refine_manifest(stage):
    path = e.DATA / f'{stage}_manifest.json'
    if path.exists():
        return read(path)
    assert stage in ('coarse', 'polish')
    rows = collect(('initial',) if stage == 'coarse' else ('initial', 'coarse'))
    strict = read(e.PARENT / 'outputs/data/validated_selection.json')
    seen = {r['id'] for r in rows}
    tasks, starts_by_wave = [], {}
    scale = .08 if stage == 'coarse' else .02
    for wave in e.WAVES:
        full = sorted((r for r in rows if r['wave'] == wave and r['full']),
                      key=lambda r: (not r['feasible'], len(r['failed']), -r['D']))
        starts = []
        for row in full:
            x = np.array([row['amplitudes'][key] for key in KEYS])
            if any(np.linalg.norm((x - y) / HIGH) < .08 for y, _ in starts):
                continue
            starts.append((x, row))
            if len(starts) == 2:
                break
        old = strict['waves'][f'{wave:g}']['selected']
        old_id = old['nominal']['id'] if old else e.task_id(next(t for t in manifest()['references'] if t['wave'] == wave))
        anchor = next((r for r in full if r['id'] == old_id), None)
        if anchor:
            x = np.array([anchor['amplitudes'][key] for key in KEYS])
            if all(np.linalg.norm((x-y)/HIGH) >= .08 for y, _ in starts):
                starts.append((x, anchor))
        starts_by_wave[f'{wave:g}'] = [r['id'] for _, r in starts]
        for x, row in starts:
            base = read(row['record'])['task']
            directions = list(itertools.product((-1., 1.), repeat=3))
            directions += [sign * np.eye(3)[dim] for dim in range(3) for sign in (-1., 1.)]
            for direction in directions:
                value = np.clip(x + scale * HIGH * np.asarray(direction), LOW, HIGH)
                task = {k: v for k, v in base.items() if k not in ('source_record', 'source_sha256')}
                task.update(stage='search', trace=False, amplitudes=dict(zip(KEYS, map(float, value))),
                            origin=dict(stage=stage, seed=row['id']))
                tid = e.task_id(task)
                if tid in seen:
                    continue
                seen.add(tid)
                tasks.append(task)
    result = dict(stage=stage, scale_fraction=scale, seeds=starts_by_wave, count=len(tasks), tasks=tasks,
        selection='Two separated relaxed-gate ranked seeds, plus separated old strict optimum/reference; no reward for amplitude distance')
    write(path, result)
    return result


def audit_setup():
    m = manifest()
    assert sha(m['plant']) == m['plant_sha256']
    for source in m['source_files']:
        assert sha(source['path']) == source['sha256'], source['path']
    for profile in m['PD_profiles_frozen'].values():
        assert sha(profile['source']) == profile['sha256']
    full = retained = rerun = 0
    strict_selection = read(e.PARENT / 'outputs/data/validated_selection.json')
    superset = []
    for wave, item in strict_selection['waves'].items():
        if not item['selected']:
            continue
        selected = item['selected']
        for row in [selected['nominal'], *selected['checks']]:
            a = assess(read(row['record']))
            assert a['feasible'], (wave, a['failed'])
        superset.append(float(wave))
    for task in m['tasks']:
        if not task.get('source_record'):
            continue
        r = read(task['source_record'])
        if r.get('metrics'):
            full += 1
            new = gates.absolute(r)
            for key, value in r['gates'].items():
                if key not in gates.CHANGED and key not in gates.INTERVALS:
                    assert new[key] == value
        elif gates.applicable_prefix_evidence(r):
            retained += 1
        else:
            rerun += 1
    result = dict(passed=True, full_prior_rollouts=full,
        prior_prefix_proofs_still_valid=retained, prior_prefixes_requiring_re_evaluation=rerun,
        fresh_conditions=m['count']-m['prior_condition_count'],
        all_old_strict_qualified_gaits_remain_feasible=superset,
        unchanged_absolute_gates_verified=True, physical_sources_unchanged=True,
        gate_policy_sha256=gates.POLICY_SHA)
    write(e.DATA / 'setup_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'audit', 'initial', 'coarse', 'polish'))
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    m = freeze()
    if args.command == 'freeze':
        print(json.dumps(dict(count=m['count'], prior=m['prior_condition_count'], policy=m['gate_policy']), indent=2))
    elif args.command == 'audit':
        print(json.dumps(audit_setup(), indent=2))
    else:
        assert read(e.DATA / 'setup_audit.json')['passed']
        verification = read(e.DATA / 'verification.json')
        assert verification['passed'] and verification['policy_sha256'] == gates.POLICY_SHA
        bank = m if args.command == 'initial' else refine_manifest(args.command)
        run_bank(bank['tasks'], args.command, args.workers)
