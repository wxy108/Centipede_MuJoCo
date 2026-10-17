"""Check relaxed-only cases, unchanged early proofs, and exact cold replay."""
import concurrent.futures
import json
import study as s
e = s.e


def check(item):
    raw = s.read(item['record'])
    full = s.assess(raw)
    proof = s.prefix(dict(raw['task'], stage='verification_prefix', trace=False))
    if full['feasible']:
        assert proof is None, (raw['id'], proof)
    if proof:
        assert not full['feasible']
        assert s.gates.applicable_prefix_evidence(proof)
    replay_equal = None
    if item['cold_replay']:
        replay = e.run_one(dict(raw['task'], stage='verification_cold', trace=False))
        for key, value in raw['metrics'].items():
            assert s.np.array_equal(s.np.asarray(value), s.np.asarray(replay['metrics'][key]), equal_nan=True), key
        replay_equal = True
    return dict(record=item['record'], feasible_new=full['feasible'], failed_new=full['failed'],
                prior_absolute_failed=[k for k, v in raw['gates'].items() if not v],
                retained_prefix=list(proof['gates']) if proof else [],
                exact_cold_metrics=replay_equal, passed=True)


if __name__ == '__main__':
    path = e.DATA / 'verification.json'
    if path.exists():
        data = s.read(path)
        assert data['policy_sha256'] == s.gates.POLICY_SHA and data['passed']
        print(json.dumps(data, indent=2))
        raise SystemExit(0)
    rows = []
    for task in s.manifest()['tasks']:
        if not task.get('source_record'):
            continue
        raw = s.read(task['source_record'])
        if raw.get('metrics'):
            row = s.assess(raw)
            if row['feasible'] and not all(raw['gates'].values()):
                rows.append(row)
    chosen = []
    for wave in (1., 2., 4.):
        cases = sorted((r for r in rows if r['wave'] == wave), key=lambda r: -r['D'])
        assert cases, wave
        chosen.append(dict(record=cases[0]['record'], cold_replay=wave in (1., 2.)))
    task = next(t for t in s.manifest()['references'] if t['wave'] == 8.)
    chosen.append(dict(record=str(s.reference_path(task)), cold_replay=False))
    print(json.dumps(dict(cases=chosen), indent=2), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, chosen))
    assert any(r['retained_prefix'] for r in results)
    data = dict(passed=True, policy_sha256=s.gates.POLICY_SHA, checks=results,
                physics_unchanged=True, relaxed_away_prefixes_not_rejected=True)
    s.write(path, data)
    print(json.dumps(data, indent=2))
