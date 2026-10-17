"""Read-only planning/namespace tests before freezing an experiment."""
import json
import copy
import study as s


def run():
    old=s.read(s.OLD/'outputs/data/amplitude_manifest.json')
    s._manifest=dict(old,sobol_power=8,sobol_seeds=[9302026,19302026],body_profile_deg=list(range(0,37,3)),profile_sobol_power=3)
    assert s.e.TRIALS==s.TRIALS==s.e.backend.TRIALS
    assert s.e.OUT==s.OUT and s.prior.e is s.e
    assert s.e.PARENT!=s.ROOT and s.e.PARENT!=s.OLD
    assert s.gates.POLICY_SHA==old['gate_policy_sha256']
    tasks=s.transfer_bank()
    assert len({s.e.task_id(t) for t in tasks})==len(tasks)
    for t in tasks:s.check_task(t)
    banks=[s.sobol_bank(seed) for seed in s.manifest()['sobol_seeds']]
    assert all(len(b)==14*256 for b in banks)
    first={s.e.task_id(t) for t in banks[0]}
    second={s.e.task_id(t) for t in banks[1]}
    assert len(first)==len(second)==3584 and not first.intersection(second)
    malformed=dict(tasks[0],config=dict(tasks[0]['config'],yaw_k=-1))
    try:s.check_task(malformed)
    except AssertionError:pass
    else:raise AssertionError('PD mutation was not blocked')
    for field,value in [('terrain','rough'),('topology','rigid')]:
        try:s.check_task(dict(tasks[0],**{field:value}))
        except AssertionError:pass
        else:raise AssertionError(field)
    for t in s.manifest()['references']:
        reference,path=s.load_reference(t)
        assert reference['task']['amplitudes']==t['amplitudes']
        assert not path.is_relative_to(s.ROOT)
    rows=s.old_rows()
    assert len(rows)==5036
    for item in s.incumbents():
        r=s.assess(s.read(item['record']))
        assert r['feasible'] and r['D']==item['D']
    # Test the adaptive poll logic on an isolated synthetic coupled objective.
    # This is NOT locomotion evidence and writes no experiment result.
    np=s.np
    target=np.array([.5,.5,.5])
    def synthetic(t):
        x=np.array([t['amplitudes'][k] for k in s.KEYS])/s.HIGH
        delta=x-target
        value=100-float(delta@np.array([[4.,1.,.5],[1.,3.,.4],[.5,.4,2.]])@delta)
        return dict(id=s.e.task_id(t),wave=t['wave'],amplitudes=t['amplitudes'],feasible=True,D=value)
    center=synthetic(s.task(2.,s.HIGH*np.array([.46,.46,.46]),'synthetic_selftest'))
    original_choose,original_bank,original_write=s.choose_starts,s.run_bank,s.write
    s._manifest.update(three_d_max_polls=16,profile_max_polls=8,three_d_min_step=.0025,profile_min_step=.01,improvement_mm=1e-6)
    captured=[]
    try:
        s.choose_starts=lambda kind:[dict(key='synthetic',wave=2.,center=center,step=.04,history=[],stop=None)]
        s.run_bank=lambda tasks,name,workers:[synthetic(t) for t in tasks]
        s.write=lambda path,value:captured.append((str(path),copy.deepcopy(value)))
        refined=s.refine('three_d',1)[0]
        assert refined['center']['D']>center['D']
        assert refined['center']['D']==100.
        assert len(refined['history'])>2
        assert refined['stop']=='poll_stationary_at_declared_resolution_not_global_proof'
        profile=s.refine('profile',1)[0]
        assert profile['center']['amplitudes']['body']==center['amplitudes']['body']
        assert all(row['step']>=.01 for row in profile['history'])
    finally:
        s.choose_starts,s.run_bank,s.write=original_choose,original_bank,original_write
    print(json.dumps(dict(passed=True,transfer_tasks=len(tasks),fresh_global_tasks=7168,
        frozen_gates_equal=True,namespace_isolated=True,old_conditions=5036,
        adaptive_poll_and_fixed_body_profile_tested=True,
        accidental_PD_topology_terrain_changes_rejected=True)))


if __name__=='__main__':run()
