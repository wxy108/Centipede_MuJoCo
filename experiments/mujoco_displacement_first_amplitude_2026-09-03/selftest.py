"""Read-only policy, isolation and search-plan checks before freezing."""
import copy,json,math
import study as s

def run():
    old=s.read(s.OLD/'outputs/data/amplitude_manifest.json')
    s._manifest=dict(old,source_records=[],second_sobol_seed=39302026,second_sobol_power=7,
      local_starts=3,local_step=.04,local_min_step=.0025,local_max_polls=16,improvement_mm=1e-6,
      body_profile_deg=[],gate_policy_sha256=s.gates.POLICY_SHA)
    assert s.e.TRIALS==s.TRIALS==s.e.backend.TRIALS and s.e.OUT==s.OUT
    assert s.gates.POLICY_SHA!=old['gate_policy_sha256']
    assert set(s.gates.SPEC['diagnostics_not_gates'])
    banks=s.second_sobol();assert len(banks)==14*128==1792 and len({s.e.task_id(t) for t in banks})==1792
    inherited=s.first_sobol();assert len(inherited)==14*256==3584
    for t in inherited[:8]+banks[:8]:s.check_task(t)
    bad=dict(banks[0],config=dict(banks[0]['config'],yaw_k=-1))
    try:s.check_task(bad)
    except AssertionError:pass
    else:raise AssertionError('PD mutation not blocked')
    sample=None
    for row in s.source_rows().values():
        raw=s.read(row['record'])
        if raw.get('metrics') and raw['metrics']['displacement_mm']>0:sample=raw;break
    assert sample
    assessed=s.gates.assess(sample);assert assessed['D']==sample['metrics']['displacement_mm']
    # Diagnostics cannot silently enter the gate: changing slip alone must not alter feasibility.
    altered=copy.deepcopy(sample);altered['metrics']['loaded_foot_slip_p95_mm_s']=1e12
    assert s.gates.assess(altered)['feasible']==assessed['feasible']
    # Every declared numeric limit is active: a gross roll must fail explicitly.
    altered=copy.deepcopy(sample);altered['metrics']['root_roll_abs_peak_deg']=36
    assert not s.gates.assess(altered)['gates']['root_roll_peak']
    print(json.dumps(dict(passed=True,isolated=True,model_controller_PD_frozen=True,
      inherited_sobol=3584,independent_sobol=1792,objective='D only',diagnostics_not_gates=True)))

if __name__=='__main__':run()
