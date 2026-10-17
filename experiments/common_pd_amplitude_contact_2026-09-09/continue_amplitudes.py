"""Finish authorized CPU amplitude work after independent coverage banks complete."""
import argparse,datetime,json,time
import physical_search as ps
import core as c
import search as s
import qualify as q


def status(stage,**extra):
    record=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=stage,**extra)
    c.write(c.DATA/'pipeline_status.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    dependencies=[ps.SOURCE_DATA/'completed/broad_midpoint.json',c.DATA/'completed/supplement_sobol_midpoint.json',c.DATA/'completed/spine_rescue_01.json']
    status('waiting_for_coverage',dependencies=list(map(str,dependencies)))
    while not all(p.exists() for p in dependencies):time.sleep(5)
    status('coverage_complete_import_and_rescue')
    # No overlapping assessment writers remain after the dependency barrier.
    ps.import_completed();ps.rescue(a.workers,2)
    for it in range(3):
        name=f'adaptive_midpoint_{it:02d}';status(name)
        path=c.DATA/'banks'/f'{name}.json'
        tasks=c.read(path) if path.exists() else s.propose('midpoint',it)
        c.bank(tasks,name,a.workers);s.snapshot('midpoint')
    status('multistart_local');s.local('midpoint',a.workers,8);s.snapshot('midpoint')
    status('coupled_three_amplitude_probes');q.coupled(a.workers,2)
    status('fresh_nominal_long_halfstep_qualification');q.qualify(a.workers,10)
    status('amplitude_validation_finished',selection=str(c.DATA/'validated_selection.json'),
        next='Assess trend and search coverage before deciding MuJoCo contact tests or original Isaac fallback. No contact erasure dispatched automatically.')
