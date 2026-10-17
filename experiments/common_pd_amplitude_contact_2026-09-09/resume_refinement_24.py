"""Change worker allocation only at the completed adaptive-bank boundary."""
import time,psutil
import physical_search as ps
import core as c
import search as s
import qualify as q
from continue_amplitudes import status


def main():
    pid=65360;old=psutil.Process(pid);born=old.create_time()
    assert old.cmdline()[2].endswith('common_pd_amplitude_contact_2026-09-09/continue_amplitudes.py')
    assert old.cmdline()[-2:]==['--workers','16']
    boundary=c.DATA/'completed/adaptive_midpoint_02.json'
    while not boundary.exists():
        assert old.is_running() and old.create_time()==born
        time.sleep(2)
    assert old.is_running() and old.create_time()==born
    old.suspend()
    try:
        children=old.children(recursive=True)
        assert all(p.name().lower()=='python.exe' for p in children)
        receipt=dict(reason='Increase CPU throughput after the third adaptive bank completes; no physics, PD, objective, candidates, or seed changes.',
            old_pid=pid,old_create_time=born,old_cmdline=old.cmdline(),child_pids=[p.pid for p in children],
            completed_boundary_sha256=c.sha(boundary),previous_workers=16,new_workers=24,
            retained_assessment_files=len(list((c.OUT/'assessments').glob('*.json'))),
            note='A just-started local-bank worker may be interrupted. Every complete assessment is cached; any incomplete task is rerun at the same input identity.')
        c.write(c.DATA/'worker_allocation_switch.json',receipt)
        for p in children:p.terminate()
        old.terminate()
        _,alive=psutil.wait_procs([old,*children],timeout=10)
        for p in alive:p.kill()
    except BaseException:
        if old.is_running():old.resume()
        raise
    status('multistart_local',workers=24)
    s.local('midpoint',24,8);s.snapshot('midpoint')
    status('coupled_three_amplitude_probes',workers=24);q.coupled(24,2)
    status('fresh_nominal_long_halfstep_qualification',workers=24);q.qualify(24,10)
    status('amplitude_validation_finished',workers=24,selection=str(c.DATA/'validated_selection.json'),
        next='Assess validated trend and coverage; amplitude freeze still precedes contact erasure.')


if __name__=='__main__':main()
