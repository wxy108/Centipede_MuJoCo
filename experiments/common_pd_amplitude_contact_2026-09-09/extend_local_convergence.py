"""Continue unresolved starts uniformly, then qualify before any contact work."""
import os,time,shutil,psutil
import physical_search as ps
import core as c
import search as s
import qualify as q
from continue_amplitudes import status

def handover():
    old=psutil.Process(59616);cmd=old.cmdline();born=old.create_time()
    assert any(x.endswith('resume_refinement_24.py') for x in cmd),cmd
    for channel in ('contact_erase','contact_erase_transient'):
        assert not list((ps.SOURCE_OUT/channel/'assessments').glob('*.json'))
    checkpoint=c.DATA/'local_midpoint_07.json'
    while not checkpoint.exists():
        assert old.is_running() and old.create_time()==born
        time.sleep(.25)
    old.suspend()
    try:
        assert old.create_time()==born
        assert not (c.DATA/'validated_selection.json').exists()
        children=old.children(recursive=True)
        assert all(p.name().lower()=='python.exe' for p in children)
        states=c.read(checkpoint)
        receipt=dict(time=c.datetime.datetime.now().astimezone().isoformat(),old_pid=old.pid,old_command=cmd,
            old_create_time=born,child_pids=[p.pid for p in children],completed_checkpoint=str(checkpoint),
            checkpoint_sha256=c.sha(checkpoint),unresolved_starts=sum(not st['stop'] for st in states),
            old_total_poll_limit=8,new_total_poll_limit=24,workers=24,
            reason='Recent starts still improve at coarse steps. Uniform continuation applies existing convergence criterion to all unresolved starts, independent of the desired trend.',
            source_hashes={str(p):c.sha(p) for p in (c.ROOT/'extend_local_convergence.py',c.ROOT/'CONVERGENCE_EXTENSION.md',c.ROOT/'search.py',c.ROOT/'qualify.py')})
        c.write(c.DATA/'convergence_extension_handover.json',receipt)
        previous=c.DATA/'local_midpoint_complete.json'
        if previous.exists():
            archive=c.OUT/'audit/local_budget08_complete.json';archive.parent.mkdir(parents=True,exist_ok=True)
            assert not archive.exists();shutil.copy2(previous,archive)
        for p in children:p.terminate()
        old.terminate()
        _,alive=psutil.wait_procs([old,*children],timeout=10)
        for p in alive:p.kill()
    except BaseException:
        if old.is_running():old.resume()
        raise

def coupled_after_extension():
    for it in range(2):
        name=f'coupled_extended_midpoint_{it:02d}';path=c.DATA/'banks'/f'{name}.json';step=.02/(2**it)
        if path.exists():tasks=c.read(path)
        else:
            tasks=[];rows=c.nominal('midpoint')
            for wave in c.WAVES:
                candidates=[r for r in rows if r['wave']==wave and r['feasible']]
                if not candidates:continue
                center=max(candidates,key=lambda r:r['D'])
                for direction in s.itertools.product((-1,1),repeat=3):
                    x=c.np.clip(s.vec(center)+c.np.array(direction)*step*c.HIGH,0,c.HIGH)
                    tasks.append(c.task(wave,x,name,phase='midpoint'))
            c.write(c.DATA/f'{name}_declaration.json',dict(source_sha256=c.sha(__file__),step=step,
                purpose='Same two coupled three-amplitude probes after uniform local convergence extension; no amplitude-direction preference.'))
        c.bank(tasks,name,24);s.snapshot('midpoint')

def main():
    handover();status('uniform_local_convergence_extension',workers=24,total_poll_safety_limit=24)
    s.local('midpoint',24,24);s.snapshot('midpoint')
    states=c.read(c.DATA/'local_midpoint_complete.json')
    audit=dict(starts=len(states),converged=sum(st['stop']=='no_axis_improvement_at_resolution' for st in states),
        budget_limited=[dict(key=st['key'],wave=st['wave'],step=st['step'],last_gain_mm=st['history'][-1]['gain_mm']) for st in states if st['stop']=='poll_budget_reached'],
        finite_search_only=True,stopping_tolerance_mm=.2,normalized_probe_resolution=.005)
    c.write(c.DATA/'local_convergence_audit.json',audit)
    status('coupled_three_amplitude_probes_after_extension',workers=24);coupled_after_extension()
    status('fresh_nominal_long_halfstep_qualification',workers=24);q.qualify(24,10)
    status('amplitude_validation_finished',workers=24,selection=str(c.DATA/'validated_selection.json'),
        next='Only now freeze amplitudes and run paired contact-erasure comparisons.')

if __name__=='__main__':main()
