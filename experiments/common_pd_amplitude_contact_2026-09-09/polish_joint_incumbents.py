"""One declared local polish of improved joint-probe incumbents; then qualify."""
import argparse,itertools,time
import psutil
import physical_search as ps
import core as c
import search as s
import qualify as q
from continue_amplitudes import status

def no_contact():
    for channel in ('contact_erase','contact_erase_transient'):
        assert not list((ps.SOURCE_OUT/channel/'assessments').glob('*.json'))

def handover():
    old=psutil.Process(53728);cmd=old.cmdline();born=old.create_time()
    assert any(x.endswith('extend_local_convergence.py') for x in cmd),cmd
    checkpoint=c.DATA/'completed/coupled_extended_midpoint_01.json'
    while not checkpoint.exists():
        assert old.is_running() and old.create_time()==born
        time.sleep(.02)
    old.suspend()
    try:
        no_contact();assert not (c.DATA/'validated_selection.json').exists()
        assert c.read(c.DATA/'pipeline_status.json')['stage']=='coupled_three_amplitude_probes_after_extension'
        assert old.create_time()==born and not old.children(recursive=True)
        receipt=dict(time=c.datetime.datetime.now().astimezone().isoformat(),old_pid=old.pid,old_command=cmd,
                     old_create_time=born,completed_checkpoint=str(checkpoint),checkpoint_sha256=c.sha(checkpoint),
                     unchanged_qualification_source_sha256=c.sha(c.ROOT/'qualify.py'),
                     source_hashes={str(p):c.sha(p) for p in [c.ROOT/'polish_joint_incumbents.py',c.ROOT/'JOINT_POLISH_PROTOCOL.md']})
        c.write(c.DATA/'joint_polish_handover.json',receipt)
        old.terminate();psutil.wait_procs([old],timeout=10)
        assert not old.is_running()
    except BaseException:
        if old.is_running():old.resume()
        raise

def polish():
    path=c.DATA/'joint_polish_initial.json'
    if path.exists():states=c.read(path)
    else:
        local=c.read(c.DATA/'local_midpoint_complete.json');rows=c.nominal('midpoint');states=[]
        for w in c.WAVES:
            before=max(st['center']['D'] for st in local if st['wave']==w)
            eligible=[r for r in rows if r['wave']==w and r['feasible']]
            best=max(eligible,key=lambda r:r['D'])
            if best['D']-before>.2:
                states.append(dict(wave=w,center=best,pre_probe_D=before,initial_joint_probe_gain_mm=best['D']-before,
                                   stop=False,history=[]))
        c.write(path,states)
    directions=[sign*c.np.eye(3)[d] for d in range(3) for sign in (-1,1)]
    directions += [c.np.array(v) for v in itertools.product((-1,1),repeat=3)]
    assert len(directions)==14
    for poll in range(12):
        checkpoint=c.DATA/f'joint_polish_{poll:02d}.json'
        if checkpoint.exists():states=c.read(checkpoint);continue
        name=f'joint_polish_{poll:02d}';tasks=[];mapping={}
        for i,st in enumerate(states):
            if st['stop']:continue
            proposed=[c.task(st['wave'],c.np.clip(s.vec(st['center'])+.005*c.HIGH*direction,0,c.HIGH),name,phase='midpoint') for direction in directions]
            tasks += proposed;mapping[i]=[c.identity(t) for t in proposed]
        if not tasks:break
        result={r['id']:r for r in c.bank(tasks,name,24)}
        for i,ids in mapping.items():
            st=states[i];old=st['center'];best=max([old,*[result[k] for k in ids if result[k]['feasible']]],key=lambda r:r['D'])
            gain=best['D']-old['D'];st['center']=best
            st['history'].append(dict(poll=poll,before=old['id'],after=best['id'],gain_mm=gain))
            if gain<=.2:st['stop']='no_improvement_in_14_directions_at_resolution'
        c.write(checkpoint,states)
    for st in states:
        if not st['stop']:st['stop']='poll_budget_reached'
    c.write(c.DATA/'joint_polish_complete.json',states)
    c.write(c.DATA/'joint_polish_audit.json',dict(starts=len(states),waves=[st['wave'] for st in states],
        converged=sum(st['stop']=='no_improvement_in_14_directions_at_resolution' for st in states),
        budget_limited=[st['wave'] for st in states if st['stop']=='poll_budget_reached'],directions=14,
        normalized_step=.005,gain_tolerance_mm=.2,finite_search_only=True,source_sha256=c.sha(__file__)))
    s.snapshot('midpoint')

def main(resume=False):
    if resume:
        receipt=c.read(c.DATA/'joint_polish_handover.json')
        for p,h in receipt['source_hashes'].items():assert c.sha(p)==h
        no_contact()
        if psutil.pid_exists(receipt['old_pid']):assert psutil.Process(receipt['old_pid']).create_time()!=receipt['old_create_time']
    else:handover()
    status('polishing_improved_joint_probe_incumbents',workers=24,normalized_step=.005,directions=14,poll_safety_limit=12)
    polish();status('fresh_nominal_long_halfstep_qualification',workers=24)
    q.qualify(24,10)
    status('amplitude_validation_finished',workers=24,selection=str(c.DATA/'validated_selection.json'),
           next='Freeze amplitudes and run the paired contact channels only now.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.resume)
