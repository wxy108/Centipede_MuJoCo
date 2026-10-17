"""Only finish genuinely budget-limited local starts, before qualification."""
import argparse,itertools,time
import psutil
import physical_search as ps
import core as c
import search as s
import qualify as q
from continue_amplitudes import status


def sources():
    return {str(p):c.sha(p) for p in [c.ROOT/'finish_budget_limited_polish.py',c.ROOT/'JOINT_FINISH_PROTOCOL.md']}


def no_contact():
    for channel in ('contact_erase','contact_erase_transient'):
        assert not list((ps.SOURCE_OUT/channel/'assessments').glob('*.json'))


def decision_and_handover():
    old=psutil.Process(58960);born=old.create_time();cmd=old.cmdline()
    assert born==1788952851.4244545 and any(v.endswith('polish_joint_incumbents.py') for v in cmd)
    audit=c.DATA/'joint_polish_audit.json'
    assert not audit.exists(),'Conditional monitor must start before the original audit.'
    declared=dict(time=c.datetime.datetime.now().astimezone().isoformat(),source_hashes=sources(),
        original_pid=old.pid,original_create_time=born,trigger='Original joint-polish budget exhausted with gain greater than0.2mm.')
    declaration=c.DATA/'joint_finish_declaration.json';assert not declaration.exists()
    c.write(declaration,declared)
    print('Conditional monitor prepared; no extra search unless a start actually exhausts its budget while improving.',flush=True)
    while not audit.exists():
        assert old.is_running() and old.create_time()==born
        time.sleep(.02)
    result=c.read(audit)
    assert sources()==declared['source_hashes'],'Conditional source changed after declaration.'
    decision=dict(time=c.datetime.datetime.now().astimezone().isoformat(),
        extension_needed=bool(result['budget_limited']),waves=result['budget_limited'],
        original_audit_sha256=c.sha(audit),source_hashes=sources(),
        condition='Every original budget-limited start still improving by more than0.2mm, independent of wave or amplitude trend.')
    if not decision['extension_needed']:
        c.write(c.DATA/'joint_finish_decision.json',decision)
        print('Original joint polish converged under its criterion; no continuation needed.',flush=True)
        return False
    old.suspend()
    try:
        no_contact();assert not (c.DATA/'validated_selection.json').exists()
        assert c.read(c.DATA/'pipeline_status.json')['stage']=='polishing_improved_joint_probe_incumbents'
        assert old.create_time()==born and not old.children(recursive=True)
        state=c.read(c.DATA/'joint_polish_complete.json')
        for st in state:
            if st['wave'] in decision['waves']:
                assert st['stop']=='poll_budget_reached' and st['history'][-1]['gain_mm']>.2
        receipt=dict(**decision,old_pid=old.pid,old_command=cmd,old_create_time=born,
            unchanged_qualification_source_sha256=c.sha(c.ROOT/'qualify.py'))
        c.write(c.DATA/'joint_finish_handover.json',receipt)
        c.write(c.DATA/'joint_finish_decision.json',decision)
        old.terminate();psutil.wait_procs([old],timeout=10);assert not old.is_running()
    except BaseException:
        if old.is_running():old.resume()
        raise
    return True


def finish():
    initial=c.DATA/'joint_finish_initial.json'
    if initial.exists():states=c.read(initial)
    else:
        states=[st for st in c.read(c.DATA/'joint_polish_complete.json') if st['stop']=='poll_budget_reached']
        for st in states:st['stop']=False;st['additional_history']=[]
        c.write(initial,states)
    directions=[sign*c.np.eye(3)[d] for d in range(3) for sign in (-1,1)]
    directions += [c.np.array(v) for v in itertools.product((-1,1),repeat=3)]
    for poll in range(12):
        name=f'joint_finish_{poll:02d}';checkpoint=c.DATA/f'{name}.json'
        if checkpoint.exists():states=c.read(checkpoint);continue
        tasks=[];mapping={}
        for i,st in enumerate(states):
            if st['stop']:continue
            center=s.vec(st['center'])
            last=(st['additional_history'] or st['history'])[-1]
            before=c.read(c.OUT/'assessments'/f"{last['before']}.json")
            delta=(center-s.vec(before))/c.HIGH;norm=float(c.np.max(c.np.abs(delta)))
            assert norm>0
            pattern=delta/norm
            proposed=[c.np.clip(center+.005*c.HIGH*v,0,c.HIGH) for v in directions]
            proposed += [c.np.clip(center+step*c.HIGH*pattern,0,c.HIGH) for step in (.01,.02,.04)]
            planned=[c.task(st['wave'],x,name,phase='midpoint') for x in proposed]
            tasks.extend(planned);mapping[i]=[c.identity(t) for t in planned]
        if not tasks:break
        result={r['id']:r for r in c.bank(tasks,name,24)}
        for i,ids in mapping.items():
            st=states[i];old=st['center']
            best=max([old,*[result[k] for k in ids if result[k]['feasible']]],key=lambda r:r['D'])
            gain=best['D']-old['D'];st['center']=best
            st['additional_history'].append(dict(poll=poll,before=old['id'],after=best['id'],gain_mm=gain))
            if gain<=.2:st['stop']='no_improvement_in_stencil_and_pattern_at_tolerance'
        c.write(checkpoint,states)
    for st in states:
        if not st['stop']:st['stop']='additional_poll_budget_reached'
    c.write(c.DATA/'joint_finish_complete.json',states)
    c.write(c.DATA/'joint_finish_audit.json',dict(starts=len(states),waves=[st['wave'] for st in states],
        converged=sum(st['stop']=='no_improvement_in_stencil_and_pattern_at_tolerance' for st in states),
        budget_limited=[st['wave'] for st in states if st['stop']=='additional_poll_budget_reached'],
        gain_tolerance_mm=.2,normalized_stencil_step=.005,pattern_distances=[.01,.02,.04],
        max_additional_polls=12,finite_search_only=True,source_hashes=sources()))
    s.snapshot('midpoint')


def main(resume=False):
    if resume:
        receipt=c.read(c.DATA/'joint_finish_handover.json')
        for p,h in receipt['source_hashes'].items():assert c.sha(p)==h
        no_contact()
        if psutil.pid_exists(receipt['old_pid']):assert psutil.Process(receipt['old_pid']).create_time()!=receipt['old_create_time']
    elif not decision_and_handover():return
    status('finishing_budget_limited_joint_starts',workers=24,max_additional_polls=12)
    finish();status('fresh_nominal_long_halfstep_qualification',workers=24)
    q.qualify(24,10)
    status('amplitude_validation_finished',workers=24,selection=str(c.DATA/'validated_selection.json'),
        next='Freeze amplitudes and run the paired contact channels only now.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');main(parser.parse_args().resume)
