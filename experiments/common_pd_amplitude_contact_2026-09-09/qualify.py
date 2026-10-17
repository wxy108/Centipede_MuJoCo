"""Fresh physical validation, independent of the amplitude trend."""
import argparse,json
import physical_search as ps
import search as s
import core as c
np=c.np


def coupled(workers,rounds):
    for it in range(rounds):
        name=f'coupled_midpoint_{it:02d}';path=c.DATA/'banks'/f'{name}.json'
        if path.exists():tasks=c.read(path)
        else:
            tasks=[];rows=c.nominal('midpoint');step=.02/(2**it)
            for w in c.WAVES:
                candidates=[r for r in rows if r['wave']==w and r['feasible']]
                if not candidates:continue
                center=max(candidates,key=lambda r:r['D'])
                for direction in s.itertools.product((-1,1),repeat=3):
                    x=np.clip(s.vec(center)+np.array(direction)*step*c.HIGH,0,c.HIGH)
                    tasks.append(c.task(w,x,name,phase='midpoint'))
            c.write(c.DATA/f'{name}_declaration.json',dict(source_sha256=c.sha(__file__),step=step,
                purpose='Joint three-amplitude perturbations after coordinate refinement; no monotonicity selection.'))
        c.bank(tasks,name,workers);s.snapshot('midpoint')


def qualify(workers,attempts):
    path=c.DATA/'qualification_progress.json'
    state=c.read(path) if path.exists() else dict(selected={},attempts=[],unresolved=[],complete=False)
    for iteration in range(attempts):
        previous={r['nominal_id'] for r in state['attempts']};rows=c.nominal('midpoint')
        chosen=[]
        for w in c.WAVES:
            if str(w) in state['selected']:continue
            eligible=[r for r in rows if r['wave']==w and r['feasible'] and r['id'] not in previous]
            if eligible:chosen.append(max(eligible,key=lambda r:r['D']))
        if not chosen:break
        # A fresh trace also forces imported records to receive continuous collision observation.
        jobs={}
        for r in chosen:
            jobs[r['id']]=[c.task(r['wave'],s.vec(r),f'qualification_{len(state["attempts"]):03d}',phase='midpoint',
                duration=d,scale=dt,trace=True,prefix=False) for d,dt in [(6,.5),(12,.5),(6,.25)]]
        bankname=f'qualification_{len(state["attempts"]):03d}'
        result={r['id']:r for r in c.bank([t for ts in jobs.values() for t in ts],bankname,workers)}
        for source in chosen:
            nominal,long,half=[result[c.identity(t)] for t in jobs[source['id']]]
            exact=nominal['D'] is not None and abs(nominal['D']-source['D'])<1e-8
            assert exact,('Nominal reproducibility failure',source['id'],nominal['D'],source['D'])
            assert all('nonadjacent_chassis_contact_steps' in r['metrics'] for r in (nominal,long,half))
            relative=abs(half['D']-nominal['D'])/max(abs(nominal['D']),1e-12)
            accepted=all(r['feasible'] for r in (nominal,long,half)) and relative<=.1
            item=dict(wave=source['wave'],nominal_id=nominal['id'],nominal=nominal,long=long,half=half,
                amplitude=nominal['amplitudes'],qualified=accepted,half_step_relative_difference=relative,
                source_search_id=source['id'],nominal_exact=exact)
            state['attempts'].append(item)
            if accepted:state['selected'][str(source['wave'])]=item
        c.write(path,state)
        print(json.dumps(dict(stage='qualification',attempted=len(state['attempts']),qualified=len(state['selected'])),ensure_ascii=False),flush=True)
    state['unresolved']=[w for w in c.WAVES if str(w) not in state['selected']]
    state['complete']=True;state['finite_bank_only']=True
    state['scope']='Best qualified observed displacement per wave; no proof of a continuous global optimum.'
    state['code_sha256']=c.sha(__file__)
    state['all_shared_pd_sha256']=c.read(c.DATA/'manifest.json')['active_pd_sha256']
    c.write(path,state);c.write(c.DATA/'validated_selection.json',state)
    print(json.dumps(dict(qualified=len(state['selected']),unresolved=state['unresolved'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['coupled','qualify'])
    p.add_argument('--workers',type=int,default=12);p.add_argument('--rounds',type=int,default=2);p.add_argument('--attempts',type=int,default=10)
    a=p.parse_args();assert c.read(c.DATA/'physical_parity_check.json')['passed']
    if a.action=='coupled':coupled(a.workers,a.rounds)
    else:qualify(a.workers,a.attempts)
