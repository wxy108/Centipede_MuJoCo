"""Finer-step global proposals and bounded multistart refinement."""
import argparse,itertools
import core as c
import acquisition
import qualify_checkpoint as q
import validate
from prepare import vec,diverse
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
np=c.np

def checked():return c.read(c.DATA/'validation_index.json') if (c.DATA/'validation_index.json').exists() else {}
def eligible(rows,validation):return [r for r in rows if r['feasible'] and (r['id'] not in validation or validation[r['id']]['qualified'])]
def incumbents():
    rows=eligible(c.nominal(),checked())
    return {w:diverse([r for r in rows if r['wave']==w],2,.06) for w in c.WAVES}

def broad():
    path=c.DATA/'banks/broad.json'
    if path.exists():return c.read(path)
    points=qmc.Sobol(3,scramble=True,seed=90820261).random_base2(4)*c.HIGH
    anchors=[[0,0,0],[0,18.75,22.918311805232932],[0,49.37746770784939,22.918311805232932],
        [1,70,36],[36,90,60],[36,0,30],[18,90,0],[18,45,60]]
    best=incumbents();available=[w for w in c.WAVES if best[w]];tasks=[]
    for w in c.WAVES:
        tasks += [c.task(w,x,'broad') for x in [*points,*anchors]]
        transfers=list(dict.fromkeys([*sorted(available,key=lambda v:abs(v-w))[:2],*([available[0],available[-1]] if available else [])]))
        tasks += [c.task(w,vec(best[v][0]),'broad') for v in transfers if v!=w]
    return tasks

def propose(iteration):
    path=c.DATA/'banks'/f'adaptive_{iteration:02d}.json'
    if path.exists():return c.read(path)
    fresh=c.nominal();validation=checked();tasks=[];logs=[]
    for w in c.WAVES:
        rows=[r for r in fresh if r['wave']==w and 'error' not in r]
        full=[r for r in rows if r['full']];assert full
        X=np.array([vec(r)/c.HIGH for r in rows]);yes=np.array([r['feasible'] and (r['id'] not in validation or validation[r['id']]['qualified']) for r in rows])
        xr=np.array([vec(r)/c.HIGH for r in full]);yr=np.array([r['D'] for r in full])
        seed=90820261+1000*iteration+int(2*w);rng=np.random.default_rng(seed)
        centers=diverse(eligible(rows,validation),3,.06) or diverse(full,3,.06)
        pool=np.vstack([qmc.Sobol(3,scramble=True,seed=seed).random_base2(11),
            *[np.clip(vec(r)/c.HIGH+rng.normal(0,sigma,(128,3)),0,1) for r in centers for sigma in [.02,.06,.15]]])
        distance=np.min(np.linalg.norm(pool[:,None]-X[None,:],axis=2),axis=1);keep=distance>1e-5;pool=pool[keep];distance=distance[keep]
        reg=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=1).fit(xr,yr)
        pred=np.array([tr.predict(pool) for tr in reg.estimators_]);mean=pred.mean(axis=0);std=pred.std(axis=0)
        if len(np.unique(yes))==2:
            clf=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=1).fit(X,yes)
            probability=clf.predict_proba(pool)[:,list(clf.classes_).index(True)]
        else:probability=np.full(len(pool),.9 if yes.any() else .25)
        inc=max([r['D'] for r in eligible(rows,validation)],default=0.)
        performance,gain=acquisition.scores(mean,std,probability,inc)
        categories=[('predicted_performance',performance)]*3+[('expected_gain',gain)]*3+[
            ('uncertainty',std*(.2+.8*probability)),('coverage',distance)]
        chosen=[];log=[]
        for kind,score in categories:
            for j in np.argsort(score)[::-1]:
                if any(np.linalg.norm(pool[j]-pool[k])<.025 for k in chosen):continue
                chosen.append(int(j));tasks.append(c.task(w,pool[j]*c.HIGH,f'adaptive_{iteration:02d}'))
                log.append(dict(kind=kind,pool_index=int(j),predicted_D=float(mean[j]),ensemble_spread=float(std[j]),estimated_feasibility=float(probability[j]),amplitudes=(pool[j]*c.HIGH).tolist()));break
        assert len(chosen)==8
        file=c.DATA/'proposals'/f'wave{w:g}_round{iteration}.npz';file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,pool=pool,mean=mean,std=std,probability=probability,performance=performance,expected_gain=gain,training_x=X,training_feasible=yes,regression_x=xr,regression_D=yr)
        logs.append(dict(wave=w,seed=seed,training=len(rows),incumbent=inc,chosen=log,pool=str(file),sha256=c.sha(file)))
    c.write(c.DATA/'proposals'/f'round{iteration}.json',dict(rule='Three predicted feasible performance, three approximate expected improvement, one uncertainty and one coverage proposal. Fine-step training data only; actual full simulation decides performance.',waves=logs))
    return tasks

def replace_failed(states,poll):
    validation=checked();starts=incumbents()
    for st in states:
        if st['center'] is None:continue
        old=validation.get(st['center']['id'])
        if not old or old['qualified']:continue
        other=[a['center'] for a in states if a['wave']==st['wave'] and a['key']!=st['key'] and a['center'] is not None]
        choices=sorted(starts[st['wave']],key=lambda r:all(np.linalg.norm((vec(r)-vec(a))/c.HIGH)>=.03 for a in other),reverse=True)
        if not choices:st['stop']='no_start_after_qualification_failure';continue
        st.setdefault('qualification_replacements',[]).append(dict(before=st['center']['id'],after=choices[0]['id'],before_poll=poll,reasons=old['reasons']))
        st['center']=choices[0];st['step']=.04;st['stop']=None
    return validation

def local(workers):
    q.main('prelocal_qualification',c.WAVES,workers)
    path=c.DATA/'local_initial.json'
    if path.exists():states=c.read(path)
    else:
        states=[]
        for w,rr in incumbents().items():
            for i,r in enumerate(rr):states.append(dict(key=f'w{w:g}_s{i}',wave=w,center=r,step=.04,history=[],diagonal_checked=False,stop=None))
            if not rr:states.append(dict(key=f'w{w:g}_none',wave=w,center=None,step=.04,history=[],diagonal_checked=False,stop='no_fine_feasible_start'))
        c.write(path,states)
    axes=[sign*np.eye(3)[j] for j in range(3) for sign in [-1,1]]
    for poll in range(6):
        after=c.DATA/f'local_after_{poll:02d}.json'
        if after.exists():
            states=c.read(after)
            if poll in (0,2,4):q.main(f'local_qualification_{poll:02d}',c.WAVES,workers)
            continue
        validation=replace_failed(states,poll);tasks=[];ids={}
        for st in states:
            if st['stop']:continue
            ids[st['key']]=[]
            for d in axes:
                t=c.task(st['wave'],np.clip(vec(st['center'])+st['step']*c.HIGH*d,0,c.HIGH),f'local_{poll:02d}');tasks.append(t);ids[st['key']].append(c.identity(t))
        if not tasks:break
        got={r['id']:r for r in c.bank(tasks,f'local_{poll:02d}',workers)};diagonal=[];extra={}
        for st in states:
            if st['stop']:continue
            best=max([st['center'],*eligible([got[k] for k in ids[st['key']]],validation)],key=lambda r:r['D'])
            if best['D']-st['center']['D']<=.2 and not st['diagonal_checked']:
                extra[st['key']]=[]
                for d in itertools.product([-1,1],repeat=3):
                    t=c.task(st['wave'],np.clip(vec(st['center'])+st['step']*c.HIGH*np.array(d),0,c.HIGH),f'local_diagonal_{poll:02d}');diagonal.append(t);extra[st['key']].append(c.identity(t))
                st['diagonal_checked']=True
        if diagonal:got.update({r['id']:r for r in c.bank(diagonal,f'local_diagonal_{poll:02d}',workers)})
        for st in states:
            if st['stop']:continue
            old=st['center'];candidates=eligible([got[k] for k in ids[st['key']]+extra.get(st['key'],[])],validation)
            best=max([old,*candidates],key=lambda r:r['D']);gain=best['D']-old['D']
            st['history'].append(dict(poll=poll,step=st['step'],before=old['id'],after=best['id'],gain_mm=gain,diagonal_challenge=st['key'] in extra));st['center']=best
            if gain<=.2:
                if st['step']<=.005+1e-12:st['stop']='no_material_improvement_at_declared_resolution'
                else:st['step']/=2
        c.write(after,states)
        if poll in (0,2,4):q.main(f'local_qualification_{poll:02d}',c.WAVES,workers)
    for st in states:
        if not st['stop']:st['stop']='poll_budget_reached_not_converged'
    c.write(c.DATA/'local_complete.json',states)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    assert (c.DATA/'completed/warm.json').exists() and c.read(c.DATA/'screening_verification.json')['passed']
    if not (c.DATA/'warm_fresh_results.json').exists():c.write(c.DATA/'warm_fresh_results.json',c.nominal())
    validate.early(a.workers)
    c.bank(broad(),'broad',a.workers)
    for i in range(3):c.bank(propose(i),f'adaptive_{i:02d}',a.workers)
    local(a.workers)
