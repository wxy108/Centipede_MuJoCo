"""Balanced global exploration, surrogate proposals and explicit multistart polls."""
import argparse,itertools,math
import core as c
np=c.np
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

def vector(r):return np.array([r['amplitudes'][k] for k in c.KEYS])
def diverse(rows,n,distance=.06):
    out=[]
    for r in sorted(rows,key=lambda r:float(r['D']),reverse=True):
        if all(np.linalg.norm((vector(r)-vector(a))/c.HIGH)>=distance for a in out):out.append(r)
        if len(out)==n:break
    return out
def incumbents():
    rows=c.nominal();v=c.read(c.DATA/'validation_index.json') if (c.DATA/'validation_index.json').exists() else {}
    return {w:diverse([r for r in rows if r['wave']==w and r['feasible'] and (r['id'] not in v or v[r['id']]['qualified'])],2) for w in c.WAVES}

def broad():
    saved=c.DATA/'banks/broad.json'
    if saved.exists():return c.read(saved)
    points=qmc.Sobol(3,scramble=True,seed=8092026).random_base2(5)*c.HIGH
    anchors=[[0,0,0],[0,18.75,22.918311805232932],[0,49.37746770784939,22.918311805232932],[1,70,36],
        [36,90,60],[36,0,30],[18,90,0],[18,45,60]]
    best=incumbents();available=[w for w in c.WAVES if best[w]];tasks=[]
    for w in c.WAVES:
        tasks += [c.task(w,x,'broad') for x in [*points,*anchors]]
        transfer=list(dict.fromkeys([*sorted(available,key=lambda v:abs(v-w))[:2],*([available[0],available[-1]] if available else [])]))
        tasks += [c.task(w,vector(best[v][0]),'broad') for v in transfer if v!=w]
    return tasks

def propose(iteration):
    saved=c.DATA/'banks'/f'adaptive_{iteration:02d}.json'
    if saved.exists():return c.read(saved)
    fresh=c.nominal();old=c.history()['full_rows'];tasks=[];ledgers=[]
    validation=c.read(c.DATA/'validation_index.json') if (c.DATA/'validation_index.json').exists() else {}
    for w in c.WAVES:
        history={tuple(vector(r)):r for r in old if r['wave']==w}
        history.update({tuple(vector(r)):r for r in fresh if r['wave']==w and ('error' not in r)})
        training=list(history.values());X=np.array([vector(r)/c.HIGH for r in training]);yes=np.array([r['feasible'] and (r['id'] not in validation or validation[r['id']]['qualified']) for r in training])
        full=[r for r in training if r['D'] is not None];xr=np.array([vector(r)/c.HIGH for r in full]);yr=np.array([r['D'] for r in full])
        seed=8092026+1000*iteration+int(w*2);rng=np.random.default_rng(seed)
        glob=qmc.Sobol(3,scramble=True,seed=seed).random_base2(11)
        qualified=diverse([r for r in fresh if r['wave']==w and r['feasible'] and (r['id'] not in validation or validation[r['id']]['qualified'])],3)
        if not qualified:qualified=diverse([r for r in full if r['feasible']],3)
        if not qualified:qualified=diverse(full,3)
        local=[]
        for row in qualified:
            for sigma in [.02,.06,.15]:local.append(np.clip(vector(row)/c.HIGH+rng.normal(0,sigma,(128,3)),0,1))
        pool=np.vstack([glob,*local]);tested=np.array([vector(r)/c.HIGH for r in fresh if r['wave']==w])
        min_distance=np.min(np.linalg.norm(pool[:,None,:]-tested[None,:,:],axis=2),axis=1)
        keep=min_distance>1e-5;pool=pool[keep];min_distance=min_distance[keep]
        reg=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=2,max_features=1.,random_state=seed,n_jobs=1).fit(xr,yr)
        preds=np.array([tree.predict(pool) for tree in reg.estimators_]);mean=preds.mean(axis=0);std=preds.std(axis=0)
        if len(np.unique(yes))==2:
            clf=ExtraTreesClassifier(n_estimators=96,min_samples_leaf=2,random_state=seed,n_jobs=1).fit(X,yes)
            probability=clf.predict_proba(pool)[:,list(clf.classes_).index(True)]
        else:probability=np.full(len(pool),.25 if not yes.any() else .9)
        incumbent=max([r['D'] for r in fresh if r['wave']==w and r['feasible'] and (r['id'] not in validation or validation[r['id']]['qualified'])],default=0)
        scores=(np.maximum(0,mean-incumbent)+1.25*std+1)*(.1+.9*probability)
        categories=[('model_gain',scores)]*6+[('uncertainty',std*(.2+.8*probability)),('coverage',min_distance)]
        chosen=[];log=[]
        for kind,score in categories:
            for idx in np.argsort(score)[::-1]:
                x=pool[idx]
                if any(np.linalg.norm(x-pool[j])<.025 for j in chosen):continue
                chosen.append(int(idx));tasks.append(c.task(w,x*c.HIGH,f'adaptive_{iteration:02d}'))
                log.append(dict(kind=kind,pool_index=int(idx),predicted_D=float(mean[idx]),model_std=float(std[idx]),estimated_feasibility=float(probability[idx]),x_deg=(x*c.HIGH).tolist()));break
        assert len(chosen)==8
        pool_path=c.DATA/'proposals'/f'wave{w:g}_round{iteration}.npz';pool_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(pool_path,pool_normalized=pool,mean=mean,std=std,feasibility=probability,score=scores,training_x=X,training_feasible=yes)
        ledgers.append(dict(wave=w,seed=seed,training=len(training),fresh=sum(r['wave']==w for r in fresh),incumbent=incumbent,chosen=log,pool_file=str(pool_path),pool_sha256=c.sha(pool_path)))
    c.write(c.DATA/'proposals'/f'round{iteration}.json',dict(rule='Six diverse upper-confidence gain/feasibility proposals, one uncertainty proposal and one maximin exploration proposal. Surrogates select experiments only; actual displacement and unchanged gates decide ranking.',waves=ledgers))
    return tasks

def local(workers):
    import qualify_checkpoint
    qualify_checkpoint.main('prelocal_qualification',c.WAVES,workers)
    initial=c.DATA/'local_initial.json'
    if initial.exists():states=c.read(initial)
    else:
        states=[]
        for w,rr in incumbents().items():
            for i,r in enumerate(rr):states.append(dict(key=f'w{w:g}_s{i}',wave=w,center=r,step=.04,history=[],stop=None))
            if not rr:states.append(dict(key=f'w{w:g}_none',wave=w,center=None,step=.04,history=[],stop='no_current_feasible_start'))
        c.write(initial,states)
    directions=[sign*np.eye(3)[j] for j in range(3) for sign in [-1,1]]+list(itertools.product([-1,1],repeat=3))
    for poll in range(8):
        after=c.DATA/f'local_after_{poll:02d}.json'
        if after.exists():
            states=c.read(after)
            if poll in (0,2,4,6):qualify_checkpoint.main(f'local_qualification_{poll:02d}',c.WAVES,workers)
            continue
        validation=c.read(c.DATA/'validation_index.json');eligible=incumbents()
        for st in states:
            if st['center'] is None:continue
            known=validation.get(st['center']['id'])
            if known and not known['qualified']:
                other=[a['center'] for a in states if a['wave']==st['wave'] and a['key']!=st['key'] and a['center'] is not None]
                choices=eligible[st['wave']]
                choices=sorted(choices,key=lambda r:all(np.linalg.norm((vector(r)-vector(a))/c.HIGH)>=.03 for a in other),reverse=True)
                if choices:
                    st.setdefault('qualification_replacements',[]).append(dict(before=st['center']['id'],after=choices[0]['id'],before_poll=poll,failed_reasons=known['reasons']))
                    st['center']=choices[0];st['step']=.04;st['stop']=None
                else:st['stop']='no_start_after_qualification_failure'
        tasks=[];ids={}
        for st in states:
            if st['stop']:continue
            ids[st['key']]=[];x=vector(st['center'])
            for d in directions:
                t=c.task(st['wave'],np.clip(x+st['step']*c.HIGH*np.array(d),0,c.HIGH),f'local_{poll:02d}')
                tasks.append(t);ids[st['key']].append(c.identity(t))
        if not tasks:break
        got={r['id']:r for r in c.bank(tasks,f'local_{poll:02d}',workers)}
        for st in states:
            if st['stop']:continue
            old=st['center'];available=[got[key] for key in ids[st['key']] if got[key]['feasible'] and (key not in validation or validation[key]['qualified'])];best=max([old,*available],key=lambda r:r['D']);gain=best['D']-old['D']
            st['history'].append(dict(poll=poll,step=st['step'],before=old['id'],after=best['id'],gain_mm=gain));st['center']=best
            if gain<=.2:
                if st['step']<=.005+1e-12:st['stop']='no_material_improvement_at_declared_resolution'
                else:st['step']/=2
        c.write(after,states)
        if poll in (0,2,4,6):qualify_checkpoint.main(f'local_qualification_{poll:02d}',c.WAVES,workers)
    for st in states:
        if not st['stop']:st['stop']='poll_budget_reached_not_converged'
    c.write(c.DATA/'local_complete.json',states)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);ap.add_argument('--stage',choices=['all','broad','adaptive','local'],default='all');args=ap.parse_args()
    assert c.read(c.DATA/'screening_verification.json')['passed']
    if not (c.DATA/'warm_fresh_results.json').exists():c.write(c.DATA/'warm_fresh_results.json',c.nominal())
    if args.stage in ['all','broad']:c.bank(broad(),'broad',args.workers)
    if args.stage in ['all','adaptive']:
        for it in range(4):c.bank(propose(it),f'adaptive_{it:02d}',args.workers)
    if args.stage in ['all','local']:local(args.workers)
