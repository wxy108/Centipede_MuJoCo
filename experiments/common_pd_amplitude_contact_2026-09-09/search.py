"""Common-PD amplitude search; no amplitude-trend or crossover score."""
import argparse,itertools,json
import core as c
from prepare import anchors
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
import fast_metrics
fast_metrics.install()
np=c.np

def vec(r):return np.array([r['amplitudes'][k] for k in c.KEYS])
def diverse(rows,n,distance=.07):
    selected=[]
    for r in sorted(rows,key=lambda x:x['D'],reverse=True):
        if all(np.linalg.norm((vec(r)-vec(p))/c.HIGH)>distance for p in selected):selected.append(r)
        if len(selected)==n:break
    return selected

def broad(phase,power):
    history=anchors()
    q=qmc.Sobol(3,scramble=True,seed=90920261).random_base2(power)*c.HIGH
    fixed=[[0,18.75,23],[12,18.75,23],[24,18.75,23],[36,18.75,23],
           [0,60,30],[36,0,30],[18,90,0],[18,45,60]]
    tasks=[]
    for w in c.WAVES:
        points=[*q,*fixed]
        for name,mapping in history.items():
            nearest=sorted(mapping,key=lambda v:abs(v-w))[:2]
            points.extend(mapping[v] for v in nearest)
        for sourcew in (1.5,4.,7.):
            if sourcew in history['recovery']:points.append(history['recovery'][sourcew])
        tasks.extend(c.task(w,x,'broad_'+phase,phase=phase) for x in points)
    return tasks

def propose(phase,iteration):
    allrows=c.nominal(phase);tasks=[];notes=[]
    for w in c.WAVES:
        rows=[r for r in allrows if r['wave']==w and 'error' not in r]
        full=[r for r in rows if r['full']]
        if not full:continue
        X=np.array([vec(r)/c.HIGH for r in rows]);ok=np.array([r['feasible'] for r in rows])
        seed=90920261+100*iteration+int(w*10);rng=np.random.default_rng(seed)
        centers=diverse([r for r in rows if r['feasible']],3) or diverse(full,2)
        pool=np.vstack([qmc.Sobol(3,scramble=True,seed=seed).random_base2(10),
            *[np.clip(vec(r)/c.HIGH+rng.normal(0,s,(128,3)),0,1) for r in centers for s in (.03,.1)]])
        distance=np.min(np.linalg.norm(pool[:,None]-X[None,:],axis=2),axis=1)
        good=distance>1e-6;pool=pool[good];distance=distance[good]
        reg=ExtraTreesRegressor(n_estimators=96,random_state=seed,n_jobs=1).fit(np.array([vec(r)/c.HIGH for r in full]),[r['D'] for r in full])
        predictions=np.array([t.predict(pool) for t in reg.estimators_]);mean=predictions.mean(0);spread=predictions.std(0)
        if len(np.unique(ok))==2:
            clf=ExtraTreesClassifier(n_estimators=96,random_state=seed,n_jobs=1).fit(X,ok)
            probability=clf.predict_proba(pool)[:,list(clf.classes_).index(True)]
        else:probability=np.full(len(pool),.9 if ok.any() else .25)
        incumbent=max([r['D'] for r in rows if r['feasible']],default=0)
        scores=[np.maximum(mean,0)*(.1+.9*probability)]*3+[(np.maximum(mean-incumbent,0)+spread)*(.1+.9*probability)]*3+[spread,distance]
        chosen=[]
        for score in scores:
            for j in np.argsort(score)[::-1]:
                if any(np.linalg.norm(pool[j]-pool[k])<.025 for k in chosen):continue
                chosen.append(int(j));tasks.append(c.task(w,pool[j]*c.HIGH,f'adaptive_{phase}_{iteration:02d}',phase=phase));break
        notes.append(dict(wave=w,training=len(rows),full=len(full),feasible=int(ok.sum()),incumbent_D=incumbent,
           chosen=[dict(x=(pool[j]*c.HIGH).tolist(),predicted_D=float(mean[j]),heuristic_spread=float(spread[j]),heuristic_feasible=float(probability[j])) for j in chosen]))
    c.write(c.DATA/f'adaptive_{phase}_{iteration:02d}_proposals.json',notes)
    return tasks

def local(phase,workers,polls=8):
    path=c.DATA/f'local_{phase}_initial.json'
    if path.exists():states=c.read(path)
    else:
        rows=c.nominal(phase);states=[]
        for w in c.WAVES:
            for i,r in enumerate(diverse([r for r in rows if r['wave']==w and r['feasible']],2)):
                states.append(dict(key=f'w{w:g}_s{i}',wave=w,center=r,step=.04,history=[],stop=False))
        c.write(path,states)
    directions=[sign*np.eye(3)[j] for j in range(3) for sign in (-1,1)]
    for poll in range(polls):
        checkpoint=c.DATA/f'local_{phase}_{poll:02d}.json'
        if checkpoint.exists():states=c.read(checkpoint);continue
        tasks=[];keys={}
        for st in states:
            if st['stop']:continue
            keys[st['key']]=[]
            for delta in directions:
                t=c.task(st['wave'],np.clip(vec(st['center'])+st['step']*c.HIGH*delta,0,c.HIGH),f'local_{phase}_{poll:02d}',phase=phase)
                tasks.append(t);keys[st['key']].append(c.identity(t))
        if not tasks:break
        measured={r['id']:r for r in c.bank(tasks,f'local_{phase}_{poll:02d}',workers)}
        for st in states:
            if st['stop']:continue
            candidates=[measured[k] for k in keys[st['key']] if measured[k]['feasible']]
            old=st['center'];best=max([old,*candidates],key=lambda r:r['D']);gain=best['D']-old['D']
            st['history'].append(dict(poll=poll,before=old['id'],after=best['id'],gain_mm=gain,step=st['step']))
            st['center']=best
            if gain<=.2:
                if st['step']<=.005:st['stop']='no_axis_improvement_at_resolution'
                else:st['step']/=2
        c.write(checkpoint,states)
    for st in states:
        if not st['stop']:st['stop']='poll_budget_reached'
    c.write(c.DATA/f'local_{phase}_complete.json',states)

def snapshot(phase):
    rows=c.nominal(phase);out=[]
    for w in c.WAVES:
        candidates=[r for r in rows if r['wave']==w]
        eligible=[r for r in candidates if r['feasible']]
        best=max(eligible,key=lambda r:r['D']) if eligible else None
        out.append(dict(wave=w,evaluated=len(candidates),feasible=len(eligible),selected=best))
    c.write(c.DATA/f'checkpoint_{phase}.json',dict(status='SEARCH_CHECKPOINT_NOT_VALIDATED',waves=out))
    print(json.dumps([dict(wave=r['wave'],n=r['evaluated'],feasible=r['feasible'],D=r['selected']['D'] if r['selected'] else None,
        amplitude=r['selected']['amplitudes'] if r['selected'] else None) for r in out],indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['broad','adaptive','local','snapshot','all'])
    parser.add_argument('--phase',choices=['legacy','midpoint'],default='midpoint');parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--power',type=int,default=5);parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--polls',type=int,default=8)
    args=parser.parse_args();assert c.read(c.DATA/'fast_trajectory_check.json')['passed']
    if args.action in ('broad','all'):c.bank(broad(args.phase,args.power),'broad_'+args.phase,args.workers);snapshot(args.phase)
    if args.action in ('adaptive','all'):
        for i in range(args.rounds):
            p=c.DATA/'banks'/f'adaptive_{args.phase}_{i:02d}.json'
            tasks=c.read(p) if p.exists() else propose(args.phase,i)
            c.bank(tasks,f'adaptive_{args.phase}_{i:02d}',args.workers);snapshot(args.phase)
    if args.action in ('local','all'):local(args.phase,args.workers,args.polls);snapshot(args.phase)
    if args.action=='snapshot':snapshot(args.phase)
