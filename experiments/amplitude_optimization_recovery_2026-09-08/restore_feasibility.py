"""Additional amplitude-only proposals for waves without any feasible gait."""
import argparse,datetime
import core as c
from search import vector
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor
np=c.np

def violation(row):
    m=row['metrics'];values=[]
    for key,limit in c.policy.SPEC.items():
        if key not in m or isinstance(limit,bool) or not isinstance(limit,(int,float)):continue
        values.append(limit/max(m[key],1e-12) if key=='min_upright' else m[key]/limit)
    if 'finite_warning_free' in row['failed'] or row['D']<=0:return None
    return max(values)

def proposal(iteration):
    path=c.DATA/'banks'/f'feasibility_{iteration:02d}.json'
    if path.exists():return c.read(path)
    fresh=c.nominal();tasks=[];logs=[]
    for w in c.WAVES:
        if any(r['wave']==w and r['feasible'] for r in fresh):continue
        if not any(r['wave']==w and r['full'] for r in fresh):continue
        combined={tuple(vector(r)):r for r in c.history()['full_rows'] if r['wave']==w}
        combined.update({tuple(vector(r)):r for r in fresh if r['wave']==w and r['full']})
        rows=[r for r in combined.values() if violation(r) is not None]
        if len(rows)<5:continue
        x=np.array([vector(r)/c.HIGH for r in rows]);y=np.array([violation(r) for r in rows]);centers=[]
        for j in np.argsort(y):
            if all(np.linalg.norm(x[j]-p)>.05 for p in centers):centers.append(x[j])
            if len(centers)==3:break
        seed=918026+iteration*100+int(w*2);rng=np.random.default_rng(seed)
        pool=np.vstack([qmc.Sobol(3,scramble=True,seed=seed).random_base2(11),
            *[np.clip(p+rng.normal(0,sigma,(128,3)),0,1) for p in centers for sigma in [.02,.05,.1]]])
        tested=np.array([vector(r)/c.HIGH for r in fresh if r['wave']==w]);distance=np.min(np.linalg.norm(pool[:,None]-tested[None,:],axis=2),axis=1)
        pool=pool[distance>1e-5]
        reg=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=2,random_state=seed,n_jobs=1).fit(x,np.log(y))
        pred=np.array([tr.predict(pool) for tr in reg.estimators_]);mean=pred.mean(axis=0);std=pred.std(axis=0)
        chosen=[]
        # Half exploit the predicted constraint margin, half explore its uncertainty.
        for score in [mean]*4+[mean-std]*4:
            for j in np.argsort(score):
                if all(np.linalg.norm(pool[j]-pool[k])>.025 for k in chosen):chosen.append(int(j));break
        assert len(chosen)==8
        tasks += [c.task(w,pool[j]*c.HIGH,f'feasibility_{iteration:02d}') for j in chosen]
        file=c.DATA/'proposals'/f'feasibility_w{w:g}_{iteration}.npz'
        file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,pool=pool,mean_log_violation=mean,std=std,training_x=x,training_max_ratio=y,chosen=chosen)
        logs.append(dict(wave=w,minimum_observed_constraint_ratio=float(y.min()),pool=str(file),sha256=c.sha(file),seed=seed))
    c.write(c.DATA/'proposals'/f'feasibility_{iteration:02d}.json',logs)
    return tasks

def main(workers):
    path=c.DATA/'feasibility_amendment.json'
    if not path.exists():c.write(path,dict(time=datetime.datetime.now().astimezone().isoformat(),
        reason='When no candidate passes, a binary feasibility model has no positive examples. Add targeted amplitude proposals that approach the existing constraint boundary, so such waves receive a meaningful attempt to find feasible starts.',
        maximum_rounds=3,proposals_per_unresolved_wave_per_round=8,
        no_policy_or_physics_change=True,final_objective='Actual displacement among fully qualified candidates; proposal constraint ratios never rank a final winner.',
        files=[dict(path=str(c.ROOT/'restore_feasibility.py'),sha256=c.sha(c.ROOT/'restore_feasibility.py'))]))
    for i in range(3):
        tasks=proposal(i)
        if not tasks:break
        c.bank(tasks,f'feasibility_{i:02d}',workers)
    c.write(c.DATA/'feasibility_complete.json',dict(completed=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();main(a.workers)
