"""Extra finer-step proposals for waves with no nominal feasible gait."""
import datetime
import core as c
from prepare import vec
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor
np=c.np

def violation(r):
    m=r['metrics'];values=[1-r['D']/10.]
    for k,limit in c.policy.SPEC.items():
        if k not in m or isinstance(limit,bool) or not isinstance(limit,(int,float)):continue
        value=(1-m[k])/(1-limit) if k=='min_upright' else m[k]/limit
        if not np.isfinite(value):return None
        values.append(value)
    if 'finite_warning_free' in r['failed']:values.append(5.)
    return max(1e-9,max(values))

def proposal(iteration):
    path=c.DATA/'banks'/f'feasibility_{iteration:02d}.json'
    if path.exists():return c.read(path)
    fresh=c.nominal();tasks=[];logs=[]
    for w in c.WAVES:
        observed=[r for r in fresh if r['wave']==w]
        if any(r['feasible'] for r in observed):continue
        rows=[r for r in observed if r['full'] and violation(r) is not None]
        if not rows:continue
        x=np.array([vec(r)/c.HIGH for r in rows]);y=np.array([violation(r) for r in rows]);centers=[]
        if len(rows)<5:
            center=x[np.argmin(y)];seed=91820261+iteration*100+int(w*2)
            pool=np.clip(center+.08*(2*qmc.Sobol(3,scramble=True,seed=seed).random_base2(5)-1),0,1)
            tested=np.array([vec(r)/c.HIGH for r in observed]);chosen=[]
            for point in pool:
                if np.min(np.linalg.norm(tested-point,axis=1))<1e-5 or any(np.linalg.norm(point-a)<.02 for a in chosen):continue
                chosen.append(point);tasks.append(c.task(w,point*c.HIGH,f'feasibility_{iteration:02d}'))
                if len(chosen)==8:break
            assert len(chosen)==8
            logs.append(dict(wave=w,seed=seed,training=len(rows),rule='Sparse full evidence: eight diverse joint-amplitude perturbations around the least-violating complete condition; no surrogate confidence claimed.',amplitudes=(np.array(chosen)*c.HIGH).tolist()))
            continue
        for j in np.argsort(y):
            if all(np.linalg.norm(x[j]-p)>.05 for p in centers):centers.append(x[j])
            if len(centers)==3:break
        seed=91820261+iteration*100+int(w*2);rng=np.random.default_rng(seed)
        pool=np.vstack([qmc.Sobol(3,scramble=True,seed=seed).random_base2(11),
            *[np.clip(p+rng.normal(0,sigma,(128,3)),0,1) for p in centers for sigma in [.02,.05,.1]]])
        tested=np.array([vec(r)/c.HIGH for r in observed]);distance=np.min(np.linalg.norm(pool[:,None]-tested[None,:],axis=2),axis=1)
        keep=distance>1e-5;pool=pool[keep];distance=distance[keep]
        model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=1).fit(x,np.log(y))
        pred=np.array([tree.predict(pool) for tree in model.estimators_]);mean=pred.mean(axis=0);std=pred.std(axis=0)
        categories=[('predicted_constraint_margin',-mean)]*3+[('optimistic_constraint_margin',-mean+.75*std)]*3+[
            ('constraint_uncertainty',std),('coverage',distance)]
        chosen=[];log=[]
        for kind,score in categories:
            for j in np.argsort(score)[::-1]:
                if any(np.linalg.norm(pool[j]-pool[k])<.025 for k in chosen):continue
                chosen.append(int(j));tasks.append(c.task(w,pool[j]*c.HIGH,f'feasibility_{iteration:02d}'))
                log.append(dict(kind=kind,pool_index=int(j),predicted_log_ratio=float(mean[j]),ensemble_spread=float(std[j]),amplitudes=(pool[j]*c.HIGH).tolist()));break
        assert len(chosen)==8
        file=c.DATA/'proposals'/f'feasibility_w{w:g}_{iteration}.npz';file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,pool=pool,mean=mean,std=std,training_x=x,training_max_ratio=y)
        logs.append(dict(wave=w,seed=seed,training=len(rows),minimum_observed_ratio=float(y.min()),chosen=log,pool=str(file),sha256=c.sha(file)))
    c.write(c.DATA/'proposals'/f'feasibility_{iteration:02d}.json',logs)
    return tasks

def main():
    path=c.DATA/'feasibility_amendment.json'
    if not path.exists():c.write(path,dict(time=datetime.datetime.now().astimezone().isoformat(),maximum_rounds=2,proposals_per_unresolved_wave=8,
        reason='Binary feasibility has no positive example for waves 8 and 9. Target existing constraint margins using only exact finer full evaluations. Include negative displacement as a violation rather than dropping those examples.',
        positive_progress_proposal_ratio='1 - D/10 mm; this sets proposal units only, not a new minimum-speed gate.',
        physical_policy_unchanged=True,final_ranking='Actual displacement only among fully qualified gaits',source_sha256=c.sha(c.ROOT/'feasibility.py')))
    for i in range(2):
        tasks=proposal(i)
        if not tasks:break
        c.bank(tasks,f'feasibility_{i:02d}',2)
    c.write(c.DATA/'feasibility_complete.json',dict(completed=True))

if __name__=='__main__':main()
