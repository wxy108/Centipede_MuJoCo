"""A separate no-restoration diagnostic; never pooled with burst-loss curves."""
import hashlib
import study as s

s.TRIALS=s.OUT/'persistent_loss_diagnostic'

def persistent_schedule(nu,seed,duration=6.):
    uniforms=s.np.random.default_rng(9082026+int(seed)).random(19)
    events=[dict(pair=i,window=0,start=1.5,end=duration,u=float(u)) for i,u in enumerate(uniforms) if u<nu]
    return dict(nu=nu,q=nu,seed=seed,duration=duration,expected_erased_load=nu,
        channel_kind='persistent_bilateral_pair_loss_no_restoration',opportunities=19,
        uniform_sha256=hashlib.sha256(uniforms.tobytes()).hexdigest(),events=events)

s.schedule=persistent_schedule

if __name__=='__main__':
    plan=dict(created=s.now(),waves=[2,3,7],pair_probabilities=[.25,.5,.75],seeds=[0,1,2],
        interpretation='Small sensitivity diagnostic, not independent crossover confirmation. Each pair is drawn once at t=1.5 s and remains collision-disabled to the horizon. No contact restoration occurs. Same frozen plant and per-wave controller. Different channel from burst loss, so no pooled curve.',
        source_sha256=s.sha(__file__),study_sha256=s.sha(s.ROOT/'study.py'))
    path=s.DATA/'persistent_diagnostic_plan.json'
    if not path.exists():s.write(path,plan)
    tasks=[s.task(w,p,se,stage='persistent_diagnostic') for p in plan['pair_probabilities'] for se in plan['seeds'] for w in plan['waves']]
    s.bank(tasks,'persistent_diagnostic',6)
