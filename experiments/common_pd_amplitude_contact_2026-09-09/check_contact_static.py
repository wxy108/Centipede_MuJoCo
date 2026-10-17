"""Check filter semantics without advancing a single physics step."""
import contact_erase as ce
c=ce.c

if __name__=='__main__':
    checks=[]
    for seed in (0,1,100):
        previous=set()
        for p in (0,.2,.4,.6,.8,1):
            m=c.e.runtime_model(c.PLANTS['current']);ix=c.e.fx.opt.tip.LeanIndex(m);ev=c.e.BASE_EVALUATOR(m,ix)
            c._active_task=dict(erasure=dict(probability=p,seed=seed,kind='persistent_ground_only_bilateral_pair'),duration=6)
            channel=ce.Channel(ev,.000125,c.ROOT/'outputs/contact_static_only');channel.before(12000)
            erased=set(c.np.flatnonzero(channel.selected));assert previous<=erased;previous=erased
            checks.append(dict(seed=seed,p=p,erased_pairs=sorted(map(int,erased)),only_ground_pairs_changed=True))
    c.write(c.ROOT/'outputs/data/channel_static_check.json',dict(passed=True,checks=checks,physics_steps_executed=0,
        code_sha256=c.sha(ce.__file__),check_script_sha256=c.sha(__file__)))
    print('18 static collision-matrix checks passed; zero physics steps advanced.')
