"""Opt-in physical-geometry policy; leaves the legacy branch untouched."""
import argparse,copy,inspect,json,textwrap
import search as s
import core as c
import screening
from numba import njit
np=c.np
SOURCE_DATA=c.DATA;SOURCE_OUT=c.OUT
POLICY='physical_geometry_v2'
ORIGINAL_TASK=c.task
ORIGINAL_ASSESS=c.policy.assess


@njit(cache=False)
def self_contacts(g1,g2,dist,categories):
    count=0;penetration=0.
    for j in range(len(g1)):
        a=categories[g1[j]];b=categories[g2[j]]
        if 38<=a<59 and 38<=b<59 and abs(a-b)>1 and dist[j]<0:
            count+=1
            if -dist[j]>penetration:penetration=-dist[j]
    return count,penetration


def assess(raw):
    a=ORIGINAL_ASSESS(raw)
    if not a['full']:return a
    a['gates'].pop('spine')
    a['gates']['nonadjacent_chassis_contact']=raw['metrics']['nonadjacent_chassis_contact_steps']==0
    a['failed']=[k for k,v in a['gates'].items() if not v]
    a['feasible']=not a['failed'];return a


def task(*args,**kwargs):
    t=ORIGINAL_TASK(*args,**kwargs);t['viability_policy']=POLICY;return t


def install():
    source=(c.ROOT/'fast_run_task.py').read_text()
    source=source.replace('    finite=True\n','    self_contact_steps=0;self_contact_points=0;self_penetration_peak=0.\n    finite=True\n')
    marker='            int(m.opt.cone)==0,force[k],foot_mask[k])\n'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'''        if k*dt>=1.5:
            count,depth=OBSERVE_SELF_CONTACT(d.contact.geom1,d.contact.geom2,d.contact.dist,ground_categories)
            self_contact_steps+=int(count>0);self_contact_points+=count
            self_penetration_peak=max(self_penetration_peak,depth)
''')
    marker='    warnings=opt.tip.stage1.warning_counts(d)\n'
    assert source.count(marker)==1
    source=source.replace(marker,'''    mtr.update(nonadjacent_chassis_contact_steps=int(self_contact_steps),
        nonadjacent_chassis_contact_points=int(self_contact_points),
        nonadjacent_chassis_penetration_peak_mm=float(self_penetration_peak*1000))
'''+marker)
    path=c.ROOT/'physical_run_task.py'
    if path.exists():assert path.read_text()==source
    else:path.write_text(source,encoding='utf-8')
    pilot=c.e.fx.pilot;ns=dict(vars(pilot),OBSERVE_SELF_CONTACT=self_contacts)
    exec(compile(source,str(path),'exec'),ns)
    pilot.run_task=ns['run_task'];pilot.OBSERVE_SELF_CONTACT=self_contacts
    guard=textwrap.dedent(inspect.getsource(screening.Guard.after))
    marker=",('spine',spine,1.2)";assert guard.count(marker)==1
    guard=guard.replace(marker,'')
    ns=dict(vars(screening));exec(compile(guard,str(c.ROOT/'physical_guard_after'),'exec'),ns)
    screening.Guard.after=ns['after']
    c.task=task;c.policy.assess=assess
    c.OUT=SOURCE_OUT/POLICY;c.DATA=c.OUT/'data';c.e.TRIALS=c.OUT/'trials'


install()


def prepare():
    m=c.read(SOURCE_DATA/'manifest.json')
    m.update(viability_policy=POLICY,physical_policy_note=str(c.ROOT/'PHYSICAL_POLICY_AMENDMENT.md'))
    for p in (c.ROOT/'PHYSICAL_POLICY_AMENDMENT.md',c.ROOT/'physical_search.py',c.ROOT/'physical_run_task.py'):
        m['frozen_inputs'][str(p)]=c.sha(p)
    path=c.DATA/'manifest.json'
    if path.exists():assert c.read(path)==m
    else:c.write(path,m)


def verify(workers):
    bridge=c.read(SOURCE_DATA/'bridge_analysis.json')['rows']
    old=[r for r in bridge if r['plant']=='current' and r['phase']=='midpoint' and
         ((r['family']=='rl_v4' and r['wave'] in (1.5,2,3)) or (r['family']=='recovery' and r['wave'] in (1.5,7)))]
    ts=[c.task(r['wave'],list(r['amplitude'].values()),'physical_parity',phase='midpoint',trace=True,prefix=False) for r in old]
    rows=c.bank(ts,'physical_parity',workers);checks=[]
    for r,t,reference in zip(sorted(rows,key=lambda x:x['id']),sorted(ts,key=c.identity),sorted(old,key=lambda x:c.identity(c.task(x['wave'],list(x['amplitude'].values()),'physical_parity',phase='midpoint',trace=True,prefix=False)))):
        assert r['id']==c.identity(t)
        errors={k:float(np.max(np.abs(np.asarray(v)-np.asarray(r['metrics'][k])))) for k,v in reference['metrics'].items()}
        z=np.load(r['motion_path']);before=np.load(reference['motion'])
        trajectory_exact=all(np.array_equal(z[k],before[k]) for k in z.files)
        passed=trajectory_exact and max(errors.values())<1e-12
        checks.append(dict(id=r['id'],reference=reference['id'],D_difference=r['D']-reference['D'],
            max_metric_difference=max(errors.values()),trajectory_exact=trajectory_exact,passed=passed,
            self_contact_steps=r['metrics']['nonadjacent_chassis_contact_steps'],failed=r['failed']))
    result=dict(passed=all(x['passed'] for x in checks),checks=checks)
    c.write(c.DATA/'physical_parity_check.json',result);print(json.dumps(result,indent=2));assert result['passed']


def import_completed():
    imported=[]
    for path in sorted((SOURCE_OUT/'assessments').glob('*.json')):
        r=c.read(path)
        if r['task']['phase']!='midpoint' or r['task']['plant']!='current' or 'error' in r:continue
        t=dict(r['task'],viability_policy=POLICY);key=c.identity(t);target=c.OUT/'assessments'/f'{key}.json'
        if target.exists():continue
        remaining=[f for f in r['failed'] if f!='spine']
        if not r['full'] and not remaining:continue
        row=copy.deepcopy(r);row.update(id=key,task=t,failed=remaining,feasible=bool(r['full'] and not remaining),
            motion_path=None,imported_from=r['id'],self_contact_verified=False,
            eligibility_note='PROVISIONAL: same physical trajectory, continuous self-contact must be verified before final selection')
        if not r['full']:
            row['proof']['violations']=[v for v in row['proof']['violations'] if v['gate']!='spine']
        c.write(target,row);imported.append(key)
    return imported


def rescue(workers,batch):
    import_completed();tasks=[]
    for path in (SOURCE_OUT/'assessments').glob('*.json'):
        r=c.read(path)
        if r['task']['phase']=='midpoint' and r['task']['plant']=='current' and not r['full'] and r['failed']==['spine']:
            t=dict(r['task'],viability_policy=POLICY,stage=f'spine_rescue_{batch:02d}')
            if not (c.OUT/'assessments'/f'{c.identity(t)}.json').exists():tasks.append(t)
    c.bank(tasks,f'spine_rescue_{batch:02d}',workers);s.snapshot('midpoint')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify','import','rescue','adaptive','local','snapshot'])
    p.add_argument('--workers',type=int,default=12);p.add_argument('--batch',type=int,default=0)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--polls',type=int,default=8);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='verify':verify(a.workers)
    else:
        assert c.read(c.DATA/'physical_parity_check.json')['passed']
        if a.action=='import':print(json.dumps(dict(imported=len(import_completed()))))
        elif a.action=='rescue':rescue(a.workers,a.batch)
        elif a.action=='adaptive':
            import_completed()
            for i in range(a.rounds):
                path=c.DATA/'banks'/f'adaptive_midpoint_{i:02d}.json'
                tasks=c.read(path) if path.exists() else s.propose('midpoint',i)
                c.bank(tasks,f'adaptive_midpoint_{i:02d}',a.workers);s.snapshot('midpoint')
        elif a.action=='local':s.local('midpoint',a.workers,a.polls);s.snapshot('midpoint')
        elif a.action=='snapshot':s.snapshot('midpoint')
