"""Bounded, registered multistart amplitude refinements. Never changes PD."""
import argparse
import itertools
import json
import search as s
e,np=s.e,s.np


def collect(stages):
    rows=[];seen=set()
    for stage in stages:
        path=e.DATA/f'{stage}_summary.json'
        clarified=e.DATA/f'{stage}_objective_summary.json'
        if clarified.exists():path=clarified
        if not path.exists():continue
        for row in json.loads(path.read_text())['assessments']:
            if row['id'] not in seen:rows.append(row);seen.add(row['id'])
    return rows


def freeze(stage,parents,scale,seeds):
    path=e.DATA/f'{stage}_manifest.json'
    if path.exists():return json.loads(path.read_text())
    rows=collect(parents);tasks=[];origins={};seen={row['id'] for row in rows}
    for wave in e.WAVES:
        full=[r for r in rows if r['wave']==wave and r['full']]
        # Gate counts prioritize diagnostic follow-up only. Never a reward or objective.
        ranked=sorted(full,key=lambda r:(not r['feasible'],len(r['failed']),-r['D']))
        starts=[]
        for row in ranked:
            x=np.array([row['amplitudes'][k] for k in s.KEYS])
            if any(np.linalg.norm((x-y)/s.HIGH)<.08 for y,_ in starts):continue
            starts.append((x,row))
            if len(starts)>=seeds:break
        # Preserve a high-displacement starting point even if low-motion cases
        # have fewer failed diagnostic gates. Do not let gate counts erase it.
        baseline_task=next(t for t in json.loads(s.MANIFEST.read_text())['references'] if t['wave']==wave)
        baseline_id=e.task_id(baseline_task)
        baseline=next((r for r in full if r['id']==baseline_id),None)
        if baseline is not None and all(r['id']!=baseline_id for _,r in starts):
            starts.append((np.array([baseline['amplitudes'][k] for k in s.KEYS]),baseline))
        origins[f'{wave:g}']=[r['id'] for _,r in starts]
        for x,row in starts:
            raw=json.loads(open(row['record'],encoding='utf-8').read());base=raw['task']
            directions=[np.array(v,dtype=float) for v in itertools.product((-1,1),repeat=3)]
            directions += [sign*np.eye(3)[dim] for dim in range(3) for sign in (-1,1)]
            for delta in directions:
                value=np.clip(x+scale*s.HIGH*delta,s.LOW,s.HIGH)
                task=dict(base,stage=stage,trace=False,amplitudes=dict(zip(s.KEYS,map(float,value))),origin=row['id'])
                tid=e.task_id(task)
                if tid in seen:continue
                seen.add(tid);tasks.append(task)
    payload=dict(stage=stage,parent_stages=parents,scale_fraction=scale,seeds_per_wave=seeds,
        selection_rule='all-gates feasible first, then least failed diagnostic gates, then displacement; normalized seed separation 0.08; always retain current fixed-amplitude reference as an additional seed if missing',
        seed_ids=origins,tasks=tasks,count=len(tasks),global_optimum_proved=False)
    path.write_text(json.dumps(payload,indent=2));return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('refine','polish'))
    p.add_argument('--workers',type=int,default=20);args=p.parse_args()
    if args.stage == 'polish':
        proof=json.loads((e.DATA/'chassis_contact_pruning_verification.json').read_text())
        assert len(proof)==3 and all(row['consistent'] for row in proof)
        exercised=json.loads((e.DATA/'chassis_contact_pruning_branch_exercised.json').read_text())
        assert exercised['consistent'] and 'chassis_contact_irreversible' in exercised['proof']['gates']
    parents=['broad'] if args.stage=='refine' else ['broad','refine']
    manifest=freeze(args.stage,parents,.08 if args.stage=='refine' else .02,2)
    print(json.dumps({k:v for k,v in manifest.items() if k!='tasks'}),flush=True)
    s.run_bank(manifest['tasks'],args.workers,args.stage)
