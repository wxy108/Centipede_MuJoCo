import argparse,itertools,json
import core as c
np=c.np

def warm():
    tasks=[];selected=c.read(c.ROOT.parent/'mujoco_displacement_first_amplitude_2026-09-03/outputs/data/interim_validated_selection.json')
    for w in c.WAVES:
        rows=[r for r in c.history()['full_rows'] if r['wave']==w];ranked=sorted(rows,key=lambda r:(not r['feasible'],len(r['failed']),-r['D']))
        chosen=[]
        for r in ranked:
            x=np.array([r['amplitudes'][k] for k in c.KEYS])
            if any(np.linalg.norm((x-y)/c.HIGH)<.08 for y in chosen):continue
            chosen.append(x);tasks.append(c.task(w,x,'warm'))
            if len(chosen)==3:break
        old=selected['waves'][f'{w:g}']['selected']
        if old and old['qualified']:
            raw=c.read(old['nominal']['record']);tasks.append(c.task(w,[raw['task']['amplitudes'][k] for k in c.KEYS],'warm'))
        tasks.append(c.task(w,[16.918311805232932,49.37746770784939,22.918311805232932],'warm'))
    return tasks

def freeze():
    p=c.DATA/'manifest.json'
    if p.exists():return
    old=c.read(c.ROOT.parent/'mujoco_displacement_first_amplitude_2026-09-03/outputs/data/amplitude_manifest.json')
    sources=[r['path'] for r in old['frozen_files']]+[c.history()['plant'],c.e.__file__,c.e.fx.pilot.__file__,c.ROOT/'PROTOCOL.md',c.DATA/'history.json']
    c.write(p,dict(plant=c.history()['plant'],plant_sha256=c.history()['plant_sha256'],policy=c.policy.SPEC,
        waves=c.WAVES,search_lower_deg=[0,0,0],search_upper_deg=c.HIGH.tolist(),
        frozen_inputs=[dict(path=str(p),sha256=c.sha(p)) for p in dict.fromkeys(sources)],
        authorization='2026-09-08 user requests improving the previously unsatisfactory per-wave amplitude optimization',
        source_runtime_only_old_results_are_proposals=True,no_contact_loss_selection=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['warm']);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    freeze();c.bank(warm(),'warm',a.workers)
