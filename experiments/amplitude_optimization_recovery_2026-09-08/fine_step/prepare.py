"""Freeze the finer experiment, import exact-condition evidence, and warm start."""
import argparse,datetime
from pathlib import Path
import core as c
np=c.np

def coarse_rows():return [c.read(p) for p in (c.ROOT.parent/'outputs/assessments').glob('*.json')]
def vec(r):return np.array([r['amplitudes'][k] for k in c.KEYS])
def diverse(rows,n,distance=.075):
    out=[]
    for r in sorted(rows,key=lambda r:r['D'],reverse=True):
        if all(np.linalg.norm((vec(r)-vec(a))/c.HIGH)>=distance for a in out):out.append(r)
        if len(out)==n:break
    return out

def freeze():
    path=c.DATA/'manifest.json'
    if path.exists():return
    parent=c.read(c.ROOT.parent/'outputs/data/manifest.json')
    files=list(parent['frozen_inputs'])
    files += [dict(path=str(p),sha256=c.sha(p)) for p in [c.ROOT/'PROTOCOL.md',c.ROOT.parent/'outputs/data/resolution_diagnostic.json',c.ROOT.parent/'outputs/data/runtime.json']]
    c.write(path,dict(time=datetime.datetime.now().astimezone().isoformat(),nominal_dt_s=.000125,half_dt_s=.0000625,
        model_sha256=c.history()['plant_sha256'],frozen_inputs=files,coarse_scores_used_only_for_proposals=True))
    imported=[]
    for r in coarse_rows():
        if r['task']['duration']!=6 or r['task']['dt_scale'] not in (.5,.25):continue
        assert c.sha(r['record'])==r['raw_sha256']
        raw=c.read(r['record']);assert raw['model_sha256']==c.history()['plant_sha256']
        c.write(c.OUT/'assessments'/f"{r['id']}.json",r)
        imported.append(dict(id=r['id'],record=r['record'],sha256=r['raw_sha256'],dt_scale=r['task']['dt_scale']))
    c.write(c.DATA/'imported_exact_conditions.json',dict(current_session_only=True,runtime_sha256=c.sha(c.ROOT.parent/'outputs/data/runtime.json'),conditions=imported))
    c.write(c.DATA/'coarse_proposals.json',[r for r in coarse_rows() if r['task']['duration']==6 and r['task']['dt_scale']==1])

def warm():
    path=c.DATA/'banks/warm.json'
    if path.exists():return c.read(path)
    coarse=c.read(c.DATA/'coarse_proposals.json');old=c.history()['full_rows'];tasks=[]
    baseline=c.read(c.EXPERIMENTS/'mujoco_displacement_first_amplitude_2026-09-03/outputs/data/interim_validated_selection.json')
    for w in c.WAVES:
        current=diverse([r for r in coarse if r['wave']==w and r['feasible']],3)
        historical=diverse([r for r in old if r['wave']==w and r['feasible']],2)
        tasks += [c.task(w,vec(r),'warm') for r in [*current,*historical]]
        sel=baseline['waves'][f'{w:g}']['selected']
        if sel and sel['qualified']:
            raw=c.read(sel['nominal']['record']);tasks.append(c.task(w,[raw['task']['amplitudes'][k] for k in c.KEYS],'warm'))
        tasks.append(c.task(w,[16.918311805232932,49.37746770784939,22.918311805232932],'warm'))
    return list({c.identity(t):t for t in tasks}.values())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);a=ap.parse_args();freeze();c.bank(warm(),'warm',a.workers)
