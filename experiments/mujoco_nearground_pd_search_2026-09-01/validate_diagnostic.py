"""Longer/half-step diagnostic replay. Height-gate failure is not waived."""
import concurrent.futures
import json
from run_search import ROOT,OUT,cfg,run_one

if __name__=='__main__':
    source=OUT/'common/basin18_lpk2_lpd0p25_rk14p082_w2_t6_dt0p00025.json'
    candidate=json.loads(source.read_text())['config']
    baseline=cfg('baseline')
    tasks=[(candidate,2.,6.,1.,True,'common'),
        (candidate,2.,12.,1.,True,'common'),(baseline,2.,12.,1.,False,'common'),
        (candidate,2.,6.,.5,True,'common'),(baseline,2.,6.,.5,False,'common')]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(run_one,tasks):
            results.append(r['id'])
            print(json.dumps(dict(id=r['id'],D=r['metrics']['displacement_mm'],failed=[k for k,v in r['gates'].items() if not v])),flush=True)
    (OUT/'diagnostic_validation_ids.json').write_text(json.dumps(results,indent=2))
