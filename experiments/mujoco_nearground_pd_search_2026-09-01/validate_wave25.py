"""Long/half-step replay of the top two original-gates-feasible wave2.5 points."""
import concurrent.futures
import json
from run_search import OUT,cfg,run_one

def load(name):return json.loads((OUT/'common'/name).read_text())['config']

if __name__=='__main__':
    candidates=[load('axis_leg_yaw_k_0p25_w2p5_t6_dt0p00025.json'),load('legyaw_boundary_k0p2_w2p5_t6_dt0p00025.json')]
    tasks=[]
    for c in candidates:tasks += [(c,2.5,6.,1.,True,'common'),(c,2.5,12.,1.,True,'common'),(c,2.5,6.,.5,True,'common')]
    tasks += [(cfg('baseline'),2.5,12.,1.,False,'common'),(cfg('baseline'),2.5,6.,.5,False,'common')]
    ids=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(run_one,tasks):
            ids.append(r['id']);print(json.dumps(dict(id=r['id'],D=r['metrics']['displacement_mm'],failed=[k for k,v in r['gates'].items() if not v])),flush=True)
    (OUT/'wave25_validation_ids.json').write_text(json.dumps(ids,indent=2))
