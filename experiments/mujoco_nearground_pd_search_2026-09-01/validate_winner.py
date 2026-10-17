"""Exact validation of the best tested nominal wave2.5 feasible point."""
import concurrent.futures,json
from run_search import OUT,run_one
candidate=json.loads((OUT/'common/legyaw_refine_k0p29_w2p5_t6_dt0p00025.json').read_text())['config']
if __name__=='__main__':
    tasks=[(candidate,2.5,12.,1.,True,'common'),(candidate,2.5,6.,.5,True,'common')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as p:
        rows=list(p.map(run_one,tasks))
    (OUT/'winner_validation_ids.json').write_text(json.dumps([r['id'] for r in rows],indent=2))
    print(json.dumps([dict(id=r['id'],D=r['metrics']['displacement_mm'],failed=[k for k,v in r['gates'].items() if not v]) for r in rows]))
