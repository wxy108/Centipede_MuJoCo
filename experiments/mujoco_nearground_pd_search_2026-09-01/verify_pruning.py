"""Full replay of four rejected trials, plus original-start regression check."""
import concurrent.futures
import json
from run_search import OUT,OLD,pilot,cfg,near_ground,amplitude_case

def worker(args):
    kind,name,wave=args
    if kind=='original':
        r=pilot.run_task((cfg('baseline'),2.5,6.,1.,False),output_dir=OUT/'verification_original')
        old=json.loads((OLD/'outputs/baseline_w2p5_t6_dt0p00025.json').read_text())
        assert r['metrics']==old['metrics'],'original reset regression'
        return dict(kind=kind,all_original_metrics_identical=True)
    proof=json.loads((OUT/'common'/f'{name}_w{wave:g}_t6_dt0.00025'.replace('.','p')).with_suffix('.json').read_text())
    r=pilot.run_task((proof['config'],wave,6.,1.,False),initialize=near_ground,
        case_override=amplitude_case(wave,'common'),output_dir=OUT/'verification_pruning')
    for failure in proof['failure_evidence']:
        assert not r['gates'][failure['gate']],failure
    return dict(id=proof['id'],full_replay_confirms_all_pruning_failures=True)

if __name__=='__main__':
    tasks=[('prune','sobol_s1901_0000',2.),('prune','sobol_s1901_0000',2.5),
           ('prune','sobol_s1901_0000',1.5),('prune','sobol_s1901_0002',.5),('original','baseline',2.5)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as p:results=list(p.map(worker,tasks))
    (OUT/'pruning_regression_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
