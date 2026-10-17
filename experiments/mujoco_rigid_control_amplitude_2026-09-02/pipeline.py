"""Finish the current one-shot experiment, after its in-flight broad bank."""
from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'outputs'/'data'
log=[]

def record(stage,status):
    item=dict(stage=stage,status=status,time=datetime.datetime.now().astimezone().isoformat())
    log.append(item);(DATA/'pipeline_progress.json').write_text(json.dumps(log,indent=2))
    print(json.dumps(item),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-from',choices=('refine','polish','validation','delivery'))
    args=parser.parse_args()
    if args.resume_from and (DATA/'pipeline_progress.json').exists():
        log=json.loads((DATA/'pipeline_progress.json').read_text())
        record(args.resume_from,'resuming interrupted coordinator; frozen manifests and completed trials retained')
    else:
        record('broad','waiting for the already running complete registered bank')
    expected=json.loads((DATA/'amplitude_manifest.json').read_text())['candidate_count']
    while True:
        path=DATA/'broad_summary.json'
        if path.exists():
            try:
                data=json.loads(path.read_text())
                if len(data['assessments'])==expected:break
                raise RuntimeError('Broad bank incomplete; resource errors must be resolved before refinement')
            except json.JSONDecodeError:pass
        time.sleep(3)
    if not args.resume_from:record('broad','complete')
    stages=[('clarified objective',['reassess.py','broad']),('refine',['refine.py','refine','--workers','20']),
            ('polish',['refine.py','polish','--workers','20']),('validation',['validate.py','--workers','20']),('delivery',['delivery.py'])]
    if args.resume_from:stages=stages[[stage for stage,_ in stages].index(args.resume_from):]
    for stage,stage_args in stages:
        record(stage,'running')
        result=subprocess.run([sys.executable,str(ROOT/stage_args[0]),*stage_args[1:]],cwd=ROOT)
        if result.returncode:
            record(stage,f'failed with exit {result.returncode}')
            raise SystemExit(result.returncode)
        record(stage,'complete')
    record('experiment','complete; no canonical baseline overwritten')
