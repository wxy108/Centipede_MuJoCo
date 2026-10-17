"""Resumable displacement-first pipeline."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',default='20');p.add_argument('--resume-from',choices=(
      'audit','verify','pilot','pilot_validation','pilot_delivery','global1','global2','transfer','local','validation','delivery'))
    a=p.parse_args();stages=[('audit',['study.py','audit']),('verify',['study.py','verify']),
      ('pilot',['study.py','pilot']),('pilot_validation',['validate.py','--label','interim']),
      ('pilot_delivery',['report.py','--label','interim']),('global1',['study.py','global1']),
      ('global2',['study.py','global2']),('transfer',['study.py','transfer']),
      ('local',['study.py','local']),('validation',['validate.py','--label','final']),
      ('delivery',['report.py','--label','final'])]
    if a.resume_from:stages=stages[[x for x,y in stages].index(a.resume_from):]
    path=ROOT/'outputs/data/pipeline_progress.json';history=json.loads(path.read_text()) if path.exists() else []
    def log(stage,status):
        row=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),stage=stage,status=status)
        history.append(row);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(history,indent=2));tmp.replace(path);print(json.dumps(row),flush=True)
    for stage,command in stages:
        log(stage,'running');extra=[] if command[0]=='report.py' else ['--workers',a.workers]
        run=subprocess.run([sys.executable,str(ROOT/command[0]),*command[1:],*extra],cwd=ROOT)
        if run.returncode:log(stage,f'failed:{run.returncode}; completed evidence retained');raise SystemExit(run.returncode)
        log(stage,'complete')
    log('experiment','complete; finite-tested-set only; no canonical/Clean promotion')
