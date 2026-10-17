"""Resumable coordinator; each stage is a separate process and evidence file."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',default='20')
    p.add_argument('--resume-from',choices=('audit','verify','transfer','interim_validation','interim_delivery','global','profile','local','validation','delivery'))
    args=p.parse_args()
    stages=[('audit',['study.py','audit']),('verify',['study.py','verify']),
        ('transfer',['study.py','transfer']),
        ('interim_validation',['validate.py','--label','interim']),
        ('interim_delivery',['report.py','--label','interim']),
        ('global',['study.py','global']),('profile',['study.py','profile']),
        ('local',['study.py','local']),('validation',['validate.py','--label','final']),
        ('delivery',['report.py','--label','final'])]
    if args.resume_from:stages=stages[[k for k,v in stages].index(args.resume_from):]
    path=ROOT/'outputs/data/pipeline_progress.json'
    history=json.loads(path.read_text()) if path.exists() else []
    def event(stage,status):
        entry=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),stage=stage,status=status)
        history.append(entry);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(history,indent=2));temporary.replace(path)
        print(json.dumps(entry),flush=True)
    for stage,cmd in stages:
        event(stage,'running')
        extra=[] if cmd[0]=='report.py' else ['--workers',args.workers]
        run=subprocess.run([sys.executable,str(ROOT/cmd[0]),*cmd[1:],*extra],cwd=ROOT)
        if run.returncode:
            event(stage,f'failed:{run.returncode}; previous evidence retained');sys.exit(run.returncode)
        event(stage,'complete')
    event('experiment','complete; bounded search only; no canonical or Clean promotion')
