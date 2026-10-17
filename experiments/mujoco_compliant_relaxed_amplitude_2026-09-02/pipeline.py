"""One-shot resumable coordinator for the user-requested relaxed study."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LOG = ROOT / 'outputs/data/pipeline_progress.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', choices=('verification', 'initial', 'coarse', 'polish', 'validation', 'delivery'))
    args = parser.parse_args()
    history = json.loads(LOG.read_text()) if LOG.exists() else []
    stages = [('verification', ['verify.py']),
              ('initial', ['study.py', 'initial', '--workers', '20']),
              ('coarse', ['study.py', 'coarse', '--workers', '20']),
              ('polish', ['study.py', 'polish', '--workers', '20']),
              ('validation', ['validate.py', '--workers', '20']),
              ('delivery', ['delivery.py'])]
    if args.resume_from:
        stages = stages[[name for name, _ in stages].index(args.resume_from):]

    def record(stage, status):
        event = dict(stage=stage, status=status, time=datetime.datetime.now().astimezone().isoformat())
        history.append(event)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(history, indent=2))
        print(json.dumps(event), flush=True)

    for stage, commands in stages:
        record(stage, 'running')
        result = subprocess.run([sys.executable, str(ROOT / commands[0]), *commands[1:]], cwd=ROOT)
        if result.returncode:
            record(stage, f'failed: {result.returncode}; completed evidence retained')
            raise SystemExit(result.returncode)
        record(stage, 'complete')
    record('experiment', 'complete; previous strict study and canonical model unchanged')
