"""Retain old floor-based output; apply clarified displacement-objective policy."""
import argparse
import json
from pathlib import Path
import search as s
e=s.e
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');a=p.parse_args()
    source=e.DATA/f'{a.stage}_summary.json';previous=json.loads(source.read_text())
    records=[json.loads(Path(r['record']).read_text()) for r in previous['assessments']]
    result=s.summarize(records,a.stage+'_objective')
    print(json.dumps({w:dict(count=r['count'],feasible=r['feasible'],D=r['best']['D'] if r['best'] else None) for w,r in result['waves'].items()},indent=2))
