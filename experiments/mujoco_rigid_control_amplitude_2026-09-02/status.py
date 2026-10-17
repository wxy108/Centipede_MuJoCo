"""Read-only compact progress snapshot for the registered amplitude bank."""
import argparse
import json
import search as s
e=s.e
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('broad','refine','polish','boundary'),default='broad')
    args=parser.parse_args()
    path=s.MANIFEST if args.stage=='broad' else e.DATA/f'{args.stage}_manifest.json'
    manifest=json.loads(path.read_text());out={}
    for w in e.WAVES:
        tasks=[t for t in manifest['tasks'] if t['wave']==w];full=pruned=passed=0;best=None
        if not tasks:
            out[f'{w:g}']=dict(completed=0,total=0,full=0,pruned=0,feasible=0,best=None)
            continue
        base,_=s.load_reference(tasks[0])
        for t in tasks:
            tid=e.task_id(t);path=e.TRIALS/args.stage/tid/f'{tid}.json'
            if not path.exists():
                if base['id']==tid:r=base
                else:continue
            else:
                try:r=json.loads(path.read_text())
                except json.JSONDecodeError:continue
            if r.get('metrics'):
                full+=1;a=s.assess(r)
                if a['feasible']:
                    passed+=1
                    if best is None or a['D']>best['D']:best=dict(D=a['D'],amplitudes=a['amplitudes'])
            else:pruned+=1
        out[f'{w:g}']=dict(completed=full+pruned,total=len(tasks),full=full,pruned=pruned,feasible=passed,best=best)
    print(json.dumps(dict(completed=sum(x['completed'] for x in out.values()),total=len(manifest['tasks']),waves=out),indent=2))
