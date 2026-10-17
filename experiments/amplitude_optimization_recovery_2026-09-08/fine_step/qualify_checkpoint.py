"""Qualification feedback on actual contenders while the nominal search proceeds."""
import argparse
import core as c
import validate as v

def main(name,waves,workers,diverse_unqualified=True):
    path=c.DATA/f'{name}_plan.json'
    if path.exists():plan=c.read(path)
    else:
        checked=v.update_validation_index();plan=[]
        for w in waves:
            rr=sorted([r for r in c.nominal() if r['wave']==w and r['feasible']],key=lambda r:r['D'],reverse=True)
            best=max([q['nominal']['D'] for q in checked.values() if q['qualified'] and q['nominal']['wave']==w],default=0)
            candidates=[r for r in rr if r['id'] not in checked and r['D']>best+.2]
            if best==0 and diverse_unqualified:
                failed=[q['nominal'] for q in checked.values() if not q['qualified'] and q['nominal']['wave']==w]
                diverse=[r for r in candidates if all(c.np.linalg.norm((c.np.array([r['amplitudes'][k] for k in c.KEYS])-c.np.array([a['amplitudes'][k] for k in c.KEYS]))/c.HIGH)>=.075 for a in failed)]
                if diverse:candidates=diverse
            candidate=next(iter(candidates),None)
            if candidate:plan.append(candidate)
        c.write(path,plan)
    c.bank([t for r in plan for t in v.pair_tasks(r,name)],name,workers)
    checked=v.update_validation_index()
    print({r['wave']:dict(D=r['D'],qualified=checked[r['id']]['qualified'],reasons=checked[r['id']]['reasons']) for r in plan},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--waves',type=float,nargs='+',default=c.WAVES);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();main(a.name,a.waves,a.workers)
