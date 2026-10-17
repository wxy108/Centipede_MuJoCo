"""Re-record selected trajectories and verify them against their scored runs."""
import argparse
import core as c
from report import baseline_rows,vector


def trace_rows(rows,name,workers):
    tasks=[c.task(r['wave'],vector(r),name,trace=True) for r in rows]
    got=c.bank(tasks,name,workers)
    before={r['id']:r for r in rows};checks=[]
    for r in got:
        old=before[r['id']]
        check=dict(id=r['id'],D_exact=r['D']==old['D'],metrics_exact=r.get('metrics')==old.get('metrics'),
            feasibility_exact=r['feasible']==old['feasible'],record=r.get('record'))
        checks.append(check)
        assert all(check[k] for k in ['D_exact','metrics_exact','feasibility_exact']),check
        assert (c.Path(r['record']).parent/'motion.npz').exists()
    c.write(c.DATA/f'{name}_verification.json',dict(passed=True,checks=checks))
    return {r['id']:r for r in got}


def baseline(workers):
    trace_rows(list(baseline_rows().values()),'baseline_motion',workers)


def selected(workers):
    path=c.DATA/'validated_selection.json';v=c.read(path)
    rows=[r['selected']['nominal'] for r in v['waves'].values() if r['qualified']]
    got=trace_rows(rows,'selected_motion',workers)
    for cell in v['waves'].values():
        if cell['qualified']:cell['selected']['nominal']=got[cell['selected']['nominal']['id']]
    c.write(path,v)
    # A complete nominal failure is a diagnostic replay, never a selected gait.
    diagnostic=[]
    for w,cell in v['waves'].items():
        if cell['qualified']:continue
        full=[r for r in c.nominal() if r['wave']==float(w) and r['full']]
        if full:diagnostic.append(max(full,key=lambda r:r['D']))
    # Some failed full runs would be prefix-rejected on a fresh guarded replay.
    # Reconstruct their original full qpos using a separately verified trace pass.
    c.write(c.DATA/'unresolved_diagnostic_plan.json',diagnostic)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baseline','selected']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    (baseline if a.stage=='baseline' else selected)(a.workers)
