"""Refresh presentation labels using existing recorded poses only."""
import contextlib,datetime,time
import core as c
from render import make
from report import baseline_rows
from finish_delivery import WAVES

def main():
    start=time.monotonic();selection=c.read(c.DATA/'validated_selection.json');baseline=baseline_rows()
    validation=c.read(c.DATA/'validation_index.json');index={r['id']:r for r in c.records()};jobs=[]
    for wave in WAVES:
        old=baseline.get(wave);rows=[]
        if old:rows.append((old,'PREVIOUS AMPLITUDES / CURRENT RUNTIME',validation.get(old['id'],{}).get('qualified',False)))
        rows.append((selection['waves'][f'{wave:g}']['selected']['nominal'],'NEW AMPLITUDE SELECTION',True))
        jobs.append((f'wave{wave:g}_before_after',rows))
    groups={}
    for old in c.read(c.DATA/'unresolved_diagnostic_plan.json'):
        groups.setdefault(old['wave'],[]).append((index[old['id']],old['diagnostic_label'],False))
    jobs += [(f'unresolved_wave{wave:g}',rows) for wave,rows in groups.items()]
    with (c.OUT/'logs/presentation_refresh.log').open('w',encoding='utf-8') as log:
        for i,(name,rows) in enumerate(jobs):
            with contextlib.redirect_stdout(log):make(rows,name)
            print(f'{i+1}/{len(jobs)}: {name}',flush=True)
    c.write(c.DATA/'presentation_refresh.json',dict(time=datetime.datetime.now().astimezone().isoformat(),completed=True,
        videos=[name for name,_ in jobs],physics_rerun=False,render_source_sha256=c.sha(c.ROOT/'render.py'),elapsed_s=time.monotonic()-start))
    print('Presentation refresh complete.',flush=True)

if __name__=='__main__':main()
