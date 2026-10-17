"""Choose a discovery pair, freeze the choice, then run new seeds and resolution checks."""
import study as s

def choose():
    assert s.read(s.DATA/'main_accepted9_complete.json')['count']==180
    summary=s.read(s.DATA/'main_accepted9_summary.json');candidates=[]
    baseline={x['wave']:x['mean_D'] for x in summary['cells'] if x['nu']==0}
    for pair in summary['crossings']:
        if pair['low_wave']>3.5 or pair['no_loss_difference']/baseline[pair['high_wave']]<.1:continue
        for cross in pair['crossings']:
            upper=cross['upper']
            if upper['nu']>.75 or min(upper['low_mean'],upper['high_mean'])<20:continue
            score=[upper['both_upright'],upper['high_win_fraction'],min(upper['low_mean'],upper['high_mean'])]
            candidates.append(dict(waves=[pair['low_wave'],pair['high_wave']],score=score,crossing=cross))
    candidates.sort(key=lambda r:tuple(r['score']),reverse=True)
    if not candidates:raise RuntimeError('No eligible positive-displacement discovery crossing; do not manufacture one')
    selected=candidates[0];lo=selected['crossing']['lower']['nu'];hi=selected['crossing']['upper']['nu']
    levels=sorted({lo,(lo+hi)/2,hi,min(.75,hi+.25)}-{0})
    plan=dict(created=s.now(),waves=selected['waves'],nu=levels,seeds=list(range(100,108)),
        selection_rule='Post-screen selection, before independent results: lower wave <=3.5, >=10% no-loss advantage, observed reversal at nu<=.75 while both mean displacements >=20 mm; prioritize paired upright survival, then high-wave win fraction, then minimum mean displacement. This is a discovery rule, not a preregistered scientific test.',
        candidates=candidates,selected=selected,source_sha256=s.sha(__file__),
        validation='New seeds 100..107 at each listed nu; separate seed 100 at half timestep and 12 s, at nu=0 and the discovery reversal level. Each 12 s outcome uses its own 12 s no-loss reference. No retuning or changed viability gates.')
    s.write(s.DATA/'confirmation_plan.json',plan);return plan

if __name__=='__main__':
    path=s.DATA/'confirmation_plan.json';plan=s.read(path) if path.exists() else choose()
    print(plan['waves'],plan['nu'],flush=True)
    tasks=[s.task(w,nu,se,stage='independent_confirmation') for nu in plan['nu'] for se in plan['seeds'] for w in plan['waves']]
    s.bank(tasks,'independent_confirmation',12)
    upper=plan['selected']['crossing']['upper']['nu']
    checks=[s.task(w,nu,100,duration,scale,stage='resolution_duration_check') for duration,scale in [(6.,.5),(12.,1.)] for nu in [0,upper] for w in plan['waves']]
    s.bank(checks,'resolution_duration_check',8)
