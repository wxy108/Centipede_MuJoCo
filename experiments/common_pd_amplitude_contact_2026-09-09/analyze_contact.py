"""Discovery is exploratory; a crossing is confirmed only on new paired masks."""
from pathlib import Path
import argparse,hashlib,itertools,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/contact_erase';DATA=OUT/'data'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def boot(values,alpha=.05):
    a=np.asarray(values,float)
    if len(a)==1:return [float(a[0]),float(a[0])]
    rng=np.random.default_rng(9092026)
    means=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)
    return list(map(float,np.quantile(means,[alpha/2,1-alpha/2])))

def get_rows():
    return [read(p) for p in (OUT/'assessments').glob('*.json')]

def summarize(rows,w,p,seeds):
    expected=[0] if p in (0,1) else seeds
    chosen=[next((r for r in rows if r['wave']==w and r['task']['erasure']['probability']==p and r['task']['erasure']['seed']==seed),None) for seed in expected]
    full=all(r is not None and r['full'] and r['D'] is not None for r in chosen)
    result=dict(wave=w,p=p,expected_seeds=expected,complete=full,available=sum(r is not None for r in chosen))
    if not full:return result
    values=[r['D'] for r in chosen];channels=[read(r['record'])['channel'] for r in chosen]
    result.update(mean_D=float(np.mean(values)),D_CI95=boot(values),values=values,
        upright_fraction=float(np.mean([r['metrics']['min_upright']>=np.cos(np.radians(50)) for r in chosen])),
        engineering_fraction=float(np.mean([r['feasible'] for r in chosen])),
        chassis_contact_fraction=float(np.mean([r['metrics']['chassis_contact_fraction'] for r in chosen])),
        mean_erased_fraction=float(np.mean([r['actual_erased_fraction'] for r in channels])),
        ids=[r['id'] for r in chosen])
    return result

def discovery():
    m=read(DATA/'manifest.json');completion=read(DATA/'completed/discovery.json');allrows=get_rows()
    ids=set(completion['ids']);rows=[r for r in allrows if r['id'] in ids]
    waves=sorted(map(float,m['profiles']));ps=m['probabilities'];seeds=m['discovery_seeds']
    summary=[summarize(rows,w,p,seeds) for w in waves for p in ps]
    lookup={(r['wave'],r['p']):r for r in summary};candidates=[]
    for low,high in itertools.combinations(waves,2):
        left=lookup[(low,0)];right=lookup[(high,0)]
        if not (left['complete'] and right['complete']):continue
        if left['mean_D']<=right['mean_D']:continue
        for p0,p1 in zip(ps,ps[1:]):
            group=[lookup[(w,p)] for p in (p0,p1) for w in (low,high)]
            if not all(r['complete'] for r in group):continue
            diff0=group[0]['mean_D']-group[1]['mean_D'];diff1=group[2]['mean_D']-group[3]['mean_D']
            if diff0>0 and diff1<0:
                clean=all(r['upright_fraction']==1 and r['chassis_contact_fraction']<=.15 and r['mean_D']>0 for r in group)
                useful_high=all(r['upright_fraction']>=.75 and r['mean_D']>=max(5.,.1*right['mean_D']) for r in (group[1],group[3]))
                body_low=m['profiles'][str(float(low))]['amplitude']['body']
                body_high=m['profiles'][str(float(high))]['amplitude']['body']
                candidates.append(dict(low_wave=low,high_wave=high,bracket=[p0,p1],
                    baseline_body_amplitudes_deg=[body_low,body_high],baseline_body_amplitude_descends=body_low>body_high,
                    discovery_differences_mm=[diff0,diff1],margin_mm=min(diff0,-diff1),
                    clean_exploratory=clean,high_wave_remains_useful=useful_high,group=group,
                    interpretation='Exploratory sign change; engineering metrics remain separate. Not independently confirmed.'))
                break
    candidates.sort(key=lambda r:(r['clean_exploratory'],r['baseline_body_amplitude_descends'],r['margin_mm']),reverse=True)
    choice=next((r for r in candidates if r['clean_exploratory'] and r['high_wave_remains_useful']),None)
    if choice is None:choice=next((r for r in candidates if r['high_wave_remains_useful']),None)
    result=dict(status='DISCOVERY_ONLY',summary=summary,candidates=candidates,selected_for_confirmation=choice,
        no_failure_conditioned_means=True,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(DATA/'discovery_analysis.json',result)
    if choice:
        record=dict(choice,confirmation_seeds=m['confirmation_seeds'],
            confirmation_probabilities=sorted(set([*choice['bracket'],sum(choice['bracket'])/2])),
            selection_rule='Prefer both waves upright and mean chassis-contact fraction <= .15. Within that category first prefer pairs whose frozen low-wave body amplitude exceeds the high-wave body amplitude, then the largest two-sided discovery sign margin. If absent, apply the same amplitude preference and margin ordering to crossings with the high wave upright in >=75% of discovery masks and mean D >= max(5mm,10% of its baseline) at both sides. All pairs remain in the report. A low-wave failure crossing is explicitly labeled; it is not clean-locomotion evidence.',
            interpretation='Frozen before new seeds; no guarantee of confirmation.')
        path=DATA/'crossing_choice.json'
        if path.exists():assert read(path)==record
        else:write(path,record)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for i,w in enumerate(waves):
        selected=[lookup[(w,p)] for p in ps];y=[r['mean_D'] if r['complete'] else np.nan for r in selected]
        axes[0].plot(ps,y,'o-',label=f'w={w:g}',color=plt.cm.viridis(i/max(1,len(waves)-1)))
        base=lookup[(w,0)].get('mean_D',np.nan)
        axes[1].plot(ps,np.array(y)/base,'o-',color=plt.cm.viridis(i/max(1,len(waves)-1)))
    axes[0].set(ylabel='Full-window displacement (mm)',title='Exploratory ground-contact loss; all outcomes')
    axes[1].set(ylabel='Displacement / no-loss displacement',title='Retention; no survivor-conditioned averaging')
    axes[0].legend(fontsize=7,ncol=3)
    for ax in axes:ax.set_xlabel('Bernoulli probability p per bilateral pair');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'contact_discovery.png',dpi=180);fig.savefig(ROOT/'contact_discovery.pdf');plt.close(fig)
    print(json.dumps(dict(exploratory_crossings=len(candidates),clean_candidates=sum(r['clean_exploratory'] for r in candidates),choice=None if choice is None else [choice['low_wave'],choice['high_wave'],choice['bracket']]),ensure_ascii=False))

def confirmation():
    choice=read(DATA/'crossing_choice.json');rows=get_rows();seeds=choice['confirmation_seeds'];ps=choice['confirmation_probabilities']
    waves=[choice['low_wave'],choice['high_wave']]
    selected=[r for r in rows if r['task']['duration']==6 and r['task']['dt_scale']==.5]
    zero_loss=[summarize(selected,w,0,[0]) for w in waves]
    assert all(g['complete'] for g in zero_loss)
    points=[]
    for p in ps:
        group=[summarize(selected,w,p,seeds) for w in waves]
        if not all(r['complete'] for r in group):points.append(dict(p=p,complete=False,group=group));continue
        diff=np.array(group[0]['values'])-np.array(group[1]['values'])
        points.append(dict(p=p,complete=True,mean_difference_mm=float(diff.mean()),
            paired_difference_CI97p5=boot(diff,alpha=.025),group=group))
    endpoints=[next(r for r in points if r['p']==p) for p in choice['bracket']]
    sign_evidence=all(r['complete'] for r in endpoints) and endpoints[0]['paired_difference_CI97p5'][0]>0 and endpoints[1]['paired_difference_CI97p5'][1]<0
    upright_clean=all(r['complete'] and all(g['upright_fraction']>=.95 and g['chassis_contact_fraction']<=.15 and g['mean_D']>0 for g in r['group']) for r in endpoints)
    all_engineering=all(r['complete'] and all(g['engineering_fraction']==1 for g in r['group']) for r in endpoints)
    result=dict(status='INDEPENDENT_PAIRED_CONFIRMATION',choice=choice,points=points,opposite_sign_intervals=sign_evidence,
        zero_loss_baseline=zero_loss,
        upright_and_low_chassis_support=upright_clean,confirmed_at_nominal_step=sign_evidence,
        upright_low_chassis_crossing_at_nominal_step=sign_evidence and upright_clean,
        all_engineering_gates_at_bracket=all_engineering,
        engineering_qualified_crossing_at_nominal_step=sign_evidence and all_engineering,
        posture_label_note='Upright with low chassis support is a posture category. It does not certify every force, torque, velocity, penetration or self-contact engineering gate.',
        halfstep_duration_checks_still_required=True,
        interval_note='Paired percentile bootstrap; 97.5% interval at each endpoint gives a Bonferroni two-endpoint convention. Approximate sampling uncertainty, not deterministic proof.')
    write(DATA/'confirmation_analysis.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for j,w in enumerate(waves):
        x=[];y=[];lo=[];hi=[]
        if 0 not in ps:
            g=zero_loss[j];x.append(0.);y.append(g['mean_D']);lo.append(g['mean_D']);hi.append(g['mean_D'])
        for r in points:
            if r['complete']:
                g=r['group'][j];x.append(r['p']);y.append(g['mean_D']);lo.append(g['D_CI95'][0]);hi.append(g['D_CI95'][1])
        axes[0].plot(x,y,'o-',label=f'w={w:g}');axes[0].fill_between(x,lo,hi,alpha=.18)
    x=[r['p'] for r in points if r['complete']];y=[r['mean_difference_mm'] for r in points if r['complete']]
    lo=[r['paired_difference_CI97p5'][0] for r in points if r['complete']];hi=[r['paired_difference_CI97p5'][1] for r in points if r['complete']]
    axes[1].plot(x,y,'o-');axes[1].fill_between(x,lo,hi,alpha=.2);axes[1].axhline(0,color='black',lw=1)
    axes[0].set(ylabel='Full-window displacement (mm)',title='Held-out means and 95% intervals; p=0 baseline');axes[0].legend()
    axes[1].set(ylabel='Low wave minus high wave (mm)',title='Paired difference; 97.5% intervals')
    for ax in axes:ax.set_xlabel('Bernoulli probability p');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'contact_confirmation.png',dpi=180);fig.savefig(ROOT/'contact_confirmation.pdf');plt.close(fig)
    print(json.dumps(dict(opposite_sign_intervals=sign_evidence,clean=upright_clean,halfstep_duration_checks_still_required=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['discovery','confirmation']);a=p.parse_args()
    discovery() if a.action=='discovery' else confirmation()
