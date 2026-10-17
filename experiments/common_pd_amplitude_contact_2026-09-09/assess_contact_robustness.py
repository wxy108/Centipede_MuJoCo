"""Independent contact validation: retain failures and label timestep sensitivity."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_contact as ac

ROOT=Path(__file__).resolve().parent;DATA=ac.DATA


def analyze():
    declaration=ac.read(DATA/'step_duration_declaration.json')
    completion=ac.read(DATA/'completed/step_duration.json')
    choice=ac.read(DATA/'crossing_choice.json')
    manifest=ac.read(DATA/'manifest.json')
    waves=[choice['low_wave'],choice['high_wave']]
    allrows=ac.get_rows();ids=set(completion['ids'])
    rows=[r for r in allrows if r['id'] in ids]
    seeds=declaration['seeds'];ps=declaration['probabilities']
    conditions=[dict(duration=6,scale=.5,label='Nominal, same eight masks'),
                *[dict(**v,label=f"{v['duration']:g}s, dt scale {v['scale']:g}") for v in declaration['conditions']]]
    report=[]
    for cond in conditions:
        source=allrows if cond['duration']==6 and cond['scale']==.5 else rows
        selected=[r for r in source if r['task']['duration']==cond['duration'] and r['task']['dt_scale']==cond['scale']]
        points=[]
        for p in ps:
            group=[ac.summarize(selected,w,p,seeds) for w in waves]
            point=dict(p=p,group=group,complete=all(g['complete'] for g in group))
            if point['complete']:
                delta=np.array(group[0]['values'])-np.array(group[1]['values'])
                # All records are sampled at 1ms: evaluate actual saved sample window.
                elapsed=[]
                for g in group:
                    per_mask=[]
                    for rid in g['ids']:
                        row=next(r for r in selected if r['id']==rid)
                        with np.load(Path(row['record']).parent/'contact_trajectory.npz') as z:
                            tt=z['t'][z['t']>=1.5];per_mask.append(float(tt[-1]-tt[0]))
                    elapsed.append(per_mask)
                np.testing.assert_array_equal(elapsed[0],elapsed[1])
                speed_delta=delta/np.array(elapsed[0])
                point.update(mean_difference_mm=float(delta.mean()),CI97p5_mm=ac.boot(delta,.025),
                    elapsed_s=elapsed,mean_speed_difference_mm_s=float(speed_delta.mean()),CI97p5_mm_s=ac.boot(speed_delta,.025))
            points.append(point)
        signs=all(v['complete'] for v in points) and points[0]['CI97p5_mm'][0]>0 and points[1]['CI97p5_mm'][1]<0
        clean=all(v['complete'] and all(g['upright_fraction']>=.95 and g['chassis_contact_fraction']<=.15 and g['mean_D']>0 for g in v['group']) for v in points)
        engineering=all(v['complete'] and all(g['engineering_fraction']==1 for g in v['group']) for v in points)
        report.append(dict(condition=cond,points=points,opposite_sign_intervals=signs,clean_posture=clean,
            all_engineering_gates_pass=engineering))
    nominal,half,long=report
    sensitivity=[]
    for j,p in enumerate(ps):
        for k,w in enumerate(waves):
            a=nominal['points'][j]['group'][k];b=half['points'][j]['group'][k]
            if a['complete'] and b['complete']:
                reference=manifest['profiles'][str(float(w))]['nominal']['D']
                differences=np.array(b['values'])-np.array(a['values'])
                sensitivity.append(dict(wave=w,p=p,mean_half_minus_nominal_mm=float(differences.mean()),
                    paired_CI95_mm=ac.boot(differences),max_abs_paired_difference_mm=float(np.max(np.abs(differences))),
                    absolute_mean_difference_as_no_loss_fraction=float(abs(differences.mean())/abs(reference))))
    result=dict(status='CONTACT_ROBUSTNESS_ASSESSED',conditions=report,timestep_sensitivity=sensitivity,
        crossing_sign_supported_all_conditions=all(r['opposite_sign_intervals'] for r in report),
        clean_posture_all_conditions=all(r['clean_posture'] for r in report),
        all_engineering_gates_all_conditions=all(r['all_engineering_gates_pass'] for r in report),
        engineering_qualified_crossing_all_conditions=all(r['opposite_sign_intervals'] and r['all_engineering_gates_pass'] for r in report),
        posture_label_note='Posture and sign support do not certify every engineering gate or numerical convergence; inspect the separate gate fractions and timestep sensitivity.',
        interpretation='Eight predeclared paired masks. Intervals are sampling diagnostics; failure or sign loss is reported. Long-horizon speeds use actual sample duration. Timestep sensitivity is descriptive and is not normalized by a near-zero erased displacement.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    ac.write(DATA/'robustness_analysis.json',result)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,r in zip(axes,report):
        points=[v for v in r['points'] if v['complete']]
        x=[v['p'] for v in points];y=[v['mean_speed_difference_mm_s'] for v in points]
        lo=[v['CI97p5_mm_s'][0] for v in points];hi=[v['CI97p5_mm_s'][1] for v in points]
        ax.plot(x,y,'o-');ax.fill_between(x,lo,hi,alpha=.2);ax.axhline(0,color='black',lw=1)
        ax.set(title=r['condition']['label'],xlabel='Erasure probability p',ylabel='Low minus high wave speed (mm/s)')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'contact_robustness.png',dpi=180);fig.savefig(ROOT/'contact_robustness.pdf');plt.close(fig)
    print(json.dumps({k:result[k] for k in ('crossing_sign_supported_all_conditions','clean_posture_all_conditions')}))


if __name__=='__main__':analyze()
