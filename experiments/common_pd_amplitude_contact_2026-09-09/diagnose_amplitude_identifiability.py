"""Describe near-best amplitude ambiguity; never selects a gait for erasure."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/physical_geometry_v2'

def read(p):return json.loads(p.read_text())

def main():
    rows=[read(p) for p in (OUT/'assessments').glob('*.json')]
    rows=[r for r in rows if r['full'] and r['feasible'] and r['task']['phase']=='midpoint' and r['task']['duration']==6 and r['task']['dt_scale']==.5 and r.get('self_contact_verified') is not False]
    waves=[1.5,2,2.5,3,3.5,4,4.5];groups=[];bands=[]
    for w in waves:
        group=sorted([r for r in rows if r['wave']==w],key=lambda r:r['D'],reverse=True);assert group
        groups.append(group);best=group[0]['D']
        for fraction in (.01,.02,.03,.05):
            near=[r for r in group if r['D']>=(1-fraction)*best]
            amp=[r['amplitudes']['body'] for r in near]
            bands.append(dict(wave=w,relative_speed_loss=fraction,n=len(near),body_min_deg=min(amp),body_max_deg=max(amp),best_D_mm=best))
    # Dynamic programming minimizes the largest relative loss over a fully
    # non-increasing chain in the observed finite candidate set. This diagnoses
    # the historical smoothing mechanism, and is NOT adopted as our optimizer.
    costs=np.zeros(len(groups[0]));paths=[[r['id']] for r in groups[0]]
    costs=np.array([1-r['D']/groups[0][0]['D'] for r in groups[0]])
    for prev,group in zip(groups,groups[1:]):
        new=np.full(len(group),np.inf);newpaths=[None]*len(group)
        for j,r in enumerate(group):
            allowed=[i for i,p in enumerate(prev) if p['amplitudes']['body']>=r['amplitudes']['body']-1e-10 and np.isfinite(costs[i])]
            if allowed:
                values=[max(costs[i],1-r['D']/group[0]['D']) for i in allowed]
                i=allowed[int(np.argmin(values))];new[j]=min(values);newpaths[j]=paths[i]+[r['id']]
        costs=new;paths=newpaths
    feasible=bool(np.isfinite(costs).any());chain=[]
    if feasible:
        j=int(np.argmin(costs));lookup={r['id']:r for r in rows}
        for i,key in enumerate(paths[j]):
            r=lookup[key];chain.append(dict(wave=r['wave'],body_deg=r['amplitudes']['body'],D_mm=r['D'],id=key,
                loss_from_independent_nominal_best=1-r['D']/groups[i][0]['D']))
    result=dict(status='PROVISIONAL_NOMINAL_DIAGNOSTIC_NOT_SELECTED',bands=bands,
        observed_monotonic_chain_exists=feasible,minimax_relative_speed_loss=float(costs.min()) if feasible else None,
        diagnostic_chain=chain,used_for_contact=False,changes_to_optimizer=False,
        scope='Full6s nominal candidates with observed chassis-contact checks only. Not a long/halfstep-qualified amplitude bank. Finite sampled intervals are not confidence intervals. This diagnostic quantifies how a monotonic near-best path could differ from independently maximizing displacement; it does not adopt that path.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'data/amplitude_identifiability.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    best=[g[0]['amplitudes']['body'] for g in groups]
    axes[0].plot(waves,best,'o-',label='Independent best observed',color='#245e80')
    for loss,color,width in [(.03,'#b4c9d6',7),(.01,'#6393af',3)]:
        selected=[r for r in bands if r['relative_speed_loss']==loss]
        axes[0].vlines(waves,[r['body_min_deg'] for r in selected],[r['body_max_deg'] for r in selected],color=color,lw=width,alpha=.7,label=f'Observed min/max within {loss:.0%} of best')
    axes[0].set(xlabel='Coupled wave number',ylabel='Body command amplitude (deg)',title='Near-best ranges do not remove the reversal')
    axes[0].legend(fontsize=8)
    for w,color in [(2,'#337fa1'),(2.5,'#bd7132')]:
        group=groups[waves.index(w)];maximum=group[0]['D']
        axes[1].scatter([r['amplitudes']['body'] for r in group],[r['D']/maximum for r in group],s=20,alpha=.5,c=color,label=f'w={w:g}; all eligible nominal points')
        axes[1].scatter([group[0]['amplitudes']['body']],[1.],s=110,marker='*',c=color,edgecolors='black')
    axes[1].axhline(.97,color='grey',ls='--',lw=1)
    axes[1].set(xlabel='Body command amplitude (deg)',ylabel='Displacement / own-wave best',title='Wave 2 and 2.5 prefer different sampled regions')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('PROVISIONAL diagnostic — no monotonic path adopted; fixed PD',fontsize=12)
    fig.savefig(ROOT/'amplitude_identifiability.png',dpi=180);fig.savefig(ROOT/'amplitude_identifiability.pdf');plt.close(fig)
    print(json.dumps(dict(n_candidates=len(rows),monotonic_chain_exists=feasible,minimax_relative_speed_loss=result['minimax_relative_speed_loss'],diagnostic_only=True)))

if __name__=='__main__':main()
