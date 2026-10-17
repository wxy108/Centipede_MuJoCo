"""Describe recorded movement and support on the same held-out cohorts."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

def main(channel):
    out=ROOT/'outputs'/channel;data=out/'data'
    result=read(data/'confirmation_analysis.json')
    rows={r['id']:r for r in [read(p) for p in (out/'assessments').glob('*.json')]}
    groups=[*result['zero_loss_baseline'],*[g for point in result['points'] for g in point['group']]]
    groups=list({(g['wave'],g['p']):g for g in groups}.values())
    report=[];sources={}
    for group in groups:
        item=dict(wave=group['wave'],p=group['p'],complete=group['complete'])
        if not group['complete']:report.append(item);continue
        measured=[]
        for rid in group['ids']:
            row=rows[rid];assert row['task']['duration']==6 and row['task']['dt_scale']==.5
            path=Path(row['record']).parent/'contact_trajectory.npz'
            with np.load(path,allow_pickle=False) as z:
                good=z['t']>=1.5;xyz=z['center'][good];t=z['t'][good]
                distance=float(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1).sum()*1000)
                D=float(-(xyz[-1,0]-xyz[0,0])*1000)
                lateral=float((xyz[-1,1]-xyz[0,1])*1000)
            assert D==row['D']
            efficiency=D/max(distance,1e-12)
            assert abs(efficiency-row['metrics']['path_efficiency'])<1e-10
            measured.append(dict(id=rid,seed=row['task']['erasure']['seed'],forward_mm=D,
                                  path_xy_mm=distance,lateral_net_mm=lateral,elapsed_s=float(t[-1]-t[0]),
                                  path_efficiency=efficiency,all_engineering_gates=row['feasible'],
                                  mean_contacting_feet=row['metrics']['mean_contacting_feet'],
                                  chassis_contact_fraction=row['metrics']['chassis_contact_fraction'],
                                  root_roll_abs_peak_deg=row['metrics']['root_roll_abs_peak_deg'],
                                  ground_force_peak_bw=row['metrics']['ground_force_peak_bw'],
                                  mean_actual_erased_fraction=read(row['record'])['channel']['actual_erased_fraction']))
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        means={k:float(np.mean([r[k] for r in measured])) for k in measured[0] if k not in ('id','seed')}
        item.update(values=measured,means=means,mean_forward_over_mean_path=means['forward_mm']/means['path_xy_mm'] if means['path_xy_mm']>0 else None)
        report.append(item)
    waves=sorted({g['wave'] for g in report});baselines={w:next(g for g in report if g['wave']==w and g['p']==0) for w in waves}
    for g in report:
        if not g['complete']:continue
        base=baselines[g['wave']]
        g['forward_retention']=g['means']['forward_mm']/base['means']['forward_mm']
        g['path_retention']=g['means']['path_xy_mm']/base['means']['path_xy_mm']
        g['direction_efficiency_retention']=g['mean_forward_over_mean_path']/base['mean_forward_over_mean_path'] if g['mean_forward_over_mean_path'] is not None else None
        if g['direction_efficiency_retention'] is not None:
            np.testing.assert_allclose(g['path_retention']*g['direction_efficiency_retention'],g['forward_retention'],rtol=1e-12,atol=1e-12)
    info=dict(channel=channel,groups=report,same_heldout_cohorts=True,survivor_conditioning=False,
              interpretation='Exact algebraic decomposition: mean forward displacement retention = mean XY path retention times the retention of mean-forward/mean-path direction efficiency. This describes movement, not a causal proof of contact redundancy. Complete failed outcomes stay in every mean; incomplete cohorts have no estimated mean.',
              source_hashes={**sources,**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),data/'confirmation_analysis.json']}})
    (data/'mechanism_analysis.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
    colors=['#28618d','#c56b30']
    for w,color in zip(waves,colors):
        selected=sorted((g for g in report if g['wave']==w),key=lambda g:g['p'])
        x=[g['p'] for g in selected]
        for ax,key,title in [(axes[0,0],'path_retention','XY path length / baseline'),
                             (axes[0,1],'direction_efficiency_retention','Direction efficiency / baseline')]:
            ax.plot(x,[g.get(key,np.nan) if g.get(key) is not None else np.nan for g in selected],'o-',color=color,label=f'w={w:g}')
            ax.set(ylabel='Ratio',title=title);ax.axhline(1,color='#aaaaaa',ls=':',lw=1)
        for ax,key,title in [(axes[1,0],'mean_contacting_feet','Mean number of feet contacting the ground'),
                             (axes[1,1],'chassis_contact_fraction','Fraction of time with chassis support')]:
            ax.plot(x,[g['means'][key] if g['complete'] else np.nan for g in selected],'o-',color=color,label=f'w={w:g}')
            ax.set(title=title)
    for ax in axes.flat:ax.set_xlabel('Requested erasure probability p');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend();axes[1,1].axhline(.15,color='#999999',ls=':',lw=1)
    title='Persistent ground loss' if channel=='contact_erase' else 'Transient ground loss with safe restoration'
    fig.suptitle(title+' — recorded movement and support',fontsize=13)
    fig.supxlabel('Same held-out cohorts, including complete failed outcomes. Descriptive means; no new significance tests.',fontsize=8)
    fig.savefig(out/'crossing_mechanism.png',dpi=180);fig.savefig(out/'crossing_mechanism.pdf');plt.close(fig)
    print(out/'crossing_mechanism.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('channel',choices=['contact_erase','contact_erase_transient']);main(p.parse_args().channel)
