"""Presentation of completed statistics; no refitting, selection or new physics."""
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
    a=read(ROOT/'outputs/physical_geometry_v2/data/validated_amplitudes.json')
    assert a['status']=='VALIDATED_FINITE_SEARCH'
    discovery=read(data/'discovery_analysis.json')
    assert discovery['selected_for_confirmation'],'No selected crossing: retain the all-wave discovery figure.'
    result=read(data/'confirmation_analysis.json');choice=result['choice']
    waves=[choice['low_wave'],choice['high_wave']];colors=['#28618d','#c56b30']
    fig,axes=plt.subplots(1,3,figsize=(14.2,4.8),layout='constrained')
    x=[r['wave'] for r in a['selected']]
    y=[r['amplitude']['body'] for r in a['selected']]
    axes[0].plot(x,y,'o-',color='#727e85',ms=4,lw=1.4,label='Independent qualified choices')
    axes[0].plot(x,[r['realized_body_median_deg'] for r in a['selected']],'.--',color='#b0b5b8',lw=1,label='Realized body, 1 Hz median')
    for w,color in zip(waves,colors):
        chosen=next(r for r in a['selected'] if r['wave']==w)
        axes[0].scatter(w,chosen['amplitude']['body'],s=65,color=color,zorder=5,label=f'w={w:g}')
    axes[0].set(xlabel='Coupled wave number',ylabel='Body amplitude (deg)',title='A  One complete PD at every wave')
    axes[0].legend(fontsize=7,loc='best')
    gate_labels=[]
    for j,(w,color) in enumerate(zip(waves,colors)):
        explored=sorted((r for r in discovery['summary'] if r['wave']==w),key=lambda r:r['p'])
        axes[1].plot([r['p'] for r in explored],[r.get('mean_D',np.nan) for r in explored],
                     ls=':',lw=1.1,color=color,alpha=.6)
        points=[r for r in result['points'] if r['complete']]
        xp=[r['p'] for r in points];groups=[r['group'][j] for r in points]
        if 0 not in xp:xp=[0.,*xp];groups=[result['zero_loss_baseline'][j],*groups]
        yp=np.array([g['mean_D'] for g in groups]);ci=np.array([g['D_CI95'] for g in groups])
        # Percentile intervals need not contain the sample mean: draw their
        # exact bounds independently instead of clamping negative error sizes.
        axes[1].plot(xp,yp,'o-',color=color,lw=2,ms=5,label=f'w={w:g}')
        axes[1].vlines(xp,ci[:,0],ci[:,1],color=color,lw=1.2)
        gate_labels.append(f"w={w:g}: "+', '.join(f"p={p:g} {g['engineering_fraction']:.0%}" for p,g in zip(xp,groups)))
    points=[r for r in result['points'] if r['complete']]
    xp=[r['p'] for r in points];yp=[r['mean_difference_mm'] for r in points]
    ci=np.array([r['paired_difference_CI97p5'] for r in points])
    axes[2].plot(xp,yp,'o-',color='#485663',ms=5)
    if len(points):axes[2].vlines(xp,ci[:,0],ci[:,1],color='#485663',lw=1.5)
    axes[2].axhline(0,color='black',lw=1)
    low,high=choice['bracket']
    for ax in axes[1:]:ax.axvspan(low,high,color='#829cb1',alpha=.08)
    axes[1].set(xlabel='Requested erasure probability p',ylabel='Complete-window displacement (mm)',
                title='B  Frozen gaits; new paired masks')
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel='Requested erasure probability p',ylabel='Low wave minus high wave (mm)',
                title='C  Paired 97.5% intervals')
    title='Persistent bilateral-pair ground loss' if channel=='contact_erase' else 'Transient loss: 0.5 s requests, legwise safe restoration'
    status='Held-out sign change confirmed at nominal timestep' if result['confirmed_at_nominal_step'] else 'Held-out sign change NOT confirmed at nominal timestep'
    fig.suptitle(title+'\n'+status,fontsize=13)
    footer='Solid: held-out means / 95% intervals; dotted: four-mask discovery.\nEngineering pass fractions: '+' | '.join(gate_labels)
    if channel.endswith('transient'):footer+='\nRequested p is not identical actual erased time; see the exposure report.'
    fig.supxlabel(footer,fontsize=7)
    for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    path=out/'final_crossing'
    fig.savefig(path.with_suffix('.png'),dpi=200);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
    inputs=[ROOT/'outputs/physical_geometry_v2/data/validated_amplitudes.json',data/'discovery_analysis.json',data/'confirmation_analysis.json',Path(__file__)]
    receipt=dict(no_new_statistics=True,no_selected_noise_seeds=True,channel=channel,
                 source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (data/'final_crossing_figure_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(path.with_suffix('.png'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('channel',choices=['contact_erase','contact_erase_transient']);main(p.parse_args().channel)
