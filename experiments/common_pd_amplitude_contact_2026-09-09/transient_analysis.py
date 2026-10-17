"""Reuse declared paired analysis in a separate transient output namespace."""
from pathlib import Path
import argparse,json
import numpy as np
import analyze_contact as a
ROOT=Path(__file__).resolve().parent
a.OUT=ROOT/'outputs/contact_erase_transient';a.DATA=a.OUT/'data';a.ROOT=a.OUT

def exposure():
    m=a.read(a.DATA/'manifest.json');completed=a.read(a.DATA/'completed/discovery.json')
    ids=set(completed['ids']);rows=[r for r in a.get_rows() if r['id'] in ids]
    result=[]
    fig,axes=a.plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for w in sorted(map(float,m['profiles'])):
        points=[]
        for p in m['probabilities']:
            group=[r for r in rows if r['wave']==w and r['task']['erasure']['probability']==p]
            complete=all(r['full'] for r in group)
            channels=[a.read(r['record'])['channel'] for r in group if r['full']]
            expected=1 if p in (0,1) else len(m['discovery_seeds'])
            item=dict(wave=w,p=p,complete=complete and len(channels)==expected)
            if item['complete']:
                item.update(actual_erased_fraction=float(np.mean([r['actual_erased_fraction'] for r in channels])),
                    desired_erased_fraction=float(np.mean([r['desired_erased_fraction'] for r in channels])),
                    delayed_extra_erased_fraction=float(np.mean([r['delayed_extra_erased_fraction'] for r in channels])),
                    restorations=sum(r['restorations'] for r in channels))
            points.append(item);result.append(item)
        ps=[v['p'] for v in points]
        axes[0].plot(ps,[v.get('actual_erased_fraction',np.nan) for v in points],'o-',label=f'w={w:g}')
        axes[1].plot(ps,[v.get('delayed_extra_erased_fraction',np.nan) for v in points],'o-')
    axes[0].plot([0,1],[0,1],'k--',lw=1,label='desired probability p')
    axes[0].set(ylabel='Actual erased leg-time fraction',title='Actual exposure can differ across waves')
    axes[1].set(ylabel='Extra erased fraction from restoration delay',title='Report recovery dependence alongside crossings')
    axes[0].legend(fontsize=7,ncol=3)
    for ax in axes:ax.set_xlabel('Desired Bernoulli probability p');ax.grid(alpha=.2)
    fig.savefig(a.OUT/'transient_exposure.png',dpi=180);fig.savefig(a.OUT/'transient_exposure.pdf');a.plt.close(fig)
    a.write(a.DATA/'exposure_analysis.json',dict(summary=result,no_failure_conditioned_means=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['discovery','confirmation','robustness']);args=p.parse_args()
    if args.action=='discovery':a.discovery();exposure()
    elif args.action=='confirmation':a.confirmation()
    else:
        import assess_contact_robustness as ar
        ar.ROOT=a.OUT;ar.DATA=a.DATA;ar.analyze()
