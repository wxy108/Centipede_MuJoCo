"""Clear two-wave exposure display; reuses recorded cohorts without reanalysis."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/contact_erase_transient';DATA=OUT/'data'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    choice=read(DATA/'crossing_choice.json');discovery=read(DATA/'exposure_analysis.json')
    waves=[choice['low_wave'],choice['high_wave']];sources={}
    for p in (DATA/'crossing_choice.json',DATA/'exposure_analysis.json',Path(__file__)):sources[str(p)]=sha(p)
    confirmed=DATA/'confirmation_analysis.json';held=[]
    if confirmed.exists():
        result=read(confirmed);sources[str(confirmed)]=sha(confirmed)
        rows={r['id']:r for r in [read(p) for p in (OUT/'assessments').glob('*.json')]}
        for point in result['points']:
            for group in point['group']:
                if not group['complete']:continue
                channels=[]
                for rid in group['ids']:
                    path=Path(rows[rid]['record']);channels.append(read(path)['channel']);sources[str(path)]=sha(path)
                item=dict(wave=group['wave'],p=group['p'],n=len(channels),
                    actual_erased_fraction=float(np.mean([ch['actual_erased_fraction'] for ch in channels])),
                    delayed_extra_erased_fraction=float(np.mean([ch['delayed_extra_erased_fraction'] for ch in channels])))
                assert item['actual_erased_fraction']==group['mean_erased_fraction']
                held.append(item)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    for wave,color in zip(waves,('#28618d','#c56b30')):
        points=sorted([r for r in discovery['summary'] if r['wave']==wave],key=lambda r:r['p'])
        assert all(p['complete'] for p in points)
        for ax,key in zip(axes,('actual_erased_fraction','delayed_extra_erased_fraction')):
            ax.plot([r['p'] for r in points],[100*r[key] for r in points],':',color=color,alpha=.7,
                    label=f'w={wave:g}, discovery')
            h=sorted([r for r in held if r['wave']==wave],key=lambda r:r['p'])
            if h:ax.plot([r['p'] for r in h],[100*r[key] for r in h],'o-',color=color,label=f'w={wave:g}, held-out')
    axes[0].plot([0,1],[0,100],'--',color='#777777',lw=1,label='requested probability')
    axes[0].set(ylabel='Actual erased leg-time (%)',title='Exposure in the frozen comparison')
    axes[1].set(ylabel='Additional erased leg-time (percentage points)',title='Extra time caused by safe restoration delay')
    for ax in axes:ax.set_xlabel('Requested erasure probability p');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.supxlabel('Dotted: four-mask discovery (deterministic endpoints). Solid points: the same new masks used for displacement confirmation.\nRight: actual minus sampled desired exposure, not actual minus p. Complete failed outcomes remain included.',fontsize=8)
    fig.savefig(OUT/'selected_transient_exposure.png',dpi=180);fig.savefig(OUT/'selected_transient_exposure.pdf');plt.close(fig)
    receipt=dict(heldout_analysis_present=confirmed.exists(),heldout_included=bool(held),heldout_summary=held,changed_statistics=False,
                 existing_frozen_comparison=True,source_hashes=sources)
    (DATA/'selected_exposure_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(heldout_included=bool(held),path=str(OUT/'selected_transient_exposure.png'))))

if __name__=='__main__':main()
