"""Exact reference-phase grouping; not a measured contact-redundancy claim."""
from pathlib import Path
from fractions import Fraction
import collections,datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def main():
    manifest=ROOT/'outputs/physical_geometry_v2/data/manifest.json'
    waves=read(manifest)['waves'];rows=[];grid=[]
    for w in waves:
        # The midpoint-corrected leg-pitch oscillator reference, before
        # amplitude weighting and left/right joint-sign conventions.
        wave=Fraction(str(w))
        phase=[(-wave*i/18+wave/36+Fraction(1,2))%1 for i in range(19)]
        groups=collections.defaultdict(list)
        for i,p in enumerate(phase):groups[str(p)].append(i)
        values=np.array(list(map(float,phase)))
        rows.append(dict(wave=w,distinct_pair_reference_phases=len(groups),
            maximum_pair_positions_in_one_phase=max(map(len,groups.values())),
            groups=dict(groups),phase_turns=values.tolist(),
            first_phase_concentration=float(abs(np.exp(2j*np.pi*values).mean()))))
        grid.append(values)
    expected={.5:19,1:18,1.5:12,2:9,2.5:19,3:6,3.5:19,4:9,4.5:4,5:18,6:3,7:18,8:9,9:2}
    assert all(r['distinct_pair_reference_phases']==expected[r['wave']] for r in rows)
    files=[Path(__file__),manifest,ROOT/'core.py',ROOT.parent/'mujoco_wave_amplitude_global_optimization_2026-08-31/run_wave_amplitude_global.py']
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),physics_steps=0,n_pair_positions=19,
        phase_increment_magnitude='2*pi*wave/18',rows=rows,
        scope='Steady reference oscillator phases, after the onset blend; no amplitude or contact outcome selection.',
        interpretation='Distinct reference phases are not independent stochastic samples, actual stance phases, support counts, or a causal explanation of a measured crossing. Mirrored left/right joint signs do not by themselves determine physical contact timing. Group counts alone also omit spatial arrangement and phase coverage.',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (ROOT/'outputs/data/spatial_phase_groups.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.1),layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
    im=axes[0].imshow(np.array(grid).T,origin='lower',aspect='auto',cmap='twilight',vmin=0,vmax=1)
    axes[0].set(xticks=range(len(waves)),xticklabels=[f'{w:g}' for w in waves],yticks=[0,3,6,9,12,15,18],
        xlabel='Coupled wave number',ylabel='Leg-pair position index',title='Midpoint-corrected pitch reference phase')
    cb=fig.colorbar(im,ax=axes[0],fraction=.035,pad=.025);cb.set_label('Phase / one cycle')
    axes[1].bar(range(len(waves)),[r['distinct_pair_reference_phases'] for r in rows],color='#477c91')
    axes[1].set(xticks=range(len(waves)),xticklabels=[f'{w:g}' for w in waves],ylim=(0,21),yticks=[0,3,6,9,12,15,18],
        xlabel='Coupled wave number',ylabel='Number of distinct reference phases',title='Discrete phase grouping is not monotonic')
    for i,r in enumerate(rows):axes[1].text(i,r['distinct_pair_reference_phases']+.3,str(r['distinct_pair_reference_phases']),ha='center',fontsize=8)
    axes[1].grid(axis='y',alpha=.15);axes[1].spines[['top','right']].set_visible(False)
    fig.suptitle('19 paired positions; reference-phase geometry only',fontsize=13)
    fig.supxlabel('No physics run. Reference grouping is not a measurement of actual contact timing or support redundancy.',fontsize=8)
    fig.savefig(ROOT/'spatial_phase_groups.png',dpi=180);fig.savefig(ROOT/'spatial_phase_groups.pdf');plt.close(fig)
    print(json.dumps({str(r['wave']):r['distinct_pair_reference_phases'] for r in rows}))


if __name__=='__main__':main()
