import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
f=s.ROOT/'runs/continuous_slide_tgs_cpu_dt62_000'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.3))
for file,label,color,ls in [('Mu_baseline.npz','MuJoCo original cone coupling','#c45144','-'),('Mu_corrected.npz','MuJoCo normal-flow correction','#2878a8','-'),('trajectory.npz','PhysX TGS, external forces each iteration','#27986d','--')]:
    tr={k:v for k,v in np.load(f/file).items()};names=tr['joint_names'].tolist();iz=names.index('z');t=tr['t']*1e3
    axes[0].plot(t,(tr['q'][:,iz]-.00045)*1e6,label=label,color=color,ls=ls,lw=1.5)
    axes[1].plot(t,tr['v'][:,iz]*1e3,label=label,color=color,ls=ls,lw=1.5)
axes[0].set(ylabel='Sphere surface height (µm)',ylim=(-1,12));axes[1].set(ylabel='Normal velocity (mm/s)',ylim=(-18,18))
for ax in axes:ax.set(xlabel='Time (ms)',xlim=(0,30));ax.set_xticks(range(0,31,5));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Continuous sliding under the same PD, mass and friction — free normal motion',fontsize=11)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.5,-.015))
fig.subplots_adjust(bottom=.30,top=.85,wspace=.28)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/18_continuous_slide_normal_response.{ext}',bbox_inches='tight')
summary=json.loads((f/'comparison.json').read_text());s.write(s.ROOT/'audit/desaxce_contact_evidence.json',dict(continuous_slide=summary,prescribed_slide=json.loads((s.ROOT/'runs/prescribed_slide_tgs_cpu_dt62_000/desaxce_diagnostic.json').read_text()),mechanical_suite=json.loads((s.ROOT/'audit/desaxce_mechanical_suite.json').read_text()),scope='Isolated contact checks only; complete locomotion validation remains required.',source_sha=s.sha(__file__)))
print('Saved isolated contact-law evidence and standalone figures.')
