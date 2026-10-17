import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
f=s.ROOT/'runs/normal_drive_contact_pgs_cpu_dt62_000';tr={k:v for k,v in np.load(f/'trajectory.npz').items()};mu={k:v for k,v in np.load(f/'mu_comparison.npz').items()};names=tr['joint_names'].tolist();ix=names.index('x');t=tr['t']*1000
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].plot(t,tr['q'][:,ix]*1e6,color='#222',label='PhysX PGS')
ax[0].plot(t,mu['split_q'][:,ix]*1e6,color='#c45346',ls='--',label='MuJoCo split update')
ax[0].plot(t,mu['coupled_q'][:,ix]*1e6,color='#2878a8',ls=':',label='MuJoCo coupled update')
ax[0].set_ylabel('Horizontal position (µm)');ax[0].set_ylim(0,30);ax[0].set_yticks([0,10,20,30])
for key,color,label in [('split','#c45346','Split update'),('coupled','#2878a8','Coupled update')]:ax[1].plot(t,(mu[key+'_q'][:,ix]-tr['q'][:,ix])*1e6,color=color,label=label)
ax[1].set_ylabel('Error relative to PhysX (µm)');ax[1].set_ylim(-.4,.4);ax[1].set_yticks([-.4,-.2,0,.2,.4]);ax[1].axhline(0,color='#888',lw=.6)
for a in ax:a.set_xlim(0,30);a.set_xticks([0,10,20,30]);a.set_xlabel('Time (ms)');a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
ax[0].set_title('Same normal and tangential PD');ax[1].set_title('Normalized RMS: 0.602% → 0.075%');h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,0));fig.subplots_adjust(bottom=.24,wspace=.3)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/10_coupled_contact_response.{ext}',bbox_inches='tight')
print('Saved contact coupling response figure.')
