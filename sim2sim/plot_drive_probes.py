import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
fig,axs=plt.subplots(1,2,figsize=(9.4,4),sharex=True,sharey=True,layout='constrained');rows=[]
for ax,(label,step) in zip(axs,[('dt250',250),('dt62',62.5)]):
    f=s.ROOT/'runs'/f'drive_probe_actual_cpu_{label}_000';tr=np.load(f/'paired_trajectory.npz');meta=json.loads((f/'comparison.json').read_text());j=next(i for i,r in enumerate(meta['rows']) if r['family']=='tip');time=(tr['t']-.01)*1000
    for y,color,style,name in [(tr['ideal'][:,j],'#333333','--','Analytical ODE'),(tr['mu'][:,j,0],'#237f8d','-','MuJoCo'),(tr['native'][:,j],'#d95f02','-','PhysX CPU')]:ax.plot(time,np.degrees(y),style,color=color,label=name,lw=1.5)
    ax.set_xlim(-.1,2);ax.set_ylim(-.05,1.15);ax.set_xticks([0,.5,1,1.5,2]);ax.set_xlabel('Time after target step (ms)');ax.grid(axis='y',alpha=.15);ax.set_title(f'Physics step = {step:g} µs');ax.legend(fontsize=8)
    mask=(time>=0)&(time<=5);delta=tr['mu'][mask,j,0]-tr['native'][mask,j]
    rows.append(dict(dt_us=step,tip_inertia=meta['rows'][j]['inertia'],max_transient_error_deg=float(np.degrees(np.abs(delta).max())),first_5ms_nrmse=float(np.sqrt(np.mean(delta**2))/np.radians(1.)),full_window_nrmse=meta['rows'][j]['mu_native_nrmse']))
axs[0].set_ylabel('Passive tip probe angle (deg)');s.write(s.ROOT/'audit/tip_transient_comparison.json',dict(rows=rows))
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/'figures'/('04_tip_drive_transient.'+ext),dpi=240)
print(json.dumps(rows,indent=2))
