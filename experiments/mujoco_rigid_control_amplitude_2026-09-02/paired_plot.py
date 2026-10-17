"""Plot unsmoothed paired flat traces from the stored simulations."""
import json
from pathlib import Path
import engine as e
np=e.np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.loads((e.DATA/'paired_ablation.json').read_text())
fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
for col,w in enumerate((2.5,3.)):
    for topology,color in (('flexible','#e67700'),('rigid','#007ea8')):
        r=next(r for r in rows if r['terrain']=='flat' and r['task']['wave']==w and r['topology']==topology)
        a=np.load(Path(r['record_path']).with_suffix('.npz'));t=a['t'];good=t>=1.5
        center=a['xpos'].mean(axis=1)
        dx=e.fx.opt.base.FORWARD_SIGN*(center[:,0]-center[np.flatnonzero(good)[0],0])*1000
        wr=np.radians(a['worldroll']);roll=np.degrees(np.arctan2(np.sin(wr).mean(axis=1),np.cos(wr).mean(axis=1)))
        axes[0,col].plot(t[good],dx[good],color=color,label=topology)
        axes[1,col].plot(t[good],roll[good],color=color,lw=.9)
        force=a['force'];tf=np.arange(len(force))*r['dt'];fine=tf>=1.5
        bw=0.0025360310917800356*9.81
        axes[2,col].plot(tf[fine],force[fine].sum(axis=1)/bw,color=color,lw=.65,alpha=.85)
    axes[0,col].set_title(f'Flat, wave={w:g}; identical amplitudes and PD')
    axes[0,col].legend();axes[2,col].set_xlabel('Time (s)')
    for row in range(3):axes[row,col].grid(alpha=.2)
for row,label in enumerate(('Forward displacement (mm)','Coherent body roll (deg)','Total ground normal force / body weight')):axes[row,0].set_ylabel(label)
fig.suptitle('Flexible tip vs fully rigid distal leg — raw traces, no smoothing',fontsize=14)
fig.tight_layout();out=e.OUT/'figures';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'paired_flat_raw_traces.png',dpi=160)
print(str(out/'paired_flat_raw_traces.png'))
