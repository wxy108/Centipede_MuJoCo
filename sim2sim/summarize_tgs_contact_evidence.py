import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
drop_names=['contact_drop_slide_cpu_dt62_000','tgs_drop_external_each_iter_cpu_000','tgs_drop_external_each_iter_gpu_000']
drop=[]
for name in drop_names:
    f=s.ROOT/'runs'/name;meta=json.loads((f/'results.json').read_text());tr={k:v for k,v in np.load(f/'trajectory.npz').items()};ix=tr['joint_names'].tolist().index('x')
    drop.append(dict(name=name,final_x_um=meta['final_x']*1e6,minimum_vx_mm_s=float(tr['v'][:,ix].min()*1000),final_vx_mm_s=float(tr['v'][-1,ix]*1000)))
friction=[]
for name in ['friction_cone_cpu_angle45_000','friction_cone_gpu_angle45_000','tgs_friction_angle45_default_cpu_000','tgs_friction_angle45_external_each_iter_gpu_000']:
    r=json.loads((s.ROOT/'runs'/name/'results.json').read_text());friction.append(dict(name=name,final_xy_um=[r['final_x']*1e6,r['final_y']*1e6]))
s.write(s.ROOT/'audit/tgs_contact_evidence.json',dict(drop=drop,friction=friction,source_sha=s.sha(__file__)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
for name,color,ls,label in zip(drop_names,['#c45346','#27986d','#d95f02'],['--','-',':'],['TGS default external forces','TGS each iteration, CPU','TGS each iteration, GPU']):
    tr={k:v for k,v in np.load(s.ROOT/'runs'/name/'trajectory.npz').items()};ix=tr['joint_names'].tolist().index('x');axes[0].plot(tr['t']*1000,tr['v'][:,ix]*1000,color=color,ls=ls,label=label,lw=1.7)
axes[0].set_xlim(5,9);axes[0].set_ylim(-15,25);axes[0].set_xticks([5,6,7,8,9]);axes[0].set_xlabel('Time (ms)');axes[0].set_ylabel('Horizontal velocity (mm/s)');axes[0].set_title('Impact response with identical physical inputs');axes[0].axhline(0,color='#888',lw=.6)
series=[('friction_cone_cpu_angle45_000','trajectory.npz','#777','--','PGS CPU / square limit'),('friction_cone_gpu_angle45_000','trajectory.npz','#aaa',':','PGS GPU / square limit'),('tgs_friction_angle45_default_cpu_000','trajectory.npz','#27986d','-','TGS CPU / circular limit'),('tgs_friction_angle45_external_each_iter_gpu_000','trajectory.npz','#d95f02',':','TGS GPU / circular limit'),('friction_cone_cpu_angle45_000','mu_elliptic.npz','#2878a8','-','MuJoCo / circular limit')]
for name,file,color,ls,label in series:
    tr={k:v for k,v in np.load(s.ROOT/'runs'/name/file).items()};names=tr['joint_names'].tolist();xy=tr['q'][:,[names.index('x'),names.index('y')]];axes[1].plot(tr['t']*1000,np.linalg.norm(xy,axis=1)*1e6,color=color,ls=ls,label=label,lw=1.5)
axes[1].set_xlim(0,10);axes[1].set_ylim(-2,55);axes[1].set_xticks([0,2,4,6,8,10]);axes[1].set_yticks([0,10,20,30,40,50]);axes[1].set_xlabel('Time (ms)');axes[1].set_ylabel('Horizontal motion (µm)');axes[1].set_title('45° force, magnitude 1.2 × μN')
for a in axes:a.grid(axis='y',alpha=.2);a.set_axisbelow(True);a.legend(loc='upper left',bbox_to_anchor=(0,-.23),frameon=False,fontsize=8)
fig.subplots_adjust(bottom=.37,wspace=.3)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/13_tgs_contact_evidence.{ext}',bbox_inches='tight')
print(json.dumps(dict(drop=drop,friction=friction),indent=2))
