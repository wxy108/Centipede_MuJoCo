import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.3,4.2));a,b=axes
angle=np.linspace(0,2*np.pi,361)
a.plot(np.cos(angle),np.sin(angle),color='#2878a8',lw=2,label='Circular Coulomb limit')
a.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],color='#d95f02',lw=2,label='Separate tangent limits')
for theta,label in [(0,'0° load'),(np.pi/4,'45° load')]:
    xy=1.2*np.array([np.cos(theta),np.sin(theta)]);a.annotate('',xy=xy,xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#333',lw=1.2));a.text(xy[0]+.03,xy[1]+.06,label,fontsize=9)
a.set_xlim(-1.5,1.6);a.set_ylim(-1.4,1.4);a.set_aspect('equal');a.set_xticks([-1,0,1]);a.set_yticks([-1,0,1]);a.set_xlabel(r'$F_x/(\mu N)$');a.set_ylabel(r'$F_y/(\mu N)$');a.axhline(0,color='#ddd',lw=.7);a.axvline(0,color='#ddd',lw=.7);a.set_title('Same axis coefficient, different force limits')
folder=s.ROOT/'runs/friction_cone_cpu_angle45_000'
for file,color,ls,label in [('trajectory.npz','#d95f02','-','PhysX CPU PGS'),('mu_elliptic.npz','#2878a8','-','MuJoCo circular'),('mu_box_derived.npz','#444',':','MuJoCo derived square')]:
    tr={k:v for k,v in np.load(folder/file).items()};names=tr['joint_names'].tolist();xy=tr['q'][:,[names.index('x'),names.index('y')]]
    b.plot(tr['t']*1000,np.linalg.norm(xy,axis=1)*1e6,color=color,ls=ls,label=label,lw=2)
b.set_xlim(0,10);b.set_ylim(-2,55);b.set_xticks([0,2,4,6,8,10]);b.set_yticks([0,10,20,30,40,50]);b.set_xlabel('Time (ms)');b.set_ylabel('Horizontal motion (µm)');b.set_title('45° load: 1.2 × μN');b.grid(axis='y',alpha=.2);b.set_axisbelow(True)
b.text(.04,.94,'0° check: PhysX 49.406 µm\nMuJoCo square 49.167 µm',transform=b.transAxes,va='top',fontsize=9)
h,l=b.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02));fig.subplots_adjust(bottom=.25,wspace=.36)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/12_friction_limit_probe.{ext}',bbox_inches='tight')
print('Saved independent friction-limit figure.')
