import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
R=s.ROOT;j=json.loads((R/'audit/common_contact_coordinate_precision.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.1,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.6,4.5));colors=['#2878a8','#d95f02'];xs=np.arange(2)
for e,y in enumerate([0.,1.75]):
    for ax,key,value in [(axes[0],'translation_rows','measured_displacement_mm'),(axes[1],'gait_rows','max_xy_difference_mm')]:
        values=[next((r[value] for r in j[key] if r['global_y_m']==y and r['grid_us']==g),np.nan) for g in [31.25,15.625]]
        ax.bar(xs+(e-.5)*.32,values,.32,color=colors[e],label=f'Initial global Y = {y:g} m')
        for x,v in zip(xs+(e-.5)*.32,values):
            if np.isfinite(v):ax.text(x,v+.05,f'{v:.4f}',ha='center',fontsize=8)
for ax in axes:
    ax.set_xticks(xs,['31.25','15.625']);ax.set_xlabel('Physics time step (µs)');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
axes[0].set_ylim(0,2);axes[0].set_yticks(np.arange(0,2.01,.5));axes[0].axhline(1.5,color='#666',ls='--',lw=1,label='Analytical translation: 1.5 mm');axes[0].set_ylabel('Measured Y translation (mm)');axes[0].set_title('Constant 3 mm/s, 0.5 s\nNo gravity or contact; no joint motion',fontsize=10)
axes[1].set_ylim(0,4);axes[1].set_yticks(range(5));axes[1].set_ylabel('Maximum XY path difference from MuJoCo (mm)');axes[1].set_title('Full six-second gait: wave 4.5 / A\nRelative centroid path over 2–6 s',fontsize=10)
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.025))
fig.suptitle('CPU world-origin sensitivity — same physical plant, PD and contact law',fontsize=11)
fig.text(.5,.005,'Selected causal diagnostics; all four paired gait and translation checks completed. Original bank retained.' if len(j['gait_rows'])==4 and len(j['translation_rows'])==4 else 'Selected causal diagnostics; the frozen full-bank results are retained. Missing bars are pending.',ha='center',fontsize=8,color='#666')
fig.subplots_adjust(top=.77,bottom=.27,wspace=.27)
for ext in ['png','pdf','svg']:fig.savefig(R/f'figures/22_common_contact_coordinate_precision.{ext}',bbox_inches='tight')
print('figures/22_common_contact_coordinate_precision.png')
