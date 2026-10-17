import json,itertools,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
R=s.ROOT;j=json.loads((R/'audit/common_contact_centered_motion_gates.json').read_text())
keys=[(a,g,b,g,'cross_engine') for g in [31.25,15.625] for a,b in itertools.combinations(['mu','cpu','gpu'],2)]+[(d,31.25,d,15.625,'half_step') for d in ['mu','cpu','gpu']]
labels=[f'{a.upper()}/{b.upper()}\n{ga:g} µs' if kind=='cross_engine' else f'{a.upper()}\nhalf-step' for a,ga,b,gb,kind in keys]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':1.1,'savefig.dpi':250})
fig,axes=plt.subplots(2,1,figsize=(11.5,6.8),sharex=True)
for ax,metric,limit,color,label in [(axes[0],'max_xy_difference_mm',.5,'#2878a8','Maximum XY path difference (mm)'),(axes[1],'max_heading_difference_deg',.25,'#d95f02','Maximum heading difference (deg)')]:
    values=[]
    for a,ga,b,gb,kind in keys:
        group=[r for r in j['rows'] if (r['first_engine'],r['first_grid_us'],r['second_engine'],r['second_grid_us'],r['kind'])==(a,ga,b,gb,kind)]
        values.append(max(r[metric] for r in group) if len(group)==8 else np.nan)
    ax.bar(range(9),values,.62,color=color)
    for x,v in enumerate(values):
        if np.isfinite(v):
            near_limit=limit*.88<v<=limit
            ax.text(x,v-limit*.07 if near_limit else v+limit*.028,f'{v:.4f}',ha='center',fontsize=8,color='white' if near_limit else '#222',va='top' if near_limit else 'bottom')
    ax.axhline(limit,color='#a33232',ls='--',lw=1);ax.text(8.5,limit*1.025,f'limit {limit:g}',ha='left',va='bottom',fontsize=8,color='#a33232');ax.set_ylim(0,limit*1.22);ax.set_xlim(-.6,9.2);ax.set_ylabel(label);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
axes[1].set_xticks(range(9),labels);axes[1].set_xlabel('Backend pair / time step comparison')
fig.suptitle('Centered common-contact motion checks — maximum across all eight fixed gaits',fontsize=11)
fig.text(.5,.012,'Relative 21-body centroid path and initial-relative head–tail chord heading, 2–6 s. Missing bars are pending.' if j['missing'] else 'All 72 complete comparisons. Relative centroid path and head–tail chord heading, 2–6 s.',ha='center',fontsize=8,color='#666')
fig.subplots_adjust(left=.095,right=.95,top=.90,bottom=.16,hspace=.18)
for ext in ['png','pdf','svg']:fig.savefig(R/f'figures/26_common_contact_centered_motion_checks.{ext}',bbox_inches='tight')
print(json.dumps(dict(status=j['status'],missing=len(j['missing']))))
