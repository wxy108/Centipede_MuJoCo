import json,math,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
j=json.loads((s.ROOT/'audit/common_contact_summary.json').read_text());rows=j['rows'];series=[('cpu_gap_pct','MuJoCo / CPU gap','#2878a8'),('gpu_gap_pct','MuJoCo / GPU gap','#d95f02'),('mu_refinement_pct','MuJoCo half-step','#72ac90'),('cpu_refinement_pct','CPU half-step','#58a7b4'),('gpu_refinement_pct','GPU half-step','#e6b65b')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(8);width=.15;largest=max(r[k] or 0 for r in rows for k,_,_ in series);ceiling=max(.05,largest*1.3)
for i,(key,label,color) in enumerate(series):
    y=[np.nan if r[key] is None else r[key] for r in rows];ax.bar(x+(i-2)*width,y,width,label=label,color=color)
    for k,v in enumerate(y):
        if np.isfinite(v):ax.text(x[k]+(i-2)*width,v+ceiling*.018,f'{v:.3f}',ha='center',fontsize=6.5,rotation=90)
ax.set_xticks(x,[f"{r['wave']} / {r['family']}" for r in rows]);ax.set_xlabel('Wave number / amplitude group');ax.set_ylabel('Relative difference (%)');ax.set_ylim(0,ceiling);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title('Shared compliant contact: measured differences — expanded y scale',fontsize=11)
fig.legend(loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025));fig.subplots_adjust(bottom=.29)
fig.text(.5,.006,'Limits: cross-engine 10%; half-step 5%.'+(' Missing bars are pending.' if j['missing'] else ' All eight fixed gaits are included.'),ha='center',fontsize=8,color='#666')
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/21_common_contact_difference_detail.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps(dict(status=j['status'],maximum_plotted_pct=largest,missing=len(j['missing']))))
