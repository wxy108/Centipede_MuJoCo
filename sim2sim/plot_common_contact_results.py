import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
R=s.ROOT;j=json.loads((R/'audit/common_contact_summary.json').read_text());rows=j['rows']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6),sharex=True,sharey=True)
series=[('mu_dt31_mm','MuJoCo 31.25 µs','#2878a8','-','o'),('cpu_dt31_mm','PhysX CPU 31.25 µs','#27986d','-','s'),('gpu_dt31_mm','PhysX GPU 31.25 µs','#d95f02','-','^'),('mu_dt15_mm','MuJoCo 15.625 µs','#2878a8',':','o'),('cpu_dt15_mm','PhysX CPU 15.625 µs','#27986d',':','s'),('gpu_dt15_mm','PhysX GPU 15.625 µs','#d95f02',':','^')]
for family,ax in enumerate(axes):
    bank=rows[family::2]
    for key,label,color,ls,marker in series:
        values=[np.nan if r[key] is None else r[key] for r in bank]
        ax.plot([r['wave'] for r in bank],values,label=label,color=color,ls=ls,marker=marker,fillstyle='none' if '15' in key else 'full',lw=1.65,ms=5)
    ax.set_title(['Body / leg yaw / leg pitch = 32 / 18 / 26°','Body / leg yaw / leg pitch = 16 / 72 / 24°'][family],fontsize=9.5)
    ax.set_xlim(1.25,4.75);ax.set_xticks([1.5,2.5,3.5,4.5]);ax.set_ylim(0,250);ax.set_yticks(range(0,251,50));ax.set_xlabel('Wave number');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Forward displacement, 2–6 s (mm)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.038))
fig.suptitle('Shared compliant contact — identical physical plant, PD and force law',fontsize=11)
fig.text(.5,.007,'Incomplete validation: missing curves are pending' if j['missing'] else 'All eight fixed gaits; independent half-step checks',ha='center',fontsize=8,color='#666');fig.subplots_adjust(bottom=.29,top=.82,wspace=.12)
for ext in ['png','pdf','svg']:fig.savefig(R/f'figures/19_common_contact_full_bank.{ext}',bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.3));x=np.arange(8);width=.15
for i,(key,label,color) in enumerate([('cpu_gap_pct','MuJoCo / CPU gap','#2878a8'),('gpu_gap_pct','MuJoCo / GPU gap','#d95f02'),('mu_refinement_pct','MuJoCo half-step','#72ac90'),('cpu_refinement_pct','CPU half-step','#58a7b4'),('gpu_refinement_pct','GPU half-step','#e6b65b')]):
    y=[np.nan if r[key] is None else r[key] for r in rows];ax.bar(x+(i-2)*width,y,width,label=label,color=color)
    for k,v in enumerate(y):
        if np.isfinite(v):ax.text(x[k]+(i-2)*width,v+.13,f'{v:.2f}',ha='center',fontsize=6.3,rotation=90)
ax.axhline(10,color='#a33232',ls='--',lw=1,label='Gap limit 10%');ax.axhline(5,color='#777',ls=':',lw=1,label='Refinement limit 5%');ax.set_xticks(x,[f"{r['wave']} / {r['family']}" for r in rows]);ax.set_xlabel('Wave number / amplitude group');ax.set_ylabel('Relative difference (%)');ax.set_ylim(0,max(12,max(r[k] or 0 for r in rows for k in j['criteria'])+2));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title('Shared compliant-contact acceptance checks'+(' — incomplete' if j['missing'] else ''),fontsize=11);fig.legend(loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005));fig.subplots_adjust(bottom=.27)
for ext in ['png','pdf','svg']:fig.savefig(R/f'figures/20_common_contact_acceptance_checks.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps(dict(status=j['status'],missing=len(j['missing']))))
