"""Evidence table: missing outcomes stay missing; failed criteria stay failed."""
import json,csv
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s

def result(name,row=0):
    p=s.ROOT/'runs'/name/'results.json'
    if not p.exists():return None
    r=json.loads(p.read_text());v=r['rows'][row]
    return v['D_mm'] if v['finite'] and v['complete_s']==6. else None

def pct(x,y,symmetric=False):
    if x is None or y is None:return None
    return 100*abs(x-y)/max((abs(x)+abs(y))/2 if symmetric else abs(x),10.)

rows=[]
for i,c in enumerate(s.cases('holdout')):
    mu={g:result(f'mu_be_rigid_{g}_case{i:02d}_000') for g in ['dt62','dt31','dt15','dt7']}
    native={k:result(n,i+1) for k,n in {'cpu62':'native_pgs_validation_cpu_dt62_000','cpu31':'native_pgs_validation_cpu_dt31_000','gpu62':'native_pgs_validation_gpu_dt62_000','iter64':'native_pgs_validation_cpu_dt62_iter64_000'}.items()}
    r=dict(case=i,wave=c['wave'],family='AB'[i%2],amplitudes=c['amplitudes'],**{f'mu_{k}_mm':v for k,v in mu.items()},**{f'native_{k}_mm':v for k,v in native.items()})
    r.update(mu_62_to_31_pct=pct(mu['dt62'],mu['dt31']),mu_31_to_15_pct=pct(mu['dt31'],mu['dt15']),mu_15_to_7_pct=pct(mu['dt15'],mu['dt7']),native_62_to_31_pct=pct(native['cpu62'],native['cpu31']),native_16_to_64_iters_pct=pct(native['cpu62'],native['iter64']),cpu_gpu_gap_pct=pct(native['cpu62'],native['gpu62'],True),mu31_cpu62_gap_pct=pct(mu['dt31'],native['cpu62'],True),mu15_cpu62_gap_pct=pct(mu['dt15'],native['cpu62'],True),mu15_gpu62_gap_pct=pct(mu['dt15'],native['gpu62'],True))
    rows.append(r)
summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='UNDER_VALIDATION_NOT_ACCEPTED',rows=rows,thresholds=dict(sim2sim_symmetric_gap_pct=10,step_change_pct=5,minimum_forward_mm=50),gpu_half_step='not performed',note='All fixed eight gaits retained. Single calibration agreement is not acceptance. Missing values are not zero or failures. Native CPU half-step also refines command sampling, as disclosed in amendment 02.')
s.write(s.ROOT/'audit/pgs_candidate_summary.json',summary)
with (s.ROOT/'pgs_validation_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.25),sharex=True,sharey=True)
series=[('mu_dt62_mm','MuJoCo 62.5 µs','#b7d4e5','--','o'),('mu_dt31_mm','MuJoCo 31.25 µs','#348abd','-','o'),('mu_dt15_mm','MuJoCo 15.625 µs','#174a70','-','s'),('native_cpu62_mm','PhysX CPU 62.5 µs','#d95f02','-','^'),('native_gpu62_mm','PhysX GPU 62.5 µs','#b22970',':','D')]
for j,ax in enumerate(axes):
    bank=rows[j::2]
    for key,label,color,ls,marker in series:
        values=np.array([np.nan if r[key] is None else r[key] for r in bank])
        if np.isfinite(values).any():ax.plot([r['wave'] for r in bank],values,label=label,color=color,ls=ls,marker=marker,lw=1.7,ms=5)
    ax.set_title('ABC amplitudes = '+str(bank[0]['amplitudes'])+'°',fontsize=10.5);ax.set_xlim(1.25,4.75);ax.set_xticks([1.5,2.5,3.5,4.5]);ax.set_ylim(0,250);ax.set_yticks(np.arange(0,251,50));ax.set_xlabel('Wave number');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Forward displacement, 2–6 s (mm)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.01));fig.suptitle('Same physical plant and PD — candidate under validation',fontsize=12);fig.subplots_adjust(bottom=.27,top=.81,wspace=.12)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/06_pgs_full_bank.{ext}',bbox_inches='tight')
plt.close(fig)
cols=['mu_62_to_31_pct','mu_31_to_15_pct','native_62_to_31_pct','native_16_to_64_iters_pct','cpu_gpu_gap_pct'];names=['Mu 62.5→31.25 µs','Mu 31.25→15.625 µs','PhysX 62.5→31.25 µs','PhysX iterations 16→64','PhysX CPU / GPU']
fig,ax=plt.subplots(figsize=(10.2,4.4));data=np.array([[np.nan if r[k] is None else r[k] for k in cols] for r in rows]);im=ax.imshow(np.ma.masked_invalid(data),vmin=0,vmax=20,cmap='YlOrRd',aspect='auto');ax.set_xticks(range(len(cols)),names,rotation=18,ha='right',fontsize=9);ax.set_yticks(range(8),[f"w={r['wave']}, {r['family']}" for r in rows]);ax.set_title('Numerical sensitivity (%) — missing tests remain blank')
for y in range(8):
    for x in range(len(cols)):
        ax.text(x,y,'pending' if np.isnan(data[y,x]) else f'{data[y,x]:.2f}',ha='center',va='center',fontsize=9,color='white' if data[y,x]>13 else 'black')
fig.colorbar(im,ax=ax,label='Absolute relative change (%)');fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/07_pgs_numerical_checks.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps([dict(case=r['case'],mu31=r['mu_dt31_mm'],mu15=r['mu_dt15_mm'],cpu62=r['native_cpu62_mm'],gap31=r['mu31_cpu62_gap_pct'],step=r['mu_31_to_15_pct']) for r in rows],indent=2))
