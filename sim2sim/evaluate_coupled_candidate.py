import csv,json
from datetime import datetime,timezone
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s

def read(name):
    p=s.ROOT/'runs'/name/'results.json'
    return json.loads(p.read_text()) if p.exists() else None
def dvalue(result,e):
    if result is None:return None
    r=result['rows'][e]
    return r['D_mm'] if r['finite'] and r['complete_s']==6 else None
def rel(x,y,sym=False):
    if x is None or y is None:return None
    return 100*abs(x-y)/max((abs(x)+abs(y))/2 if sym else abs(x),10.)
native={k:read(name) for k,name in {'gpu62':'native_pgs_validation_gpu_dt62_000','gpu31':'native_pgs_validation_gpu_dt31_000','cpu62':'native_pgs_validation_cpu_dt62_000','cpu31':'native_pgs_validation_cpu_dt31_000','cpu64iters':'native_pgs_validation_cpu_dt62_iter64_000'}.items()}
rows=[]
for i,c in enumerate(s.cases('holdout')):
    mu={g:read(f'mu_coupled_rigid_{g}_case{i:02d}_000') for g in ['dt62','dt31','dt15']}
    r=dict(case=i,wave=c['wave'],family='AB'[i%2],amplitudes=c['amplitudes'],**{f'mu_{k}_mm':dvalue(v,0) for k,v in mu.items()},**{f'native_{k}_mm':dvalue(v,i+1) for k,v in native.items()})
    for key,v in mu.items():
        if v:assert v['rows'][0]['wave']==c['wave'] and v['rows'][0]['amplitudes']==c['amplitudes'] and v['numerical_effective_inertia_adapter']
    r.update(primary_gap_pct=rel(r['mu_dt31_mm'],r['native_gpu62_mm'],True),mu_refinement_pct=rel(r['mu_dt31_mm'],r['mu_dt15_mm']),gpu_refinement_pct=rel(r['native_gpu62_mm'],r['native_gpu31_mm']),cpu_gpu_gap_pct=rel(r['native_cpu62_mm'],r['native_gpu62_mm'],True),mu_cpu_gap_pct=rel(r['mu_dt31_mm'],r['native_cpu62_mm'],True),cpu_step_change_pct=rel(r['native_cpu62_mm'],r['native_cpu31_mm']),cpu_iteration_change_pct=rel(r['native_cpu62_mm'],r['native_cpu64iters_mm']))
    rows.append(r)
missing=[];failed=[]
for r in rows:
    for key,limit in [('primary_gap_pct',10),('mu_refinement_pct',5),('gpu_refinement_pct',5)]:
        if r[key] is None:missing.append(f"case {r['case']}: {key}")
        elif r[key]>limit:failed.append(dict(case=r['case'],criterion=key,value_pct=r[key],limit_pct=limit))
    for key in ['mu_dt31_mm','mu_dt15_mm','native_gpu62_mm','native_gpu31_mm']:
        if r[key] is not None and r[key]<50:failed.append(dict(case=r['case'],criterion='minimum forward 50 mm',field=key,value_mm=r[key]))
status='FAILED_CRITERIA' if failed else 'PENDING_REFINEMENT' if missing else 'NUMERICAL_GATES_PASSED_GPU_SCOPE_QUALITY_REVIEW_REQUIRED'
summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,rows=rows,missing=missing,failed=failed,main_profile=dict(mujoco_dt_s=.00003125,native_gpu_dt_s=.0000625,command_dt_s=.0000625,mujoco_integration='coupled M+hD+h²K, physical armature remains zero',native_solver='PGS 16/4'),scope='Native GPU benchmark only. Native CPU differences are retained and must not be hidden or substituted. Original RL training pipeline has not been migrated.',source_sha=s.sha(__file__))
s.write(s.ROOT/'audit/coupled_candidate_summary.json',summary)
with (s.ROOT/'coupled_validation_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),sharex=True,sharey=True)
series=[('mu_dt31_mm','MuJoCo 31.25 µs','#2878a8','-','o'),('native_gpu62_mm','PhysX GPU 62.5 µs','#d95f02','-','^'),('mu_dt15_mm','MuJoCo 15.625 µs','#2878a8',':','s'),('native_gpu31_mm','PhysX GPU 31.25 µs','#d95f02',':','D'),('native_cpu62_mm','PhysX CPU (diagnostic)','#9a9a9a','--','x')]
for j,ax in enumerate(axes):
    bank=rows[j::2]
    for key,label,color,ls,marker in series:
        values=np.array([np.nan if r[key] is None else r[key] for r in bank])
        if np.isfinite(values).any():ax.plot([r['wave'] for r in bank],values,label=label,color=color,ls=ls,marker=marker,lw=1.8,ms=5)
    ax.set_title('Body / leg yaw / leg pitch = '+str(bank[0]['amplitudes'])+'°',fontsize=9.5);ax.set_xlim(1.25,4.75);ax.set_xticks([1.5,2.5,3.5,4.5]);ax.set_ylim(0,250);ax.set_yticks(np.arange(0,251,50));ax.set_xlabel('Wave number');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Forward displacement, 2–6 s (mm)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005));fig.suptitle('Shared physical model and PD — coupled integration',fontsize=12);fig.subplots_adjust(bottom=.27,top=.81,wspace=.12)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/08_coupled_full_bank.{ext}',bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(10.2,3.8));x=np.arange(8);width=.25
for j,(key,label,color) in enumerate([('primary_gap_pct','MuJoCo / GPU gap','#2878a8'),('mu_refinement_pct','MuJoCo half-step change','#60aa80'),('gpu_refinement_pct','GPU half-step change','#e49348')]):
    y=[np.nan if r[key] is None else r[key] for r in rows];ax.bar(x+(j-1)*width,y,width,label=label,color=color)
    for i,v in enumerate(y):
        if np.isfinite(v):ax.text(x[i]+(j-1)*width,v+.12,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
ax.axhline(10,color='#a33232',ls='--',lw=1,label='Gap limit 10%');ax.axhline(5,color='#777',ls=':',lw=1,label='Refinement limit 5%');ax.set_xticks(x,[f"{r['wave']} / {r['family']}" for r in rows]);ax.set_xlabel('Wave number / amplitude group');ax.set_ylabel('Relative difference (%)');ax.set_ylim(0,max(12,max([r[k] or 0 for r in rows for k in ['primary_gap_pct','mu_refinement_pct','gpu_refinement_pct']])+2));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title('All eight fixed gaits' + (' — missing refinement bars are pending' if missing else ' — complete numerical checks'),fontsize=11);fig.legend(loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005));fig.subplots_adjust(bottom=.29)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/09_coupled_acceptance_checks.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps(dict(status=status,missing=len(missing),failed=failed,max_primary_gap_pct=max(r['primary_gap_pct'] or 0 for r in rows)),indent=2))
