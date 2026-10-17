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
native={f'{device}{grid}':read(f'native_tgs_ext_validation_{device}_dt{grid}_000') for device in ['cpu','gpu'] for grid in [250,125]}
rows=[]
for i,c in enumerate(s.cases('holdout')):
    mu={g:read(f'mu_coupled_cmd250_{g}_case{i:02d}_000') for g in ['dt31','dt15']}
    r=dict(case=i,wave=c['wave'],family='AB'[i%2],amplitudes=c['amplitudes'],**{f'mu_{k}_mm':dvalue(v,0) for k,v in mu.items()},**{f'native_{k}_mm':dvalue(v,i+1) for k,v in native.items()})
    for v in mu.values():
        if v:assert v['rows'][0]['wave']==c['wave'] and v['rows'][0]['amplitudes']==c['amplitudes'] and v['arguments']['command_dt']==.00025 and v['numerical_effective_inertia_adapter']
    for v in native.values():
        if v:assert v['solver']=='TGS' and v['rows'][i+1]['wave']==c['wave'] and v['rows'][i+1]['amplitudes']==c['amplitudes']
    r.update(cpu_gap_pct=rel(r['mu_dt31_mm'],r['native_cpu250_mm'],True),gpu_gap_pct=rel(r['mu_dt31_mm'],r['native_gpu250_mm'],True),
        mu_refinement_pct=rel(r['mu_dt31_mm'],r['mu_dt15_mm']),cpu_refinement_pct=rel(r['native_cpu250_mm'],r['native_cpu125_mm']),gpu_refinement_pct=rel(r['native_gpu250_mm'],r['native_gpu125_mm']),cpu_gpu_gap_pct=rel(r['native_cpu250_mm'],r['native_gpu250_mm'],True))
    rows.append(r)
missing=[];failed=[]
criteria=[('cpu_gap_pct',10),('gpu_gap_pct',10),('mu_refinement_pct',5),('cpu_refinement_pct',5),('gpu_refinement_pct',5),('cpu_gpu_gap_pct',10)]
for r in rows:
    for key,limit in criteria:
        if r[key] is None:missing.append(f"case {r['case']}: {key}")
        elif r[key]>limit:failed.append(dict(case=r['case'],criterion=key,value_pct=r[key],limit_pct=limit))
    for key in ['mu_dt31_mm','mu_dt15_mm','native_cpu250_mm','native_cpu125_mm','native_gpu250_mm','native_gpu125_mm']:
        if r[key] is not None and r[key]<50:failed.append(dict(case=r['case'],criterion='minimum forward 50 mm',field=key,value_mm=r[key]))
status='FAILED_CRITERIA' if failed else 'PENDING_RESULTS' if missing else 'NUMERICAL_GATES_PASSED_QUALITY_REVIEW_REQUIRED'
summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,rows=rows,missing=missing,failed=failed,criteria=dict(criteria),
    main_profile=dict(mujoco_dt_s=.00003125,native_dt_s=.00025,command_dt_s=.00025,native_solver='TGS 16/4',external_forces_every_iteration=True),source_sha=s.sha(__file__))
s.write(s.ROOT/'audit/tgs_candidate_summary.json',summary)
with (s.ROOT/'tgs_validation_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),sharex=True,sharey=True)
series=[('mu_dt31_mm','MuJoCo 31.25 µs','#2878a8','-','o'),('native_cpu250_mm','PhysX CPU 250 µs','#27986d','-','s'),('native_gpu250_mm','PhysX GPU 250 µs','#d95f02','-','^'),('mu_dt15_mm','MuJoCo 15.625 µs','#2878a8',':','o'),('native_cpu125_mm','PhysX CPU 125 µs','#27986d',':','s'),('native_gpu125_mm','PhysX GPU 125 µs','#d95f02',':','^')]
for j,ax in enumerate(axes):
    bank=rows[j::2]
    for key,label,color,ls,marker in series:
        y=[np.nan if r[key] is None else r[key] for r in bank]
        if np.isfinite(y).any():ax.plot([r['wave'] for r in bank],y,label=label,color=color,ls=ls,marker=marker,lw=1.7,ms=4.5)
    ax.set_title('Body / leg yaw / leg pitch = '+str(bank[0]['amplitudes'])+'°',fontsize=9.5);ax.set_xlim(1.25,4.75);ax.set_xticks([1.5,2.5,3.5,4.5]);ax.set_ylim(0,250);ax.set_yticks(range(0,251,50));ax.set_xlabel('Wave number');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Forward displacement, 2–6 s (mm)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.04));fig.suptitle('Shared physical model and PD — circular friction, TGS external forces each iteration',fontsize=10.5);fig.subplots_adjust(bottom=.30,top=.81,wspace=.12)
fig.text(.5,.007,'Rejected candidate: time-step refinement exceeds the frozen limits' if failed else 'Validation pending' if missing else 'Numerical gates passed; quality review required',ha='center',fontsize=8,color='#a33232' if failed else '#666')
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/14_tgs_full_bank.{ext}',bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4));x=np.arange(8);width=.15
for j,(key,label,color) in enumerate([('cpu_gap_pct','MuJoCo / CPU gap','#2878a8'),('gpu_gap_pct','MuJoCo / GPU gap','#d95f02'),('mu_refinement_pct','MuJoCo half-step','#72ac90'),('cpu_refinement_pct','CPU half-step','#58a7b4'),('gpu_refinement_pct','GPU half-step','#e6b65b')]):
    y=[np.nan if r[key] is None else r[key] for r in rows];ax.bar(x+(j-2)*width,y,width,label=label,color=color)
    for i,v in enumerate(y):
        if np.isfinite(v):ax.text(x[i]+(j-2)*width,v+.1,f'{v:.1f}',ha='center',fontsize=6.5)
ax.axhline(10,color='#a33232',ls='--',lw=1,label='Gap limit 10%');ax.axhline(5,color='#777',ls=':',lw=1,label='Refinement limit 5%');ax.set_xticks(x,[f"{r['wave']} / {r['family']}" for r in rows]);ax.set_xlabel('Wave number / amplitude group');ax.set_ylabel('Relative difference (%)');ax.set_ylim(0,max(12,max(r[k] or 0 for r in rows for k,_ in criteria)+2));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title('Complete eight-gait criteria' + (' — missing bars are pending' if missing else ''),fontsize=11);fig.legend(loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.005));fig.subplots_adjust(bottom=.29)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/15_tgs_acceptance_checks.{ext}',bbox_inches='tight')
plt.close(fig)
print(json.dumps(dict(status=status,missing=len(missing),failed=failed),indent=2))
