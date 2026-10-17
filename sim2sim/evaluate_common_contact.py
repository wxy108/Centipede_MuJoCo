"""Frozen eight-gait gates; missing or failed runs never count as passes."""
import csv,json
from datetime import datetime,timezone
import numpy as np
import shared as s
R=s.ROOT
def read(name):
    f=R/'runs'/name/'results.json'
    return json.loads(f.read_text()) if f.exists() else None
def value(j,e,c):
    if j is None:return None
    r=j['rows'][e]
    assert r['wave']==c['wave'] and r['amplitudes']==c['amplitudes']
    assert 'COMMON_CONTACT_V2' in j['contact_model'] and j['arguments']['command_dt']==.00025
    return r['D_mm'] if r['finite'] and r['complete_s']==6 else None
def rel(x,y,symmetric=False):
    if x is None or y is None:return None
    return 100*abs(x-y)/max((abs(x)+abs(y))/2 if symmetric else abs(x),10.)
native={f'{d}_dt{g}':read(f'native_common_contact_validation_{d}_dt{g}_000') for d in ['cpu','gpu'] for g in [31,15]}
rows=[];missing=[];failed=[]
criteria={f'{d}_gap_pct':10 for d in ['cpu','gpu']}
criteria.update({f'{d}_refinement_pct':5 for d in ['mu','cpu','gpu']});criteria['cpu_gpu_gap_pct']=10
for i,c in enumerate(s.cases('holdout')):
    mu={f'mu_dt{g}_mm':value(read(f'mu_common_contact_dt{g}_case{i:02d}_000'),0,c) for g in [31,15]}
    r=dict(case=i,wave=c['wave'],family='AB'[i%2],**mu,**{f'{k}_mm':value(v,i+1,c) for k,v in native.items()})
    for d in ['cpu','gpu']:r[f'{d}_gap_pct']=rel(r['mu_dt31_mm'],r[f'{d}_dt31_mm'],True)
    for d in ['mu','cpu','gpu']:r[f'{d}_refinement_pct']=rel(r[f'{d}_dt31_mm'],r[f'{d}_dt15_mm'])
    r['cpu_gpu_gap_pct']=rel(r['cpu_dt31_mm'],r['gpu_dt31_mm'],True)
    for k,limit in criteria.items():
        if r[k] is None:missing.append(dict(case=i,criterion=k))
        elif r[k]>limit:failed.append(dict(case=i,criterion=k,value_pct=r[k],limit_pct=limit))
    for k,v in list(r.items()):
        if k.endswith('_mm') and v is not None and v<50:failed.append(dict(case=i,criterion=k,value_mm=v,minimum_mm=50))
    rows.append(r)
status='FAILED_CRITERIA' if failed else 'PENDING_RESULTS' if missing else 'NUMERICAL_GATES_PASSED_QUALITY_REVIEW_REQUIRED'
summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=status,physical_model='COMMON_CONTACT_V2.json',scope='New shared compliant contact law; does not validate the original native rigid contacts',criteria=criteria,rows=rows,missing=missing,failed=failed,main_dt_s=.00003125,refinement_dt_s=.000015625,command_dt_s=.00025,source_sha=s.sha(__file__))
s.write(R/'audit/common_contact_summary.json',summary)
with (R/'common_contact_validation_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(dict(status=status,missing=len(missing),failed=failed,main_mu_mm=[r['mu_dt31_mm'] for r in rows])))
