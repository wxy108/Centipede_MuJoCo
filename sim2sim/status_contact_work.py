import json
from pathlib import Path
R=Path(__file__).resolve().parent
names=['native_tgs_ext_validation_cpu_dt125_iter64_000','native_tgs_ext_validation_cpu_dt125_cmd250_000','native_tgs_pos1_cpu_dt125_vi32_000','native_tgs_pos1_cpu_dt125_vi64_000','native_tgs_pos1_cpu_dt62_vi32_000','mu_desaxceAA_cmd250_dt125_case03_000','mu_desaxceAA_cmd250_dt62_case03_000']
rows=[]
for name in names:
    f=R/'runs'/name;l=R/(name+'.log');r=dict(name=name,created=f.exists())
    for key,file in [('error','error.json'),('interrupted','interrupted.json')]:
        if (f/file).exists():r[key]=json.loads((f/file).read_text(encoding='utf-8-sig'))
    if (f/'results.json').exists():
        j=json.loads((f/'results.json').read_text());r.update(done=True,D_mm=[v['D_mm'] for v in j['rows']],solver_diagnostics=j.get('contact_solver_diagnostics'))
    elif l.exists():r['recent_log']='\n'.join(l.read_text(encoding='utf-8',errors='replace').splitlines()[-2:])
    rows.append(r)
print(json.dumps(rows,indent=2))
