import json,subprocess,sys
import shared as s
jobs=[
    ('friction_cone_gpu_angle45_000','run_friction_cone_probe.py',['--force-angle','45']),
    ('tgs_friction_angle45_external_each_iter_gpu_000','run_tgs_friction_force_iteration.py',['--force-angle','45','--external-iter','1']),
    ('tgs_drop_external_each_iter_gpu_000','run_tgs_drop_force_iteration.py',['--external-iter','1']),
]
receipt=[]
for name,script,extra in jobs:
    if (s.ROOT/'STOP_SIM2SIM').exists():raise RuntimeError('Stop requested; GPU diagnostic queue cancelled.')
    args=[sys.executable,'-B',str(s.ROOT/script),'--name',name,'--device','cuda:0','--headless']+extra
    with (s.ROOT/(name+'.log')).open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    receipt.append(dict(name=name,returncode=r.returncode));print(json.dumps(receipt[-1]),flush=True)
    if r.returncode:raise RuntimeError(name)
s.write(s.ROOT/'audit/gpu_contact_battery_execution.json',dict(rows=receipt))
