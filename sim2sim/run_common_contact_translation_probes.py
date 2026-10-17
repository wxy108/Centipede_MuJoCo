import json,os,subprocess,sys
import shared as s
R=s.ROOT;env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';rows=[]
for grid,dt in [(31,.00003125),(15,.000015625)]:
    if (R/'STOP_SIM2SIM').exists():break
    name=f'native_common_contact_translation_cpu_dt{grid}_001';args=[sys.executable,'-B',str(R/'run_native_common_contact_translation_probe.py'),'--name',name,'--bank','translation_probe','--device','cpu','--cpu-threads','2','--dt',str(dt),'--command-dt','.00025','--duration','.5','--headless']
    with (R/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    rows.append(dict(name=name,pid=p.pid,returncode=code,command=args));print(json.dumps(rows[-1]),flush=True)
    if code:break
s.write(R/'audit/common_contact_translation_execution.json',dict(rows=rows))
