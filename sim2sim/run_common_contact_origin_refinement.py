import os,subprocess,sys
import shared as s
R=s.ROOT
if (R/'STOP_SIM2SIM').exists():raise SystemExit('Stopped before start')
name='native_common_contact_origin_probe_cpu_dt15_000'
args=[sys.executable,'-B',str(R/'run_native_common_contact_origin_probe.py'),'--name',name,'--bank','origin_probe','--device','cpu','--cpu-threads','2','--dt','.000015625','--command-dt','.00025','--duration','6','--headless']
env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
with (R/(name+'.log')).open('w') as log:
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
s.write(R/'audit/common_contact_origin_refinement_execution.json',dict(name=name,pid=p.pid,returncode=code,command=args))
