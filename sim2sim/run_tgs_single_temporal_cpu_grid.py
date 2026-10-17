import json,os,subprocess,sys
import shared as s
rows=[]
for label,dt,vi in [('dt125_vi32',.000125,32),('dt125_vi64',.000125,64),('dt62_vi32',.0000625,32)]:
    name='native_tgs_pos1_cpu_'+label+'_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():break
    args=[sys.executable,'-B',str(s.ROOT/'run_native_v14.py'),'--name',name,'--bank','validation','--device','cpu','--cpu-threads','2','--headless','--duration','6','--dt',str(dt),'--command-dt','.00025','--pos-iters','1','--vel-iters',str(vi)]
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (s.ROOT/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    row=dict(name=name,pid=p.pid,returncode=code,command=args);rows.append(row);print(json.dumps(row),flush=True)
    if code:break
s.write(s.ROOT/'audit/tgs_single_temporal_cpu_execution.json',dict(rows=rows))
