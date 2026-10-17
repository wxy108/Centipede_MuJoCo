import concurrent.futures,json,os,subprocess,sys
import shared as s
jobs=[('native_tgs_ext_validation_cpu_dt62_000',['--dt','.0000625']),('native_tgs_ext_validation_cpu_dt125_iter64_000',['--dt','.000125','--pos-iters','64','--vel-iters','16']),('native_tgs_ext_validation_cpu_dt125_cmd250_000',['--dt','.000125','--command-dt','.00025'])]
def run(job):
    name,extra=job
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled=True)
    args=[sys.executable,'-B',str(s.ROOT/'run_native_v14.py'),'--name',name,'--bank','validation','--device','cpu','--cpu-threads','2','--headless','--duration','6',*extra]
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (s.ROOT/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    result=dict(name=name,returncode=code,pid=p.pid,command=args);print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
s.write(s.ROOT/'audit/cpu_tgs_diagnostic_execution.json',dict(rows=results))
