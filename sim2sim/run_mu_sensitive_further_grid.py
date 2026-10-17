import concurrent.futures,json,os,subprocess,sys
import shared as s
def run(i):
    name=f'mu_coupled_cmd250_dt7_case{i:02d}_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled=True)
    args=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v8.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt','.0000078125','--command-dt','.00025','--duration','6','--coupled-drive-inertia']
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (s.ROOT/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    result=dict(name=name,returncode=code,pid=p.pid,command=args);print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[1,3]))
s.write(s.ROOT/'audit/mu_sensitive_further_grid_execution.json',dict(rows=rows))
