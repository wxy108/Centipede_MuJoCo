import concurrent.futures,json,os,subprocess,sys
import shared as s
jobs=[(g,dt,i) for g,dt in [('dt31',.00003125),('dt15',.000015625)] for i in range(8)]
def run(job):
    g,dt,i=job;name=f'mu_coupled_cmd250_{g}_case{i:02d}_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled=True)
    if (s.ROOT/'runs'/name).exists():raise FileExistsError(name)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    args=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v8.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt',str(dt),'--command-dt','.00025','--duration','6','--coupled-drive-inertia']
    with (s.ROOT/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    print(json.dumps(dict(name=name,returncode=code,pid=p.pid)),flush=True)
    return dict(name=name,returncode=code,pid=p.pid,command=args)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipt=list(pool.map(run,jobs))
s.write(s.ROOT/'audit/mu_command250_bank_execution.json',dict(rows=receipt))
