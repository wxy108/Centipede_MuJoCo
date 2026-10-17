import concurrent.futures,json,os,subprocess,sys
import shared as s
def run(i):
    name=f'mu_common_contact_dt15_case{i:02d}_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled=True)
    args=[sys.executable,'-B',str(s.ROOT/'run_mujoco_common_contact.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt','.000015625','--command-dt','.00025','--duration','6','--coupled-drive-inertia']
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (s.ROOT/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    result=dict(name=name,pid=p.pid,returncode=code,command=args);print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,range(8)))
s.write(s.ROOT/'audit/mu_common_contact_refinement_execution.json',dict(rows=rows))
