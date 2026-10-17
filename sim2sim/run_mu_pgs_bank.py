import concurrent.futures,subprocess,sys,os,json
import shared as s
assert (s.ROOT/'NUMERICAL_AMENDMENT_04.md').exists()
def run(task):
    label,dt,i=task;name=f'mu_be_rigid_{label}_case{i:02d}_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled_before_start=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    argv=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v6.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt',str(dt),'--command-dt','.0000625','--duration','6','--position-implicit-adapter']
    with (s.ROOT/(name+'.log')).open('w') as f:p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env);code=p.wait()
    return dict(name=name,returncode=code,pid=p.pid)
if __name__=='__main__':
    jobs=[(label,dt,i) for label,dt in [('dt62',.0000625),('dt31',.00003125)] for i in range(8)];results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):results.append(f.result());print(json.dumps(results[-1]),flush=True);s.write(s.ROOT/'audit/mu_pgs_bank_execution.json',dict(rows=results))
