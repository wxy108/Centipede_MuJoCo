import concurrent.futures,subprocess,sys,os,json
import shared as s
def run(task):
    label,dt,i=task;name=f'mu_coupled_rigid_{label}_case{i:02d}_000';folder=s.ROOT/'runs'/name
    if folder.exists():return dict(name=name,already_dispatched=True)
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled_before_start=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    argv=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v8.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt',str(dt),'--command-dt','.0000625','--duration','6','--coupled-drive-inertia']
    with (s.ROOT/(name+'.log')).open('w') as f:
        proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env);code=proc.wait()
    return dict(name=name,returncode=code,pid=proc.pid)
if __name__=='__main__':
    jobs=[(label,dt,i) for label,dt in [('dt62',.0000625),('dt31',.00003125)] for i in range(8)];receipt=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            receipt.append(future.result());print(json.dumps(receipt[-1]),flush=True);s.write(s.ROOT/'audit/mu_coupled_bank_execution.json',dict(rows=receipt))
