"""Third-grid full-bank check after the first pair failed convergence."""
import concurrent.futures,subprocess,sys,os,json
import shared as s

def run(i):
    name=f'mu_be_rigid_dt15_case{i:02d}_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():return dict(name=name,cancelled_before_start=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    argv=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v6.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt','0.000015625','--command-dt','.0000625','--duration','6','--position-implicit-adapter']
    with (s.ROOT/(name+'.log')).open('w') as f:
        proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env);code=proc.wait()
    return dict(name=name,returncode=code,pid=proc.pid)

if __name__=='__main__':
    receipt=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,i) for i in range(8)]):
            receipt.append(future.result());print(json.dumps(receipt[-1]),flush=True)
            s.write(s.ROOT/'audit/mu_pgs_refinement_execution.json',dict(rows=receipt))
