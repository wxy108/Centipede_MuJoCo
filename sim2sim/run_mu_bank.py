"""Bounded local CPU worker pool; every case has an isolated MuJoCo model and evidence directory."""
from pathlib import Path
import concurrent.futures,subprocess,sys,os,json
import shared as s
assert (s.ROOT/'VALIDATION_FREEZE_01.json').exists()
def run(task):
    label,dt,i=task;name=f'mu_holdout_{label}_case{i:02d}_000'
    log=s.ROOT/(name+'.log');env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    argv=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v2.py'),'--name',name,'--bank','holdout','--case-index',str(i),'--dt',str(dt),'--duration','6']
    with log.open('w') as f:p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env);code=p.wait()
    return dict(name=name,returncode=code,log=str(log),pid=p.pid)
if __name__=='__main__':
    jobs=[(label,dt,i) for label,dt in [('dt62',.0000625),('dt31',.00003125)] for i in range(8)];results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):results.append(f.result());print(json.dumps(results[-1]),flush=True);s.write(s.ROOT/'audit/mu_bank_execution.json',dict(rows=results))
