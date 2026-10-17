from pathlib import Path
import concurrent.futures,subprocess,sys,os,json
import shared as s
assert (s.ROOT/'NUMERICAL_AMENDMENT_02.md').exists()
def run(task):
    label,dt,i=task;bank='calibration' if i<0 else 'holdout';name=f'mu_sharedclock_{label}_'+('calibration' if i<0 else f'case{i:02d}')+'_000'
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    argv=[sys.executable,'-B',str(s.ROOT/'run_mujoco_v4.py'),'--name',name,'--bank',bank,'--dt',str(dt),'--command-dt','.0000625','--duration','6']
    if i>=0:argv+=['--case-index',str(i)]
    with (s.ROOT/(name+'.log')).open('w') as f:p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env);code=p.wait()
    return dict(name=name,returncode=code,pid=p.pid)
if __name__=='__main__':
    jobs=[(label,dt,i) for label,dt in [('dt15',.000015625),('dt7',.0000078125)] for i in [-1,*range(8)]];results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):results.append(f.result());print(json.dumps(results[-1]),flush=True);s.write(s.ROOT/'audit/mu_refinement_execution.json',dict(rows=results))
