import concurrent.futures,json,os,subprocess,sys
import shared as s
names=[p.name for p in (s.ROOT/'runs').iterdir() if p.name.startswith(('native_tgs_pos1_','mu_common_contact_dt','native_common_contact_validation_')) and (p/'results.json').exists() and (p/'trajectory.npz').exists() and not (p/'quality_audit.json').exists()]
def run(name):
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (s.ROOT/'runs'/name/'quality_audit.log').open('w') as log:p=subprocess.run([sys.executable,'-B',str(s.ROOT/'quality_audit_fast.py'),name],stdout=log,stderr=subprocess.STDOUT,env=env)
    return dict(name=name,returncode=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,names))
s.write(s.ROOT/'audit/additional_quality_execution_latest.json',dict(rows=rows));print(json.dumps(rows))
