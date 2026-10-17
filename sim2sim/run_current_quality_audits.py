import concurrent.futures,subprocess,sys,os,json
import shared as s
names=[p.name for p in (s.ROOT/'runs').glob('mu_coupled_rigid_*') if (p/'trajectory.npz').exists()]
names += [n for n in ['native_pgs_validation_cpu_dt62_000','native_pgs_validation_cpu_dt31_000','native_pgs_validation_cpu_dt62_iter64_000','native_pgs_validation_gpu_dt62_000','native_pgs_validation_gpu_dt31_000'] if (s.ROOT/'runs'/n/'trajectory.npz').exists()]
def run(n):
    f=s.ROOT/'runs'/n
    if (f/'quality_audit.json').exists():return dict(name=n,already_audited=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (f/'quality_audit.log').open('w') as log:r=subprocess.run([sys.executable,'-B',str(s.ROOT/'quality_audit_fast.py'),n],stdout=log,stderr=subprocess.STDOUT,env=env)
    return dict(name=n,returncode=r.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:receipt=list(pool.map(run,names))
s.write(s.ROOT/'audit/current_quality_execution.json',dict(rows=receipt));print(json.dumps(receipt,indent=2))
