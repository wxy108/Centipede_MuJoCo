import subprocess,sys,os,json
import shared as s
env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
for p in sorted((s.ROOT/'runs').glob('mu_holdout_*/trajectory.npz'))+[s.ROOT/'runs/native_calibration_gpu_dt125_000/trajectory.npz']:
    if (p.parent/'quality_audit.json').exists():continue
    with (p.parent/'quality_audit.log').open('w') as f:r=subprocess.run([sys.executable,'-B',str(s.ROOT/'quality_audit.py'),p.parent.name],stdout=f,stderr=subprocess.STDOUT,env=env)
    print(json.dumps(dict(run=p.parent.name,returncode=r.returncode)),flush=True)
