"""Reproduce the frozen aligned benchmark, with one shared physical definition."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,os,subprocess,sys,uuid
ROOT=Path(__file__).resolve().parent
PYTHON=Path(r'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe')
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['mujoco','isaac-gpu'],required=True);p.add_argument('--name');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
profile=json.loads((ROOT/'OPERATING_PROFILE.json').read_text());manifest=json.loads((ROOT/'REPRODUCTION_MANIFEST.json').read_text())
for relative,expected in manifest['required_sha256'].items():
    path=ROOT/relative
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen benchmark input changed: {relative}')
name=a.name or ('aligned_'+a.engine.replace('-','_')+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
if not name or Path(name).name!=name or any(c in name for c in '/\\:'):raise ValueError('Output name must be a single directory name.')
if a.engine=='isaac-gpu':
    jobs=[(name,[str(PYTHON),'-B',str(ROOT/profile['native_runner']),'--name',name,'--bank','validation','--device',profile['native_device'],'--headless','--duration',str(profile['duration_s']),'--dt',str(profile['native_dt_s']),'--pos-iters',str(profile['native_position_iterations']),'--vel-iters',str(profile['native_velocity_iterations'])])]
else:
    jobs=[]
    for i in range(9):
        run_name=name+('_calibration' if i==0 else f'_case{i-1:02d}')
        argv=[str(PYTHON),'-B',str(ROOT/profile['mujoco_runner']),'--name',run_name,'--bank','calibration' if i==0 else 'holdout','--duration',str(profile['duration_s']),'--dt',str(profile['mujoco_dt_s']),'--command-dt',str(profile['command_dt_s']),'--coupled-drive-inertia']
        if i:argv+=['--case-index',str(i-1)]
        jobs.append((run_name,argv))
if a.dry_run:
    print(json.dumps(dict(profile_status=profile['status'],engine=a.engine,commands=[argv for _,argv in jobs]),indent=2));sys.exit(0)
if profile['status']!='validated_gpu_scope':raise RuntimeError('This candidate is still under validation. No production benchmark was started; see AUDIT_AND_STATUS.zh-CN.md.')
if not PYTHON.exists():raise FileNotFoundError(PYTHON)
if (ROOT/'STOP_SIM2SIM').exists():raise RuntimeError('STOP_SIM2SIM is present; no work was started.')
for run_name,_ in jobs:
    if (ROOT/'runs'/run_name).exists():raise FileExistsError(run_name)
receipt_dir=ROOT/'reproductions'/name;receipt_dir.mkdir(parents=True,exist_ok=False)
(receipt_dir/'profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
def execute(job):
    run_name,argv=job
    if (ROOT/'STOP_SIM2SIM').exists():return dict(name=run_name,cancelled_before_start=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (receipt_dir/(run_name+'.log')).open('w') as log:
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env);code=proc.wait()
    return dict(name=run_name,returncode=code,pid=proc.pid,command=argv)
if a.engine=='isaac-gpu':receipts=[execute(jobs[0])]
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:receipts=list(pool.map(execute,jobs))
(receipt_dir/'execution.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
if any(r.get('returncode',1)!=0 for r in receipts):raise RuntimeError(f'At least one run did not complete; evidence retained in {receipt_dir}')
rows=[]
for run_name,_ in jobs:
    result=json.loads((ROOT/'runs'/run_name/'results.json').read_text())
    if any(r['complete_s']!=profile['duration_s'] or r['D_mm'] is None or not r['finite'] for r in result['rows']):raise RuntimeError(f'Run stopped or did not produce a complete finite score: {run_name}')
    rows.extend(result['rows'])
(receipt_dir/'results.json').write_text(json.dumps(dict(engine=a.engine,rows=rows,runs=[n for n,_ in jobs]),indent=2),encoding='utf-8')
print(str(receipt_dir/'results.json'))
