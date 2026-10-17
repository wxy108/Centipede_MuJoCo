"""Reproduce only an accepted, checksum-frozen shared-contact benchmark."""
from pathlib import Path
import argparse,concurrent.futures,datetime,hashlib,json,os,subprocess,sys,uuid
ROOT=Path(__file__).resolve().parent
PYTHON=Path(r'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe')
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['mujoco','isaac-cpu','isaac-gpu'],required=True);p.add_argument('--name');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
accepted=ROOT/'COMMON_CONTACT_OPERATING_PROFILE.json'
candidate=ROOT/'CENTERED_COMMON_CONTACT_CANDIDATE_PROFILE.json'
profile=json.loads((accepted if accepted.exists() else candidate if candidate.exists() else ROOT/'COMMON_CONTACT_CANDIDATE_PROFILE.json').read_text())
if profile['status']=='validated_common_contact_cpu_gpu':
    manifest=json.loads((ROOT/'COMMON_CONTACT_REPRODUCTION_MANIFEST.json').read_text())
    for relative,expected in manifest['required_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen input changed: {relative}')
    if hashlib.sha256((ROOT/'audit/runtime_versions.json').read_bytes()).hexdigest()!=manifest['runtime_versions_sha256']:raise RuntimeError('Frozen runtime version record changed.')
    from verify_alignment_runtime import verify
    verify(PYTHON,check_gpu_driver=a.engine=='isaac-gpu')
name=a.name or ('common_contact_'+a.engine.replace('-','_')+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
if not name or Path(name).name!=name or any(c in name for c in '/\\:'):raise ValueError('Output name must be a single directory name.')
if a.engine.startswith('isaac-'):
    device='cpu' if a.engine=='isaac-cpu' else 'cuda:0'
    jobs=[(name,[str(PYTHON),'-B',str(ROOT/profile['native_runner']),'--name',name,'--bank','validation','--device',device,'--headless','--duration',str(profile['duration_s']),'--dt',str(profile['native_dt_s']),'--command-dt',str(profile['command_dt_s']),'--pos-iters',str(profile['native_position_iterations']),'--vel-iters',str(profile['native_velocity_iterations']),'--cpu-threads',str(profile['native_cpu_threads'])])]
else:
    jobs=[]
    for i in range(9):
        run_name=name+('_calibration' if i==0 else f'_case{i-1:02d}')
        argv=[str(PYTHON),'-B',str(ROOT/profile['mujoco_runner']),'--name',run_name,'--bank','calibration' if i==0 else 'holdout','--duration',str(profile['duration_s']),'--dt',str(profile['mujoco_dt_s']),'--command-dt',str(profile['command_dt_s']),'--coupled-drive-inertia']
        if i:argv+=['--case-index',str(i-1)]
        jobs.append((run_name,argv))
if a.dry_run:
    print(json.dumps(dict(profile_status=profile['status'],contact_contract=profile['contact_contract'],engine=a.engine,commands=[v for _,v in jobs]),indent=2));sys.exit(0)
if profile['status']!='validated_common_contact_cpu_gpu':raise RuntimeError('Shared-contact validation is incomplete. No simulation was started; see REPORT.md.')
if (ROOT/'STOP_SIM2SIM').exists():raise RuntimeError('STOP_SIM2SIM is present; no work was started.')
for run_name,_ in jobs:
    if (ROOT/'runs'/run_name).exists():raise FileExistsError(run_name)
receipt=ROOT/'reproductions'/name;receipt.mkdir(parents=True,exist_ok=False)
(receipt/'profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
def execute(job):
    run_name,argv=job
    if (ROOT/'STOP_SIM2SIM').exists():return dict(name=run_name,cancelled_before_start=True)
    env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    with (receipt/(run_name+'.log')).open('w') as log:proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env);code=proc.wait()
    return dict(name=run_name,returncode=code,pid=proc.pid,command=argv)
if a.engine.startswith('isaac-'):receipts=[execute(jobs[0])]
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(execute,jobs))
(receipt/'execution.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
if any(r.get('returncode',1)!=0 for r in receipts):raise RuntimeError(f'A run did not finish; evidence retained in {receipt}')
rows=[]
for run_name,_ in jobs:
    result=json.loads((ROOT/'runs'/run_name/'results.json').read_text())
    if any(r['complete_s']!=6 or r['D_mm'] is None or not r['finite'] for r in result['rows']):raise RuntimeError(f'No complete finite score: {run_name}')
    rows.extend(result['rows'])
(receipt/'results.json').write_text(json.dumps(dict(engine=a.engine,physical_contact_model='COMMON_CONTACT_V2.json',rows=rows,runs=[n for n,_ in jobs]),indent=2),encoding='utf-8')
print(str(receipt/'results.json'))
