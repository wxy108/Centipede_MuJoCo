"""Bounded owned GPU queue; no concurrent GPU physics workers."""
import ctypes,json,os,subprocess,sys,numpy as np
import shared as s
R=s.ROOT;previous=R/'runs/native_common_contact_validation_gpu_dt15_000';pid=json.loads((previous/'pid.json').read_text())['pid'];kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x00100000,False,pid)
if handle:
    while kernel.WaitForSingleObject(handle,1000)==258:
        if (R/'STOP_SIM2SIM').exists():kernel.CloseHandle(handle);raise SystemExit('Stopped before centered validation')
    kernel.CloseHandle(handle)
assert json.loads((previous/'complete.json').read_text())['complete']
old=json.loads((previous/'results.json').read_text());coarse=json.loads((R/'runs/native_common_contact_validation_gpu_dt31_000/results.json').read_text())
for a,b in zip(old['rows'][1:],coarse['rows'][1:]):assert a['finite'] and a['complete_s']==6 and a['D_mm']>=50 and 100*abs(a['D_mm']-b['D_mm'])/max(abs(b['D_mm']),10)<=5
assert json.loads((R/'audit/common_contact_centered_short_equivalence.json').read_text())['passed']
env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';records=[]
def launch(name,dt,duration):
    if (R/'STOP_SIM2SIM').exists():raise SystemExit('Stopped before start')
    args=[sys.executable,'-B',str(R/'run_native_common_contact_centered.py'),'--name',name,'--bank','validation','--device','cuda:0','--cpu-threads','2','--dt',str(dt),'--command-dt','.00025','--duration',str(duration),'--headless']
    with (R/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    records.append(dict(name=name,pid=p.pid,returncode=code,command=args));s.write(R/'audit/common_contact_centered_gpu_execution.json',dict(previous_owned_pid=pid,rows=records));assert code==0,name
    return R/'runs'/name
f=launch('native_common_contact_centered_gpu_short_000',.00003125,.02);tr=np.load(f/'trajectory.npz');control=np.load(R/'runs/native_common_contact_validation_gpu_dt31_000/trajectory.npz')
proof={k:bool(np.array_equal(tr[k][:,0],control[k][:len(tr['t']),0])) for k in ['q','v','body']};s.write(R/'audit/common_contact_centered_gpu_short_equivalence.json',dict(passed=all(proof.values()),bitwise=proof,control='Original GPU calibration already at Y=0'));assert all(proof.values())
for grid,dt in [(31,.00003125),(15,.000015625)]:
    f=launch(f'native_common_contact_centered_validation_gpu_dt{grid}_000',dt,6)
    result=json.loads((f/'results.json').read_text());assert all(r['finite'] and r['complete_s']==6 and r['D_mm']>=50 for r in result['rows'])
    if grid==31:
        for i,r in enumerate(result['rows'][1:]):
            mu=json.loads((R/f'runs/mu_common_contact_dt31_case{i:02d}_000/results.json').read_text())['rows'][0];assert 100*abs(mu['D_mm']-r['D_mm'])/max((abs(mu['D_mm'])+abs(r['D_mm']))/2,10)<=10
