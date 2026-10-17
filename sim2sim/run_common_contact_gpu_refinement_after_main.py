"""One-shot continuation of the current owned GPU job; never runs two GPU workers."""
import ctypes,json,os,subprocess,sys,time
import shared as s
R=s.ROOT;main=R/'runs/native_common_contact_validation_gpu_dt31_000';pid=json.loads((main/'pid.json').read_text())['pid'];kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x00100000,False,pid)
if handle:
    while kernel.WaitForSingleObject(handle,1000)==258:
        if (R/'STOP_SIM2SIM').exists():kernel.CloseHandle(handle);sys.exit('Stopped before refinement')
    kernel.CloseHandle(handle)
assert (main/'complete.json').exists() and json.loads((main/'complete.json').read_text())['complete']
result=json.loads((main/'results.json').read_text());checks=[]
for i,c in enumerate(s.cases('holdout')):
    native=result['rows'][i+1];mu=json.loads((R/f'runs/mu_common_contact_dt31_case{i:02d}_000/results.json').read_text())['rows'][0]
    gap=100*abs(native['D_mm']-mu['D_mm'])/max((abs(native['D_mm'])+abs(mu['D_mm']))/2,10.)
    checks.append(dict(case=i,gap_pct=gap,passed=native['finite'] and native['complete_s']==6 and native['D_mm']>=50 and gap<=10))
s.write(R/'audit/common_contact_gpu_refinement_dispatch_check.json',dict(main_pid=pid,checks=checks))
assert all(x['passed'] for x in checks),'Main gate failed; do not spend a full refinement run without diagnosis'
assert not (R/'STOP_SIM2SIM').exists()
name='native_common_contact_validation_gpu_dt15_000';args=[sys.executable,'-B',str(R/'run_native_common_contact_fast_recorded.py'),'--name',name,'--bank','validation','--device','cuda:0','--cpu-threads','2','--dt','.000015625','--command-dt','.00025','--duration','6','--headless']
env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
with (R/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
s.write(R/'audit/common_contact_gpu_refinement_execution.json',dict(name=name,pid=p.pid,returncode=code,command=args));print(json.dumps(dict(name=name,returncode=code)))
