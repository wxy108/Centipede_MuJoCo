import os,subprocess,sys,json,numpy as np
import shared as s
R=s.ROOT;env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';records=[]
def launch(name,dt,duration):
    if (R/'STOP_SIM2SIM').exists():raise SystemExit('Stopped before start')
    args=[sys.executable,'-B',str(R/'run_native_common_contact_centered.py'),'--name',name,'--bank','validation','--device','cpu','--cpu-threads','2','--dt',str(dt),'--command-dt','.00025','--duration',str(duration),'--headless']
    with (R/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);code=p.wait()
    records.append(dict(name=name,pid=p.pid,returncode=code,command=args));s.write(R/'audit/common_contact_centered_cpu_execution.json',dict(rows=records));assert code==0,name
    return R/'runs'/name
f=launch('native_common_contact_centered_cpu_short_000',.00003125,.02);tr=np.load(f/'trajectory.npz');proofs=[]
for name,e,control_e in [('native_common_contact_validation_cpu_dt31_000',0,0),('native_common_contact_origin_probe_cpu_dt31_000',7,0)]:
    old=np.load(R/'runs'/name/'trajectory.npz');same={k:bool(np.array_equal(tr[k][:,e],old[k][:len(tr['t']),control_e])) for k in ['q','v','body']};proofs.append(dict(control=name,environment=e,control_environment=control_e,bitwise=same));assert all(same.values()),proofs[-1]
s.write(R/'audit/common_contact_centered_short_equivalence.json',dict(passed=True,proofs=proofs))
for grid,dt in [(31,.00003125),(15,.000015625)]:launch(f'native_common_contact_centered_validation_cpu_dt{grid}_000',dt,6)
