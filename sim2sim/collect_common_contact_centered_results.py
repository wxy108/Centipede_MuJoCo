"""Collect a fixed centered validation bank; never launch physics or promote."""
import json,os,subprocess,sys,time
import shared as s
R=s.ROOT;names=[f'native_common_contact_centered_validation_{d}_dt{g}_000' for d in ['cpu','gpu'] for g in [31,15]];done=set();actions=[];env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
def run(script,*args):
    p=subprocess.run([sys.executable,'-B',str(R/script),*args],capture_output=True,text=True,env=env);actions.append(dict(script=script,args=args,returncode=p.returncode,output=p.stdout[-1800:],error=p.stderr[-1800:]));s.write(R/'audit/common_contact_centered_collection.json',dict(completed=sorted(done),actions=actions));assert p.returncode==0,p.stderr
while len(done)<4:
    if (R/'STOP_SIM2SIM').exists():raise SystemExit('Stopped collection')
    for name in names:
        f=R/'runs'/name
        if name in done or not (f/'complete.json').exists():continue
        assert json.loads((f/'complete.json').read_text())['complete']
        run('validate_common_contact_backend.py',name)
        layout=json.loads((f/'coordinate_layout.json').read_text());assert layout['global_origins_m']==[[0.,0.,0.]]*9 and not layout['periodic_rebase']
        if not (f/'quality_audit.json').exists():run('quality_audit_fast.py',name)
        if not (f/'compliance_geometry_audit.json').exists():run('audit_common_contact_compression.py',name)
        done.add(name)
        for script in ['evaluate_common_contact_centered.py','analyze_common_contact_centered_motion.py','evaluate_common_contact_centered_motion.py']:run(script)
        print(json.dumps(dict(collected=name)),flush=True)
    if len(done)<4:time.sleep(5)
s.write(R/'audit/common_contact_centered_collection.json',dict(completed=sorted(done),actions=actions,all_collected=True));print('All centered native runs collected; no promotion.',flush=True)
