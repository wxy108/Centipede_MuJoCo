"""Bounded collection of four already-planned native runs; no simulation launch or promotion."""
import json,os,subprocess,sys,time
import shared as s
R=s.ROOT;names=[f'native_common_contact_validation_{d}_dt{g}_000' for d in ['cpu','gpu'] for g in [31,15]];done=set();actions=[]
env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
def run(script,*args):
    p=subprocess.run([sys.executable,'-B',str(R/script),*args],capture_output=True,text=True,env=env)
    actions.append(dict(script=script,args=args,returncode=p.returncode,output=p.stdout[-2000:],error=p.stderr[-2000:]));s.write(R/'audit/common_contact_collection.json',dict(completed=sorted(done),actions=actions))
    assert p.returncode==0,p.stderr
while len(done)<len(names):
    if (R/'STOP_SIM2SIM').exists():raise SystemExit('Stopped collection')
    for name in names:
        f=R/'runs'/name
        if name in done or not (f/'complete.json').exists():continue
        assert json.loads((f/'complete.json').read_text())['complete'],name
        run('validate_common_contact_backend.py',name)
        if not (f/'quality_audit.json').exists():run('quality_audit_fast.py',name)
        if not (f/'compliance_geometry_audit.json').exists():run('audit_common_contact_compression.py',name)
        done.add(name);print(json.dumps(dict(collected=name)),flush=True)
        for script in ['evaluate_common_contact.py','plot_common_contact_results.py','build_current_alignment_report.py']:run(script)
    if len(done)<len(names):time.sleep(5)
s.write(R/'audit/common_contact_collection.json',dict(completed=sorted(done),actions=actions,all_collected=True));print('All four native result packages collected; no profile promotion performed.',flush=True)
