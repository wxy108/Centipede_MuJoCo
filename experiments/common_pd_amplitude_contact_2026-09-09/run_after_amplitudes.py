"""Continue already-authorized contact work only after completed amplitude qualification."""
from pathlib import Path
import datetime,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'outputs/physical_geometry_v2/data'

def read(p):return json.loads(Path(p).read_text())
def status(stage,**kw):
    value=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=stage,**kw)
    p=ROOT/'outputs/contact_workflow_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(p);print(json.dumps(value),flush=True)
def run(script,*args):
    status('running',script=script,args=list(args))
    subprocess.run([sys.executable,'-B',str(ROOT/script),*args],cwd=ROOT.parents[1],check=True)

def main():
    status('waiting_for_completed_amplitude_qualification')
    while True:
        pipeline=DATA/'pipeline_status.json';selection=DATA/'validated_selection.json'
        if pipeline.exists() and selection.exists() and read(pipeline)['stage']=='amplitude_validation_finished':break
        time.sleep(10)
    selected=read(selection);assert selected['complete'] and selected['selected']
    run('report_amplitudes.py','--final')
    for transient in (False,True):
        channel='contact_erase_transient' if transient else 'contact_erase'
        analysis='transient_analysis.py' if transient else 'analyze_contact.py'
        confirm='confirm_transient.py' if transient else 'confirm_contact.py'
        folder=ROOT/'outputs'/channel/'data'
        run(channel+'.py','freeze');run(channel+'.py','baseline','--workers','24')
        run(channel+'.py','discovery','--workers','24');run(analysis,'discovery')
        choice=folder/'crossing_choice.json'
        if choice.exists():
            run(confirm,'confirm','--workers','24');run(analysis,'confirmation')
            if read(folder/'confirmation_analysis.json')['confirmed_at_nominal_step']:
                run(confirm,'step_duration','--workers','24')
                if transient:run(analysis,'robustness')
                else:run('assess_contact_robustness.py')
    status('mujoco_amplitude_and_two_contact_channels_complete',
        note='Completion of prescribed experiments does not assert a crossing; inspect each channel result and unresolved waves.')

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        status('stopped_for_error',error=repr(ex));raise
