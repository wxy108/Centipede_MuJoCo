"""Prepare figures only after the measured contact workflow is complete."""
from pathlib import Path
import datetime,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

def status(stage,**kw):
    value=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=stage,**kw)
    (ROOT/'outputs/reporting_status.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value),flush=True)

def run(script,*args):
    status('preparing_completed_evidence_artifacts',script=script,args=args)
    subprocess.run([sys.executable,'-B',str(ROOT/script),*args],cwd=ROOT,check=True)

def main():
    status('waiting_for_completed_contact_workflow')
    while True:
        path=ROOT/'outputs/contact_workflow_status.json'
        if path.exists() and read(path)['stage']=='mujoco_amplitude_and_two_contact_channels_complete':break
        time.sleep(2)
    run('report_convergence.py')
    run('verify_evidence.py','--final')
    for channel in ('contact_erase','contact_erase_transient'):
        data=ROOT/'outputs'/channel/'data'
        if read(data/'discovery_analysis.json')['selected_for_confirmation']:
            run('plot_final_crossing.py',channel)
            run('report_crossing_mechanism.py',channel)
    status('awaiting_visual_review',final_report_assembled=False,video_rendered=False,
           note='Required statistics are complete; figures still need visual inspection, and representative motion must be reviewed before final delivery.')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('reporting_error',error=repr(error));raise
