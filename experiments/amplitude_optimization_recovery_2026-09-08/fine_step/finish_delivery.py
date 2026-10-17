"""Finish trajectory evidence, figures and representative motion comparisons."""
import argparse,datetime,subprocess,sys,time
import psutil
import core as c

WAVES=[.5,1,1.5,2,4,5,6,7,9]

def run():
    selection=c.read(c.DATA/'validated_selection.json')
    assert c.read(c.DATA/'polish_complete.json')['completed'] and selection['qualified_polish_complete']
    jobs=[('selected_motion',['delivery.py','selected','--workers','12']),
          ('unresolved_motion',['diagnostic_replay.py'])]
    jobs += [(f'video_w{wave:g}',['render.py','--wave',str(wave),'--name',f'wave{wave:g}_before_after']) for wave in WAVES]
    jobs += [('report',['final_report.py'])]
    started=time.monotonic()
    for i,(stage,args) in enumerate(jobs):
        c.write(c.DATA/'delivery_status.json',dict(time=datetime.datetime.now().astimezone().isoformat(),stage=stage,
            done=i,total=len(jobs),complete=False,elapsed_s=time.monotonic()-started))
        print('Delivery stage: '+stage,flush=True)
        subprocess.run([sys.executable,'-B',*args],cwd=c.ROOT,check=True)
    c.write(c.DATA/'delivery_status.json',dict(time=datetime.datetime.now().astimezone().isoformat(),stage='motion_and_figures_complete',
        done=len(jobs),total=len(jobs),complete=True,elapsed_s=time.monotonic()-started,final_audit_pending=True))
    print('Delivery artifacts complete; visual inspection and final audit remain.',flush=True)

def launch():
    assert c.read(c.DATA/'polish_complete.json')['completed']
    assert c.read(c.DATA/'validated_selection.json')['qualified_polish_complete']
    for p in psutil.process_iter(['pid','name']):
        if not (p.info['name'] or '').lower().startswith('python'):continue
        try:
            if p.cwd()==str(c.ROOT) and ('polish.py' in p.cmdline() or ('finish_delivery.py' in p.cmdline() and '--run' in p.cmdline())):
                raise RuntimeError(f'An experiment stage is active: {p.pid}')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    logs=c.OUT/'logs';number=1+len(list(logs.glob('delivery_run_*.log')))
    out=logs/f'delivery_run_{number:02d}.log';err=logs/f'delivery_run_{number:02d}.err'
    command=[sys.executable,'-B','finish_delivery.py','--run']
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    with out.open('wb') as stdout,err.open('wb') as stderr:
        child=subprocess.Popen(command,cwd=c.ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
            startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    p=psutil.Process(child.pid)
    owner=dict(pid=p.pid,create_time=p.create_time(),cmdline=command,cwd=str(c.ROOT),stdout=str(out),stderr=str(err))
    c.write(c.DATA/'delivery_process.json',owner);print(owner,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
    (run if args.run else launch)()
