"""Launch this experiment's qualified polish with hidden persistent logs."""
import subprocess,sys
import psutil
import core as c
import polish

def main():
    for p in psutil.process_iter(['pid','name']):
        if not (p.info['name'] or '').lower().startswith('python'):continue
        try:
            if p.cwd()==str(c.ROOT) and any(a in ('search.py','after_search.py','polish.py') for a in p.cmdline()):
                raise RuntimeError(f'An experiment stage is still active: {p.pid}')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    plan=polish.prepare()
    logs=c.OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    number=1+len(list(logs.glob('polish_run_*.log')))
    out=logs/f'polish_run_{number:02d}.log';err=logs/f'polish_run_{number:02d}.err'
    command=[sys.executable,'-B','polish.py']
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    with out.open('wb') as stdout,err.open('wb') as stderr:
        child=subprocess.Popen(command,cwd=c.ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
            startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    process=psutil.Process(child.pid)
    item=dict(pid=child.pid,create_time=process.create_time(),cmdline=command,cwd=str(c.ROOT),stdout=str(out),stderr=str(err))
    c.write(c.DATA/'supplement_process.json',item)
    print(dict(process=item,waves=[s['wave'] for s in plan['states']],already_covered=plan['already_covered_waves']),flush=True)

if __name__=='__main__':main()
