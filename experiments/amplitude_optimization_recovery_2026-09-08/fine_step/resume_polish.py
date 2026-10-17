"""Record an audited interrupted supplement before resuming saved conditions."""
import datetime,os
import psutil
import core as c
import launch_polish

def main():
    active=[]
    for p in psutil.process_iter(['pid','name']):
        if p.pid==os.getpid() or not (p.info['name'] or '').lower().startswith('python'):continue
        try:
            if p.cwd()==str(c.ROOT):active.append(dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline()))
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    assert not active,active
    audit=c.read(c.DATA/'current_audit.json');assert audit['passed'] and not audit['final']
    old=c.read(c.DATA/'supplement_process.json')
    folder=c.OUT/'audit/restarts';number=1+len(list(folder.glob('restart_*.json')))
    receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),audit=audit,last_status=c.read(c.DATA/'live_status.json'),
        previous_process=old,previous_error=c.Path(old['stderr']).read_text(encoding='utf-8'),
        reason_zh='收尾第二轮复核期间进程池意外退出，日志为 BrokenProcessPool 与 handle is closed；底层退出原因未确定。恢复前确认本实验没有存活的 Python 进程，核对全部已存记录与固定输入，仅继续保存计划中缺失的条件。',
        launch_source_sha256=c.sha(c.ROOT/'launch_polish.py'),polish_source_sha256=c.sha(c.ROOT/'polish.py'),
        memory_available_bytes=psutil.virtual_memory().available)
    c.write(folder/f'restart_{number:02d}.json',receipt)
    launch_polish.main()

if __name__=='__main__':main()
