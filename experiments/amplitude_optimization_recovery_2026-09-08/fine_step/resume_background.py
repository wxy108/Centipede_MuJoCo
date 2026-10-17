"""Resume immutable banks with persistent logs after an interrupted connection."""
import datetime,subprocess,sys,time
import psutil
import core as c

def main():
    for p in psutil.process_iter(['pid','name']):
        if p.pid==os_pid() or not (p.info['name'] or '').lower().startswith('python'):continue
        try:
            if p.cwd()==str(c.ROOT) and any(a in ('search.py','after_search.py') for a in p.cmdline()):
                raise RuntimeError(f'An experiment owner is still active: {p.pid}')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    restart_dir=c.OUT/'audit/restarts';restart_dir.mkdir(parents=True,exist_ok=True)
    number=1+len(list(restart_dir.glob('restart_*.json')))
    old=c.read(c.DATA/'search_process.json')
    prior_logs=[]
    if (c.DATA/'background_processes.json').exists():
        for item in c.read(c.DATA/'background_processes.json')['processes']:
            path=c.Path(item['stderr'])
            prior_logs.append(dict(path=str(path),sha256=c.sha(path),text=path.read_text(encoding='utf-8')))
    file_lock=any('PermissionError' in item['text'] and 'live_status.json' in item['text'] for item in prior_logs)
    reason=('Windows denied atomic replacement of live_status.json; bounded retry for transient file occupancy was added and verified.' if file_lock else
        'Execution connection no longer recognized its sessions; owned Python processes were absent. Cause of termination is unknown.')
    receipt=dict(time=datetime.datetime.now().astimezone().isoformat(),previous_owner=old,reason=reason,
        reason_zh='Windows 拒绝替换状态文件；已加入并验证短暂占用时的有限重试。' if file_lock else '运行连接失效后原进程已不存在，具体终止原因未确定。',prior_error_logs=prior_logs,
        last_status=c.read(c.DATA/'live_status.json'),audit=c.read(c.DATA/'current_audit.json'),
        completed_banks=[p.stem for p in sorted((c.DATA/'completed').glob('*.json'))],
        current_assessments=[dict(id=r['id'],raw_sha256=r['raw_sha256']) for r in c.records()],
        physics_or_search_rule_changed=False,completed_conditions_reused_by_exact_id=True)
    assert receipt['audit']['passed']
    c.write(restart_dir/f'restart_{number:02d}.json',receipt)
    logs=c.OUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    started=[]
    for script in ['search.py','after_search.py']:
        command=[sys.executable,'-B',script]+(['--workers','12'] if script=='search.py' else [])
        out=logs/f'{script[:-3]}_resume_{number:02d}.log';err=logs/f'{script[:-3]}_resume_{number:02d}.err'
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        with out.open('wb') as stdout,err.open('wb') as stderr:
            child=subprocess.Popen(command,cwd=c.ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,
                startupinfo=startup,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        process=psutil.Process(child.pid)
        item=dict(pid=child.pid,create_time=process.create_time(),cmdline=command,cwd=str(c.ROOT),stdout=str(out),stderr=str(err))
        if script=='search.py':c.write(c.DATA/'search_process.json',{k:item[k] for k in ['pid','create_time','cmdline']})
        started.append(item)
    c.write(c.DATA/'background_processes.json',dict(restart=number,processes=started))
    print(started,flush=True)

def os_pid():
    import os
    return os.getpid()

if __name__=='__main__':main()
