"""Compact read-only status, including process ownership and current error logs."""
import datetime,json
from pathlib import Path
import psutil

root=Path(__file__).resolve().parent;data=root/'outputs/data'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
owners=[];processes=read(data/'background_processes.json')['processes']
if (data/'supplement_process.json').exists():processes.append(read(data/'supplement_process.json'))
if (data/'delivery_process.json').exists():processes.append(read(data/'delivery_process.json'))
for owner in processes:
    item=dict(pid=owner['pid'])
    try:
        process=psutil.Process(owner['pid'])
        item['owned_alive']=(process.create_time()==owner['create_time'] and process.cmdline()==owner['cmdline'] and process.cwd()==owner['cwd'])
        item['children']=len(process.children()) if item['owned_alive'] else None
    except psutil.NoSuchProcess:item['owned_alive']=False
    error=Path(owner['stderr']).read_text(encoding='utf-8')
    if error:item['error_tail']=error[-1600:]
    owners.append(item)
print(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),status=read(data/'live_status.json'),owners=owners,
    delivery=read(data/'delivery_status.json') if (data/'delivery_status.json').exists() else None,
    local_complete=(data/'local_complete.json').exists(),initial_final_validation_complete=(data/'initial_final_validation_complete.json').exists(),
    polish_complete=(data/'polish_complete.json').exists(),selection_refreshed=read(data/'validated_selection.json').get('qualified_polish_complete',False) if (data/'validated_selection.json').exists() else False),indent=2))
