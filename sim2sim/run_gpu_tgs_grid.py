import json,subprocess,sys
import shared as s
receipt=[]
for label,dt in [('dt250',.00025),('dt125',.000125)]:
    name=f'native_tgs_ext_validation_gpu_{label}_000'
    if (s.ROOT/'STOP_SIM2SIM').exists():raise RuntimeError('Stop requested; remaining GPU grid cancelled.')
    if (s.ROOT/'runs'/name).exists():raise FileExistsError(name)
    args=[sys.executable,'-B',str(s.ROOT/'run_native_v13.py'),'--name',name,'--bank','validation','--device','cuda:0','--headless','--duration','6','--dt',str(dt)]
    with (s.ROOT/(name+'.log')).open('w') as log:p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
    receipt.append(dict(name=name,pid=p.pid,returncode=code));print(json.dumps(receipt[-1]),flush=True)
    if code:raise RuntimeError(name)
    result=json.loads((s.ROOT/'runs'/name/'results.json').read_text())
    if result['stopped'] or result['rows'][0]['complete_s']!=6:raise RuntimeError('Incomplete GPU grid; next run cancelled.')
s.write(s.ROOT/'audit/gpu_tgs_grid_execution.json',dict(rows=receipt))
