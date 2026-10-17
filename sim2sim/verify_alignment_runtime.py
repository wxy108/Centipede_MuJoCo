"""Read the actual target runtime without importing either simulation engine."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAMES=['mujoco','isaacsim','isaaclab','torch','warp-lang','numpy','scipy']
def verify(python,check_gpu_driver=False):
    expected=json.loads((ROOT/'audit/runtime_versions.json').read_text(encoding='utf-8'))
    code="import importlib.metadata as m,importlib.util,json,sys; names="+repr(NAMES)+"; result={n:m.version(n) for n in names}; result['python']=sys.version; result['isaaclab_origin']=importlib.util.find_spec('isaaclab').origin; print(json.dumps(result))"
    p=subprocess.run([str(python),'-B','-c',code],capture_output=True,text=True,check=True);actual=json.loads(p.stdout)
    differences={key:dict(expected=expected[key],actual=actual[key]) for key in [*NAMES,'python'] if actual[key]!=expected[key]}
    if differences:raise RuntimeError('Validated simulation runtime changed: '+json.dumps(differences))
    checked=0
    if (ROOT/'audit/runtime_code_snapshot.json').exists():
        snapshot=json.loads((ROOT/'audit/runtime_code_snapshot.json').read_text(encoding='utf-8'))
        if Path(actual['isaaclab_origin']).resolve()!=Path(snapshot['isaaclab_origin']).resolve():raise RuntimeError('Editable Isaac Lab source location changed.')
        for row in snapshot['rows']:
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Validated runtime code changed: '+row['path'])
            checked+=1
    if check_gpu_driver:
        import csv,io
        fields=['name','driver_version','pci.device_id','memory.total']
        output=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True)
        current=[dict(zip(fields,[value.strip() for value in row])) for row in csv.reader(io.StringIO(output))]
        expected_hardware=json.loads((ROOT/'audit/hardware_identity.json').read_text())['driver_devices']
        if current!=expected_hardware:raise RuntimeError('Validated GPU or driver changed; rerun numerical validation before claiming the same accepted backend.')
    return dict(passed=True,actual=actual,runtime_files_checked=checked,gpu_driver_checked=check_gpu_driver)
if __name__=='__main__':
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument('--check-gpu-driver',action='store_true');args=p.parse_args()
    print(json.dumps(verify(sys.executable,check_gpu_driver=args.check_gpu_driver)))
