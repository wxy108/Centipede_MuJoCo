"""Record hardware identity from the driver and saved native runtime readbacks."""
import csv,io,json,re,subprocess
from datetime import datetime,timezone
import shared as s

fields=['name','driver_version','pci.device_id','memory.total']
output=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True)
devices=[dict(zip(fields,[value.strip() for value in row])) for row in csv.reader(io.StringIO(output))]
native=[]
for device in ['cpu','gpu']:
    for grid in [31,15]:
        name=f'native_common_contact_centered_validation_{device}_dt{grid}_000'
        log=(s.ROOT/(name+'.log')).read_text(errors='replace')
        backend=json.loads((s.ROOT/'runs'/name/'execution_backend.json').read_text())
        native.append(dict(run=name,logged_driver_versions=re.findall(r'Driver Version:\s*([^\s|]+)',log),
                           logged_gpu_name='NVIDIA GeForce GTX 1050 Ti' if 'NVIDIA GeForce GTX 1050 Ti' in log else None,
                           backend=backend))
for row in native:
    assert row['logged_driver_versions']==[devices[0]['driver_version']]
    if row['backend']['torch_device']=='cuda:0':assert row['backend']['torch_cuda_name']==devices[0]['name']
path=s.ROOT/'audit/hardware_identity.json'
assert not path.exists(),'Hardware snapshot is immutable; review before creating another.'
s.write(path,dict(captured_utc=datetime.now(timezone.utc).isoformat(),driver_devices=devices,
                 cpu='Intel(R) Core(TM) i7-14700HX (native startup log)',native_runs=native,
                 graphics_log_discrepancy='Graphics startup table labels the card GTX 1050 Ti; driver and Torch identify RTX 5070 Laptop GPU. Scene flags and CUDA tensor readback establish the actual tested GPU backend. The cause of the graphics-label discrepancy is not diagnosed.',
                 source_sha=s.sha(__file__)))
print(json.dumps(dict(passed=True,devices=devices,runs=len(native))))
