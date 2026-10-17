from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import shared as s

folder=s.ROOT/'runs/native_readback_cpu_000'
meta=json.loads((folder/'readback.json').read_text());a=np.load(folder/'backend.npz');tr=np.load(folder/'trajectory.npz')
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];mujoco.mj_forward(m,d)
rows=[]
for k,n in enumerate(meta['body_names']):
    b=m.body(n);muR=Rotation.from_quat(np.roll(b.iquat,-1)).as_matrix();muI=muR@np.diag(b.inertia)@muR.T
    # Tensor getter returns inertia about COM expressed in LINK axes, already rotated.
    # Installed docstring's return section and measured off-diagonal terms agree;
    # rotating again by COM orientation would incorrectly double-rotate the tensor.
    c=a['coms'][0,k];nativeI=a['inertias'][0,k].reshape(3,3)
    rows.append(dict(name=n,mass_rel=float(abs(a['masses'][0,k]-b.mass[0])/b.mass[0]),com_error_m=float(np.linalg.norm(c[:3]-b.ipos)),inertia_rel=float(np.linalg.norm(nativeI-muI)/np.linalg.norm(muI)),position_error_m=float(np.linalg.norm(tr['body'][0,0,k]-d.xpos[b.id]))))
lookup={x['name']:x for x in s.M['joints']};pd=[]
for k,n in enumerate(meta['joint_names']):
    pd.append(dict(name=n,kp_rel=float(abs(a['dof_stiffnesses'][0,k]-lookup[n]['kp'])/lookup[n]['kp']),kv_rel=float(abs(a['dof_dampings'][0,k]-lookup[n]['kv'])/lookup[n]['kv'])))
summary={key:max(r[key] for r in rows) for key in ['mass_rel','com_error_m','inertia_rel','position_error_m']}
summary.update({key:max(r[key] for r in pd) for key in ['kp_rel','kv_rel']})
summary['arrays']={k:dict(shape=list(a[k].shape),minimum=float(a[k].min()),maximum=float(a[k].max())) for k in a.files}
summary['material_unique']=np.unique(a['material_properties'].reshape(-1,3),axis=0).tolist()
summary['worst_inertia']=sorted(rows,key=lambda x:x['inertia_rel'],reverse=True)[:10]
s.write(s.ROOT/'audit/runtime_readback.json',dict(summary=summary,bodies=rows,joints=pd,source='runs/native_readback_cpu_000'))
print(json.dumps(summary,indent=2))
