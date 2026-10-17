"""Compare native limit readback with the canonical Mu model; no simulation."""
import json,numpy as np,mujoco
import shared as s
R=s.ROOT;m=mujoco.MjModel.from_xml_path(str(R/'assets/common.xml'));rows=[]
for device in ['cpu','gpu']:
    f=R/'runs'/f'native_common_contact_centered_validation_{device}_dt31_000'
    if not (f/'backend.npz').exists():continue
    meta=json.loads((f/'readback.json').read_text());b=np.load(f/'backend.npz');errors=[];unbounded=[]
    for k,name in enumerate(meta['joint_names']):
        joint=m.joint(name);native=b['dof_limits'][:,k].astype(float)
        if joint.limited[0]:errors.append(dict(joint=name,max_absolute_error_rad=float(abs(native-joint.range).max()),mujoco_range_rad=joint.range.tolist(),native_range_rad=native[0].tolist()))
        else:unbounded.append(dict(joint=name,native_min_rad=float(native[:,0].min()),native_max_rad=float(native[:,1].max())))
    rows.append(dict(device=device,bounded=errors,unbounded=unbounded,maximum_bounded_error_rad=max((r['max_absolute_error_rad'] for r in errors),default=0)))
clearances=[]
for f in sorted((R/'runs').glob('mu_common_contact_dt*_case*_000'))+sorted((R/'runs').glob('native_common_contact_centered_validation_*_000')):
    if not (f/'results.json').exists():continue
    tr=np.load(f/'trajectory.npz');names=tr['joint_names'].tolist();bounds=np.array([m.joint(n).range for n in names]);q=tr['q'][tr['t']>=2].astype(float);margin=np.minimum(q-bounds[:,0],bounds[:,1]-q);i=np.unravel_index(np.argmin(margin),margin.shape)
    clearances.append(dict(run=f.name,minimum_sampled_limit_clearance_deg=float(np.degrees(margin[i])),nearest_joint=names[i[2]],environment=int(i[1]),window_s=[2,6],sample_interval_s=.01))
s.write(R/'audit/common_joint_limit_readback.json',dict(rows=rows,trajectory_limit_clearances=clearances,source_sha=s.sha(__file__)));print(json.dumps(dict(readbacks={r['device']:dict(bounded=len(r['bounded']),unbounded=len(r['unbounded']),max_error_rad=r['maximum_bounded_error_rad']) for r in rows},minimum_sampled_clearance_deg=min(r['minimum_sampled_limit_clearance_deg'] for r in clearances),nearest=min(clearances,key=lambda r:r['minimum_sampled_limit_clearance_deg']))))
