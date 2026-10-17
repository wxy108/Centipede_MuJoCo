import json,numpy as np,mujoco
import shared as s
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));d=mujoco.MjData(m);d.qpos[:7]=s.M['initial_root_pose_wxyz'];mujoco.mj_forward(m,d);mu=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mu);rows=[]
for name in ['native_matrix_cpu_000','native_fast_equivalence_gpu_case03_000']:
    f=s.ROOT/'runs'/name;meta=json.loads((f/'readback.json').read_text());native=np.load(f/'generalized_matrices.npz')['mass'][0].astype(float)
    order=list(range(6))+[m.joint(n).dofadr[0] for n in meta['joint_names']];ref=mu[np.ix_(order,order)];assert native.shape==ref.shape,(native.shape,ref.shape)
    raw_root_diag_error=float(np.max(np.abs(np.diag(native)[:6]/np.diag(ref)[:6]-1)))
    # Native root linear velocity is at root-link COM; MuJoCo free translation is at link origin.
    x,y,z=m.body_ipos[1];skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    transform=np.eye(m.nv);transform[:3,3:6]=skew;ref=transform.T@ref@transform
    # Normalize each entry by the geometric mean of corresponding diagonal inertias.
    scale=np.sqrt(np.diag(ref)[:,None]*np.diag(ref)[None,:]);err=(native-ref)/scale
    e=np.linalg.eigvalsh(native/scale);r=dict(run=name,shape=list(native.shape),raw_unmatched_root_origin_diagonal_error=raw_root_diag_error,max_diagonal_relative_error=float(np.max(np.abs(np.diag(native)/np.diag(ref)-1))),max_entry_diagonal_normalized_error=float(np.max(np.abs(err))),normalized_frobenius_error=float(np.linalg.norm(err)/np.linalg.norm(ref/scale)),min_scaled_eigenvalue=float(e.min()))
    rows.append(r)
s.write(s.ROOT/'audit/generalized_mass_comparison.json',dict(rows=rows,frame='identity root rotation; transform MuJoCo root-origin linear velocity to native root-COM linear velocity, joints reordered by name',note='Unmatched root reference points create a spurious 9.3% diagonal error. Raw mixed linear/angular mass-matrix condition number is unit dependent and is not itself evidence of a precision failure.'));print(json.dumps(rows,indent=2))
