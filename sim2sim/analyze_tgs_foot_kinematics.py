"""Describe recorded near-plane sphere kinematics; no force/contact-state inference."""
import json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
R=s.ROOT;m=mujoco.MjModel.from_xml_path(str(R/'assets/common.xml'));d=mujoco.MjData(m)
gids=np.array([g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE]);radii=m.geom_size[gids,0]
jobs=[('mu_coupled_cmd250_dt31_case03_000',0,'MuJoCo 31.25 µs'),('native_tgs_ext_validation_cpu_dt250_000',4,'TGS CPU 250 µs'),('native_tgs_ext_validation_cpu_dt125_000',4,'TGS CPU 125 µs'),('native_tgs_ext_validation_gpu_dt250_000',4,'TGS GPU 250 µs')]
rows=[];curves=[]
for name,e,label in jobs:
    f=R/'runs'/name;tr={k:v for k,v in np.load(f/'trajectory.npz').items()};rp=np.load(f/'reconstructed_root_poses.npz')['root'];t=tr['t'];qids=[m.joint(n).qposadr[0] for n in tr['joint_names'].tolist()];centers=[];rot=[]
    for k in range(len(t)):
        d.qpos[:7]=rp[k,e];d.qpos[qids]=tr['q'][k,e];mujoco.mj_kinematics(m,d);centers.append(d.geom_xpos[gids].copy());rot.append(d.geom_xmat[gids].reshape(-1,3,3).copy())
    centers=np.array(centers);rot=np.array(rot);h=np.diff(t)
    dr=rot[1:]@rot[:-1].transpose(0,1,3,2);omega=Rotation.from_matrix(dr.reshape(-1,3,3)).as_rotvec().reshape(len(h),len(gids),3)/h[:,None,None]
    vc=np.diff(centers,axis=0)/h[:,None,None];lever=np.zeros_like(vc);lever[:,:,2]=-radii
    vs=vc+np.cross(omega,lever);height=centers[:,:,2]-radii
    active=(t[1:]>=2)&(t[1:]<=6);near=(height[1:]<3e-6)&(height[:-1]<3e-6)&active[:,None]
    speed=np.linalg.norm(vs[:,:,:2],axis=2);threshold_rows=[]
    for tol in [1e-6,3e-6,10e-6]:
        mask=(height[1:]<tol)&(height[:-1]<tol)&active[:,None]
        threshold_rows.append(dict(height_threshold_m=tol,near_plane_sample_fraction=float(mask[active].mean()),median_surface_speed_mm_s=float(np.median(speed[mask])*1e3) if mask.any() else None,mean_surface_speed_mm_s=float(np.mean(speed[mask])*1e3) if mask.any() else None))
    rows.append(dict(run=name,label=label,case=3,wave=2.5,amplitudes=[16,72,24],sphere_count=len(gids),thresholds=threshold_rows))
    curves.append((label,speed[near]*1e3,near[active].sum(1)))
s.write(R/'audit/tgs_foot_kinematics.json',dict(rows=rows,method='10 ms recorded poses; finite-difference sphere center velocity plus rotational surface velocity at plane-facing point. Both interval endpoints must be near plane.',limitations='Near-plane geometry is not a measured force-bearing contact. No contact-force, friction-coefficient or causal claim. Native reconstructed pose uncertainty can be of order 1 µm. Rotation/velocity differencing at 100 Hz misses rapid stick-slip events.',source_sha=s.sha(__file__)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':250})
fig,axes=plt.subplots(1,2,figsize=(10.5,4))
for (label,v,n),color in zip(curves,['#2878a8','#27986d','#777777','#d95f02']):
    sv=np.sort(v);axes[0].plot(sv,np.arange(1,len(v)+1)/len(v),label=label,color=color)
    axes[1].plot(np.arange(len(n))*.01+2,n,label=label,color=color,lw=.9,alpha=.8)
axes[0].set(xlabel='Estimated sphere surface speed (mm/s)',ylabel='Cumulative fraction',xlim=(0,120),ylim=(0,1));axes[1].set(xlabel='Time (s)',ylabel='Near-plane spheres (3 µm threshold)',xlim=(2,6),ylim=(0,114))
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('2.5 / B — recorded near-plane kinematics, not measured contact forces',fontsize=11);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False);fig.subplots_adjust(bottom=.26,top=.85,wspace=.25)
for ext in ['png','pdf','svg']:fig.savefig(R/f'figures/17_tgs_near_plane_kinematics.{ext}',bbox_inches='tight')
print(json.dumps(rows,indent=2))
