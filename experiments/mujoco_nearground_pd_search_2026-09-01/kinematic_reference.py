"""Diagnostic only: exact CPG tracking with horizontal rigid spine and zero tip roll.

Root height is placed kinematically at first foot touch on each sample. This
is NOT a physical rollout and must not enter optimizer feasibility/ranking.
"""
import json
from run_search import OUT,np,mujoco,opt,near_ground,amplitude_case

ev=opt.FastEvaluator();m,d,ix=ev.model,ev.data,ev.index
mujoco.mj_forward(m,d);near_ground(ev);root=d.qpos[:7].copy()
sites=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'site_foot_{i}_{j}') for i in range(19) for j in range(2)])
radius=m.geom_size[ev.foot_geoms,0].copy()
rows=[]
for wave in (.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9):
    c=amplitude_case(wave,'common')
    ctl=opt.ThreeAmplitudeController(m,ix,wave,c['body'],c['leg_yaw'],c['leg_pitch'])
    centers=[];bottoms=[];swing=[]
    for t in np.arange(2.,3.,.002):
        d.qpos[:]=m.qpos0;d.qpos[:7]=root;d.qvel[:]=0;d.time=float(t)
        ctl.step(d);d.qpos[ix.qpos_addresses]=ctl.target
        mujoco.mj_kinematics(m,d)
        z=d.site_xpos[sites,2].copy()
        shift=-float(np.min(z-radius));z+=shift
        centers.extend(z);bottoms.extend(z-radius);swing.extend(z[(z-radius)>1e-6])
    rows.append(dict(wave=wave,site_center_all_p95_mm=float(np.quantile(centers,.95)*1000),
        site_center_swing_p95_mm=float(np.quantile(swing,.95)*1000),
        sphere_bottom_all_p95_mm=float(np.quantile(bottoms,.95)*1000),
        site_center_peak_mm=float(np.max(centers)*1000)))
result=dict(description=__doc__,tip_roll_rad=0,body_pitch_roll_rad=0,perfect_target_tracking=True,
    foot_radius_mm=(radius*1000).tolist(),dynamic_rollout=False,gate_modified=False,rows=rows)
(OUT/'kinematic_height_reference.json').write_text(json.dumps(result,indent=2))
print(json.dumps(rows,indent=2))
