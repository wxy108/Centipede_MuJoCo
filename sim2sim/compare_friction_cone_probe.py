import json
import numpy as np,mujoco
import shared as s
out=[]
for angle in [0,45]:
    folder=s.ROOT/f'runs/friction_cone_cpu_angle{angle}_000';meta=json.loads((folder/'results.json').read_text());native={k:v for k,v in np.load(folder/'trajectory.npz').items()};names=meta['joint_names']
    for mode in ['elliptic','pyramidal','box_derived']:
        m=mujoco.MjModel.from_xml_path(str(folder/'probe.xml'));m.geom_solref[:,0]=2*meta['dt'];m.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC if mode=='elliptic' else mujoco.mjtCone.mjCONE_PYRAMIDAL
        if mode=='box_derived':m.geom_friction[:,0]*=np.sqrt(2)
        d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=native['q'][0];d.qvel[ids]=native['v'][0];d.qfrc_applied[ids]=meta['applied_force_N'];qs=[d.qpos[ids].copy()]
        for k in range(len(native['t'])-1):
            if mode=='box_derived':
                mujoco.mj_step1(m,d)
                for c in d.contact:
                    c.frame[3:6]=[1/np.sqrt(2),1/np.sqrt(2),0];c.frame[6:9]=np.cross(c.frame[:3],c.frame[3:6])
                mujoco.mj_makeConstraint(m,d);mujoco.mj_projectConstraint(m,d);mujoco.mj_referenceConstraint(m,d);mujoco.mj_step2(m,d)
            else:mujoco.mj_step(m,d)
            qs.append(d.qpos[ids].copy())
        q=np.array(qs);out.append(dict(engine='mujoco',mode=mode,angle=angle,final_xy_um=(q[-1,[names.index('x'),names.index('y')]]*1e6).tolist(),physical_axis_mu=.5,numeric_pyramid_edge_mu=.5*np.sqrt(2) if mode=='box_derived' else .5))
        np.savez_compressed(folder/f'mu_{mode}.npz',t=native['t'],q=q,joint_names=names)
    out.append(dict(engine='physx_cpu',angle=angle,final_xy_um=[meta['final_x']*1e6,meta['final_y']*1e6],physical_axis_mu=.5))
s.write(s.ROOT/'audit/friction_cone_comparison.json',dict(rows=out,probe='Force magnitude 1.2*mu*normal weight; independently chosen to lie outside the circle but inside the axis-aligned square at 45 degrees.',source_sha=s.sha(__file__)))
print(json.dumps(out,indent=2))
