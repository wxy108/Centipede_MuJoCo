import json,xml.etree.ElementTree as ET
import numpy as np,mujoco
import shared as s
f=s.ROOT/'runs/normal_drive_contact_pgs_cpu_dt62_000';meta=json.loads((f/'results.json').read_text());tr={k:v.copy() for k,v in np.load(f/'trajectory.npz').items()};names=tr['joint_names'].tolist();native_x=tr['q'][:,names.index('x')];root=ET.parse(f/'probe.xml').getroot();act=ET.SubElement(root,'actuator');ET.SubElement(act,'general',name='x_drive',joint='x',gaintype='fixed',biastype='affine',gainprm='2',biasprm='0 -2 -.01');ET.SubElement(act,'general',name='z_drive',joint='z',gaintype='fixed',biastype='affine',gainprm='2',biasprm='0 -2 -.01');xml=ET.tostring(root,encoding='unicode');out=[];arrays={}
for mode in ['split','coupled']:
    m=mujoco.MjModel.from_xml_string(xml);dt=meta['dt'];m.geom_solref[:,0]=2*dt;ix=m.joint('x').dofadr[0];kp=float(np.float32(2.));kd=float(np.float32(.01));m.actuator_gainprm[:,0]=kp;m.actuator_biasprm[:,1]=-kp;m.actuator_biasprm[:,2]=-(kd+dt*kp)
    if mode=='coupled':
        m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.dof_armature[:]+=dt*kd+dt**2*kp;mujoco.mj_setConst(m,mujoco.MjData(m))
    d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=tr['q'][0];d.qvel[ids]=tr['v'][0];qs=[d.qpos[ids].copy()];vs=[d.qvel[ids].copy()]
    for k in range(len(tr['t'])-1):
        t=k*dt;qt=float(np.float32(.00002*np.sin(2*np.pi*50*t)));vt=float(np.float32(.00002*2*np.pi*50*np.cos(2*np.pi*50*t)));d.ctrl[0]=qt+kd/kp*vt;zt=float(np.float32(meta['radius']-.00002+.00004*np.sin(2*np.pi*50*t)));zv=float(np.float32(.00004*2*np.pi*50*np.cos(2*np.pi*50*t)));d.ctrl[1]=zt+kd/kp*zv;mujoco.mj_step(m,d);qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    q=np.array(qs);arrays[f'{mode}_q']=q;arrays[f'{mode}_v']=vs;error=q[:,names.index('x')]-native_x
    out.append(dict(mode=mode,x_rmse_m=float(np.sqrt(np.mean(error**2))),x_normalized_rmse=float(np.sqrt(np.mean(error**2))/np.ptp(native_x)),x_max_error_m=float(np.max(np.abs(error))),z_rmse_m=float(np.sqrt(np.mean((q[:,names.index('z')]-tr['q'][:,names.index('z')])**2))),min_surface_z_m=float(q[:,names.index('z')].min()-meta['radius']),warnings=[int(w.number) for w in d.warning],physical_kp=kp,physical_kd=kd,numerical_added_inertia=dt*kd+dt**2*kp if mode=='coupled' else 0))
np.savez_compressed(f/'mu_comparison.npz',t=tr['t'],**arrays);s.write(f/'comparison.json',dict(rows=out,source_sha=s.sha(__file__)));print(json.dumps(out,indent=2))
