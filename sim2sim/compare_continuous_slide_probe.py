import json,xml.etree.ElementTree as ET
import numpy as np,mujoco
import shared as s
from desaxce_contact import step
f=s.ROOT/'runs/continuous_slide_tgs_cpu_dt62_000';meta=json.loads((f/'results.json').read_text());tr={k:v for k,v in np.load(f/'trajectory.npz').items()};names=tr['joint_names'].tolist();root=ET.parse(f/'probe.xml').getroot();act=ET.SubElement(root,'actuator');ET.SubElement(act,'general',joint='x',gainprm='100',biastype='affine',biasprm='0 -100 -.01');xml=ET.tostring(root,encoding='unicode');rows=[]
def score(label,t,q,v):
    iz=names.index('z');ix=names.index('x');active=t>=.005;return dict(label=label,final_x_um=float(q[-1,ix]*1e6),mean_height_um=float((q[active,iz]-meta['radius']).mean()*1e6),max_height_um=float((q[active,iz]-meta['radius']).max()*1e6),normal_velocity_rms_mm_s=float(np.sqrt(np.mean(v[active,iz]**2))*1000))
for label,dt,correct in [('Mu baseline',.0000625,False),('Mu corrected',.0000625,True),('Mu corrected fine',.000015625,True)]:
    m=mujoco.MjModel.from_xml_string(xml);m.opt.timestep=dt;m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER;m.geom_solref[:,0]=2*dt;ix=m.joint('x').dofadr[0];kp=np.float32(100.);kd=np.float32(.01);m.actuator_biasprm[:,2]=-(kd+dt*kp);m.dof_armature[ix]=dt*kd+dt**2*kp;mujoco.mj_setConst(m,mujoco.MjData(m));d=mujoco.MjData(m);ids=[m.joint(n).qposadr[0] for n in names];d.qpos[ids]=tr['q'][0];d.qvel[ids]=tr['v'][0];ts=[0.];qs=[d.qpos[ids].copy()];vs=[d.qvel[ids].copy()];fail=0;maxres=0.
    for k in range(round(.03/dt)):
        # Hold the same 62.5 microsecond command values on the finer integration grid.
        ct=(k//round(.0000625/dt))*.0000625;d.ctrl[0]=float(np.float32(ct*.03))+kd/kp*float(np.float32(.03))
        if correct:r=step(m,d);fail+=not r['converged'];maxres=max(maxres,r['residual_m_s'])
        else:mujoco.mj_step(m,d)
        ts.append((k+1)*dt);qs.append(d.qpos[ids].copy());vs.append(d.qvel[ids].copy())
    q=np.array(qs);v=np.array(vs);row=score(label,np.array(ts),q,v);row.update(unconverged_steps=fail,max_correction_residual_m_s=maxres);rows.append(row);np.savez_compressed(f/(label.replace(' ','_')+'.npz'),t=ts,q=q,v=v,joint_names=names)
rows.append(score('TGS CPU',tr['t'],tr['q'],tr['v']));s.write(f/'comparison.json',dict(rows=rows,interpretation='Free Z, continuously PD-driven X at 30 mm/s. No per-step joint-state overwrite. Same physical K=100 N/m, D=.01 N s/m, mu=.5 and mass in both.',source_sha=s.sha(__file__)));print(json.dumps(rows,indent=2))
