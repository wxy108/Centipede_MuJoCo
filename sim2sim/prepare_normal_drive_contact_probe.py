import shared as s
source=(s.ROOT/'run_coupled_drive_contact_probe.py').read_text()
changes=[("stiffness={'x':2.,'z':0.},damping={'x':.01,'z':0.}","stiffness={'x':2.,'z':2.},damping={'x':.01,'z':.01}"),
("target[0,ix]=.00002*np.sin(2*np.pi*50*at);velocity=", "target[0,ix]=.00002*np.sin(2*np.pi*50*at);target[0,iz]=radius-.00002+.00004*np.sin(2*np.pi*50*at);velocity="),
("velocity[0,ix]=.00002*2*np.pi*50*np.cos(2*np.pi*50*at);robot.set", "velocity[0,ix]=.00002*2*np.pi*50*np.cos(2*np.pi*50*at);velocity[0,iz]=.00004*2*np.pi*50*np.cos(2*np.pi*50*at);robot.set"),
("horizontal_kp=2.,horizontal_kd=.01,", "horizontal_kp=2.,horizontal_kd=.01,vertical_kp=2.,vertical_kd=.01,")]
for old,new in changes:
    assert old in source,old;source=source.replace(old,new)
(s.ROOT/'run_normal_drive_contact_probe.py').write_text(source,encoding='utf-8')
source=(s.ROOT/'compare_coupled_drive_contact.py').read_text();source=source.replace("coupled_drive_contact_pgs_cpu_dt62_000","normal_drive_contact_pgs_cpu_dt62_000")
source=source.replace("xml=ET.tostring(root,encoding='unicode')","ET.SubElement(act,'general',name='z_drive',joint='z',gaintype='fixed',biastype='affine',gainprm='2',biasprm='0 -2 -.01');xml=ET.tostring(root,encoding='unicode')")
source=source.replace("m.actuator_gainprm[0,0]=kp;m.actuator_biasprm[0,1]=-kp;m.actuator_biasprm[0,2]=-(kd+dt*kp)","m.actuator_gainprm[:,0]=kp;m.actuator_biasprm[:,1]=-kp;m.actuator_biasprm[:,2]=-(kd+dt*kp)")
source=source.replace("m.dof_armature[ix]+=dt*kd+dt**2*kp", "m.dof_armature[:]+=dt*kd+dt**2*kp")
source=source.replace("d.ctrl[0]=qt+kd/kp*vt;mujoco.mj_step", "d.ctrl[0]=qt+kd/kp*vt;zt=float(np.float32(meta['radius']-.00002+.00004*np.sin(2*np.pi*50*t)));zv=float(np.float32(.00004*2*np.pi*50*np.cos(2*np.pi*50*t)));d.ctrl[1]=zt+kd/kp*zv;mujoco.mj_step")
source=source.replace("x_max_error_m=float(np.max(np.abs(error))),", "x_max_error_m=float(np.max(np.abs(error))),z_rmse_m=float(np.sqrt(np.mean((q[:,names.index('z')]-tr['q'][:,names.index('z')])**2))),")
(s.ROOT/'compare_normal_drive_contact.py').write_text(source,encoding='utf-8')
print('Prepared normal-and-tangential driven contact diagnostic.')
