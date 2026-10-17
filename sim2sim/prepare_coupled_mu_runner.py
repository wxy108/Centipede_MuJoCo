import shared as s
source=(s.ROOT/'run_mujoco_v6.py').read_text()
changes=[("p.add_argument('--position-implicit-adapter',action='store_true');", "p.add_argument('--position-implicit-adapter',action='store_true');p.add_argument('--coupled-drive-inertia',action='store_true');"),
("if a.position_implicit_adapter:m.actuator_biasprm[ids,2]-=a.dt*kp", "if a.position_implicit_adapter or a.coupled_drive_inertia:m.actuator_biasprm[ids,2]-=a.dt*kp\nif a.coupled_drive_inertia:\n    m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER\n    m.dof_armature[dids]+=a.dt*kv+a.dt*a.dt*kp\n    mujoco.mj_setConst(m,mujoco.MjData(m))"),
("discrete_pd_velocity_coefficient='D + dt*K' if a.position_implicit_adapter else 'D'", "discrete_pd_velocity_coefficient='D + dt*K' if a.position_implicit_adapter or a.coupled_drive_inertia else 'D',numerical_effective_inertia_adapter=bool(a.coupled_drive_inertia),numerical_armature_increment_kgm2=(a.dt*kv+a.dt*a.dt*kp).tolist() if a.coupled_drive_inertia else [],physical_armature_kgm2=0.,effective_integrator='Euler with shared M+hD+h²K in contact and velocity solves' if a.coupled_drive_inertia else a.integrator")]
for old,new in changes:
    assert old in source,old;source=source.replace(old,new)
(s.ROOT/'run_mujoco_v8.py').write_text(source,encoding='utf-8')
print('Prepared coupled effective-inertia runner, derived from unchanged physical PD.')
