"""Prepare a fixed-rotation, force-loaded XY sphere on a plane."""
from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'run_contact_probe_pgs.py').read_text(encoding='utf-8')
replacements={
'Matched small sphere, fixed rotation, free vertical/horizontal translation: drop and slide.':'Independent friction-limit test: constant horizontal force on a rotation-locked sphere.',
"AppLauncher.add_app_launcher_args(p);":"p.add_argument('--force-ratio',type=float,default=1.2);p.add_argument('--force-angle',type=float,default=45.)\nAppLauncher.add_app_launcher_args(p);",
"for n in ['base','carrier','ball']:":"for n in ['base','carrier','carrier_x','ball']:",
"[('z','base','carrier','Z'),('x','carrier','ball','X')]":"[('z','base','carrier','Z'),('x','carrier','carrier_x','X'),('y','carrier_x','ball','Y')]",
"q=torch.zeros((1,2),device=sim.device)":"q=torch.zeros((1,3),device=sim.device)",
"q[0,iz]=radius+clearance;v[0,ix]=.02":"q[0,iz]=radius;v[0,ix]=0.",
"    for k in range(round(.03/a.dt)):":"    force=torch.zeros_like(q);force_magnitude=a.force_ratio*.5*(mass+2*carrier)*9.81;force[0,ix]=force_magnitude*np.cos(np.radians(a.force_angle));force[0,names.index('y')]=force_magnitude*np.sin(np.radians(a.force_angle));robot.set_joint_effort_target(force)\n    for k in range(round(.01/a.dt)):",
"<body><inertial mass=\"{mass}\" diaginertia=\"{inertia} {inertia} {inertia}\" pos=\"0 0 0\"/><joint type=\"slide\" name=\"x\" axis=\"1 0 0\"/><geom type=\"sphere\" size=\"{radius}\"/></body></body>":"<body><inertial mass=\"{carrier}\" diaginertia=\"1e-15 1e-15 1e-15\" pos=\"0 0 0\"/><joint type=\"slide\" name=\"x\" axis=\"1 0 0\"/><body><inertial mass=\"{mass}\" diaginertia=\"{inertia} {inertia} {inertia}\" pos=\"0 0 0\"/><joint type=\"slide\" name=\"y\" axis=\"0 1 0\"/><geom type=\"sphere\" size=\"{radius}\"/></body></body></body>",
"initial_clearance=clearance,initial_vx=.02,":"initial_clearance=0.,initial_vx=0.,applied_force_N=arr(force)[0].tolist(),joint_names=names,force_ratio=a.force_ratio,force_angle_deg=a.force_angle,",
"final_x=float(qs[-1][ix]),":"final_x=float(qs[-1][ix]),final_y=float(qs[-1][names.index('y')]),",
}
for old,new in replacements.items():
    assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
(root/'run_friction_cone_probe.py').write_text(src,encoding='utf-8')
print('Prepared independent friction-limit probe.')
