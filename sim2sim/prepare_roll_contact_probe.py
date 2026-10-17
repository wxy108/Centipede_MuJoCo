"""Prepare an independent free-rolling sphere probe; preserve previous probes."""
from pathlib import Path
root=Path(__file__).resolve().parent
src=(root/'run_contact_probe_pgs.py').read_text(encoding='utf-8')
replacements={
'Matched small sphere, fixed rotation, free vertical/horizontal translation: drop and slide.':'Matched sphere with free rolling about Y and free X/Z translation: drop and roll.',
"for n in ['base','carrier','ball']:":"for n in ['base','carrier','carrier_x','ball']:",
"[('z','base','carrier','Z'),('x','carrier','ball','X')]":"[('z','base','carrier','Z'),('x','carrier','carrier_x','X')]",
"    shape=UsdGeom.Sphere.Define(st,'/Probe/ball/shape');":"    roll=UsdPhysics.RevoluteJoint.Define(st,'/Probe/roll');roll.CreateBody0Rel().SetTargets(['/Probe/carrier_x']);roll.CreateBody1Rel().SetTargets(['/Probe/ball']);roll.CreateAxisAttr('Y')\n    shape=UsdGeom.Sphere.Define(st,'/Probe/ball/shape');",
"q=torch.zeros((1,2),device=sim.device)":"q=torch.zeros((1,3),device=sim.device)",
"<body><inertial mass=\"{mass}\" diaginertia=\"{inertia} {inertia} {inertia}\" pos=\"0 0 0\"/><joint type=\"slide\" name=\"x\" axis=\"1 0 0\"/><geom type=\"sphere\" size=\"{radius}\"/></body></body>":"<body><inertial mass=\"{carrier}\" diaginertia=\"1e-15 1e-15 1e-15\" pos=\"0 0 0\"/><joint type=\"slide\" name=\"x\" axis=\"1 0 0\"/><body><inertial mass=\"{mass}\" diaginertia=\"{inertia} {inertia} {inertia}\" pos=\"0 0 0\"/><joint type=\"hinge\" name=\"roll\" axis=\"0 1 0\"/><geom type=\"sphere\" size=\"{radius}\"/></body></body></body>",
"final_x=float(qs[-1][ix]),":"final_x=float(qs[-1][ix]),final_vx=float(vs[-1][ix]),final_roll_velocity=float(vs[-1][names.index('roll')]),",
}
for old,new in replacements.items():
    assert src.count(old)==1,(old,src.count(old))
    src=src.replace(old,new)
(root/'run_contact_roll_probe.py').write_text(src,encoding='utf-8')
print('Prepared independent rolling probe.')
