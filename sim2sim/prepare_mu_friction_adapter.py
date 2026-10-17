from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'run_mujoco_v8.py').read_text(encoding='utf-8')
replacements={
'from command_cache import CommandCache':'from command_cache import CommandCache\nfrom friction_box_adapter import BoxFrictionAdapter',
";a=p.parse_args()":";p.add_argument('--friction-basis',choices=['none','fixed','velocity'],default='none');a=p.parse_args()",
"stride=round(.01/a.dt)":"friction_adapter=BoxFrictionAdapter(m,a.friction_basis) if a.friction_basis!='none' else None\nstride=round(.01/a.dt)",
"        mujoco.mj_step(m,d)":"        if friction_adapter is None:mujoco.mj_step(m,d)\n        else:friction_adapter.step(m,d)",
"contact_time_constant_s=2*a.dt,":"contact_time_constant_s=2*a.dt,friction_basis=a.friction_basis,physical_tangent_axis_mu=.5,numeric_pyramid_edge_mu=float(.5*np.sqrt(2)) if friction_adapter else None,friction_constitutive_approximation='square tangent force bound' if friction_adapter else 'elliptic Coulomb cone',",
"[Path(__file__),s.ROOT/'shared.py'":"[Path(__file__),s.ROOT/'friction_box_adapter.py',s.ROOT/'shared.py'",
}
for old,new in replacements.items():
    assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
(root/'run_mujoco_v10.py').write_text(src,encoding='utf-8')
print('Prepared square-friction experimental runner.')
