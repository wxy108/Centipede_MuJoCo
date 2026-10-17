from pathlib import Path
import ast
R=Path(__file__).resolve().parent
src=(R/'run_mujoco_v8.py').read_text(encoding='utf-8')
src=src.replace('from command_cache import CommandCache','from command_cache import CommandCache\nfrom desaxce_contact import step as contact_step')
src=src.replace("p.add_argument('--case-index',type=int);a=p.parse_args()", "p.add_argument('--case-index',type=int);p.add_argument('--contact-iterations',type=int,default=20);p.add_argument('--contact-velocity-tolerance',type=float,default=1e-8);a=p.parse_args();assert a.coupled_drive_inertia")
src=src.replace('peak_force=0.;stopped=False','peak_force=0.;stopped=False;max_contact_residual=0.;contact_unconverged=0;max_contact_iterations=0;contact_iteration_sum=0')
src=src.replace('        mujoco.mj_step(m,d)', "        cr=contact_step(m,d,a.contact_iterations,a.contact_velocity_tolerance);max_contact_residual=max(max_contact_residual,cr['residual_m_s']);contact_unconverged+=not cr['converged'];max_contact_iterations=max(max_contact_iterations,cr['iterations']);contact_iteration_sum+=cr['iterations']")
src=src.replace('stopped=stopped))',"stopped=stopped,contact_max_residual_m_s=max_contact_residual,contact_unconverged_steps=contact_unconverged,contact_max_iterations=max_contact_iterations,contact_mean_iterations=contact_iteration_sum/(k+1)))")
src=src.replace("s.ROOT/'COMMON_CONTRACT.json']", "s.ROOT/'COMMON_CONTRACT.json',s.ROOT/'desaxce_contact.py']")
src=src.replace("stopped=[x['stopped'] for x in traces]))", "stopped=[x['stopped'] for x in traces],contact_correction='De Saxce non-associated Coulomb fixed point',contact_solver_diagnostics=[{key:x[key] for key in ['contact_max_residual_m_s','contact_unconverged_steps','contact_max_iterations','contact_mean_iterations']} for x in traces]))")
ast.parse(src);p=R/'run_mujoco_v12.py';assert not p.exists();p.write_text(src,encoding='utf-8')
print('Prepared contact-law diagnostic robot runner, same plant and physical PD.')
