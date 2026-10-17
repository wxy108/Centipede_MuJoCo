from pathlib import Path
import ast
R=Path(__file__).resolve().parent
src=(R/'run_mujoco_v12.py').read_text(encoding='utf-8')
src=src.replace('from desaxce_contact import step as contact_step','from desaxce_contact_anderson import step as contact_step').replace("'--contact-iterations',type=int,default=20","'--contact-iterations',type=int,default=120").replace("s.ROOT/'desaxce_contact.py'","s.ROOT/'desaxce_contact_anderson.py'")
src=src.replace("contact_iteration_sum+=cr['iterations']", "contact_iteration_sum+=cr['iterations']\n        if not cr['converged']:\n            np.savez_compressed(folder/'nonconverged_state.npz',qpos=d.qpos,qvel=d.qvel,qacc=d.qacc,efc_aref=d.efc_aref,efc_force=d.efc_force)\n            s.write(folder/'error.json',dict(reason='Nonlinear contact root did not converge; no complete gait score accepted.',step=k+1,time_s=d.time,diagnostic=cr))\n            raise RuntimeError('Contact root did not converge; evidence preserved in '+str(folder))")
src=src.replace('round(1/a.dt)','round(.25/a.dt)')
src=src.replace('sim_time=(k+1)*a.dt,wall_s=time.monotonic()-started)', 'sim_time=(k+1)*a.dt,wall_s=time.monotonic()-started,contact_max_residual_m_s=max_contact_residual,contact_mean_iterations=contact_iteration_sum/(k+1))')
src=src.replace("contact_correction='De Saxce non-associated Coulomb fixed point'", "contact_correction='De Saxce non-associated Coulomb fixed point with Anderson acceleration'")
ast.parse(src);p=R/'run_mujoco_v13.py';assert not p.exists();p.write_text(src,encoding='utf-8')
