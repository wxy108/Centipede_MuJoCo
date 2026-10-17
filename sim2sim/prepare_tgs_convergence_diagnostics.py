from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent
native=(R/'run_native_v13.py').read_text()
native=native.replace("p.add_argument('--fk-check',action='store_true')", "p.add_argument('--fk-check',action='store_true');p.add_argument('--command-dt',type=float)")
native=native.replace("cache=CommandCache(names,a.dt,rows);clock_start=sim.current_time", "command_dt=a.dt if a.command_dt is None else a.command_dt;command_stride=round(command_dt/a.dt);assert command_stride>=1 and abs(command_stride*a.dt-command_dt)<1e-12;cache=CommandCache(names,command_dt,rows);clock_start=sim.current_time")
native=native.replace('q,v=cache.get(k)', 'q,v=cache.get(k//command_stride)')
mu=(R/'run_mujoco_v8.py').read_text()
mu=mu.replace("p.add_argument('--case-index',type=int);a=p.parse_args()", "p.add_argument('--case-index',type=int);p.add_argument('--noslip-iters',type=int,default=20);a=p.parse_args()")
mu=mu.replace('m.geom_solref[:,0]=2*a.dt', 'm.geom_solref[:,0]=2*a.dt;m.opt.noslip_iterations=a.noslip_iters;m.opt.noslip_tolerance=1e-12')
for name,source in [('run_native_v14.py',native),('run_mujoco_v11.py',mu)]:
    ast.parse(source);p=R/name;assert not p.exists();p.write_text(source,encoding='utf-8')
print('Prepared new, frozen diagnostic runners.')
