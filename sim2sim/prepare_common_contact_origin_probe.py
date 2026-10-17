from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact_fast_recorded.py').read_text()
old="rows=s.cases('calibration')+s.cases('holdout') if a.bank=='validation' else s.cases(a.bank);N=len(rows);started=time.monotonic()"
assert old in src
src=src.replace(old,"rows=[s.cases('holdout')[6]]*2;N=len(rows);started=time.monotonic();s.write(folder/'origin_probe_setup.json',dict(global_y_m=[0.,1.75],gait=rows[0],control_run='native_common_contact_validation_cpu_dt31_000',control_environment=7,adopted=False))")
src=src.replace('e*.25','[0.,1.75][e]').replace('torch.arange(N,device=sim.device)*.25','torch.tensor([0.,1.75],device=sim.device)')
(R/'run_native_common_contact_origin_probe.py').write_text(src)
