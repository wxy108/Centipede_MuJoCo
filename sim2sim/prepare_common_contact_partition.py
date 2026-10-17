from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact_fast.py').read_text()
src=src.replace("p.add_argument('--cpu-threads',type=int,default=8)","p.add_argument('--cpu-threads',type=int,default=8);p.add_argument('--case-indices',type=int,nargs='+')")
src=src.replace("rows=s.cases('calibration')+s.cases('holdout') if a.bank=='validation' else s.cases(a.bank);N=len(rows);started=time.monotonic()", "full_rows=s.cases('calibration')+s.cases('holdout') if a.bank=='validation' else s.cases(a.bank);case_indices=list(range(len(full_rows))) if a.case_indices is None else a.case_indices;rows=[full_rows[i] for i in case_indices];N=len(rows);started=time.monotonic()")
src=src.replace('e*.25','case_indices[e]*.25').replace('torch.arange(N,device=sim.device)*.25','torch.tensor(case_indices,device=sim.device)*.25')
(R/'run_native_common_contact_partition.py').write_text(src)
