"""Boundary expansion around the first all-original-gates-feasible point."""
import json
from run_search import ROOT,OUT
source=json.loads((OUT/'common/axis_leg_yaw_k_0p25_w2p5_t6_dt0p00025.json').read_text())
bank=[dict(source['config'],name=f'legyaw_boundary_k{v:g}',leg_yaw_k=v) for v in (.05,.1,.15,.2,.3,.35,.4,.5)]
(ROOT/'legyaw_boundary_bank.json').write_text(json.dumps(bank,indent=2));print(len(bank))
