"""Refine a high-displacement, low-impact but not-yet-feasible independent basin."""
import json
from itertools import product
from run_search import ROOT,OUT

source=json.loads((OUT/'common/sobol_s1901_0018_w2_t6_dt0p00025.json').read_text())
bank=[]
for pk,pd,rk in product((2.,4.,8.,source['config']['leg_pitch_k']),(.25,1.,4.),(4.,8.,source['config']['roll_k'])):
    c=dict(source['config'],name=f'basin18_lpk{pk:.5g}_lpd{pd:g}_rk{rk:.5g}',leg_pitch_k=pk,leg_pitch_d=pd,roll_k=rk)
    bank.append(c)
(ROOT/'basin18_bank.json').write_text(json.dumps(bank,indent=2));print(len(bank))
