"""Target the low-impact basin discovered independently of the previous PD candidate."""
import json
from itertools import product
from run_search import ROOT,OUT

source=json.loads((OUT/'common/sobol_s1901_0015_w1_t6_dt0p00025.json').read_text())
bank=[]
for yk,pd,rk in product((.6,1.,2.),(1.,2.,4.),(.5,1.,2.)):
    c=dict(source['config'],name=f'basin15_yk{yk:g}_pd{pd:g}_rk{rk:g}',yaw_k=yk,pitch_d=pd,roll_k=rk)
    bank.append(c)
(ROOT/'basin15_bank.json').write_text(json.dumps(bank,indent=2));print(len(bank))
