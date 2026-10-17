"""Four body-yaw-stiffness probes in the second seed's low-impact basin."""
import json
from run_search import ROOT,OUT
source=json.loads((OUT/'common/sobol_s2901_0006_w3p5_t6_dt0p00025.json').read_text())
bank=[dict(source['config'],name=f'basin2906_yk{v:g}',yaw_k=v) for v in (.6,1.,2.,4.)]
(ROOT/'basin2906_bank.json').write_text(json.dumps(bank,indent=2));print(len(bank))
