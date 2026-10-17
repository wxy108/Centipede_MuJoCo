"""Final one-coordinate bracket refinement around Kp factor0.25."""
import json
from run_search import ROOT,OUT
source=json.loads((OUT/'common/axis_leg_yaw_k_0p25_w2p5_t6_dt0p00025.json').read_text())
bank=[dict(source['config'],name=f'legyaw_refine_k{v:g}',leg_yaw_k=v) for v in (.26,.27,.275,.28,.29)]
(ROOT/'legyaw_refine_bank.json').write_text(json.dumps(bank,indent=2));print(len(bank))
