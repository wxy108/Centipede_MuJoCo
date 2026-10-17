"""Registered coordinated damping probes around the prior engineering candidate."""
import json
from run_search import ROOT,cfg,PREVIOUS

bank=[]
for i,(yk,yd,pk,pd,rk,rd,lyk,lyd,lpk,lpd) in enumerate([
    (1,2,4,4,1,4,1,2,2,4),
    (1,2,4,4,1,4,1,4,2,8),
    (1,2,8,8,2,8,1,4,1,8),
    (2,4,8,8,2,8,2,4,2,4),
    (4,4,8,8,2,8,4,4,2,4),
    (2,8,8,16,2,16,2,8,2,8),
    (2,8,8,16,2,16,2,8,1,8),
    (2,8,8,16,2,16,1,8,2,8),
    (2,4,8,8,2,8,2,4,4,4),
    (4,8,4,8,4,8,4,8,4,8),
    (4,16,4,16,4,16,4,16,4,16),
    (2,4,16,16,4,16,2,4,2,8),
]):
    bank.append(cfg(f'coordinated_{i:02d}',yaw_k=yk,yaw_d=yd,pitch_k=pk,pitch_d=pd,roll_k=rk,roll_d=rd,
        leg_yaw_k=lyk,leg_yaw_d=lyd,leg_pitch_k=lpk,leg_pitch_d=lpd))
(ROOT/'coordinated_bank.json').write_text(json.dumps(bank,indent=2));print(len(bank))
