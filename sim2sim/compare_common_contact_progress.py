import json
import numpy as np
import shared as s
R=s.ROOT;comparisons=[]
for device,grid in [('cpu',31),('gpu',31),('cpu',15),('gpu',15)]:
    folder=R/f'runs/native_common_contact_validation_{device}_dt{grid}_000'
    p=folder/'trajectory.npz'
    if not p.exists():p=folder/'partial_trajectory.npz'
    if not p.exists():continue
    a=np.load(p);end=float(a['t'][-1]);start=2. if end>2 else 0.;ai=int(round(start/.01));bi=int(round(end/.01));ids=[a['body_names'].tolist().index(f'link_body_{k}') for k in range(21)];rows=[]
    for i in range(8):
        b=np.load(R/f'runs/mu_common_contact_dt{grid}_case{i:02d}_000/trajectory.npz');mi=[b['body_names'].tolist().index(f'link_body_{k}') for k in range(21)];qi=[b['joint_names'].tolist().index(n) for n in a['joint_names']]
        Dn=-float(a['body'][bi,i+1,ids,0].astype(float).mean()-a['body'][ai,i+1,ids,0].astype(float).mean())*1000;Dm=-float(b['body'][bi,0,mi,0].mean()-b['body'][ai,0,mi,0].mean())*1000
        rows.append(dict(case=i,native_D_mm=Dn,mu_D_mm=Dm,absolute_difference_mm=abs(Dn-Dm),symmetric_gap_pct=100*abs(Dn-Dm)/max((abs(Dn)+abs(Dm))/2,10.),endpoint_q_max_error_deg=float(np.degrees(np.max(abs(a['q'][bi,i+1]-b['q'][bi,0,qi]))))))
    comparisons.append(dict(device=device,grid_us=31.25 if grid==31 else 15.625,window_s=[start,end],complete_score_window=start==2 and end==6,rows=rows))
s.write(R/'audit/common_contact_progress.json',dict(qualification='Partial windows are diagnostic only, never substitute for the frozen 2–6 s gate.',comparisons=comparisons))
print(json.dumps([dict(device=c['device'],grid_us=c['grid_us'],window_s=c['window_s'],maximum_gap_pct=max(r['symmetric_gap_pct'] for r in c['rows']),maximum_absolute_difference_mm=max(r['absolute_difference_mm'] for r in c['rows']),maximum_endpoint_q_error_deg=max(r['endpoint_q_max_error_deg'] for r in c['rows'])) for c in comparisons]))
