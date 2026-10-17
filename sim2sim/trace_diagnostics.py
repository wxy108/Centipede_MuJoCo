"""Read-only matched scoring and tracking diagnostics; preserves all raw traces."""
import argparse,json
import numpy as np
import shared as s
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args()
rows=[]
for name in a.runs:
    folder=s.ROOT/'runs'/name;meta=json.loads((folder/'results.json').read_text());tr=np.load(folder/'trajectory.npz');t=tr['t'];names=tr['joint_names'].tolist();bn=tr['body_names'].tolist()
    ids=[bn.index(f'link_body_{i}') for i in range(21)];xyz=tr['body'][:,:,ids].mean(2);active=t>=2
    cases=[dict(wave=r['wave'],amplitudes=r['amplitudes']) for r in meta['rows']]
    ref=np.array([s.targets(names,at,meta['arguments']['dt'],cases)[0] for at in t])
    for e,r in enumerate(meta['rows']):
        out=dict(run=name,**r,lateral_drift_mm=float((xyz[-1,e,1]-xyz[active,e,1][0])*1000))
        for family in ['body_yaw','body_pitch','body_roll','legbase_yaw','legbase_pitch','tarsus_yaw']:
            js=[k for k,n in enumerate(names) if n.startswith(family+'_')]
            if js:out[family+'_tracking_rmse_deg']=float(np.degrees(np.sqrt(np.mean((tr['q'][active,e][:,js]-ref[active,e][:,js])**2))))
        headtail=tr['body'][:,e,bn.index('link_body_0')]-tr['body'][:,e,bn.index('link_body_20')]
        out['head_tail_yaw_span_deg']=float(np.degrees(np.ptp(np.unwrap(np.arctan2(headtail[:,1],-headtail[:,0])))))
        rows.append(out)
s.write(s.ROOT/'audit/latest_trace_diagnostics.json',dict(rows=rows));print(json.dumps(rows,indent=2))
