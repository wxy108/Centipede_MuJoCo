"""Retain every consecutive gait cycle; do not choose a favourable scoring window."""
import json
import numpy as np
import shared as s
rows=[]
jobs=[(f'mu_coupled_cmd250_dt31_case{i:02d}_000',i,0) for i in range(8)]
jobs += [(f'native_tgs_ext_validation_{dev}_dt{dt}_000',i,i+1) for dev,dt in [('cpu',250),('cpu',125),('gpu',250),('gpu',125)] for i in range(8)]
cache={}
for name,i,e in jobs:
    f=s.ROOT/'runs'/name
    if not (f/'results.json').exists():continue
    if name not in cache:cache[name]={k:v for k,v in np.load(f/'trajectory.npz').items()}
    tr=cache[name];bn=tr['body_names'].tolist();ids=[bn.index(f'link_body_{j}') for j in range(21)];x=tr['body'][:,e,ids,0].mean(1);t=tr['t'];cycles=[]
    for start in range(1,6):
        ia=np.flatnonzero(np.isclose(t,start,atol=1e-9));ib=np.flatnonzero(np.isclose(t,start+1,atol=1e-9))
        if len(ia) and len(ib):cycles.append(dict(window_s=[start,start+1],D_mm=float(-(x[ib[0]]-x[ia[0]])*1000)))
    rows.append(dict(run=name,case=i,wave=s.cases('holdout')[i]['wave'],family='AB'[i%2],cycles=cycles))
s.write(s.ROOT/'audit/cycle_displacements.json',dict(rows=rows,interpretation='All completed consecutive cycles shown. Official acceptance still uses the frozen 2–6 s window.',source_sha=s.sha(__file__)))
print(json.dumps([r for r in rows if r['case'] in [1,3]],indent=2))
