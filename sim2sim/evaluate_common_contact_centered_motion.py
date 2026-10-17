"""Additional predeclared complete XY-path and chord-heading gates."""
import itertools,json,numpy as np
import shared as s
R=s.ROOT
def read(engine,grid,case):
    name=f'mu_common_contact_dt{grid}_case{case:02d}_000' if engine=='mu' else f'native_common_contact_centered_validation_{engine}_dt{grid}_000'
    f=R/'runs'/name
    if not (f/'results.json').exists():return None
    result=json.loads((f/'results.json').read_text());e=0 if engine=='mu' else case+1
    if not result['rows'][e]['finite'] or result['rows'][e]['complete_s']!=6:return None
    tr=np.load(f/'trajectory.npz');bn=tr['body_names'].tolist();b=tr['body'][:,e].astype(float);assert b.shape[0]==601
    centroid=b[:,[bn.index(f'link_body_{i}') for i in range(21)]].mean(1);xy=(centroid[200:,:2]-centroid[200,:2])*1000
    chord=b[:,bn.index('link_body_20')]-b[:,bn.index('link_body_0')];h=np.unwrap(np.arctan2(chord[:,1],chord[:,0]));h=np.degrees(h[200:]-h[0])
    return dict(xy=xy,h=h)
rows=[];missing=[];failed=[]
for case,c in enumerate(s.cases('holdout')):
    traces={(d,g):read(d,g,case) for d in ['mu','cpu','gpu'] for g in [31,15]}
    pairs=[(a,g,b,g,'cross_engine') for g in [31,15] for a,b in itertools.combinations(['mu','cpu','gpu'],2)]+[(d,31,d,15,'half_step') for d in ['mu','cpu','gpu']]
    for a,ga,b,gb,kind in pairs:
        identity=dict(case=case,wave=c['wave'],family='AB'[case%2],kind=kind,first_engine=a,first_grid_us=31.25 if ga==31 else 15.625,second_engine=b,second_grid_us=31.25 if gb==31 else 15.625)
        x,y=traces[a,ga],traces[b,gb]
        if x is None or y is None:missing.append(identity);continue
        r=dict(**identity,max_xy_difference_mm=float(np.linalg.norm(x['xy']-y['xy'],axis=1).max()),max_heading_difference_deg=float(abs(x['h']-y['h']).max()));rows.append(r)
        for key,limit in [('max_xy_difference_mm',.5),('max_heading_difference_deg',.25)]:
            if r[key]>limit:failed.append(dict(**identity,criterion=key,value=r[key],limit=limit))
status='FAILED_CRITERIA' if failed else 'PENDING_RESULTS' if missing else 'PASSED'
s.write(R/'audit/common_contact_centered_motion_gates.json',dict(status=status,expected_comparisons=72,rows=rows,missing=missing,failed=failed,criteria=dict(max_xy_difference_mm=.5,max_heading_difference_deg=.25),definitions='Relative unweighted 21-body centroid XY path over 2–6 seconds, sampled every 0.01 s. Chord heading is relative to initial heading. All three backend pairs at both grids and all three independent refinements.',source_sha=s.sha(__file__)))
print(json.dumps(dict(status=status,completed=len(rows),missing=len(missing),failed=failed)))
