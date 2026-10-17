"""Read checkpoints for progress only; never supplies final acceptance scores."""
import json
import numpy as np
import shared as s
fine=s.ROOT/'runs/native_pgs_validation_gpu_dt31_000'
path=fine/'trajectory.npz'
if not path.exists():path=fine/'partial_trajectory.npz'
b={k:v for k,v in np.load(path).items()}
a={k:v for k,v in np.load(s.ROOT/'runs/native_pgs_validation_gpu_dt62_000/trajectory.npz').items()}
end=float(b['t'][-1]);start=2.
out=dict(provisional=end<6,interval_s=[start,end],source=str(path),interpretation='This shortened interval is not the six-second acceptance score.',rows=[])
if end>start:
    def displ(tr):
        names=tr['body_names'].tolist();ids=[names.index(f'link_body_{i}') for i in range(21)]
        i=np.flatnonzero(np.isclose(tr['t'],start))[0];j=np.flatnonzero(np.isclose(tr['t'],end))[0]
        # Float64 mean prevents avoidable summation roundoff in this diagnostic.
        return -(tr['body'][j][:,ids,0].astype(float).mean(1)-tr['body'][i][:,ids,0].astype(float).mean(1))*1000
    da,db=displ(a),displ(b)
    for i in range(1,9):out['rows'].append(dict(case=i-1,coarse_partial_mm=float(da[i]),fine_partial_mm=float(db[i]),partial_change_pct=float(100*abs(db[i]-da[i])/max(abs(da[i]),10.))))
print(json.dumps(out,indent=2))
