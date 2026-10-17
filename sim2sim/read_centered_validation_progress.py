"""Read existing partial traces; no simulation launch and no acceptance writes."""
import io,json
import numpy as np
import shared as s

def features(data,environment,count):
    names=data['body_names'].tolist()
    bodies=data['body'][:count,environment].astype(float)
    center=bodies[:,[names.index(f'link_body_{i}') for i in range(21)]].mean(1)
    chord=bodies[:,names.index('link_body_20')]-bodies[:,names.index('link_body_0')]
    angle=np.unwrap(np.arctan2(chord[:,1],chord[:,0]))
    return (center[200:,:2]-center[200,:2])*1000,np.degrees(angle[200:]-angle[0])

rows=[]
for device in ['cpu','gpu']:
    folder=s.ROOT/f'runs/native_common_contact_centered_validation_{device}_dt15_000'
    path=folder/('trajectory.npz' if (folder/'complete.json').exists() else 'partial_trajectory.npz')
    if not path.exists():rows.append(dict(device=device,saved_time_s=None));continue
    try:
        data=np.load(io.BytesIO(path.read_bytes()))
        row=dict(device=device,saved_time_s=float(data['t'][-1]),finite=all(np.isfinite(data[k]).all() for k in ['q','v','body']))
        count=len(data['t'])
        if count>200:
            xy_gaps=[];heading_gaps=[]
            for case in range(8):
                reference=np.load(s.ROOT/f'runs/mu_common_contact_dt15_case{case:02d}_000/trajectory.npz')
                xy,angle=features(data,case+1,count);xy_reference,angle_reference=features(reference,0,count)
                xy_gaps.append(float(np.linalg.norm(xy-xy_reference,axis=1).max()))
                heading_gaps.append(float(abs(angle-angle_reference).max()))
            row.update(partial_max_xy_gap_mm=max(xy_gaps),partial_max_heading_gap_deg=max(heading_gaps))
        rows.append(row)
    except (OSError,ValueError,EOFError) as error:
        rows.append(dict(device=device,read_unavailable=type(error).__name__))
print(json.dumps(dict(qualification='Partial observations only; this reader never changes final acceptance files.',rows=rows)))
