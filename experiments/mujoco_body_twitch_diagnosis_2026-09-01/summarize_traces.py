"""Postprocess replay traces without modifying the simulator or prior results."""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(sys.executable).resolve().parent/'Lib/site-packages'))
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
result=[]
for path in sorted(OUT.glob('*_trace.npz')):
    z=np.load(path)
    t=z['time']; dt=float(t[1]-t[0]); mask=(t>=2)&(t<6)
    q=z['body_q'][mask]; qv=z['body_qvel'][mask]
    pitch=q[:,:,1]
    lag=int(round(.01/dt)); frame=int(round(1/30/dt))
    om=z['omega_world'][mask]
    delta=np.linalg.norm(om[lag:]-om[:-lag],axis=-1)
    bc=z['body_contact_n'][mask]
    row={'id':path.stem.replace('_trace',''),
         'pitch_angle_abs_p99_deg':float(np.quantile(np.abs(np.degrees(pitch)),.99)),
         'pitch_change_10ms_max_deg':float(np.degrees(np.abs(pitch[lag:]-pitch[:-lag])).max()),
         'pitch_change_frame_max_deg':float(np.degrees(np.abs(pitch[frame:]-pitch[:-frame])).max()),
         'per_joint_highfreq_over_10hz':{}, 'regional_events':{}}
    steady=(t>=1.5)&(t<6)
    force=z['total_contact_n'][steady]
    steady_bc=z['body_contact_n'][steady]
    k=int(np.argmax(force))
    row['largest_contact_force_event']={
        'time':float(t[steady][k]),'normal_force_mN':float(force[k]*1000),
        'chassis_force_mN':float(steady_bc[k].sum()*1000),
        'chassis_segments':np.flatnonzero(steady_bc[k]>1e-7).tolist(),
        'body_pitch_speed_max_rad_s':float(np.abs(z['body_qvel'][steady][k,:,1]).max())}
    for a,axis in enumerate(('yaw','pitch','roll')):
        h=np.hanning(len(q))[:,None]
        spec=np.abs(np.fft.rfft((qv[:,:,a]-qv[:,:,a].mean(axis=0))*h,axis=0))**2
        freq=np.fft.rfftfreq(len(q),dt)
        ratio=spec[freq>10].sum(axis=0)/np.maximum(spec.sum(axis=0),1e-30)
        row['per_joint_highfreq_over_10hz'][axis]={'median_velocity_energy_fraction':float(np.median(ratio))}
    for label,segments in [('front',range(0,7)),('middle',range(7,15)),('tail',range(15,21))]:
        sub=delta[:,list(segments)]
        at,local=np.unravel_index(np.argmax(sub),sub.shape)
        seg=list(segments)[local]
        start=at; end=at+lag
        row['regional_events'][label]={'time':float(t[mask][end]),'segment':seg,
            'delta_omega_10ms_rad_s':float(sub[at,local]),
            'body_contact_segments_in_window':np.flatnonzero(bc[start:end+1].max(axis=0)>1e-7).tolist(),
            'pitch_joint_speed_peak_in_window_rad_s':float(np.abs(qv[start:end+1,:,1]).max())}
    result.append(row)
(OUT/'comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
