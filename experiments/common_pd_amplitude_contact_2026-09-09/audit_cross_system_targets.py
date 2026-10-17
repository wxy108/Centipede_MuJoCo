"""Check saved native target phases against the audited Mu midpoint formula."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
RL=ROOT.parents[2]/'Centipede_IsaacLab_RL'
SOURCE=RL/'outputs/analysis/common_pd_amplitude_contact_20260909/initial/requests/verify_repeat_a'

def wrap(x):return np.angle(np.exp(1j*x))

def main():
    request=json.loads((SOURCE/'request.json').read_text())
    rows=[]
    with np.load(SOURCE/'tracking.npz') as z:
        names=list(z['joint_names']);good=z['t']>=4.;t=z['t'][good]
        targets=z['target'][good]
        X=np.stack([np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.ones_like(t)],axis=1)
        for e,g in enumerate(request['gains'][:28]):
            w=g['wave_abs'];table=[]
            for j in range(1,19):table.append((f'body_yaw_{j+1}','body',-2*np.pi*w*j/18,g['body_amp']*.4))
            for seg in range(19):
                for side,sign in [('L',1),('R',-1)]:
                    for family,offset,amplitude in [('legbase_yaw',-np.pi/2,g['leg_yaw_amp']*.6),('legbase_pitch',np.pi,g['leg_pitch_amp']*.4)]:
                        phase=-2*np.pi*w*seg/18+np.pi*w/18+offset+(0 if sign>0 else np.pi)
                        table.append((f'{family}_{seg}_{side}',family,phase,amplitude))
            family={}
            for name,kind,phase,amplitude in table:
                Y=targets[:,e,names.index(name)];b=np.linalg.lstsq(X,Y,rcond=None)[0]
                family.setdefault(kind,[]).append(dict(name=name,phase_offset_rad=float(wrap(np.arctan2(b[1],b[0])-phase)),
                    amplitude_relative_error=float(np.hypot(b[0],b[1])/amplitude-1),residual_rms_rad=float(np.sqrt(np.mean((Y-X@b)**2)))))
            body_offset=float(np.angle(np.mean(np.exp(1j*np.array([v['phase_offset_rad'] for v in family['body']])))))
            summary={kind:dict(max_relative_phase_error_after_common_body_clock_deg=float(np.max(np.abs(np.degrees(wrap(np.array([v['phase_offset_rad'] for v in values])-body_offset))))),
                max_amplitude_relative_error=float(max(abs(v['amplitude_relative_error']) for v in values)),
                max_residual_rms_rad=max(v['residual_rms_rad'] for v in values)) for kind,values in family.items()}
            rows.append(dict(slot=e,wave=w,common_body_clock_offset_deg=float(np.degrees(body_offset)),families=summary))
    result=dict(rows=rows,scope='Saved control targets only, not actual mechanical tracking or a dynamics-parity claim. Common temporal phase offset removed when comparing relative body-leg spatial phase.',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SOURCE/'tracking.npz',SOURCE/'request.json',ROOT/'core.py',ROOT.parent/'mujoco_wave_amplitude_global_optimization_2026-08-31/run_wave_amplitude_global.py',RL/'controllers/impedance_controller_torch.py']})
    (ROOT/'outputs/data/cross_system_target_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(max_relative_phase_error_deg=max(v['max_relative_phase_error_after_common_body_clock_deg'] for r in rows for v in r['families'].values()),max_relative_amplitude_error=max(v['max_amplitude_relative_error'] for r in rows for v in r['families'].values()))))

if __name__=='__main__':main()
