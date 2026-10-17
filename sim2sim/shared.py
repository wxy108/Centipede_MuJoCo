from pathlib import Path
import json,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'COMMON_CONTRACT.json').read_text())
M=json.loads((ROOT/'assets/mapping.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def cases(bank):
    if bank=='calibration':return [dict(wave=3.,amplitudes=[32.,18.,26.])]
    if bank=='holdout':return [dict(wave=w,amplitudes=a) for w in C['holdout_gait_bank']['waves'] for a in C['holdout_gait_bank']['amplitude_families_deg']]
    raise ValueError(bank)
def targets(names,t,dt,rows):
    def pos(at):
        out=np.zeros((len(rows),len(names)),float)
        for e,row in enumerate(rows):
            w=row['wave'];amps=np.radians(row['amplitudes'])
            start=1+np.arange(19)*.125/18
            blend=.5*(1-np.cos(np.pi*np.clip(at-start,0,.125)/.125))
            phase=2*np.pi*at-2*np.pi*w*np.arange(19)/18
            for k,name in enumerate(names):
                if name=='body_pitch_0':out[e,k]=np.radians(C['head']['pitch_target_deg'])*blend[0]
                elif name.startswith('body_yaw_'):
                    seg=int(name.rsplit('_',1)[1])-1
                    out[e,k]=0 if seg==0 else blend[seg]*amps[0]*np.sin(phase[seg])
                elif name.startswith('legbase_'):
                    _,typ,seg,side=name.split('_');seg=int(seg);j=0 if typ=='yaw' else 1
                    out[e,k]=blend[seg]*amps[j+1]*(1 if side=='L' else -1)*np.sin(phase[seg]+np.pi*w/18+[-np.pi/2,np.pi][j])
        return out
    q=pos(t);v=np.zeros_like(q) if t==0 else (q-pos(t-dt))/dt
    return q.astype(np.float32),v.astype(np.float32)
def score(t,body,q,rows,names,body_names):
    ids=[body_names.index(f'link_body_{i}') for i in range(21)]
    a=np.flatnonzero(np.isclose(t,2,atol=1e-8));b=np.flatnonzero(np.isclose(t,6,atol=1e-8));out=[]
    for e,row in enumerate(rows):
        xyz=body[:,e,ids].mean(1)
        out.append(dict(**row,complete_s=float(t[-1]),D_mm=float(-(xyz[b[0],0]-xyz[a[0],0])*1000) if len(a) and len(b) else None,
                        forward_since_start_mm=float(-(xyz[-1,0]-xyz[0,0])*1000),finite=bool(np.isfinite(q[:,e]).all() and np.isfinite(xyz).all()),
                        min_spine_center_z_m=float(body[:,e,ids,2].min()),max_abs_joint_deg=float(np.degrees(np.abs(q[:,e])).max())))
    return out
