"""Ensure demo reconstruction preserves measured body/foot positions."""
import json
from render_pd_comparison import OUT,BANK,state,replay_pose,np

checks=[]
for name in ('baseline','p4_leg_k2_yaw_d2_roll_d4'):
    cfg=next(c for c in BANK if c['name']==name)
    s=state(cfg,2.5)
    try:
        with np.load(OUT/f'{name}_w2p5_t6_dt0p00025.npz') as archive:
            tr={key:archive[key] for key in ('t','xpos','rootmat','bq','lq','tq','fq')}
        for t in (1.5,2.75,3.5,5.999):
            replay_pose(s,tr,t)
            i=int(np.argmin(np.abs(tr['t']-t)))
            error=float(np.max(np.abs(s.data.xpos[s.index.chassis_body_ids]-tr['xpos'][i])))
            assert error<5e-7,(name,t,error)
            checks.append({'case':name,'t':t,'body_position_max_error_m':error})
    finally:s.close()
(OUT/'recorded_pose_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps({'checked_poses':len(checks),'max_body_position_error_m':max(r['body_position_max_error_m'] for r in checks)}))
