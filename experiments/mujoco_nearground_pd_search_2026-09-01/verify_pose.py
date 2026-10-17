"""Check recorded-pose reconstruction against body and foot coordinates."""
import json
from run_search import OUT,np,mujoco,opt

results=[]
for tag in (
    'baseline_w2_t6_dt0p00025',
    'basin18_lpk2_lpd0p25_rk14p082_w2_t6_dt0p00025',
    'basin18_lpk2_lpd0p25_rk14p082_w2_t12_dt0p00025',
    'basin18_lpk2_lpd0p25_rk14p082_w2_t6_dt0p000125',
    'baseline_w2p5_t6_dt0p00025',
    'legyaw_refine_k0p29_w2p5_t6_dt0p00025',
    'legyaw_refine_k0p29_w2p5_t12_dt0p00025',
    'legyaw_refine_k0p29_w2p5_t6_dt0p000125',
    'baseline_w3_t6_dt0p00025',
    'basin15_yk2_pd4_rk1_w3_t6_dt0p00025',
    'basin15_yk2_pd4_rk1_w3_t12_dt0p00025',
    'basin15_yk2_pd4_rk1_w3_t6_dt0p000125',
):
    path=OUT/'common'/f'{tag}.npz'
    if not path.exists():continue
    tr=np.load(path);ev=opt.FastEvaluator();m,d,ix=ev.model,ev.data,ev.index
    sites=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'site_foot_{i}_{j}') for i in range(19) for j in range(2)])
    be=fe=0.
    for i in np.linspace(0,len(tr['t'])-1,16,dtype=int):
        d.qpos[:3]=tr['xpos'][i,0]
        q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(tr['rootmat'][i],dtype=float).reshape(9));d.qpos[3:7]=q
        d.qpos[ix.qpos_addresses[ix.body_positions]]=tr['bq'][i,:,0]
        d.qpos[ix.qpos_addresses[ix.pitch_positions]]=tr['bq'][i,:,1]
        d.qpos[ix.roll_qpos_addresses]=tr['bq'][i,:,2]
        d.qpos[ix.qpos_addresses[ix.leg_positions.reshape(-1)]]=tr['lq'][i]
        d.qpos[ix.tip_qpos_addresses]=tr['tq'][i]
        mujoco.mj_kinematics(m,d)
        be=max(be,float(np.abs(d.xpos[ix.chassis_body_ids]-tr['xpos'][i]).max()))
        fe=max(fe,float(np.abs(d.site_xpos[sites]-tr['fq'][i]).max()))
    assert max(be,fe)<1e-7
    results.append(dict(id=tag,checked_poses=16,max_body_error_m=be,max_foot_error_m=fe))
(OUT/'pose_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
