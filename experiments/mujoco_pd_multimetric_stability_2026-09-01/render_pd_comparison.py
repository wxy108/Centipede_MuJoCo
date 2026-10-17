"""Matched before/after demos; rendering changes never affect physics."""
import argparse
import json
from pathlib import Path
from run_pd_pilot import ROOT,OUT,SOURCE,BANK,load,apply_impedance,wave_case,np,mujoco

demo=load(SOURCE/'render_audit_demos.py','pd_visual_comparison_original')
demo.WIDTH=640;demo.HEIGHT=360

def state(cfg,wave):
    case=wave_case(wave);case['label']=cfg['name']
    s=demo.DemoState(case)
    apply_impedance(s.model,s.index,cfg)
    # Reduce overexposed ground and illuminate the actual body surface.
    s.model.light_diffuse[:]*=.35
    s.model.light_specular[:]=0
    s.model.light_castshadow[:]=0
    s.model.vis.headlight.ambient[:]=.30
    s.model.vis.headlight.diffuse[:]=.45
    s.model.vis.headlight.specular[:]=0
    s.model.geom_matid[s.ground]=-1
    chassis=set(int(b) for b in s.index.chassis_body_ids)
    for g in range(s.model.ngeom):
        if int(s.model.geom_bodyid[g]) in chassis and s.original_group[g]==2 and s.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            s.original_rgba[g]=(.48,.15,.065,1)
    return s

def panel(s,azimuth,elevation,distance,segment=None):
    s.apply_colors()
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth=azimuth;cam.elevation=elevation;cam.distance=distance
    center=s.data.xpos[s.index.chassis_body_ids].mean(axis=0) if segment is None else s.data.xpos[int(s.index.chassis_body_ids[segment])]
    # Horizontal following only; fixed height and no temporal smoothing.
    cam.lookat[:]=(center[0],center[1],.004)
    s.renderer.update_scene(s.data,camera=cam,scene_option=s.option)
    s.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
    return demo.Image.fromarray(s.renderer.render())

def replay_pose(s,tr,t):
    i=int(np.argmin(np.abs(tr['t']-t)));ix=s.index;d=s.data
    d.qpos[:3]=tr['xpos'][i,0]
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(tr['rootmat'][i],dtype=float).reshape(9));d.qpos[3:7]=q
    d.qpos[ix.qpos_addresses[ix.body_positions]]=tr['bq'][i,:,0]
    d.qpos[ix.qpos_addresses[ix.pitch_positions]]=tr['bq'][i,:,1]
    d.qpos[ix.roll_qpos_addresses]=tr['bq'][i,:,2]
    d.qpos[ix.qpos_addresses[ix.leg_positions.reshape(-1)]]=tr['lq'][i]
    d.qpos[ix.tip_qpos_addresses]=tr['tq'][i]
    d.time=float(tr['t'][i]);mujoco.mj_forward(s.model,d)

def make(cfg,wave,zoom=False,recorded=False):
    baseline=next(c for c in BANK if c['name']=='baseline')
    states=[state(baseline,wave),state(cfg,wave)]
    traces=[]
    if recorded:
        for conf in (baseline,cfg):
            tag=f"{conf['name']}_w{wave:g}_t6_dt0.00025".replace('.','p')
            with np.load(OUT/f'{tag}.npz') as source:
                traces.append({key:source[key] for key in ('t','xpos','rootmat','bq','lq','tq')})
    path=OUT/f"{cfg['name']}_w{wave:g}_{'local_comparison' if zoom else 'three_view_comparison'}.mp4"
    cols=2 if zoom else 3
    writer=demo.imageio.get_writer(path,fps=30,codec='libx264',quality=9,macro_block_size=1)
    try:
        for frame in range(136):
            t=1.5+frame/30
            canvas=demo.Image.new('RGB',(cols*640,840),(23,28,34));draw=demo.ImageDraw.Draw(canvas)
            for row,s in enumerate(states):
                if recorded:replay_pose(s,traces[row],t)
                else:s.advance(t)
                y=row*420
                label='BASELINE' if row==0 else f"PD TRIAL: {cfg['name']} | FULL STABILITY GATES NOT ALL PASSED"
                draw.text((14,y+6),f'{label}   wave={wave:g}   t={t:.2f}s',font=demo.FONT_SMALL,fill=(239,239,235))
                views=[('SIDE ZOOM',90,-14,.042,9),('OBLIQUE ZOOM',135,-24,.042,9)] if zoom else [('FRONT',180,-22,.12,None),('SIDE',90,-20,.105,None),('OBLIQUE',135,-28,.12,None)]
                for col,(name,az,el,dist,seg) in enumerate(views):
                    draw.text((col*640+14,y+32),name+' | contact erasure OFF',font=demo.FONT_SMALL,fill=(150,211,235))
                    canvas.paste(panel(s,az,el,dist,seg),(col*640,y+60))
            writer.append_data(np.asarray(canvas))
            if frame==60:canvas.save(OUT/f"{cfg['name']}_w{wave:g}_{'zoom' if zoom else 'three_view'}_sample.png")
    finally:
        writer.close()
        for s in states:s.close()
    reader=demo.imageio.get_reader(path)
    try:
        count=reader.count_frames();shape=list(reader.get_data(0).shape)
    finally:reader.close()
    meta=dict(video=str(path),frames=count,shape=shape,recorded_pose_replay=recorded,pose_sample_interval_s=.001 if recorded else None,render_only_changes='body albedo/light; shadows off; fixed camera height; no smoothing')
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--wave',type=float,default=2.5);p.add_argument('--zoom',action='store_true');p.add_argument('--recorded',action='store_true')
    args=p.parse_args();cfg=next(c for c in BANK if c['name']==args.name)
    make(cfg,args.wave,args.zoom,args.recorded)
