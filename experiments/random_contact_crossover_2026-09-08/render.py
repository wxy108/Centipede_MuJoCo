"""Three-view replay of recorded states, with visible collision-loss labels."""
from __future__ import annotations
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import study as s
np,mujoco=s.np,s.mujoco
h=s.load(s.e.PREVIOUS/'render_flexion_demo.py','crossover_render_helpers')
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
FONT=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
BOLD=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',23)

class Replay:
    def __init__(self,path):
        self.path=Path(path);self.r=s.read(path);assert self.r['completed']
        self.tr=dict(np.load(self.path.with_name('trajectory.npz')))
        assert s.sha(self.r['model_path'])==self.r['model_sha256']
        self.m=s.e.runtime_model(self.r['model_path']);self.ix=s.e.fx.opt.tip.LeanIndex(self.m)
        s.pilot.apply_impedance(self.m,self.ix,self.r['config'])
        self.d=mujoco.MjData(self.m);self.ground=s.opt.tip.stage1.ground_geom_id(self.m)
        self.leg_geoms=[]
        for seg in range(19):
            roots={mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,f'link_leg_{seg}_{side}_0') for side in ('L','R')}
            descendants=set()
            for body in range(self.m.nbody):
                cur=body
                while cur>0:
                    if cur in roots:descendants.add(body);break
                    cur=int(self.m.body_parentid[cur])
            self.leg_geoms.append(np.array([g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in descendants]))
        h.light_and_color(self.m,self.ground);self.original_color=self.m.geom_rgba.copy()
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=0;self.option.geomgroup[[0,2]]=1;self.option.sitegroup[:]=0
        self.m.vis.map.znear=1e-4
        self.renderer=mujoco.Renderer(self.m,height=360,width=640);self.max_time_error=0.;self.pose_errors=[]
    def pose(self,t):
        i=int(np.argmin(np.abs(self.tr['pose_t']-t)));actual=float(self.tr['pose_t'][i]);self.max_time_error=max(self.max_time_error,abs(actual-t))
        self.d.qpos[:]=self.tr['qpos'][i];self.d.time=actual;mujoco.mj_kinematics(self.m,self.d)
        assert np.array_equal(self.d.qpos,self.tr['qpos'][i])
        self.active=self.tr['erased_pose'][i];self.m.geom_rgba[:]=self.original_color
        for j in np.flatnonzero(self.active):self.m.geom_rgba[self.leg_geoms[j]]=(1.,.025,.025,1.)
        return actual
    def panel(self,az,el,distance):
        center=self.d.xpos[self.ix.chassis_body_ids].mean(axis=0)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth=az;cam.elevation=el;cam.distance=distance;cam.lookat[:]=[center[0],center[1],.004]
        self.renderer.update_scene(self.d,camera=cam,scene_option=self.option)
        scene=self.renderer.scene;scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        for axis in (0,1):
            for j in range(int(np.floor((center[axis]-.13)/.01)),int(np.ceil((center[axis]+.13)/.01))+1):
                a=np.array([center[0]-.13,center[1]-.13,8e-6]);b=np.array([center[0]+.13,center[1]+.13,8e-6]);a[axis]=b[axis]=j*.01
                h.line(scene,a,b,[.47,.50,.53,1],1)
        return Image.fromarray(self.renderer.render())

def make(waves,nu,seed,name):
    paths=[s.TRIALS/s.condition_id(s.task(w,nu,seed))/'result.json' for w in waves]
    states=[Replay(p) for p in paths];assert len({x.r['channel']['mask_sha256'] for x in states})==1
    video=s.OUT/'videos'/f'{name}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(video,fps=30,codec='libx264',quality=8,macro_block_size=1)
    views=[('FRONT',180,-20,.12),('SIDE',90,-18,.115),('OBLIQUE',135,-30,.13)]
    try:
        for f in range(180):
            t=f/30;canvas=Image.new('RGB',(1920,470*len(states)),(23,28,34));draw=ImageDraw.Draw(canvas)
            for row,state in enumerate(states):
                actual=state.pose(t);r=state.r;y=row*470;w=r['task']['wave'];q=r['channel']['q']
                i=int(np.argmin(np.abs(state.tr['t']-max(t,1.5))));j=int(np.argmin(np.abs(state.tr['t']-1.5)))
                progress=float(r['forward_sign']*(state.tr['center'][i,0]-state.tr['center'][j,0])*1000) if t>=1.5 else 0
                status='ALL VIABILITY GATES PASS' if r['viable'] else 'DIAGNOSTIC: SOME VIABILITY GATES FAIL'
                draw.text((14,y+5),f"WAVE {w:g} | nu={nu:g} (time loss) | q={q:.3f} (event probability) | seed={seed} | t={actual:.3f}s",font=BOLD,fill=(242,242,235))
                draw.text((14,y+34),f"Forward D now={progress:.1f} mm | final={r['D']:.1f} mm | {status}",font=FONT,fill=(191,220,236))
                ids=','.join(map(str,np.flatnonzero(state.active))) or 'none'
                draw.text((14,y+62),f"RED = erased bilateral leg pairs: {ids} | fixed amplitudes / PD; flat, 1 Hz",font=FONT,fill=(255,127,120))
                for col,(label,az,el,dist) in enumerate(views):
                    draw.text((col*640+14,y+87),label,font=FONT,fill=(158,207,226));canvas.paste(state.panel(az,el,dist),(col*640,y+110))
            writer.append_data(np.asarray(canvas))
            if f in (0,60,90,150,179):canvas.save(video.with_name(video.stem+f'_frame{f:03d}.png'))
    finally:
        writer.close()
        for st in states:st.renderer.close()
    reader=imageio.get_reader(video);n=reader.count_frames();frame=reader.get_data(90);reader.close();assert n==180
    info=dict(video=str(video),frames=n,fps=30,shape=list(frame.shape),records=list(map(str,paths)),
        recorded_qpos_replayed_exactly=True,no_interpolation=True,physics_rerun=False,
        max_nearest_recorded_time_error_s=max(st.max_time_error for st in states),
        note='Recorded poses sampled near 60 Hz; nearest-state replay at 30 fps. Red states are the saved event masks, no resimulation.')
    s.write(video.with_suffix('.json'),info);print(json.dumps(info),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--waves',type=float,nargs='+',required=True);ap.add_argument('--nu',type=float,required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--name',required=True);a=ap.parse_args();make(a.waves,a.nu,a.seed,a.name)
