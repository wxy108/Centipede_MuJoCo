"""Front/side/oblique views from the exact recorded qpos, without new physics."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import core as c
from report import baseline_rows,vector
np,mj=c.np,c.mujoco
h=c.load(c.e.PREVIOUS/'render_flexion_demo.py','recovery_render_helpers')
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
FONT=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
BOLD=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',23)

class Replay:
    def __init__(self,row,label,qualified):
        self.row=row;self.label=label;self.qualified=qualified
        self.path=Path(row['record']);assert c.sha(self.path)==row['raw_sha256']
        self.tr=dict(np.load(self.path.parent/'motion.npz'))
        good=self.tr['t']>=1.5
        self.forward=c.e.fx.opt.base.FORWARD_SIGN
        D=float(self.forward*np.diff(self.tr['center'][good][[0,-1],0])[0]*1000)
        assert D==row['D']
        original=c.e.runtime_model(c.history()['plant']);tree=c.e.runtime_tree(c.history()['plant'])
        ET.SubElement(tree.getroot().find('asset'),'texture',type='skybox',builtin='gradient',rgb1='.69 .74 .79',rgb2='.69 .74 .79',width='256',height='1536')
        self.m=mj.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
        for key in ['body_mass','body_inertia','geom_size','geom_pos','geom_quat','jnt_axis','jnt_range','dof_damping']:
            assert np.array_equal(getattr(original,key),getattr(self.m,key)),key
        self.ix=c.e.fx.opt.tip.LeanIndex(self.m)
        c.e.fx.pilot.apply_impedance(self.m,self.ix,row['task']['config'])
        self.d=mj.MjData(self.m);ground=c.e.fx.opt.tip.stage1.ground_geom_id(self.m)
        h.light_and_color(self.m,ground)
        self.opt=mj.MjvOption();self.opt.geomgroup[:]=0;self.opt.geomgroup[[0,2]]=1;self.opt.sitegroup[:]=0
        self.m.vis.map.znear=1e-4;self.renderer=mj.Renderer(self.m,height=360,width=640);self.max_time_error=0.

    def pose(self,t):
        i=int(np.argmin(np.abs(self.tr['pose_t']-t)));actual=float(self.tr['pose_t'][i]);self.max_time_error=max(self.max_time_error,abs(t-actual))
        self.d.qpos[:]=self.tr['qpos'][i];self.d.time=actual;mj.mj_kinematics(self.m,self.d)
        assert np.array_equal(self.d.qpos,self.tr['qpos'][i]);return actual

    def panel(self,az,el,distance):
        center=self.d.xpos[self.ix.chassis_body_ids].mean(axis=0)
        cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;cam.azimuth=az;cam.elevation=el;cam.distance=distance;cam.lookat[:]=[center[0],center[1],.004]
        self.renderer.update_scene(self.d,camera=cam,scene_option=self.opt);scene=self.renderer.scene;scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
        for axis in (0,1):
            for j in range(int(np.floor((center[axis]-.13)/.01)),int(np.ceil((center[axis]+.13)/.01))+1):
                a=np.array([center[0]-.13,center[1]-.13,8e-6]);b=np.array([center[0]+.13,center[1]+.13,8e-6]);a[axis]=b[axis]=j*.01
                h.line(scene,a,b,[.47,.50,.53,1],1)
        return Image.fromarray(self.renderer.render())

def make(rows,name,frames=180):
    states=[Replay(*r) for r in rows];video=c.OUT/'videos'/f'{name}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(video,fps=30,codec='libx264',quality=8,macro_block_size=1)
    views=[('FRONT',180,-20,.12),('SIDE',90,-18,.115),('OBLIQUE',135,-30,.13)]
    try:
        for f in range(frames):
            t=f/30;canvas=Image.new('RGB',(1920,470*len(states)),(23,28,34));draw=ImageDraw.Draw(canvas)
            for row,state in enumerate(states):
                actual=state.pose(t);r=state.row;y=row*470
                i=int(np.argmin(np.abs(state.tr['t']-max(t,1.5))));j=int(np.argmin(np.abs(state.tr['t']-1.5)))
                progress=float(state.forward*(state.tr['center'][i,0]-state.tr['center'][j,0])*1000) if t>=1.5 else 0.
                status='6s / 12s / HALF-STEP QUALIFIED' if state.qualified else ('NOMINAL GATES PASS; QUALIFICATION NOT ESTABLISHED' if r['feasible'] else 'DIAGNOSTIC: NOT QUALIFIED')
                draw.text((14,y+5),f"{state.label} | WAVE {r['wave']:g} | t={actual:.3f}s | {status}",font=BOLD,fill=(242,242,235))
                draw.text((14,y+34),f"D now={progress:.1f} mm | final={r['D']:.1f} mm | body / leg yaw / leg pitch={vector(r)[0]:.2f} / {vector(r)[1]:.2f} / {vector(r)[2]:.2f} deg",font=FONT,fill=(191,220,236))
                draw.text((14,y+62),'Flat ground; 1 Hz; no contact loss | same per-wave PD | orange = passive compliant tips | grid = 10 mm',font=FONT,fill=(233,176,129))
                for col,(label,az,el,dist) in enumerate(views):
                    draw.text((col*640+14,y+87),label,font=FONT,fill=(158,207,226));canvas.paste(state.panel(az,el,dist),(col*640,y+110))
            writer.append_data(np.asarray(canvas))
            if f in (0,60,90,150,frames-1):canvas.save(video.with_name(video.stem+f'_frame{f:03d}.png'))
    finally:
        writer.close()
        for st in states:st.renderer.close()
    reader=imageio.get_reader(video);n=reader.count_frames();shape=reader.get_data(n-1).shape;reader.close();assert n==frames
    info=dict(video=str(video),frames=n,fps=30,shape=list(shape),records=[str(st.path) for st in states],
        recorded_qpos_replayed_exactly=True,no_interpolation=True,physics_rerun=False,
        max_nearest_recorded_time_error_s=max(st.max_time_error for st in states))
    c.write(video.with_suffix('.json'),info);print(json.dumps(info),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wave',type=float,required=True);ap.add_argument('--name',required=True);ap.add_argument('--frames',type=int,default=180);ap.add_argument('--baseline-only',action='store_true');a=ap.parse_args()
    validation=c.read(c.DATA/'validation_index.json');old=baseline_rows().get(a.wave);rows=[]
    if old:rows.append((old,'PREVIOUS AMPLITUDES / CURRENT RUNTIME',validation.get(old['id'],{}).get('qualified',False)))
    if not a.baseline_only:
        v=c.read(c.DATA/'validated_selection.json')['waves'][f'{a.wave:g}'];assert v['qualified']
        rows.append((v['selected']['nominal'],'NEW AMPLITUDE SELECTION',True))
    assert rows;make(rows,a.name,a.frames)
