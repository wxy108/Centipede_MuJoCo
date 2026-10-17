"""Replay saved contact poses at real simulation speed; never run physics."""
from pathlib import Path
import argparse,json,xml.etree.ElementTree as ET
import physical_search as ps
import core as c
np,mj=c.np,c.mujoco
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
h=c.load(c.e.PREVIOUS/'render_flexion_demo.py','common_contact_render_helpers')
FONT=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',19)
BOLD=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',22)

class Replay:
    def __init__(self,path):
        self.path=Path(path);self.row=c.read(path);assert self.row['full']
        rawpath=Path(self.row['record']);self.raw=c.read(rawpath)
        self.channel=self.raw['channel'];self.tr=dict(np.load(rawpath.parent/'contact_trajectory.npz'))
        good=self.tr['t']>=1.5
        D=float(c.e.fx.opt.base.FORWARD_SIGN*np.diff(self.tr['center'][good][[0,-1],0])[0]*1000)
        assert D==self.row['D']==self.channel['raw_D_recomputed']
        manifest=c.read(path.parents[1]/'data/manifest.json')
        for source,digest in manifest['frozen_inputs'].items():assert c.sha(source)==digest,source
        original=c.e.runtime_model(c.PLANTS['current']);tree=c.e.runtime_tree(c.PLANTS['current'])
        ET.SubElement(tree.getroot().find('asset'),'texture',type='skybox',builtin='gradient',rgb1='.69 .74 .79',rgb2='.69 .74 .79',width='256',height='1536')
        self.m=mj.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
        for key in ('body_mass','body_inertia','geom_size','geom_pos','geom_quat','jnt_axis','jnt_range'):
            np.testing.assert_array_equal(getattr(original,key),getattr(self.m,key))
        self.ix=c.e.fx.opt.tip.LeanIndex(self.m);self.d=mj.MjData(self.m)
        ground=c.e.fx.opt.tip.stage1.ground_geom_id(self.m);h.light_and_color(self.m,ground)
        self.original_color=self.m.geom_rgba.copy();self.legs=[]
        for pair in range(19):
            for side in ('L','R'):
                root=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,f'link_leg_{pair}_{side}_0');assert root>0
                bodies=set()
                for b in range(1,self.m.nbody):
                    at=b
                    while at>0:
                        if at==root:bodies.add(b);break
                        at=int(self.m.body_parentid[at])
                self.legs.append(np.flatnonzero(np.isin(self.m.geom_bodyid,list(bodies))))
        self.option=mj.MjvOption();self.option.geomgroup[:]=0;self.option.geomgroup[[0,2]]=1;self.option.sitegroup[:]=0
        self.m.vis.map.znear=1e-4;self.renderer=mj.Renderer(self.m,height=360,width=640)
        self.max_time_error=0.;self.rendered_masks=[]

    def pose(self,t):
        i=int(np.argmin(np.abs(self.tr['pose_t']-t)));actual=float(self.tr['pose_t'][i])
        self.max_time_error=max(self.max_time_error,abs(actual-t))
        self.d.qpos[:]=self.tr['qpos'][i];self.d.time=actual;mj.mj_kinematics(self.m,self.d)
        np.testing.assert_array_equal(self.d.qpos,self.tr['qpos'][i])
        self.active=np.zeros(38,bool)
        if actual>=1.5-1e-10:
            if 'erased_legs' in self.tr:
                j=np.searchsorted(self.tr['mask_t'],actual+1e-10,side='right')-1
                assert j>=0;self.active=self.tr['erased_legs'][j].copy()
            else:self.active=np.repeat(self.tr['erased_pairs'],2)
        self.m.geom_rgba[:]=self.original_color
        for leg in np.flatnonzero(self.active):self.m.geom_rgba[self.legs[leg]]=(1.,.025,.025,1.)
        self.rendered_masks.append(dict(time=actual,erased_legs=np.flatnonzero(self.active).tolist()))
        return actual

    def panel(self,az,el,distance,local=False):
        center=self.d.xpos[self.ix.chassis_body_ids[10]].copy() if local else self.d.xpos[self.ix.chassis_body_ids].mean(0)
        cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE
        cam.azimuth=az;cam.elevation=el;cam.distance=distance;cam.lookat[:]=[center[0],center[1],max(.003,center[2]) if local else .004]
        self.renderer.update_scene(self.d,camera=cam,scene_option=self.option)
        scene=self.renderer.scene;scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
        for axis in (0,1):
            for j in range(int(np.floor((center[axis]-.13)/.01)),int(np.ceil((center[axis]+.13)/.01))+1):
                a=np.array([center[0]-.13,center[1]-.13,8e-6]);b=np.array([center[0]+.13,center[1]+.13,8e-6]);a[axis]=b[axis]=j*.01
                h.line(scene,a,b,[.47,.50,.53,1],1)
        return Image.fromarray(self.renderer.render())

def make(paths,name,local=False):
    assert name.replace('_','').isalnum();states=[Replay(Path(p)) for p in paths]
    assert len({s.channel['uniform_sha256'] for s in states})==1,'Compare identical intended random masks.'
    assert len({s.row['task']['duration'] for s in states})==1
    assert len({s.row['pd_readback']['active_pd_sha256'] for s in states})==1
    video=c.ROOT/'outputs/videos'/f'{name}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
    views=[('LOCAL SIDE',90,-12,.032),('LOCAL OBLIQUE',135,-26,.035)] if local else [('FRONT',180,-20,.12),('SIDE',90,-18,.115),('OBLIQUE',135,-30,.13)]
    frames=round(states[0].row['task']['duration']*30)
    writer=imageio.get_writer(video,fps=30,codec='libx264',quality=8,macro_block_size=1)
    try:
        for f in range(frames):
            canvas=Image.new('RGB',(640*len(views),470*len(states)),(23,28,34));draw=ImageDraw.Draw(canvas)
            for row,state in enumerate(states):
                actual=state.pose(f/30);r=state.row;ch=state.channel;y=row*470
                status='ENGINEERING GATES PASS' if r['feasible'] else 'DIAGNOSTIC: ENGINEERING GATES FAIL'
                draw.text((14,y+5),f"WAVE {r['wave']:g} | p={ch['probability']:g} | seed={ch['seed']} | t={actual:.3f}s | {status}",font=BOLD,fill=(242,242,235))
                a=r['amplitudes'];draw.text((14,y+34),f"Final D={r['D']:.1f} mm | body / leg yaw / pitch={a['body']:.2f} / {a['leg_yaw']:.2f} / {a['leg_pitch']:.2f} deg",font=FONT,fill=(191,220,236))
                kind='TRANSIENT, safe restoration' if 'erased_legs' in state.tr else 'PERSISTENT'
                draw.text((14,y+60),f"{kind} | red = actual erased legs ({int(state.active.sum())}/38) | same PD, flat, 1 Hz | one sample",font=FONT,fill=(255,150,132))
                for col,(label,az,el,dist) in enumerate(views):
                    draw.text((col*640+14,y+86),label,font=FONT,fill=(158,207,226));canvas.paste(state.panel(az,el,dist,local),(col*640,y+110))
            writer.append_data(np.asarray(canvas))
            if f in (0,60,90,150,frames-1):canvas.save(video.with_name(video.stem+f'_frame{f:03d}.png'))
    finally:
        writer.close()
        for state in states:state.renderer.close()
    reader=imageio.get_reader(video);n=reader.count_frames();shape=reader.get_data(n-1).shape;reader.close();assert n==frames
    receipt=dict(video=str(video),frames=n,fps=30,shape=list(shape),assessment_files=[str(s.path) for s in states],
                 source_hashes={str(s.path):c.sha(s.path) for s in states},render_source_sha256=c.sha(__file__),
                 physics_steps=0,exact_recorded_qpos=True,no_interpolation=True,real_time_playback=True,
                 max_nearest_pose_time_error_s=max(s.max_time_error for s in states),local_views=local,
                 rendered_actual_erasure_masks=[s.rendered_masks for s in states])
    c.write(video.with_suffix('.json'),receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='rendered_actual_erasure_masks'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',nargs='+',required=True);p.add_argument('--name',required=True);p.add_argument('--local',action='store_true')
    a=p.parse_args();make(a.rows,a.name,a.local)
