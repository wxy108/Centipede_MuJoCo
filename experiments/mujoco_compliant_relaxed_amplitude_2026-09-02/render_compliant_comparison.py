"""Presentation-only copy of the preserved parent renderer.

State reconstruction, cameras, physics checks and panels are unchanged.
Only the local legend is scoped to this all-compliant comparison (version 3).
"""
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import engine as e
r=e.load(e.PREVIOUS/'render_flexion_demo.py','paired_render_helpers')
np,mujoco=e.np,e.mujoco
Image,ImageDraw,ImageFont=r.Image,r.ImageDraw,r.ImageFont
FONT=r.FONT;BOLD=r.FONT_BOLD
import imageio.v2 as imageio


class State:
    def __init__(self,path):
        self.path=Path(path);self.record=json.loads(self.path.read_text())
        source=Path(self.record['model_path']);assert e.fx.sha256(source)==self.record['model_sha256']
        self.rigid=self.record['topology']=='rigid'
        original=mujoco.MjModel.from_xml_path(str(source))
        tree=e.runtime_tree(source)
        ET.SubElement(tree.getroot().find('asset'),'texture',type='skybox',builtin='gradient',
            rgb1='.69 .74 .79',rgb2='.69 .74 .79',width='256',height='1536')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
        for key in ('body_mass','body_inertia','geom_size','geom_pos','geom_quat','jnt_axis','dof_damping'):
            assert np.array_equal(getattr(original,key),getattr(self.model,key)),key
        self.index=e.ab.RigidTipIndex(self.model) if self.rigid else e.fx.opt.tip.LeanIndex(self.model)
        e.fx.pilot.apply_impedance(self.model,self.index,self.record['config'])
        self.data=mujoco.MjData(self.model);self.tr=dict(np.load(self.path.with_suffix('.npz')))
        self.ground=e.fx.opt.tip.stage1.ground_geom_id(self.model)
        self.sites=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,f'site_foot_{s}_{side}') for s in range(19) for side in range(2)])
        r.light_and_color(self.model,self.ground)
        if self.rigid:
            for g in range(self.model.ngeom):
                name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
                if '_tip_visual' in name:self.model.geom_rgba[g]=(.03,.7,.9,1)
        self.model.vis.map.znear=1e-4
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=0;self.option.geomgroup[[0,2]]=1;self.option.sitegroup[:]=0
        self.renderer=mujoco.Renderer(self.model,height=360,width=640)

    def pose(self,t):
        i=int(np.argmin(np.abs(self.tr['t']-t)));d,ix,tr=self.data,self.index,self.tr
        if self.rigid:d.qpos[:]=tr['qpos'][i]
        else:
            d.qpos[:3]=tr['xpos'][i,0]
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.asarray(tr['rootmat'][i],dtype=float).reshape(9));d.qpos[3:7]=quat
            d.qpos[ix.qpos_addresses[ix.body_positions]]=tr['bq'][i,:,0]
            d.qpos[ix.qpos_addresses[ix.pitch_positions]]=tr['bq'][i,:,1]
            d.qpos[ix.roll_qpos_addresses]=tr['bq'][i,:,2]
            d.qpos[ix.qpos_addresses[ix.leg_positions.reshape(-1)]]=tr['lq'][i]
            d.qpos[ix.tip_qpos_addresses]=tr['tq'][i]
        d.time=float(tr['t'][i]);mujoco.mj_kinematics(self.model,d);return i

    def endpoints(self):
        m,d=self.model,self.data;actual=d.site_xpos[self.sites[18]].copy()
        if self.rigid:
            g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'link_leg_9_L_tip_visual')
            center=d.geom_xpos[g];direction=actual-center;direction/=np.linalg.norm(direction)
            pivot=center-direction*m.geom_size[g,1];neutral=actual.copy()
        else:
            j=int(self.index.tip_joint_ids[18]);b=int(m.jnt_bodyid[j]);pivot=d.xpos[b].copy()
            axis=d.xmat[b].reshape(3,3)@m.jnt_axis[j];theta=d.qpos[m.jnt_qposadr[j]];v=actual-pivot
            neutral=pivot+v*np.cos(theta)-np.cross(axis,v)*np.sin(theta)+axis*np.dot(axis,v)*(1-np.cos(theta))
        return pivot,actual,neutral

    def panel(self,az,el,distance,local):
        m,d,ix=self.model,self.data,self.index;pivot,actual,neutral=self.endpoints()
        center=(pivot+actual)*.5 if local else d.xpos[ix.chassis_body_ids].mean(axis=0)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth=az;cam.elevation=el;cam.distance=distance;cam.lookat[:]=center
        if not local:cam.lookat[2]=.004 if self.record['terrain']=='flat' else center[2]-.001
        self.renderer.update_scene(d,camera=cam,scene_option=self.option);scene=self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        if self.record['terrain']=='flat' and not local:
            for axis in (0,1):
                for j in range(int(np.floor((center[axis]-.12)/.01)),int(np.ceil((center[axis]+.12)/.01))+1):
                    a=np.array([center[0]-.12,center[1]-.12,8e-6]);b=np.array([center[0]+.12,center[1]+.12,8e-6]);a[axis]=b[axis]=j*.01
                    r.line(scene,a,b,[.47,.50,.53,1],1)
        panel=Image.fromarray(self.renderer.render())
        if local:
            draw=ImageDraw.Draw(panel);p,n,a=(r.project(scene,v) for v in (pivot,neutral,actual))
            for u in np.arange(0,1,.2):draw.line([tuple(p+u*(n-p)),tuple(p+min(1.,u+.12)*(n-p))],fill=(0,211,255),width=3)
            draw.line([tuple(p),tuple(a)],fill=(0,211,255) if self.rigid else (255,99,0),width=2)
        return panel

    def verify(self):
        body=foot=0.
        for t in np.linspace(0,float(self.tr['t'][-1]),24):
            i=self.pose(t)
            body=max(body,float(np.max(np.abs(self.data.xpos[self.index.chassis_body_ids]-self.tr['xpos'][i]))))
            foot=max(foot,float(np.max(np.abs(self.data.site_xpos[self.sites]-self.tr['fq'][i]))))
        assert max(body,foot)<1e-7,(body,foot)
        return dict(record=str(self.path),body_error_m=body,foot_error_m=foot)


def make(paths,out,local=False,end=6.,labels=None):
    out=Path(out)
    if out.exists() and out.with_suffix('.json').exists():
        info=json.loads(out.with_suffix('.json').read_text())
        if info['records']==list(map(str,paths)) and info['frames']==int(end*30) and info.get('labels')==labels and (not local or info.get('local_legend_version')==3):return info
    states=[State(p) for p in paths];checks=[s.verify() for s in states]
    views=[('TOP LOCAL',90,-90,.012),('SIDE LOCAL',0,-18,.012),('OBLIQUE LOCAL',45,-30,.014)] if local else [('FRONT',180,-20,.12),('SIDE',90,-18,.115),('OBLIQUE',135,-30,.13)]
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(out,fps=30,codec='libx264',quality=9,macro_block_size=1)
    try:
        for frame in range(int(end*30)):
            t=frame/30;canvas=Image.new('RGB',(1920,440*len(states)),(23,28,34));draw=ImageDraw.Draw(canvas)
            for row,s in enumerate(states):
                s.pose(t);rec=s.record;y=row*440;task=rec['task'];a=task['amplitudes']
                status='ABSOLUTE GATES PASSED' if rec['absolute_feasible'] else 'DIAGNOSTIC: GATES FAILED'
                label='RIGID LEG / NO TIP DOF' if s.rigid else 'FLEXIBLE TIP / LOCAL-Z'
                if labels:label=labels[row]
                objective=f" | D(1.5-6s)={rec['metrics']['displacement_mm']:.2f}mm" if labels else ''
                draw.text((12,y+4),f"{label} | {rec['terrain'].upper()} | wave={task['wave']:g} | t={t:.2f}s | {status}{objective}",font=BOLD,fill=(242,240,233))
                legend='orange: compliant tip; cyan dashed: neutral; BOTH ROWS COMPLIANT' if local else 'orange: flexible; cyan: rigid'
                if labels and not local and all(not state.rigid for state in states):
                    legend='orange: flexible tip; PD fixed within wave'
                draw.text((12,y+30),f"Amplitude body/yaw/pitch = {a['body']:.2f} / {a['leg_yaw']:.2f} / {a['leg_pitch']:.2f} deg | {legend} | erasure OFF",font=FONT,fill=(195,215,231))
                for col,(name,az,el,distance) in enumerate(views):
                    draw.text((col*640+12,y+55),name,font=FONT,fill=(152,211,237));canvas.paste(s.panel(az,el,distance,local),(col*640,y+80))
            writer.append_data(np.asarray(canvas))
            if frame in (0,90,150):canvas.save(out.with_name(out.stem+f'_frame{frame:03d}.png'))
    finally:
        writer.close()
        for s in states:s.renderer.close()
    reader=imageio.get_reader(out);count=reader.count_frames();shape=list(reader.get_data(0).shape);reader.close()
    info=dict(video=str(out),frames=count,shape=shape,pose_checks=checks,unsmoothed=True,physics_changes=False,
        camera='identical settings per row; tracking geometric center',records=list(map(str,paths)),labels=labels,local_legend_version=3 if local else None)
    out.with_suffix('.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
    return info


