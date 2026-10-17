"""Three-view comparison of recorded, unsmoothed physical trajectories."""
import argparse
import json
import sys
from pathlib import Path
from run_search import ROOT,OUT,pilot,np,mujoco
sys.path.insert(0,str(pilot.ROOT))

visual=pilot.load(pilot.ROOT/'render_pd_comparison.py','near_pd_visual')
demo=visual.demo

def state(record):
    case=dict(record['case'],label=record['config']['name'])
    s=demo.DemoState(case);pilot.apply_impedance(s.model,s.index,record['config'])
    s.model.light_diffuse[:]*=.35;s.model.light_specular[:]=0;s.model.light_castshadow[:]=0
    s.model.vis.headlight.ambient[:]=.30;s.model.vis.headlight.diffuse[:]=.45;s.model.vis.headlight.specular[:]=0
    s.model.geom_matid[s.ground]=-1
    chassis=set(map(int,s.index.chassis_body_ids))
    for g in range(s.model.ngeom):
        if int(s.model.geom_bodyid[g]) in chassis and s.original_group[g]==2 and s.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            s.original_rgba[g]=(.48,.15,.065,1)
    return s

def panel(s,az,el,distance,segment=None):
    s.apply_colors();cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth=az;cam.elevation=el;cam.distance=distance
    center=s.data.xpos[s.index.chassis_body_ids].mean(axis=0) if segment is None else s.data.xpos[int(s.index.chassis_body_ids[segment])]
    cam.lookat[:]=(center[0],center[1],.004)
    s.renderer.update_scene(s.data,camera=cam,scene_option=s.option)
    scene=s.renderer.scene;scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
    # Render-only 10mm ground grid, fixed to world coordinates. No model geom,
    # collision, mass, shadow, or force changes. Helps show foot clearance.
    for axis in (0,1):
        lo=int(np.floor((center[axis]-.13)/.01));hi=int(np.ceil((center[axis]+.13)/.01))
        for j in range(lo,hi+1):
            a=np.array([center[0]-.13,center[1]-.13,8e-6]);b=np.array([center[0]+.13,center[1]+.13,8e-6])
            a[axis]=b[axis]=j*.01
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).reshape(9),np.array([.47,.50,.53,1.]))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.,a,b)
            scene.ngeom+=1
    return demo.Image.fromarray(s.renderer.render())

def make(paths,output,start=0.,end=6.,zoom=False):
    records=[json.loads(p.read_text()) for p in paths]
    states=[state(r) for r in records];traces=[]
    for path in paths:
        with np.load(path.with_suffix('.npz')) as a:
            traces.append({key:a[key] for key in ('t','xpos','rootmat','bq','lq','tq')})
    views=[('SIDE LOCAL',90,-14,.042,9),('OBLIQUE LOCAL',135,-24,.042,9)] if zoom else [('FRONT',180,-22,.12,None),('SIDE',90,-20,.105,None),('OBLIQUE',135,-28,.12,None)]
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=demo.imageio.get_writer(output,fps=30,codec='libx264',quality=9,macro_block_size=1)
    frames=int(round((end-start)*30))+1
    try:
        for frame in range(frames):
            t=start+frame/30
            canvas=demo.Image.new('RGB',(len(views)*640,len(states)*440),(23,28,34));draw=demo.ImageDraw.Draw(canvas)
            for row,(s,r,tr) in enumerate(zip(states,records,traces)):
                visual.replay_pose(s,tr,t);y=row*440
                method=r.get('initialization',{}).get('method','original_xml_reset')
                label=('BASELINE' if r['config']['name']=='baseline' else 'PD CANDIDATE')
                failed=[k for k,v in r['gates'].items() if not v]
                status='DIAGNOSTIC: NOT ALL GATES PASSED' if failed else 'NOMINAL ABSOLUTE GATES PASSED; SEE VALIDATION'
                i=int(np.argmin(np.abs(tr['t']-t)));i0=int(np.argmin(np.abs(tr['t']-1.5)))
                displacement=pilot.opt.base.FORWARD_SIGN*float(tr['xpos'][i,:,0].mean()-tr['xpos'][i0,:,0].mean())*1000
                progress=f'D={displacement:.1f}mm' if t>=1.5 else 'SETTLING / CPG START'
                draw.text((14,y+5),f'{label} | {r["config"]["name"]} | wave={r["case"]["wave"]:g} | t={t:.2f}s | {progress}',font=demo.FONT_SMALL,fill=(239,239,235))
                draw.text((14,y+28),f'{method} | {status}',font=demo.FONT_SMALL,fill=(239,200,150))
                for col,(name,az,el,dist,seg) in enumerate(views):
                    draw.text((col*640+14,y+52),name+' | contact erasure OFF',font=demo.FONT_SMALL,fill=(150,211,235))
                    canvas.paste(panel(s,az,el,dist,seg),(col*640,y+80))
            writer.append_data(np.asarray(canvas))
            if frame in (0,min(90,frames-1)):
                canvas.save(output.with_name(output.stem+f'_frame{frame:03d}.png'))
    finally:
        writer.close()
        for s in states:s.close()
    reader=demo.imageio.get_reader(output)
    try:count=reader.count_frames();shape=list(reader.get_data(0).shape)
    finally:reader.close()
    info=dict(path=str(output),source_records=[str(p) for p in paths],frames=count,shape=shape,
              start=start,end=end,physical_pose_recording_dt=.001,smoothing=False,erasure=False,render_only_changes='lighting,albedo,shadow removal,10mm world grid at z=8um')
    output.with_suffix('.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--end',type=float,default=6.);p.add_argument('--zoom',action='store_true');a=p.parse_args()
    make(a.records,a.output,a.start,a.end,a.zoom)
