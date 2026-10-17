"""CPU-only scientific pose replay; no physics steps and no native-GPU rendering."""
import json
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import ConvexHull
from matplotlib.font_manager import findfont
import shared as s
out=s.ROOT/'demos';out.mkdir(exist_ok=True)
specs=[('mu_coupled_rigid_dt31_case03_000',0,'MuJoCo | 31.25 us',(40,120,168)),('native_pgs_validation_gpu_dt62_000',4,'Isaac Lab / PhysX GPU | 62.5 us',(217,95,2))]
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'assets/common.xml'));datasets=[]
for name,e,label,color in specs:
    f=s.ROOT/'runs'/name;tr={k:v.copy() for k,v in np.load(f/'trajectory.npz').items()};roots=np.load(f/'reconstructed_root_poses.npz')['root'];qids=[m.joint(n).qposadr[0] for n in tr['joint_names'].tolist()];bn=tr['body_names'].tolist();xyz=tr['body'][:,e,[bn.index(f'link_body_{i}') for i in range(21)]].mean(1);D=-(xyz[:,0]-xyz[200,0])*1000
    datasets.append(dict(name=name,e=e,label=label,color=color,tr=tr,roots=roots,qids=qids,origin_y=float(roots[0,e,1]),D=D,centroid_x=xyz[:,0],d=mujoco.MjData(m)))
AA=2;W,H=1280,800;scale=1170/.20;left=65;x_min=-.15
font_path=findfont('DejaVu Sans')
def font(size):return ImageFont.truetype(font_path,round(size*AA))
fonts={n:font(n) for n in [11,13,15,17,22]}
def pts(points,baseline):
    points=np.asarray(points);return np.c_[(left+(-points[:,0]-x_min)*scale)*AA,(baseline-(.6*points[:,1]+.8*points[:,2])*scale)*AA]
def line(draw,p,fill,width=1):draw.line([tuple(v) for v in p],fill=fill,width=max(1,round(width*AA)))
def text(draw,xy,t,size=15,fill='#222'):draw.text((xy[0]*AA,xy[1]*AA),t,font=fonts[size],fill=fill)
def paint(k):
    global x_min
    x_min=-float(np.mean([ds['centroid_x'][k] for ds in datasets]))-.10
    im=Image.new('RGB',(W*AA,H*AA),'white');draw=ImageDraw.Draw(im);t=k*.01
    text(draw,(40,18),'Shared physical model and PD',22);text(draw,(40,49),'Wave 2.5 | body / leg yaw / leg pitch = 16 / 72 / 24 deg | 1 Hz',15)
    text(draw,(1045,24),f't = {t:4.2f} s',17)
    for row,ds in enumerate(datasets):
        top=88+row*215;baseline=top+135
        draw.rectangle((28*AA,top*AA,1252*AA,(top+198)*AA),outline='#b7b7b7',width=AA)
        text(draw,(42,top+10),ds['label'],17,ds['color'])
        value='scoring starts at 2 s' if t<2 else f'Forward since 2 s: {ds["D"][k]:.1f} mm'
        text(draw,(785,top+13),value,15)
        for x in np.arange(-.26,.151,.02):
            p=pts([[x,-.014,0],[x,.014,0]],baseline)
            if left*AA<=p[0,0]<=(left+1170)*AA:line(draw,p,'#e5e5e5')
        for y in [-.012,0,.012]:line(draw,pts([[-(x_min+.20),y,0],[-x_min,y,0]],baseline),'#e9e9e9')
        d=ds['d'];d.qpos[:7]=ds['roots'][k,ds['e']];d.qpos[1]-=ds['origin_y'];d.qpos[ds['qids']]=ds['tr']['q'][k,ds['e']];mujoco.mj_kinematics(m,d)
        shapes=[]
        for g in range(1,m.ngeom):
            pos=d.geom_xpos[g];rot=d.geom_xmat[g].reshape(3,3);depth=-.8*pos[1]+.6*pos[2];typ=m.geom_type[g]
            if typ==mujoco.mjtGeom.mjGEOM_MESH:
                mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];vertices=m.mesh_vert[start:start+m.mesh_vertnum[mid]]@rot.T+pos;xy=pts(vertices,baseline);hull=ConvexHull(xy);shapes.append((depth,'mesh',xy[hull.vertices],ds['color'],0))
            elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:
                ends=np.array([pos-rot[:,2]*m.geom_size[g,1],pos+rot[:,2]*m.geom_size[g,1]]);shapes.append((depth,'capsule',pts(ends,baseline),(110,110,110),m.geom_size[g,0]*scale*AA))
            elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:shapes.append((depth,'sphere',pts([pos],baseline),(65,65,65),m.geom_size[g,0]*scale*AA))
        for _,typ,p,color,r in sorted(shapes,key=lambda v:v[0]):
            if typ=='mesh':draw.polygon([tuple(v) for v in p],fill=color,outline=(35,65,80))
            else:
                if typ=='capsule':draw.line([tuple(v) for v in p],fill=color,width=max(1,round(2*r)))
                for x,y in p:draw.ellipse((x-r,y-r,x+r,y+r),fill=color)
        text(draw,(44,top+177),'Forward = -world X. Same scale and shared following camera in both panels.',11,fill='#555')
    gx,gy,gw,gh=90,576,1100,145
    for value in [0,50,100,150,200]:
        yy=gy+gh-value/200*gh;line(draw,[(gx*AA,yy*AA),((gx+gw)*AA,yy*AA)],'#e4e4e4');text(draw,(45,yy-8),str(value),11,fill='#555')
    for at in [2,3,4,5,6]:
        xx=gx+(at-2)/4*gw;line(draw,[(xx*AA,(gy+gh)*AA),(xx*AA,(gy+gh+5)*AA)],'#777');text(draw,(xx-5,gy+gh+8),str(at),11)
    for ds in datasets:
        tt=ds['tr']['t'][200:];yy=ds['D'][200:];p=np.c_[(gx+(tt-2)/4*gw)*AA,(gy+gh-yy/200*gh)*AA];line(draw,p,'#d5d5d5')
        if k>=200:line(draw,p[:k-199],ds['color'],2)
    if t>=2:
        xx=gx+(t-2)/4*gw;line(draw,[(xx*AA,gy*AA),(xx*AA,(gy+gh)*AA)],'#444',1)
    text(draw,(90,548),'Forward displacement (mm), measured from 2 s',13);text(draw,(580,753),'Time (s)',13)
    text(draw,(40,779),'Recorded pose replay rendered with common geometry; no physics is recomputed.',11,fill='#666')
    return im.resize((W,H),Image.Resampling.LANCZOS)
video=out/'sim2sim_w2p5_B_closeup_replay.mp4'
with imageio.get_writer(video,fps=25,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-preset','fast']) as writer:
    for k in range(0,601,4):
        frame=paint(k);writer.append_data(np.array(frame))
        if k==400:frame.save(out/'sim2sim_w2p5_B_closeup_preview.png')
    for _ in range(25):writer.append_data(np.array(frame))
s.write(out/'sim2sim_w2p5_B_closeup_replay.json',dict(source_runs=[d['name'] for d in datasets],source_env_indices=[d['e'] for d in datasets],physics_recomputed=False,rendering='CPU orthographic projection of common collision geometry; silhouettes and capsules',physics_samples_dt_s=.01,video_frame_dt_s=.04,final_frame_hold_s=1,forward_metric='unweighted link_body_0..20 centroid, negative world X, 2..6 s',final_displacements_mm=[float(d['D'][-1]) for d in datasets],source_sha=s.sha(__file__)))
print(str(video))
