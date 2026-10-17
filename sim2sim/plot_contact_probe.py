import json
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
fig,axs=plt.subplots(1,2,figsize=(9.5,4),layout='constrained');series=[]
for name,title,color in [('contact_drop_slide_cpu_dt62_000','PhysX TGS','#b04b4b'),('contact_drop_slide_cpu_dt62_pgs_000','PhysX PGS','#d95f02')]:
    tr=np.load(s.ROOT/'runs'/name/'trajectory.npz');ix=tr['joint_names'].tolist().index('x');series.append((tr['t'],tr['q'][:,ix],tr['v'][:,ix],title,color))
m=mujoco.MjModel.from_xml_path(str(s.ROOT/'runs/contact_drop_slide_cpu_dt62_000/probe.xml'));h=.000015625;m.opt.timestep=h;m.geom_solref[:,0]=2*h;d=mujoco.MjData(m);d.qpos[0]=np.float32(.00065);d.qvel[1]=np.float32(.02);ts=[0.];xx=[0.];vv=[d.qvel[1]]
for k in range(round(.03/h)):mujoco.mj_step(m,d);ts.append((k+1)*h);xx.append(d.qpos[1]);vv.append(d.qvel[1])
series.append((np.array(ts),np.array(xx),np.array(vv),'MuJoCo rigid approximation','#237f8d'))
for t,x,v,name,c in series:axs[0].plot(t*1000,v*1000,color=c,label=name,lw=1.5);axs[1].plot(t*1000,x*1e6,color=c,label=name,lw=1.5)
for ax in axs:ax.set_xlim(0,15);ax.set_xticks([0,3,6,9,12,15]);ax.set_xlabel('Time (ms)');ax.grid(axis='y',alpha=.15)
axs[0].set_ylim(-15,25);axs[0].set_ylabel('Horizontal velocity (mm/s)');axs[0].axhline(0,color='black',lw=.7);axs[0].set_title('Impact-friction response');axs[1].set_ylim(0,160);axs[1].set_ylabel('Horizontal displacement (µm)');axs[1].set_title('Same mass, geometry and friction');axs[1].legend(fontsize=8,loc='lower right')
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/'figures'/('05_contact_solver_comparison.'+ext),dpi=240)
np.savez_compressed(s.ROOT/'audit/contact_probe_rigid_trajectory.npz',t=ts,x=xx,v=vv)
