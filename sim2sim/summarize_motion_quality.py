"""Compare recorded kinematics; no simulation or fitted parameters."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s

rows=[]
runs=[(f'mu_coupled_rigid_dt31_case{i:02d}_000','MuJoCo',i) for i in range(8)]
runs += [('native_pgs_validation_gpu_dt62_000','PhysX GPU',None),('native_pgs_validation_cpu_dt62_000','PhysX CPU diagnostic',None)]
families=['body_yaw','legbase_yaw','legbase_pitch']
for name,engine,case_index in runs:
    folder=s.ROOT/'runs'/name
    tr={k:v for k,v in np.load(folder/'trajectory.npz').items()}
    result=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    t=tr['t'];active=(t>=2)&(t<=6)
    names=tr['joint_names'].tolist();bn=tr['body_names'].tolist()
    configs=[dict(wave=r['wave'],amplitudes=r['amplitudes']) for r in result['rows']]
    # Every saved state is post-step. The last command at t was evaluated at
    # t-command_dt; Mu substeps held that same command. Diagnostic only.
    command_dt=result['arguments'].get('command_dt',result['arguments']['dt'])
    ref=np.array([s.targets(names,max(0.,at-command_dt),command_dt,configs)[0] for at in t[active]])
    ids=[bn.index(f'link_body_{i}') for i in range(21)]
    xyz=tr['body'][:,:,ids].mean(2)
    design=np.column_stack([np.ones(active.sum()),np.sin(2*np.pi*t[active]),np.cos(2*np.pi*t[active])])
    for e,c in enumerate(configs):
        if case_index is None and e==0:continue
        i=case_index if case_index is not None else e-1
        q=tr['q'][active,e]
        r=dict(engine=engine,run=name,environment=e,case=i,wave=c['wave'],family='AB'[i%2],D_mm=result['rows'][e]['D_mm'])
        for family in families:
            js=[j for j,n in enumerate(names) if n.startswith(family+'_') and n!='body_yaw_1']
            err=q[:,js]-ref[:,e,js]
            beta=np.linalg.lstsq(design,q[:,js],rcond=None)[0]
            amplitude=np.degrees(np.hypot(beta[1],beta[2]))
            r[family+'_tracking_rmse_deg']=float(np.degrees(np.sqrt(np.mean(err**2))))
            r[family+'_actual_1hz_mean_amplitude_deg']=float(amplitude.mean())
        headtail=tr['body'][active,e,bn.index('link_body_0')]-tr['body'][active,e,bn.index('link_body_20')]
        yaw=np.unwrap(np.arctan2(headtail[:,1],-headtail[:,0]))
        r.update(lateral_drift_mm=float((xyz[active,e,1][-1]-xyz[active,e,1][0])*1000),
                 head_tail_yaw_span_deg=float(np.degrees(np.ptp(yaw))),
                 head_tail_yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])))
        rows.append(r)

paired=[]
for i in range(8):
    mu=next(r for r in rows if r['case']==i and r['engine']=='MuJoCo')
    gpu=next(r for r in rows if r['case']==i and r['engine']=='PhysX GPU')
    paired.append(dict(case=i,wave=mu['wave'],family=mu['family'],
        lateral_drift_difference_mm=abs(mu['lateral_drift_mm']-gpu['lateral_drift_mm']),
        heading_change_difference_deg=abs(mu['head_tail_yaw_change_deg']-gpu['head_tail_yaw_change_deg']),
        **{family+'_actual_amplitude_difference_deg':abs(mu[family+'_actual_1hz_mean_amplitude_deg']-gpu[family+'_actual_1hz_mean_amplitude_deg']) for family in families}))
s.write(s.ROOT/'audit/motion_quality_comparison.json',dict(rows=rows,paired=paired,
    window_s=[2,6],reference='Last applied position command before each saved post-step state.',
    amplitude='Mean joint 1 Hz sine/cosine least-squares amplitude, excluding commanded-static body_yaw_1; descriptive measurement, not optimization.',
    interpretation='Kinematic diagnostics only. No new acceptance threshold or equivalent-contact claim.',source_sha=s.sha(__file__)))
with (s.ROOT/'motion_quality_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.15,'savefig.dpi':250})
fig,axes=plt.subplots(2,3,figsize=(12,6.5))
panels=[(f+'_tracking_rmse_deg',label+' tracking RMS (deg)') for f,label in zip(families,['Body yaw','Leg yaw','Leg pitch'])]
panels += [('lateral_drift_mm','Lateral drift (mm)'),('head_tail_yaw_span_deg','Heading span (deg)'),('head_tail_yaw_change_deg','Heading change (deg)')]
for ax,(key,label) in zip(axes.flat,panels):
    for engine,color,marker,ls in [('MuJoCo','#2878a8','o','-'),('PhysX GPU','#d95f02','^','-'),('PhysX CPU diagnostic','#9a9a9a','x','--')]:
        bank=sorted([r for r in rows if r['engine']==engine],key=lambda r:r['case'])
        ax.plot(range(8),[r[key] for r in bank],color=color,marker=marker,ls=ls,label=engine,ms=4,lw=1.3)
    ax.set_ylabel(label);ax.set_xticks(range(8),[f"{p['wave']}/{p['family']}" for p in paired],rotation=45,fontsize=8)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    if 'drift' in key or 'change' in key:ax.axhline(0,color='#888',lw=.6)
fig.suptitle('Recorded motion diagnostics — same physical PD and target commands',fontsize=12)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
fig.subplots_adjust(bottom=.14,top=.90,hspace=.46,wspace=.38)
for ext in ['png','pdf','svg']:fig.savefig(s.ROOT/f'figures/11_motion_diagnostics.{ext}',bbox_inches='tight')
print(json.dumps(dict(maxima={k:max(r[k] for r in paired) for k in paired[0] if k not in ['case','wave','family']}),indent=2))
