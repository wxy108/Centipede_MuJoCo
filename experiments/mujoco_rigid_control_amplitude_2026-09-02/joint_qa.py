"""Achieved joint motion versus commands; distinguish selected and diagnostic gaits."""
import json
from pathlib import Path
import engine as e
np=e.np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    selection=json.loads((e.DATA/'validated_selection.json').read_text());out=e.OUT/'figures';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for wave,row in selection['waves'].items():
        r=row['selected']['nominal'] if row['selected'] else row['diagnostic']
        if not r:continue
        qualified=bool(row['selected']);stage='delivery' if qualified else 'delivery_diagnostic'
        tid=r['id'];path=e.TRIALS/stage/tid/f'{tid}.npz'
        tr=np.load(path);good=tr['t']>=1.5
        body=np.degrees(tr['bq'][good]);leg=np.degrees(tr['lq'][good]).reshape(-1,19,2,2);tip=np.degrees(tr['tq'][good]).reshape(-1,19,2)
        half=lambda x:(np.quantile(x,.99,axis=0)-np.quantile(x,.01,axis=0))/2
        b,l,t=half(body),half(leg),half(tip);a=r['amplitudes']
        fig,axs=plt.subplots(2,3,figsize=(12,6.7),sharex=True)
        panels=[('Body joint yaw',b[:,0],a['body']),('Body joint pitch',b[:,1],0.),('Body joint roll (passive)',b[:,2],0.),
                ('Leg yaw',l[...,0],a['leg_yaw']),('Leg pitch',l[...,1],a['leg_pitch']),('Tip bend (passive)',t,0.)]
        for ax,(name,values,command) in zip(axs.flat,panels):
            if values.ndim==1:ax.plot(range(19),values,'o-',ms=3,label='Achieved')
            else:
                ax.plot(range(19),values[:,0],'o-',ms=3,label='Left');ax.plot(range(19),values[:,1],'s-',ms=3,label='Right')
            if command:
                if name == 'Body joint yaw':
                    ax.plot(range(19), [0.] + [command] * 18, color='.45', ls='--', label='Command amplitude')
                else:
                    ax.axhline(command,color='.45',ls='--',label='Command amplitude')
            ax.set(title=name,xlabel='Segment index',ylabel='Robust half-range (deg)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        status='VALIDATED' if qualified else 'DIAGNOSTIC ONLY / NOT QUALIFIED'
        fig.suptitle(f'{status} wave={wave} | achieved motion: (P99-P1)/2, t >= 1.5 s\nBody yaw segment 0 command is zero; pitch/roll/tip departures from zero are compliant motion')
        suffix='' if qualified else '_diagnostic'
        fig.tight_layout();figure=out/f"joint_motion_w{wave.replace('.','p')}{suffix}.png";fig.savefig(figure,dpi=150);plt.close(fig)
        world=tr['worldroll'][good];indices=np.arange(world.shape[1])
        world_figure=out/f"world_roll_w{wave.replace('.','p')}{suffix}.png"
        fig,axs=plt.subplots(1,2,figsize=(11,3.6),sharex=True)
        axs[0].fill_between(indices,np.quantile(world,.05,axis=0),np.quantile(world,.95,axis=0),alpha=.2,color='#007a93',label='Time P5-P95')
        axs[0].plot(indices,world.mean(axis=0),'o-',color='#007a93',ms=3,label='Time mean')
        axs[1].plot(indices,np.quantile(np.abs(world),.95,axis=0),'o-',ms=3,label='Absolute P95')
        axs[1].plot(indices,np.max(np.abs(world),axis=0),'s--',ms=3,label='Absolute peak')
        for ax in axs:
            ax.axhline(0,color='.5',lw=.7);ax.set(xlabel='Chassis body index',ylabel='World roll (deg)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{status} wave={wave} | individual body orientation, t >= 1.5 s\nNot relative joint angles and not a whole-body average')
        fig.tight_layout();fig.savefig(world_figure,dpi=150);plt.close(fig)
        records.append(dict(wave=float(wave),status=row['status'],source=str(path),command_deg=a,body_yaw_segment_0_command_deg=0,
            half_range_definition='(P99-P1)/2 over t>=1.5 s, not a fitted harmonic amplitude',
            body_yaw_pitch_roll_half_range_deg=b.tolist(),leg_yaw_pitch_half_range_deg=l.tolist(),tip_bend_half_range_deg=t.tolist(),
            body_mean_angle_deg=body.mean(axis=0).tolist(),
            body_joint_roll_abs_peak_deg=np.max(np.abs(body[...,2]),axis=0).tolist(),
            world_roll_abs_p95_deg_per_body=np.quantile(np.abs(tr['worldroll'][good]),.95,axis=0).tolist(),
            world_roll_abs_peak_deg_per_body=np.max(np.abs(tr['worldroll'][good]),axis=0).tolist(),
            geometric_center_mean_height_mm=float(tr['xpos'][good,:,2].mean()*1000),
            figure=str(figure),world_roll_figure=str(world_figure)))
    (e.DATA/'achieved_joint_motion.json').write_text(json.dumps(records,indent=2))
    print(json.dumps([dict(wave=r['wave'],figure=r['figure']) for r in records],indent=2))
