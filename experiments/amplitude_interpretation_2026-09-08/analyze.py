"""Read-only kinematic interpretation of the sealed amplitude selections."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FINE=ROOT.parent/'amplitude_optimization_recovery_2026-09-08/fine_step'
sys.path.insert(0,str(FINE))
import core as c
np,mj=c.np,c.mujoco

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

def main():
    seal=sha(FINE/'FINAL_RECEIPT.json')
    sealed={Path(r['path']).as_posix():r['sha256'] for r in c.read(FINE/'FINAL_RECEIPT.json')['files']}
    for name in ['outputs/data/validated_selection.json','outputs/data/final_summary.json']:
        assert sha(FINE/name)==sealed[name]
    frozen=c.read(c.DATA/'manifest.json')['frozen_inputs']
    for source in frozen:assert sha(source['path'])==source['sha256'],source['path']
    selected=c.read(c.DATA/'validated_selection.json');summary=c.read(c.DATA/'final_summary.json')
    previous={r['wave']:r for r in summary['waves']};rows=[];sources=[];shapes={}
    for key,cell in selected['waves'].items():
        if not cell['qualified']:continue
        r=cell['selected']['nominal'];wave=r['wave'];path=Path(r['record']).parent/'motion.npz'
        assert sha(path)==sealed[path.relative_to(FINE).as_posix()]
        z=np.load(path);sources.append(dict(path=str(path),sha256=sha(path),matched_sealed_input=True))
        m=c.e.runtime_model(c.history()['plant']);ix=c.e.fx.opt.tip.LeanIndex(m)
        c.e.fx.pilot.apply_impedance(m,ix,r['task']['config']);d=mj.MjData(m)
        mask=(z['pose_t']>=1.5)&(z['pose_t']<=5.999);tt=z['pose_t'][mask];yy=[];xx=[]
        for q in z['qpos'][mask]:
            d.qpos[:]=q;mj.mj_kinematics(m,d)
            p=d.xpos[ix.chassis_body_ids,:2].copy();p-=p.mean(axis=0)
            axis=np.linalg.svd(p,full_matrices=False)[2][0]
            if np.dot(axis,p[-1]-p[0])<0:axis=-axis
            lateral=np.array([-axis[1],axis[0]])
            xx.append(1000*p@axis);yy.append(1000*p@lateral)
        xx,yy=np.array(xx),np.array(yy)
        design=np.column_stack([np.ones(len(tt)),np.sin(2*np.pi*tt),np.cos(2*np.pi*tt)])
        coeff=np.linalg.lstsq(design,yy,rcond=None)[0]
        # RMS across nodes of each node's fitted 1-Hz amplitude in its instantaneous body frame.
        amp=np.hypot(coeff[1],coeff[2]);shape_rms_amp=float(np.sqrt(np.mean(amp**2)))
        gate=(z['t']>=1.5)&(z['t']<=5.999);bq=z['bq'][gate,1:,0];tq=z['t'][gate]
        joint_design=np.column_stack([np.ones(len(tq)),np.sin(2*np.pi*tq),np.cos(2*np.pi*tq)])
        joint_coeff=np.linalg.lstsq(joint_design,bq,rcond=None)[0]
        joint_harmonic=np.hypot(joint_coeff[1],joint_coeff[2])*180/np.pi
        complex_joint=joint_coeff[1]+1j*joint_coeff[2]
        segment=np.arange(1,19)/18
        spatial_grid=np.linspace(-9,9,1801)
        correlations=np.abs(np.exp(2j*np.pi*spatial_grid[:,None]*segment)@complex_joint)/np.sum(np.abs(complex_joint))
        command_coherence=float(np.abs(np.exp(2j*np.pi*wave*segment)@complex_joint)/np.sum(np.abs(complex_joint)))
        kinematic_tortuosity={}
        for label,angle in [('common_20deg',20.),('selected_command',r['amplitudes']['body'])]:
            ratios=[]
            for phase in np.linspace(0,2*np.pi,256,endpoint=False):
                d.qpos[:]=m.qpos0
                body=np.deg2rad(angle)*np.sin(phase-2*np.pi*wave*np.arange(19)/18);body[0]=0
                d.qpos[ix.qpos_addresses[ix.body_positions]]=body;mj.mj_kinematics(m,d)
                p=d.xpos[ix.chassis_body_ids]
                ratios.append(float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()/np.linalg.norm(p[-1]-p[0])))
            kinematic_tortuosity[label]=max(ratios)
        cfg=r['task']['config'];prev=previous[wave]
        row=dict(wave=wave,command_body_deg=r['amplitudes']['body'],actual_joint_half_P95_P05_deg=prev['motion']['actual_joint_half_P95_P05_deg'][0],
            actual_joint_median_1Hz_amplitude_deg=float(np.median(joint_harmonic)),yaw_k_multiplier=cfg['yaw_k'],yaw_d_multiplier=cfg['yaw_d'],
            actual_yaw_kp_median=float(np.median(ix.kp[ix.body_positions][1:])),actual_yaw_kv_median=float(np.median(ix.kv[ix.body_positions][1:])),
            midline_median_lateral_halfspan_mm=float(np.median(np.ptp(yy,axis=1)/2)),midline_1Hz_amplitude_rms_across_nodes_mm=shape_rms_amp,
            midline_median_spatial_rms_mm=float(np.median(np.sqrt(np.mean(yy**2,axis=1)))),nominal_id=r['id'],D_mm=r['D'],
            joint_1Hz_commanded_spatial_mode_coherence=command_coherence,joint_1Hz_best_spatial_wave=float(spatial_grid[np.argmax(correlations)]),joint_1Hz_best_spatial_mode_coherence=float(correlations.max()),
            ideal_command_kinematic_tortuosity_max=kinematic_tortuosity)
        rows.append(row);shapes[wave]=(tt,xx,yy)
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),physics_simulations_run=0,method='Stored qpos -> mj_kinematics only. All 21 chassis-node origins, XY projection, 1.5..5.999 s. Per-frame centroid removal and signed first PCA axis remove rigid translation/yaw. Lateral halfspan is a descriptive geometric extent, not fitted traveling-wave amplitude; 1-Hz amplitude is also provided separately. A separate ideal-command geometry scan uses qpos0, zero pitch/roll targets, the actual body-yaw phase law at 256 evenly spaced phases, and no contact or dynamics solution. Those ratios demonstrate the geometry of a gate, not a completed simulation or causal attribution of a selected optimum.',
        seal_sha256_before=seal,seal_sha256_after=sha(FINE/'FINAL_RECEIPT.json'),verified_frozen_input_count=len(frozen),sources=sources,rows=rows)
    assert result['seal_sha256_before']==result['seal_sha256_after']
    (ROOT/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.8));waves=list(c.WAVES);indexed={r['wave']:r for r in rows}
    values=lambda name:[indexed[w][name] if w in indexed else np.nan for w in waves]
    axes[0].plot(waves,values('command_body_deg'),'o-',label='Command joint yaw amplitude')
    axes[0].plot(waves,values('actual_joint_median_1Hz_amplitude_deg'),'s--',label='Actual joint yaw, fitted 1 Hz')
    axes[0].set(xlabel='Wave number',ylabel='Degrees',title='Joint angles');axes[0].legend(fontsize=8)
    axes[1].plot(waves,values('midline_median_lateral_halfspan_mm'),'o-',label='Median lateral halfspan')
    axes[1].plot(waves,values('midline_1Hz_amplitude_rms_across_nodes_mm'),'s--',label='1 Hz lateral amplitude, RMS over nodes')
    axes[1].set(xlabel='Wave number',ylabel='mm',title='Actual midline shape, rigid motion removed');axes[1].legend(fontsize=8)
    for wave,color in [(1.5,'#26769b'),(7.,'#c46730')]:
        tt,xx,yy=shapes[wave];k=int(np.argmax(np.ptp(yy,axis=1)))
        axes[2].plot(xx[k],yy[k],'o-',ms=3,c=color,label=f'w{wave:g}: largest observed lateral extent')
    axes[2].set(xlabel='Along-body position (mm)',ylabel='Lateral position (mm)',title='Stored shapes at the same geometric scale');axes[2].axis('equal');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.18)
    for ax in axes[:2]:ax.annotate('NQ',(8,.025),xycoords=('data','axes fraction'),ha='center',color='#a34444',fontsize=8)
    fig.suptitle('Interpretation check: a joint angle is not a midline displacement amplitude')
    fig.text(.5,.015,'No physics was rerun. Wave-specific PD and optimized leg amplitudes differ. This is a diagnostic, not a common-controller experiment.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(ROOT/'amplitude_meaning.png',dpi=170);plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
