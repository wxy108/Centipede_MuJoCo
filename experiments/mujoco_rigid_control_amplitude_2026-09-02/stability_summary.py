"""Display posture/contact outcomes separately from the displacement objective."""
import json
from pathlib import Path
import refine as rf
e,s=rf.e,rf.s
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    selected=json.loads((e.DATA/'validated_selection.json').read_text())
    assert selected['validation_complete']
    manifest=json.loads(s.MANIFEST.read_text())
    motion={r['wave']:r for r in json.loads((e.DATA/'achieved_joint_motion.json').read_text())}
    rows=[]
    keys=[('coherent_roll_p95_deg','Whole-body roll P95 (deg)',10.),
          ('front_roll_p95_deg','Front-body roll P95 (deg)',12.),
          ('center_height_ptp_mm','Center height range (mm)',2.),
          ('ground_force_peak_bw','Peak total ground force (BW)',2.5),
          ('loaded_foot_slip_p95_mm_s','Loaded foot slip P95 (mm/s)',None),
          ('body_roll_speed_peak','Body roll joint peak speed (rad/s)',5.)]
    for wave in e.WAVES:
        task=next(t for t in manifest['references'] if t['wave']==wave)
        base,_=s.load_reference(task)
        choice=selected['waves'][f'{wave:g}']['selected']
        envelope=None
        if choice:
            j=motion[wave]
            envelope=dict(worst_body_world_roll_p95_deg=max(j['world_roll_abs_p95_deg_per_body']),
                worst_body_world_roll_peak_deg=max(j['world_roll_abs_peak_deg_per_body']),
                worst_relative_roll_joint_peak_deg=max(j['body_joint_roll_abs_peak_deg']),
                geometric_center_mean_height_mm=j['geometric_center_mean_height_mm'],
                post_selection_diagnostic_not_a_new_gate=True)
        rows.append(dict(wave=wave,reference=base['metrics'],post_selection_roll_envelope=envelope,
            optimized=choice['nominal']['metrics'] if choice else None,
            selected_id=choice['nominal']['id'] if choice else None,
            dt_D_relative_difference=choice['halfstep_D_relative_difference'] if choice else None,
            validation_checks=choice['checks'] if choice else None))
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for ax,(key,title,cap) in zip(axs.flat,keys):
        ax.plot([r['wave'] for r in rows],[r['reference'][key] for r in rows],
                'o--',color='.65',ms=3,label='Prior fixed amplitude; not all stable')
        good=[r for r in rows if r['optimized'] is not None]
        ax.scatter([r['wave'] for r in good],[r['optimized'][key] for r in good],
                   s=35,color='#007a93',label='Validated finite-set best')
        if cap:ax.axhline(cap,color='#ae6030',ls=':',label='Absolute cap; relative caps can be tighter')
        ax.set(title=title,xlabel='Coupled wave number');ax.grid(alpha=.2)
        ax.set_xticks(e.WAVES);ax.tick_params(axis='x',labelrotation=60)
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Posture and contact checks | flexible tip, flat ground, frozen PD per wave\nUnresolved waves have no selected marker; these plots do not replace the full gate audit')
    fig.tight_layout();out=e.OUT/'figures';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'stability_by_wave.png',dpi=170);plt.close(fig)
    (e.DATA/'stability_comparison.json').write_text(json.dumps(rows,indent=2))
    lines=['# Selected gait stability summary','',
        'Nominal 6 s metrics, t >= 1.5 s. Selection also passed 12 s and half-timestep tests. Gates remain engineering criteria, not biological parameter calibration.','',
        '| Wave | D (mm) | Whole / front roll P95 (deg) | Height range (mm) | Peak ground force (BW) | Tip P95 / peak (deg) | Half-step D difference |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['optimized']
        if m is None:continue
        lines.append(f"| {row['wave']:g} | {m['displacement_mm']:.3f} | {m['coherent_roll_p95_deg']:.2f} / {m['front_roll_p95_deg']:.2f} | {m['center_height_ptp_mm']:.3f} | {m['ground_force_peak_bw']:.3f} | {m['tip_angle_p95_deg']:.2f} / {m['tip_angle_peak_deg']:.2f} | {100*row['dt_D_relative_difference']:.2f}% |")
    lines += ['', '## Individual-body roll envelope', '',
        'Whole-body averaging can cancel opposite local rotations. These post-selection diagnostics do not change the gates or the selected parameters. Passing is not equivalent to zero roll.', '',
        '| Wave | Worst individual body absolute P95 / peak (deg) | Worst relative roll joint peak (deg) | Mean geometric-center height (mm) |',
        '|---:|---:|---:|---:|']
    for row in rows:
        j=row['post_selection_roll_envelope']
        if j is None:continue
        lines.append(f"| {row['wave']:g} | {j['worst_body_world_roll_p95_deg']:.2f} / {j['worst_body_world_roll_peak_deg']:.2f} | {j['worst_relative_roll_joint_peak_deg']:.2f} | {j['geometric_center_mean_height_mm']:.3f} |")
    lines += ['', 'Large single-step angular acceleration remains a diagnostic in the raw records; the jump gates operate on 10 ms angle/velocity changes and explicit velocity/force limits. Passing is not a claim that every instantaneous acceleration is biologically calibrated.', '',
              'Full values and validation rollouts: outputs/data/stability_comparison.json. Actual per-joint motion: outputs/data/achieved_joint_motion.json and outputs/figures/joint_motion_w*.png.', '']
    (e.ROOT/'STABILITY_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))
