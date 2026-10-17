"""Publish transparent per-wave results and unsmoothed three-view demos."""
import json
from pathlib import Path
import refine as rf
e,s,np=rf.e,rf.s,rf.np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import render
    allrows=rf.collect(['broad','refine','polish','boundary'])
    validation=json.loads((e.DATA/'validated_selection.json').read_text())
    assert validation['validation_complete'],'Do not publish a still-running validation result'
    manifest=json.loads(s.MANIFEST.read_text())
    plot=e.OUT/'figures';plot.mkdir(parents=True,exist_ok=True)
    lines=['# Flat amplitude-only optimization','',
        'Primary plant: latest local-Z compliant tip. Rigid remains a matched ablation; it did not replace the primary model. No Robot data or code changed.','',
        'Only body yaw, leg yaw and leg pitch amplitudes vary. All other physical and controller settings remain frozen per wave, including the historically saved PD for that wave. PD differs between some wave numbers, so this is conditional per-wave optimization, not an isolated causal wave-number comparison. Body and leg wave numbers remain equal; frequency is 1 Hz; erasure OFF.','',
        'The initial domain is body [0,36], leg yaw [0,90], leg pitch [0,60] degrees. The registered broad bank is followed by two distinct-seed local refinements and one bounded coupled interpolation pass from feasible anchors toward faster near-feasible endpoints (at most two endpoints and five fractions per wave). Gates are unchanged. Only a finite tested optimum is reported; a mathematical continuous global optimum is NOT proved.','',
        'Objective: world-forward geometric chassis-center displacement from t=1.5 s to 5.999 s, maximized only after all absolute, posture and registered contact/slip/work non-regression gates pass. The inherited 90%-of-reference displacement floor is diagnostic only; see OBJECTIVE_CLARIFICATION.md. Nominal candidates are independently tested at 12 s and at half timestep (0.125 ms), in descending-displacement batches of three until a candidate qualifies or the nominal feasible list is exhausted; displacement must agree within 10% across timesteps. All nominally faster candidates than the selected gait are thus checked. Failed waves remain unresolved, not substituted with a visually attractive or merely fast failure.','',
        '| Wave | Baseline D mm | Body / leg yaw / leg pitch amplitude deg | Qualified D mm | Gain | Status |',
        '|---:|---:|---|---:|---:|---|']
    table=[];videos=[]
    for w in e.WAVES:
        result=validation['waves'][f'{w:g}'];base_task=next(t for t in manifest['references'] if t['wave']==w)
        base,_=s.load_reference(base_task);db=base['metrics']['displacement_mm'];chosen=result['selected']
        full=[r for r in allrows if r['wave']==w and r['full']]
        nominal=chosen['nominal'] if chosen else result['diagnostic']
        if chosen:
            a=nominal['amplitudes'];da=nominal['D'];gain=100*(da/db-1)
            values=' / '.join(f'{a[k]:.3f}' for k in s.KEYS)
            lines.append(f'| {w:g} | {db:.3f} | {values} | {da:.3f} | {gain:+.2f}% | qualified |')
        else:
            a=None;da=None;gain=None
            lines.append(f'| {w:g} | {db:.3f} | — | — | — | unresolved |')
        if nominal:
            tid=nominal['id'];stage='delivery' if chosen else 'delivery_diagnostic'
            after=e.TRIALS/stage/tid/f'{tid}.json'
            bt=e.task_id(base_task);before=e.TRIALS/'delivery_reference'/bt/f'{bt}.json'
            replay=json.loads(after.read_text());original=json.loads(Path(nominal['record']).read_text())
            # Handle NaNs explicitly for failed diagnostic runs; qualified
            # metrics are finite for every gating field.
            for key,x in original['metrics'].items():
                assert np.array_equal(np.asarray(x),np.asarray(replay['metrics'][key]),equal_nan=True),key
            stem=f"{'optimized' if chosen else 'unresolved_diagnostic'}_w{w:g}_three_views".replace('.','p')
            path=e.OUT/'videos'/f'{stem}.mp4'
            video=render.make([str(before),str(after)],path,labels=['PRIOR FIXED AMPLITUDES',
                'AMPLITUDE OPTIMIZED / VALIDATED' if chosen else 'DIAGNOSTIC ONLY / NOT QUALIFIED'])
            video.update(wave=w,selection_status='qualified' if chosen else 'unresolved')
            videos.append(video)
        table.append(dict(wave=w,baseline_D_mm=db,amplitudes=a,D_mm=da,gain_percent=gain,status=result['status'],
            diagnostic=nominal if not chosen else None,search_count=sum(r['wave']==w for r in allrows),
            nominal_pass_count=sum(r['wave']==w and r['feasible'] for r in allrows)))
    qualified=[r for r in table if r['status']=='qualified']
    lines += ['',f'Tested distinct amplitude conditions: {len(allrows)}. Waves qualified after all checks: {len(qualified)} / {len(e.WAVES)}.','',
        'Baseline displacement alone does not indicate baseline gate feasibility. Diagnostic records are retained for unresolved waves. A failed amplitude search is not proof that no feasible amplitude exists; it is evidence that the declared finite search did not find and validate one with the frozen PD.','',
        '## Coverage and unresolved waves','']
    for row in table:
        lines.append(f"- wave={row['wave']:g}: {row['search_count']} candidates, {row['nominal_pass_count']} nominal all-gate passes; {row['status']}.")
        if row['diagnostic']:
            dr=row['diagnostic'];reason=', '.join(dr['failed']) if dr['failed'] else 'nominal passes, but long-horizon or half-step validation fails'
            lines.append('  Best diagnostic (NOT a selected gait) failed: '+reason+'.')
    lines += ['', '## Artifacts','',
        '- Paired rigid/flexible experiment: PAIRED_RESULTS.md and outputs/data/paired_ablation.json.',
        '- Exact parameter banks and provenance: outputs/data/*_manifest.json.',
        '- All candidate outcomes, including strict early-failure proofs: outputs/trials/.',
        '- Validation details: outputs/data/validated_selection.json.',
        '- Posture/contact summary: STABILITY_RESULTS.md and outputs/figures/stability_by_wave.png.',
        '- Achieved joint motion, distinct from commanded amplitude: outputs/figures/joint_motion_w*.png.',
        '- Final provenance/selection/video audit: outputs/data/final_delivery_audit.json.',
        '- Videos: outputs/videos/. Every delivery demo contains front, side and oblique views. Unresolved wave demos are explicitly labelled DIAGNOSTIC ONLY / NOT QUALIFIED.',
        '- Prior models and selected profile files were not overwritten. No new candidate was silently promoted to the canonical baseline.','']
    (e.ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (e.DATA/'amplitude_results_table.json').write_text(json.dumps(table,indent=2))
    reusable=dict(scope='Centipede Model; flat ground; flexible local-Z tip; erasure OFF',
        model_path=manifest['plant'],model_sha256=manifest['plant_sha256'],
        controller_frequency_hz=manifest['frequency_hz'],body_leg_wave_coupled=True,
        controller_law_unchanged=True,canonical_baseline_replaced=False,
        global_optimum_proved=False,selected_profiles={},unresolved_waves=[])
    for wave,result in validation['waves'].items():
        if result['selected']:
            choice=result['selected'];nominal=choice['nominal']
            reusable['selected_profiles'][wave]=dict(amplitudes_deg=nominal['amplitudes'],
                frozen_PD=manifest['PD_profiles_frozen'][wave],nominal_record=nominal['record'],
                validation_records=[check['record'] for check in choice['checks']],
                nominal_displacement_mm=nominal['D'],
                halfstep_displacement_relative_difference=choice['halfstep_D_relative_difference'])
        else:reusable['unresolved_waves'].append(float(wave))
    (e.DATA/'selected_amplitude_profiles.json').write_text(json.dumps(reusable,indent=2))
    (e.DATA/'delivery_video_manifest.json').write_text(json.dumps(videos,indent=2))
    (e.DATA/'optimized_video_manifest.json').write_text(json.dumps([v for v in videos if v['selection_status']=='qualified'],indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True)
    for ax,key,title in zip(axes,s.KEYS,('Body yaw','Leg yaw','Leg pitch')):
        ax.axhline(manifest['references'][0]['amplitudes'][key],color='.5',ls='--',label='Prior fixed amplitude')
        ax.scatter([r['wave'] for r in qualified],[r['amplitudes'][key] for r in qualified],color='#007a93',s=45,label='Validated finite-set best')
        ax.set(xlabel='Coupled wave number',ylabel='Amplitude (deg)',title=title);ax.grid(alpha=.2)
        ax.set_xticks(e.WAVES);ax.tick_params(axis='x',labelrotation=60)
    axes[0].legend(fontsize=8);fig.suptitle('Only validated amplitudes shown; unresolved waves are blank')
    fig.tight_layout();fig.savefig(plot/'amplitudes_by_wave.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot([r['wave'] for r in table],[r['baseline_D_mm'] for r in table],'o--',color='.5',label='Fixed amplitude reference (not all stable)')
    ax.scatter([r['wave'] for r in qualified],[r['D_mm'] for r in qualified],color='#007a93',s=60,label='Validated optimized amplitude')
    ax.set(xlabel='Coupled wave number',ylabel='Forward displacement (mm, t=1.5 to 5.999 s)',title='Stable-pose constrained displacement; flexible tip, flat ground')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(plot/'displacement_by_wave.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(qualified=len(qualified),table=table,videos=len(videos)),indent=2))


if __name__=='__main__':
    main()
    import subprocess
    import sys
    for script in ('joint_qa.py','stability_summary.py','audit_delivery.py','summary_zh.py'):
        subprocess.run([sys.executable,str(e.ROOT/script)],cwd=e.ROOT,check=True)
