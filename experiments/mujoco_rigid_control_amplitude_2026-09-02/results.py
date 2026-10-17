"""Evidence-linked paired-ablation report and search summaries."""
import json
import engine as e


def paired_report():
    rows=json.loads((e.DATA/'paired_ablation.json').read_text())
    lines=['# Fully rigid leg versus current compliant tip','',
        'Centipede Model only. Same amplitudes, wave-specific PD, mass, neutral leg geometry, frequency, phases, contact settings, initialization procedure and simulation step. No knee reinstated; all 38 distal tip joints removed by rigid aggregation, not by an artificially large spring.','',
        'Forward displacement is the world-forward displacement of the unweighted mean of the 21 chassis-body positions (geometric center, not mass-weighted COM), from 1.5 s to 5.999 s of a 6 s simulation. Values below are not full-duration displacement. Rough is diagnostic only.','',
        '| Terrain | Wave | Flexible D (mm) | Rigid D (mm) | Change | Absolute gates flexible / rigid |',
        '|---|---:|---:|---:|---:|---|']
    for terrain in ('flat','rough'):
        for w in (2.5,3.):
            a,b=[next(r for r in rows if r['terrain']==terrain and r['task']['wave']==w and r['topology']==t) for t in ('flexible','rigid')]
            da,db=a['metrics']['displacement_mm'],b['metrics']['displacement_mm']
            lines.append(f"| {terrain} | {w:g} | {da:.3f} | {db:.3f} | {100*(db/da-1):+.2f}% | {'PASS' if a['absolute_feasible'] else 'FAIL'} / {'PASS' if b['absolute_feasible'] else 'FAIL'} |")
    lines+=['','At the tested identical PD and amplitudes, removing tip compliance did not improve either forward displacement or gate feasibility. This does not prove rigid legs can never work with a different controller; no rigid-specific tuning is mixed into this controlled comparison. The flexible model remains the primary plant.','',
        'The flat flexible cases pass absolute gates, which is not the same as passing every legacy non-regression / horizon / timestep check. Neither rough case is certified.','',
        '## Individual records and metrics','']
    keys=('root_roll_abs_p95_deg','coherent_roll_p95_deg','front_roll_p95_deg','center_height_ptp_mm','ground_force_peak_bw','chassis_force_peak_bw','body_deltaomega_10ms_peak','loaded_foot_slip_p95_mm_s','left_right_force_imbalance_rms_bw','tip_angle_p95_deg','tip_angle_peak_deg')
    for r in rows:
        lines += [f"### {r['terrain']} — wave {r['task']['wave']:g} — {r['topology']}",'',
            f"Record: [{r['id']}]({r['record_path'].replace(chr(92),'/')})",'',
            'Failed applicable absolute gates: '+(', '.join(r['failed']) or 'none')+'.','',
            '| Metric | Value |','|---|---:|']
        for k in keys:
            if k in r['metrics']:lines.append(f"| {k} | {r['metrics'][k]:.6g} |")
        lines.append('')
    lines += ['## Demos','', 'Every video contains three simultaneous views; top row flexible (orange tip), bottom row rigid (cyan tip). Raw saved poses, no motion smoothing. Local enlargement files show neutral/actual tip direction. Contact erasure is OFF and labelled.','']
    for v in json.loads((e.DATA/'paired_video_manifest.json').read_text()):
        p=v['video'].replace('\\','/');lines.append(f'- [{p.rsplit("/",1)[-1]}]({p})')
    lines+=['','## Resource-only implementation change','',
        'Search and rendering compile an in-memory XML copy with a 64 MiB arena reservation instead of the source 2.18 GB reservation. Only legacy size/nstack and size/njmax are replaced by size/memory; nconmax and all physical settings remain unchanged. The source model files are not edited. This is memory capacity, not solver accuracy or dynamics tuning.',
        'The official [MuJoCo 3.3.7 size documentation](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#size) defines this setting. Exact model-array and dynamics checks are recorded in outputs/data/runtime_arena_equivalence.json.','']
    (e.ROOT/'PAIRED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(e.ROOT/'PAIRED_RESULTS.md'))


if __name__=='__main__':paired_report()
