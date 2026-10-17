"""Extract unedited encoded-video frames at diagnostic extrema for human QA.

This only prepares evidence. It does not claim that the frames were inspected.
It never changes trajectories, qualification, rankings, or model parameters.
"""
from pathlib import Path
import json
import study as s
import imageio.v2 as imageio

e, np = s.e, s.np


def main():
    audit = s.read(e.DATA / 'final_delivery_audit.json')
    assert audit['passed']
    table = s.read(e.DATA / 'comparison_table.json')
    videos = s.read(e.DATA / 'video_manifest.json')
    motion = s.read(e.DATA / 'achieved_joint_motion.json')
    qualified = [r for r in table if r['amplitudes']]
    unresolved = [r for r in table if not r['amplitudes']]
    chosen = {
        max(qualified, key=lambda r: r['D_mm'])['wave'],
        max(qualified, key=lambda r: max(abs(x) for x in r['amplitude_delta_deg'].values()))['wave'],
        max(qualified, key=lambda r: r['new_metrics']['coherent_roll_p95_deg'])['wave'],
        min(qualified, key=lambda r: r['wave'])['wave'],
        max(qualified, key=lambda r: r['wave'])['wave'],
    }
    chosen.update(r['wave'] for r in sorted(qualified,
        key=lambda r: max(abs(x) for x in r['amplitude_delta_deg'].values()), reverse=True)[:2])
    if unresolved:
        chosen.add(min(unresolved, key=lambda r: abs(r['wave']-8.))['wave'])
    out = e.OUT / 'visual_qa'
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    for video in videos:
        if not video['local'] and video['wave'] not in chosen:
            continue
        trace_path = Path(video['records'][1]).with_suffix('.npz')
        with np.load(trace_path) as tr:
            good = np.flatnonzero(tr['t'] >= 1.5)
            if video['local']:
                # The unchanged renderer follows tip index 18: segment 9, left.
                assert tr['tq'].shape[1] == 38
                signals = [('max_displayed_tip_bend_leg9_left', np.abs(tr['tq'][:, 18]))]
            else:
                signals = [
                    ('max_individual_world_roll', np.max(np.abs(tr['worldroll']), axis=1)),
                    ('max_geometric_center_height', tr['xpos'][:, :, 2].mean(axis=1)),
                ]
            reader = imageio.get_reader(video['video'])
            try:
                for label, values in signals:
                    index = int(good[np.argmax(values[good])])
                    t = float(tr['t'][index])
                    frame = min(video['frames']-1, max(0, round(t*30)))
                    target = out / f"{Path(video['video']).stem}_{label}_f{frame:03d}.png"
                    imageio.imwrite(target, reader.get_data(frame))
                    samples.append(dict(wave=video['wave'], selection_status=video['selection_status'],
                        local=video['local'], signal=label, trace_extremum_time_s=t,
                        encoded_frame=frame, encoded_time_s=frame/30.,
                        frame=str(target), video=video['video'], trace=str(trace_path),
                        note='Unedited encoded frame nearest the event; the event may fall between video frames'))
            finally:
                reader.close()
    summary = []
    for row in table:
        if not row['amplitudes']:
            summary.append(dict(wave=row['wave'], status=row['status'],
                                failed=row['diagnostic']['failed']))
            continue
        achieved = next(r for r in motion if r['wave'] == row['wave'])
        legs = np.asarray(achieved['leg_yaw_pitch_half_range_deg'])
        tips = np.asarray(achieved['tip_bend_half_range_deg'])
        summary.append(dict(wave=row['wave'], status=row['status'],
            max_individual_world_roll_peak_deg=max(achieved['world_roll_abs_peak_deg_per_body']),
            max_individual_world_roll_p95_deg=max(achieved['world_roll_abs_p95_deg_per_body']),
            leg_pitch_actual_half_range_min_max_deg=[float(legs[...,1].min()), float(legs[...,1].max())],
            tip_actual_half_range_min_max_deg=[float(tips.min()), float(tips.max())],
            nominal_metrics=row['new_metrics'], strict_policy_checks=row['strict_policy_checks']))
    result = dict(purpose='Prepared evidence for visual inspection, not a claim of completed visual review',
                  selected_three_view_waves=sorted(chosen), samples=samples, quality_summary=summary)
    s.write(e.DATA / 'visual_qa_candidates.json', result)
    print(json.dumps(dict(samples=len(samples), waves=sorted(chosen),
                           evidence=str(e.DATA / 'visual_qa_candidates.json')), indent=2))


if __name__ == '__main__':
    main()
