"""Actual motion diagnostics for the two interim videos; no selection changes."""
import study as s
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
e, np = s.e, s.np


def measure(record_path):
    path = s.Path(record_path).with_suffix('.npz')
    with np.load(path) as tr:
        good = tr['t'] >= 1.5
        half = lambda x: (np.quantile(x, .99, axis=0)-np.quantile(x, .01, axis=0))/2
        body = half(np.degrees(tr['bq'][good]))
        legs = half(np.degrees(tr['lq'][good]).reshape(-1, 19, 2, 2))
        tip = half(np.degrees(tr['tq'][good]).reshape(-1, 19, 2))
        world = tr['worldroll'][good]
        result = dict(record=record_path, trace_sha256=s.sha(path),
            half_range_definition='(P99-P1)/2, t>=1.5 s; not fitted harmonic amplitude',
            body_yaw_pitch_roll_half_range_deg=body.tolist(),
            leg_yaw_pitch_half_range_deg=legs.tolist(), tip_bend_half_range_deg=tip.tolist(),
            world_roll_abs_p95_deg_per_body=np.quantile(np.abs(world), .95, axis=0).tolist(),
            world_roll_abs_peak_deg_per_body=np.max(np.abs(world), axis=0).tolist(),
            max_body_joint_roll_abs_deg=float(np.max(np.abs(np.degrees(tr['bq'][good, :, 2])))),
            leg_pitch_half_range_min_max_deg=[float(legs[..., 1].min()), float(legs[..., 1].max())])
    return result


if __name__ == '__main__':
    demos = s.read(e.DATA / 'early_demo_results.json')
    fig, axes = plt.subplots(len(demos), 3, figsize=(13, 7), sharex=True)
    records = []
    for i, demo in enumerate(demos):
        measured = [measure(path) for path in demo['video']['records']]
        records.append(dict(wave=demo['wave'], status='interim candidate, not final optimum', old=measured[0], new=measured[1]))
        for data, label, color in zip(measured, ('Previous strict actual', 'New interim actual'), ('.5', '#df7200')):
            body = np.asarray(data['body_yaw_pitch_roll_half_range_deg'])[:, 0]
            legs = np.asarray(data['leg_yaw_pitch_half_range_deg'])
            values = (body, legs[..., 0], legs[..., 1])
            for ax, value, title in zip(axes[i], values, ('Body yaw', 'Leg yaw', 'Leg pitch')):
                mean = value if value.ndim == 1 else value.mean(axis=1)
                ax.plot(range(19), mean, 'o-', ms=3, color=color, label=label)
                if value.ndim > 1:
                    ax.fill_between(range(19), value.min(axis=1), value.max(axis=1), color=color, alpha=.15)
                ax.set(title=f'wave={demo["wave"]:g} | {title}', xlabel='Segment index', ylabel='Actual robust half-range (deg)')
                ax.grid(alpha=.2)
        axes[i, 0].legend(fontsize=8)
    fig.suptitle('INTERIM actual-motion check | (P99-P1)/2 over t>=1.5 s\nLeg lines average left/right half-ranges; shading spans left/right; no trajectory smoothing or selection changes')
    fig.tight_layout()
    out = e.OUT / 'figures/interim_actual_amplitude_comparison.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    s.write(e.DATA / 'early_joint_motion.json', records)
    print(s.json.dumps([dict(wave=r['wave'],
        old_pitch_range=r['old']['leg_pitch_half_range_min_max_deg'],
        new_pitch_range=r['new']['leg_pitch_half_range_min_max_deg'],
        old_world_roll_peak=max(r['old']['world_roll_abs_peak_deg_per_body']),
        new_world_roll_peak=max(r['new']['world_roll_abs_peak_deg_per_body'])) for r in records], indent=2))
