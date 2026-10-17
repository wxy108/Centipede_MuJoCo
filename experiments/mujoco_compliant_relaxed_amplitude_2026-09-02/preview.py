"""Explicitly provisional amplitude figure; never used by selection."""
import study as s
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


if __name__ == '__main__':
    rows = []
    for path in (s.e.OUT / 'assessments').glob('*.json'):
        try:
            rows.append(s.read(path))
        except json.JSONDecodeError:
            pass  # A worker may still be writing this live progress item.
    old = s.read(s.e.PARENT / 'outputs/data/validated_selection.json')
    waves = s.e.WAVES
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)
    selected = {}
    for wave in waves:
        feasible = [r for r in rows if r['wave'] == wave and r['feasible']]
        selected[wave] = max(feasible, key=lambda r: r['D']) if feasible else None
    for ax, key, title in zip(axes, s.KEYS, ('Body yaw', 'Leg yaw', 'Leg pitch')):
        previous = [old['waves'][f'{w:g}']['selected'] for w in waves]
        ax.plot(waves, [r['nominal']['amplitudes'][key] if r else float('nan') for r in previous],
                's--', color='.55', label='Previous strict, validated')
        ax.plot(waves, [selected[w]['amplitudes'][key] if selected[w] else float('nan') for w in waves],
                'o-', color='#df7200', label='New 6 s candidate, NOT validated')
        ax.set(title=title, xlabel='Coupled wave number', ylabel='Command amplitude (deg)')
        ax.set_xticks(waves)
        ax.tick_params(axis='x', labelrotation=60)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'PROVISIONAL PREVIEW | {len(rows):,} initial conditions assessed\nSame compliant tip and frozen PD per wave; refinement, 12 s and half-step checks pending')
    fig.tight_layout()
    out = s.e.OUT / 'figures/initial_amplitude_preview.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    s.write(s.e.DATA / 'initial_preview_snapshot.json', dict(
        status='provisional, not a selected recommendation', completed=len(rows),
        candidates=selected, affects_search_or_selection=False, figure=str(out)))
    print(out)
