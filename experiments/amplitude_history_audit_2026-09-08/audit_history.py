"""Read historical optimization evidence and plot it; no simulation or PD edits."""
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
MODEL = OUT.parents[1]
RL = MODEL.parent / 'Centipede_IsaacLab_RL'
sources = {}


def read(path):
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')


def jread(path):
    return json.loads(read(path))


fine = MODEL / 'experiments/amplitude_optimization_recovery_2026-09-08/fine_step'
seal = fine / 'FINAL_RECEIPT.json'
read(seal)
expected_seal = 'dec797624d36d81e08cdbe1496dd26901485ddb13433e2cb3897276212b84a57'
assert sources[str(seal)] == expected_seal
v1_path = RL / 'outputs/analysis/p8_wave_amplitude_map_2026-07-20/data/wave_conditioned_gait_map.json'
v3_path = RL / 'outputs/analysis/p8_wave_map_v3_confirmation_2026-07-21/data/wave_conditioned_gait_map_v3.json'
v4_path = RL / 'outputs/report/p8_flat_roll_causal_summary_2026-07-21/data/flat_locked_gait_map_v4_13knot.json'
v1, v3, v4 = [jread(p) for p in (v1_path, v3_path, v4_path)]
realized_path = RL / 'outputs/analysis/p8_realized_amplitude_audit_2026-07-21/data/realized_amplitude_by_wave.csv'
realized = [r for r in csv.DictReader(read(realized_path).splitlines()) if r['condition'] == 'flat']
assert len(realized) == 13
v3_rows = {float(r['wave']): r for r in v3['wave_map']}
for row in realized:
    assert math.isclose(float(row['body_yaw_command_rad']), .4*v3_rows[float(row['wave'])]['body_amp'], abs_tol=1e-12)

# Check the actual per-environment input files, not just report prose.
v4_run = RL / 'outputs/analysis/p8_flat_map_v4_13knot_confirmation_2026-07-21'
meta = jread(v4_run / 'meta/run_meta_MAP_flat_s5150.json')
gains = jread(v4_run / 'data/gains_MAP_flat_s5150.json')['gains']
impedance = jread(v4_run / 'data/impedance_MAP_flat_s5150.json')['impedance']
v4_rows = {float(r['wave']): r for r in v4['wave_map']}
pd_vectors = set()
for env, assignment in meta['env_to_assignment'].items():
    if assignment.get('guard'):
        continue
    k = int(env)
    row = v4_rows[float(assignment['wave'])]
    assert math.isclose(float(gains[k]['body_amp']), row['body_amp'], abs_tol=1e-12)
    for family in ('body_yaw', 'legbase_yaw'):
        assert impedance[k][family] == [row[family+'_kp'], row[family+'_kv']]
    pd_vectors.add(json.dumps(impedance[k], sort_keys=True))
assert len(pd_vectors) > 1

old_run = MODEL / 'experiments/mujoco_wave_amplitude_natural_gait_optimization_2026-08-31'
selection = jread(old_run / 'outputs/data/final_certified_selection.json')
cold = jread(old_run / 'outputs/data/final_selection_cold_replay_and_erasure.json')
old_rows = []
for wave, item in selection['waves'].items():
    row = item.get('selected')
    if row is None:
        continue
    assert cold['waves'][wave]['exact_repeat']
    assert cold['waves'][wave]['amplitudes_deg'] == [row['body_yaw_amplitude_deg'], row['leg_yaw_amplitude_deg'], row['leg_pitch_amplitude_deg']]
    old_rows.append(row)
old_rows.sort(key=lambda r: r['wave'])
assert len(old_rows) == 9

# Locate every winner in retained trial files without executing historical code.
selected_by_id = {r['candidate_id']: r for r in old_rows}
matches = {key: 0 for key in selected_by_id}
for filename in selection['candidate_files']:
    path = Path(filename)
    for line in read(path).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        winner = selected_by_id.get(row.get('candidate_id'))
        if winner is None:
            continue
        for key in ('wave', 'body_yaw_amplitude_deg', 'leg_yaw_amplitude_deg', 'leg_pitch_amplitude_deg', 'forward_displacement_m'):
            assert row[key] == winner[key], (path, key)
        matches[winner['candidate_id']] += 1
assert all(n >= 1 for n in matches.values())

common_pd_path = MODEL / 'experiments/mujoco_biophysical_terrain_contact_audit_2026-08-26/outputs/data/controller_selection.json'
common_pd = jread(common_pd_path)['selected_impedance']
current = jread(fine / 'outputs/data/selected_gaits.json')
current_vectors = {json.dumps({k:v for k,v in r['pd_config'].items() if k != 'name'}, sort_keys=True) for r in current['gaits']}

report = {
    'scope': 'Historical file analysis only; no simulation, controller, PD, plant, or gate changes.',
    'user_constraint': 'One identical complete PD configuration across all waves; hold it fixed during amplitude optimization. Do not tune PD.',
    'rl_v1': v1['wave_map'],
    'rl_v4': [dict(r, body_yaw_deg=math.degrees(.4*r['body_amp'])) for r in v4['wave_map']],
    'rl_v4_distinct_complete_pd_vectors': len(pd_vectors),
    'rl_realized_audit_source_version': 'v3; do not pair the wave-1.5 measurement with the later v4 micro-refinement',
    'rl_v3_realized_flat_rows': realized,
    'mujoco_august31_selected': old_rows,
    'mujoco_august31_raw_winner_matches': matches,
    'historical_common_pd_source': str(common_pd_path),
    'historical_common_pd': common_pd,
    'current_recovery_distinct_pd_vectors': len(current_vectors),
    'sealed_recovery_receipt_sha256': sources[str(seal)],
    'input_sha256': sources,
}

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for payload, color, style, label in [(v1, '#999999', '--', 'July 20 v1'), (v4, '#24567a', '-', 'July 21 v4')]:
    rows = payload['wave_map']
    axes[0].plot([r['wave'] for r in rows], [math.degrees(.4*r['body_amp']) for r in rows], style, marker='o', color=color, label=label, markersize=4)
axes[0].set_title('IsaacLab RL: optimized gait maps\nPD varies across wave numbers', loc='left', fontsize=12)
axes[0].set_xlim(1.25, 4.75)
axes[0].set_xticks([1.5, 2, 2.5, 3, 3.5, 4, 4.5])
axes[0].legend(loc='lower left', frameon=False)
axes[0].set_ylabel('Commanded body-yaw joint amplitude (degrees)')
axes[1].plot([r['wave'] for r in old_rows], [r['body_yaw_amplitude_deg'] for r in old_rows], 'o-', color='#167c64', markersize=5)
axes[1].set_title('MuJoCo: August 31 amplitude-only search\nOne common PD; older plant and gait gates', loc='left', fontsize=12)
for r in old_rows:
    axes[1].annotate(f"{r['body_yaw_amplitude_deg']:.1f}", (r['wave'], r['body_yaw_amplitude_deg']), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
axes[1].set_xlim(1.25, 6.25)
axes[1].set_xticks([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6])
for ax in axes:
    ax.set_ylim(0, 37)
    ax.set_xlabel('Wave number')
    ax.grid(alpha=.18)
fig.text(.075, .015, 'All selected waves are shown. Neither result is strictly monotonic; the two models are not a matched causal comparison.', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .06, 1, 1))
fig.savefig(OUT / 'historical_body_amplitudes.png', dpi=180)
fig.savefig(OUT / 'historical_body_amplitudes.pdf')
plt.close(fig)
assert hashlib.sha256(seal.read_bytes()).hexdigest() == expected_seal
(OUT / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({
    'rl_v4_degrees': [[r['wave'], round(math.degrees(.4*r['body_amp']), 3)] for r in v4['wave_map']],
    'rl_v4_distinct_pd': len(pd_vectors),
    'august31_raw_winners_verified': len(matches),
    'current_distinct_pd': len(current_vectors),
    'old_common_pd': common_pd,
    'source_files_hashed': len(sources),
    'receipt_unchanged': True,
}, indent=2))
