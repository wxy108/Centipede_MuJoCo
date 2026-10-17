"""Feasibility and separate metrics, never a weighted reward."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
rows=[json.loads(p.read_text()) for p in OUT.glob('*.json') if p.name not in ('comparison.json','selection.json')]
rows=[r for r in rows if 'config' in r and 'metrics' in r]
baselines={(r['case']['wave'],r['duration'],r['dt']):r for r in rows if r['config']['name']=='baseline'}
results=[]
for r in rows:
    m=r['metrics']; ref=baselines.get((r['case']['wave'],r['duration'],r['dt']))
    gates=dict(r['gates'])
    if ref:
        b=ref['metrics']
        gates['displacement_retention']=m['displacement_mm']>=.9*b['displacement_mm']
        gates['work_per_distance']=m['work_per_distance_n']<=1.25*b['work_per_distance_n']
        for metric in ('loaded_foot_slip_p95_mm_s','left_right_force_imbalance_rms_bw','normal_impulse_10ms_bw_s','chassis_impulse_10ms_bw_s'):
            gates['nonregression_'+metric]=m[metric]<=1.1*b[metric]+1e-12
    else:
        gates['matching_baseline_available']=False
    out=dict(id=r['id'],config=r['config'],wave=r['case']['wave'],duration=r['duration'],dt=r['dt'],
             metrics=m,gates=gates,feasible=all(gates.values()),failed=[k for k,v in gates.items() if not v])
    results.append(out)
    print(f"{r['config']['name']:26s} w={r['case']['wave']:g} D={m['displacement_mm']:6.1f} pitch={m['body_pitch_speed_peak']:5.1f} roll={m['body_roll_speed_peak']:5.1f} dW={m['body_deltaomega_10ms_peak']:5.1f} F={m['ground_force_peak_bw']:4.2f} tip={m['tip_angle_peak_deg']:4.1f}/{m['tip_speed_peak']:5.0f} failures={','.join(out['failed'])}")
(OUT/'comparison.json').write_text(json.dumps(results,indent=2))
