"""One bounded coupled-amplitude bridge pass near active non-catastrophic gates."""
import argparse
import json
import refine as rf
s,e,np=rf.s,rf.e,rf.np


# Only near-feasible shape/tracking/contact-quality failures are eligible as
# bridge endpoints. Absolute roll, force, jumping, torque, and numerical gates
# must already pass. Endpoints themselves are never selected or promoted.
ALLOWED={'swing_height','leg_yaw_speed','leg_pitch_speed','leg_yaw_reversals',
         'leg_pitch_reversals','body_wave','leg_yaw_wave','leg_pitch_wave',
         'pitch_tracking','leg_yaw_tracking','duty','lr_duty',
         'work_per_distance','loaded_foot_slip_p95_mm_s',
         'left_right_force_imbalance_rms_bw','normal_impulse_10ms_bw_s',
         'chassis_impulse_10ms_bw_s'}
WEIGHTS=(.25,.5,.75,.875,.9375)


def family(key):
    for prefix in ('fixed_amp_','legacy_axis_'):
        if key.startswith(prefix):return key[len(prefix):]
    return key


def freeze(preview=False):
    path=e.DATA/'boundary_manifest.json'
    if path.exists() and not preview:return json.loads(path.read_text())
    if not preview:assert (e.DATA/'polish_summary.json').exists()
    rows=rf.collect(['broad','refine','polish'])
    seen={r['id'] for r in rows};tasks=[];origins={}
    for wave in e.WAVES:
        full=[r for r in rows if r['wave']==wave and r['full']]
        feasible=sorted((r for r in full if r['feasible']),key=lambda r:-r['D'])
        if not feasible:
            origins[f'{wave:g}']=dict(reason='No feasible endpoint in the tested bank',bridges=[])
            continue
        anchor=feasible[0];x=np.array([anchor['amplitudes'][k] for k in s.KEYS])
        endpoints=[]
        for row in sorted(full,key=lambda r:-r['D']):
            failed={family(k) for k in row['failed']}
            if row['feasible'] or row['D']<=anchor['D'] or not failed or len(failed)>2 or not failed<=ALLOWED:continue
            y=np.array([row['amplitudes'][k] for k in s.KEYS])
            if any(np.linalg.norm((y-z)/s.HIGH)<.04 for z,_ in endpoints):continue
            endpoints.append((y,row))
            if len(endpoints)==2:break
        origins[f'{wave:g}']=dict(feasible_anchor=anchor['id'],bridges=[r['id'] for _,r in endpoints])
        base=json.loads(open(anchor['record'],encoding='utf-8').read())['task']
        for y,endpoint in endpoints:
            for weight in WEIGHTS:
                values=(1-weight)*x+weight*y
                assert np.all(values>=s.LOW) and np.all(values<=s.HIGH)
                task=dict(base,stage='boundary',trace=False,amplitudes=dict(zip(s.KEYS,map(float,values))),
                    origin=dict(feasible=anchor['id'],faster_failed=endpoint['id'],weight=weight))
                tid=e.task_id(task)
                if tid in seen:continue
                seen.add(tid);tasks.append(task)
    result=dict(stage='boundary',parent_stages=['broad','refine','polish'],weights=WEIGHTS,
        maximum_endpoints_per_wave=2,maximum_failed_gate_families=2,allowed_failed_gate_families=sorted(ALLOWED),
        selection='Highest-displacement nominal feasible anchor to up to two spatially distinct faster near-feasible endpoints; all gates unchanged',
        endpoint_normalized_separation=.04,origins=origins,tasks=tasks,count=len(tasks),
        interpolation_is_not_assumed_monotonic=True,global_optimum_proved=False)
    if not preview:path.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=20);p.add_argument('--preview',action='store_true');args=p.parse_args()
    manifest=freeze(args.preview)
    print(json.dumps({k:v for k,v in manifest.items() if k!='tasks'},indent=2),flush=True)
    if not args.preview:s.run_bank(manifest['tasks'],args.workers,'boundary')
