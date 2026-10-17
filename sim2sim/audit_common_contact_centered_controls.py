"""Full recorded controls for unchanged/independently-centered environments."""
import json,numpy as np
import shared as s
R=s.ROOT;rows=[]
for device in ['cpu','gpu']:
    for grid in [31,15]:
        name=f'native_common_contact_centered_validation_{device}_dt{grid}_000';f=R/'runs'/name
        if not (f/'complete.json').exists():continue
        tr=np.load(f/'trajectory.npz');controls=[]
        if grid==31:controls.append((f'native_common_contact_validation_{device}_dt31_000',0,0))
        if device=='cpu':controls.append((f'native_common_contact_origin_probe_cpu_dt{grid}_000',7,0))
        for old_name,e,old_e in controls:
            old=np.load(R/'runs'/old_name/'trajectory.npz');dif={k:float(abs(tr[k][:,e].astype(float)-old[k][:,old_e].astype(float)).max()) for k in ['q','v','body']};rows.append(dict(run=name,environment=e,control=old_name,control_environment=old_e,bitwise_identical=all(np.array_equal(tr[k][:,e],old[k][:,old_e]) for k in dif),maximum_absolute_differences=dif))
s.write(R/'audit/common_contact_centered_full_controls.json',dict(rows=rows,source_sha=s.sha(__file__)));print(json.dumps(rows))
