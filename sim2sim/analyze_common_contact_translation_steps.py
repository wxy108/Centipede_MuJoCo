"""Test arithmetic hypotheses against per-step native data; do not fit constants."""
import json,numpy as np
import shared as s
R=s.ROOT;rows=[]
def acc32(v):
    a=np.float32(0)
    for x in v:a=np.float32(a+np.float32(x))
    return a
for grid,dt in [(31,.00003125),(15,.000015625)]:
    folder=R/'runs'/f'native_common_contact_translation_steps_cpu_dt{grid}_000'
    if not (folder/'complete.json').exists():continue
    p=np.load(folder/'translation_probe.npz');b=np.load(folder/'backend.npz');q=np.load(folder/'trajectory.npz');res=json.loads((folder/'results.json').read_text());assert res['max_contact_point_force_N']==0
    for e,y in enumerate([0.,1.75]):
        masses=b['masses'][e].ravel().astype(np.float32);im=np.float32(1)/acc32(masses)
        pos=p['com_position'][:,e,:,1];v=p['com_velocity'][:,e,:,1];naive=[];corrected=[];corrections=[]
        for k in range(len(pos)-1):
            root_new=np.float32(pos[k,0]+np.float32(v[k,0]*np.float32(dt)))
            delta=np.float32(root_new-pos[k,0]);candidate=np.float32(pos[k]+delta)
            com=np.float32(acc32(pos[k]*masses)*im);momentum=acc32(v[k]*masses)
            predicted=np.float32(com+np.float32(momentum*np.float32(im*np.float32(dt))))
            candidate_com=np.float32(acc32(candidate*masses)*im);correction=np.float32(predicted-candidate_com)
            naive.append(float(root_new-pos[k+1,0]));corrected.append(float(np.float32(candidate[0]+correction)-pos[k+1,0]));corrections.append(float(correction))
        rows.append(dict(grid_us=dt*1e6,global_y_m=y,steps=len(pos)-1,root_y_displacement_um=float((p['position'][-1,e,1]-p['position'][0,e,1])*1e6),root_com_y_displacement_um=float((pos[-1,0]-pos[0,0])*1e6),max_naive_one_step_root_com_error_nm=float(np.max(np.abs(naive))*1e9),max_simple_com_correction_one_step_error_nm=float(np.max(np.abs(corrected))*1e9),naive_exact_step_count=int(np.sum(np.array(naive)==0)),simple_com_correction_exact_step_count=int(np.sum(np.array(corrected)==0)),simple_com_correction_range_nm=[float(min(corrections)*1e9),float(max(corrections)*1e9)],max_velocity_change_m_s=float(abs(p['velocity'][:,e]-p['velocity'][0,e]).max())))
out=dict(rows=rows,qualification='Diagnostic arithmetic models only. Sequential NumPy float32 weighted sums and a common rigid translation approximate the public COM correction structure, but do not reproduce all internal FK, instruction fusion, accumulation order or the installed binary. Mismatch is retained; no fitted constants or production change.',source_sha=s.sha(__file__))
s.write(R/'audit/common_contact_translation_step_arithmetic.json',out);print(json.dumps(out))
