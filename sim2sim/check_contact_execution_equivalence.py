import numpy as np,json
import shared as s
R=s.ROOT
def comparison(x,y):
    a=np.load(R/'runs'/x/'trajectory.npz');b=np.load(R/'runs'/y/'trajectory.npz')
    assert np.array_equal(a['t'],b['t']) and np.array_equal(a['joint_names'],b['joint_names']) and np.array_equal(a['body_names'],b['body_names'])
    err={k:float(np.max(abs(a[k]-b[k]))) for k in ['q','v','body']}
    return dict(run_a=x,run_b=y,frames=len(a['t']),duration_s=float(a['t'][-1]),max_absolute_errors=err,bitwise_identical=all(v==0 for v in err.values()))
rows=[comparison('common_contact_timing_gpu_000','common_contact_fast_timing_gpu_000'),comparison('native_tgs_ext_validation_cpu_dt125_000','native_tgs_ext_validation_cpu_dt125_threads2_000')]
s.write(R/'audit/contact_execution_equivalence.json',dict(comparisons=rows,qualification='First comparison covers 0.02 s initialization, not a full gait; shared contact function separately has 1000-state bitwise equivalence. Second isolates only CPU thread count over all nine full gaits.'))
print(json.dumps(rows))
