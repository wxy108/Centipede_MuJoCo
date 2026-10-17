import time,json
import numpy as np
import shared as s
from common_contact_v2 import CommonContact as Reference,quat_matrix
from common_contact_fast import CommonContact as Fast
names=json.loads((s.ROOT/'runs/native_common_contact_init_cpu_dt31_000/readback.json').read_text())['body_names']
rng=np.random.default_rng(915721);N=9;nb=len(names);a=Reference(names,N,1/32000);b=Fast(names,N,1/32000)
pos=rng.uniform(-.005,.005,(N,nb,3));pos[...,2]=rng.uniform(-.002,.008,(N,nb));rot=quat_matrix(rng.normal(size=(N,nb,4)));com=pos+rng.normal(0,.0003,(N,nb,3));vel=rng.normal(0,.02,(N,nb,6))
maxF=maxT=maxXi=0.;same_indices=True;elapsed=np.zeros(2)
for i in range(1000):
    if i%100==0:pos[...,2]+=rng.normal(0,.002,(N,nb));rot=quat_matrix(rng.normal(size=(N,nb,4)))
    pos+=rng.normal(0,1e-7,pos.shape);com=pos+rng.normal(0,1e-5,pos.shape);vel+=rng.normal(0,.001,vel.shape)
    t=time.perf_counter();F,T,info=a.forces(pos,rot,com,vel);elapsed[0]+=time.perf_counter()-t
    t=time.perf_counter();G,U,other=b.forces(pos,rot,com,vel);elapsed[1]+=time.perf_counter()-t
    maxF=max(maxF,float(np.max(abs(F-G))));maxT=max(maxT,float(np.max(abs(T-U))));maxXi=max(maxXi,float(np.max(abs(a.xi-b.xi))));same_indices &= np.array_equal(a.mesh_index,b.mesh_index)
result=dict(states=1000,environments=N,max_force_difference_N=maxF,max_torque_difference_Nm=maxT,max_memory_difference_m=maxXi,identical_support_indices=bool(same_indices),reference_s=elapsed[0],vectorized_s=elapsed[1],passed=maxF<1e-12 and maxT<1e-14 and maxXi<1e-12 and same_indices,source_hashes={str(p):s.sha(p) for p in [s.ROOT/'common_contact_v2.py',s.ROOT/'common_contact_fast.py',s.ROOT/'COMMON_CONTACT_V2.json',s.ROOT/'assets/common_contact_geometry.json']})
s.write(s.ROOT/'audit/common_contact_vectorization.json',result);print(json.dumps(result));assert result['passed']
