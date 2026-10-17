"""Memory reservation change only; prove exact dynamics against source XML."""
import json
import engine as e
np,mj=e.np,e.mujoco
rows=[]
for terrain,wave,amps in [('flat',2.5,[16.918311805232932,49.37746770784939,22.918311805232932]),('flat',3.,[30.,80.,50.]),('rough',3.,[16.918311805232932,49.37746770784939,22.918311805232932])]:
    source=e.model_path('flexible',terrain);e.fx.opt.MODEL=source
    a=e.BASE_EVALUATOR();b=e.bounded_evaluator()
    compared=[]
    for key in dir(a.model):
        if key.startswith('_'):continue
        x,y=getattr(a.model,key),getattr(b.model,key)
        if isinstance(x,np.ndarray):
            assert np.array_equal(x,y,equal_nan=True),key
            compared.append(key)
    for ev in (a,b):
        e.fx.pilot.apply_impedance(ev.model,ev.index,e.fx.pd_config(wave))
        mj.mj_resetData(ev.model,ev.data);mj.mj_forward(ev.model,ev.data)
        (e.fx.near.near_ground if terrain=='flat' else e.fx.rough_near_ground)(ev)
    ctrls=[e.fx.opt.ThreeAmplitudeController(ev.model,ev.index,wave,*amps) for ev in (a,b)]
    fields=('qpos','qvel','qacc','qacc_warmstart','ctrl','actuator_force','qM','efc_force')
    checks=0
    for step in range(24000):
        for ev,ctl in zip((a,b),ctrls):ctl.step(ev.data);mj.mj_step(ev.model,ev.data)
        if step%40==0 or step==23999:
            for key in fields:assert np.array_equal(getattr(a.data,key),getattr(b.data,key)),(terrain,wave,step,key)
            assert a.data.ncon==b.data.ncon
            checks+=1
    assert not np.any(a.data.warning.number) and not np.any(b.data.warning.number)
    row=dict(terrain=terrain,wave=wave,amplitudes=amps,source=str(source),model_array_fields_exact=len(compared),
        duration=6.,exact_dynamics_fields=fields,checks=checks,original_arena_bytes=int(a.model.narena),
        runtime_arena_bytes=int(b.model.narena),maxuse_arena_bytes=int(b.data.maxuse_arena),warnings=[])
    rows.append(row);print(json.dumps(row),flush=True)
    del a,b,ctrls
(e.DATA/'runtime_arena_equivalence.json').write_text(json.dumps(rows,indent=2))
