"""Prove independent settled-state replay is identical to uninterrupted stepping."""
import copy
import json
import search as s
e=s.e;np=s.np;mujoco=s.mujoco
manifest=json.loads(s.MANIFEST.read_text())
checks=[]
for wave,amps in ((2.5,[16.918311805232932,49.37746770784939,22.918311805232932]),(3.,[30.,80.,50.])):
    task=next(t for t in manifest['references'] if t['wave']==wave)
    task=dict(task,amplitudes=dict(zip(s.KEYS,amps)))
    state=s.settled(task);m=state['ev'].model;ix=state['ev'].index
    full=mujoco.MjData(m)
    ev=state['ev'];saved=ev.data;ev.data=full
    mujoco.mj_resetData(m,full);mujoco.mj_forward(m,full);e.fx.near.near_ground(ev);ev.data=saved
    cold=e.fx.opt.ThreeAmplitudeController(m,ix,wave,*amps)
    for k in range(int(round(1.8/m.opt.timestep))):
        cold.step(full);mujoco.mj_step(m,full)
    cached=copy.copy(state['snapshot'])
    hot=e.fx.opt.ThreeAmplitudeController(m,ix,wave,*amps);hot.previous_target=state['previous_target'].copy()
    for k in range(state['steps'],int(round(1.8/m.opt.timestep))):
        hot.step(cached);mujoco.mj_step(m,cached)
    fields=('qpos','qvel','qacc','qacc_warmstart','ctrl','actuator_force','qM','efc_force')
    equal={f:bool(np.array_equal(getattr(full,f),getattr(cached,f))) for f in fields}
    assert all(equal.values()),equal
    checks.append(dict(wave=wave,amplitudes=amps,arena_bytes=int(m.narena),mujoco_version=mujoco.__version__,exact_array_equality=equal,time_equal=full.time==cached.time))
(e.DATA/'settled_state_replay_verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
