"""Explicit physical PD values and source provenance for this frozen-plant study."""
import datetime
import json
import engine as e
from pathlib import Path
manifest=json.loads((e.DATA/'amplitude_manifest.json').read_text())
modules=(e.fx,e.fx.pilot,e.fx.pilot.pilot,e.fx.opt,e.fx.opt.tip,e.fx.opt.tip.stage1,e.fx.opt.base,e.fx.near,e.ab,e.builder)
paths=sorted(set(Path(m.__file__).resolve() for m in modules))
inputs=[dict(path=str(p),sha256=e.fx.sha256(p)) for p in paths]
rows={}
for task in manifest['references']:
    e.fx.opt.MODEL=e.model_path('flexible','flat');ev=e.bounded_evaluator();m,ix=ev.model,ev.index
    e.fx.pilot.apply_impedance(m,ix,task['config'])
    assert not e.np.any(m.dof_armature)
    rows[f"{task['wave']:g}"]=dict(config=task['config'],active_joint_names=list(ix.active_joint_names),
        kp_nm_per_rad=ix.kp.tolist(),kv_nms_per_rad=ix.kv.tolist(),
        passive_body_roll_stiffness_nm_per_rad=m.jnt_stiffness[ix.roll_joint_ids].tolist(),
        passive_body_roll_damping_nms_per_rad=m.dof_damping[ix.roll_dof_addresses].tolist(),
        tip_stiffness_nm_per_rad=m.jnt_stiffness[ix.tip_joint_ids].tolist(),
        tip_damping_nms_per_rad=m.dof_damping[ix.tip_dof_addresses].tolist(),
        tip_axis_local=m.jnt_axis[ix.tip_joint_ids].tolist(),tip_range_rad=m.jnt_range[ix.tip_joint_ids].tolist(),
        timestep=float(m.opt.timestep),total_mass_kg=float(m.body_mass.sum()),armature_max=float(m.dof_armature.max()),
        nq=int(m.nq),nv=int(m.nv),nu=int(m.nu),actuator_forcerange_nm=m.actuator_forcerange.tolist())
    del ev,m,ix
out=dict(snapshot_time=datetime.datetime.now().astimezone().isoformat(),
    note='Runtime provenance snapshot; baseline XML hashes equal the initial registered hashes. Historical controller/helper files were not edited in this experiment. Absolute values supplement, not replace, the pre-registered multiplier profiles.',
    model=str(e.model_path('flexible','flat')),model_sha256=e.fx.sha256(e.model_path('flexible','flat')),
    controller_frequency_hz=e.fx.opt.base.FREQUENCY_HZ,source_files=inputs,profiles=rows)
assert out['model_sha256']==manifest['plant_sha256']
(e.DATA/'runtime_parameter_snapshot.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(profiles=len(rows),source_files=len(inputs),model_sha256=out['model_sha256'],frequency_hz=out['controller_frequency_hz'])))
