"""Check that impedance overlays leave all frozen model fields unchanged."""
import json
from pathlib import Path
from run_pd_pilot import BANK,OUT,apply_impedance,opt,np

FIELDS=('body_mass','body_inertia','body_ipos','body_iquat','dof_armature',
        'geom_size','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp',
        'geom_contype','geom_conaffinity','jnt_range','jnt_axis','jnt_pos')
checks=[]
for cfg in BANK:
    ev=opt.FastEvaluator();m,ix=ev.model,ev.index
    original={field:getattr(m,field).copy() for field in FIELDS}
    tips_k=m.jnt_stiffness[ix.tip_joint_ids].copy()
    tips_d=m.dof_damping[ix.tip_dof_addresses].copy()
    apply_impedance(m,ix,cfg)
    for field,value in original.items():
        assert np.array_equal(getattr(m,field),value),(cfg['name'],field)
    assert np.array_equal(tips_k,m.jnt_stiffness[ix.tip_joint_ids])
    assert np.array_equal(tips_d,m.dof_damping[ix.tip_dof_addresses])
    assert not np.any(m.dof_armature)
    checks.append({'candidate':cfg['name'],'frozen_fields_unchanged':True})
old=json.loads((Path(__file__).resolve().parent.parent/'mujoco_body_twitch_diagnosis_2026-09-01/outputs/selected_w2p5_summary.json').read_text())
new=json.loads((OUT/'baseline_w2p5_t6_dt0p00025.json').read_text())
assert abs(old['windows']['steady']['total_ground_force_bodyweights']['max']-new['metrics']['ground_force_peak_bw'])<1e-10
assert abs(old['windows']['steady']['body_relative_speed_rad_s']['pitch']['max']-new['metrics']['body_pitch_speed_peak'])<1e-10
result={'overlay_checks':checks,'independent_baseline_force_and_pitch_peaks_match':True,'model_sha256':opt.tip.sha256(opt.MODEL)}
(OUT/'contract_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
