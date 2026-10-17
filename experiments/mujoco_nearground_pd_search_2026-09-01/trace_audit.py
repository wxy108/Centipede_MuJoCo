"""Realized joint excursion and sampled passive-roll torque; no gate changes."""
import argparse
import json
from pathlib import Path
from run_search import np,mujoco,opt,pilot

p=argparse.ArgumentParser();p.add_argument('record',type=Path);args=p.parse_args()
r=json.loads(args.record.read_text());tr=np.load(args.record.with_suffix('.npz'))
ev=opt.FastEvaluator();m,ix=ev.model,ev.index;pilot.apply_impedance(m,ix,r['config'])
good=tr['t']>=1.5;leg=np.degrees(tr['lq'][good].reshape(-1,19,2,2));body=np.degrees(tr['bq'][good])
excursion=.5*np.ptp(leg,axis=0)
roll_tau=-m.jnt_stiffness[ix.roll_joint_ids]*tr['bq'][good,:,2]-m.dof_damping[ix.roll_dof_addresses]*tr['bv'][good,:,2]
physical=dict(active={},passive_body_roll=dict(k_nm_rad=float(m.jnt_stiffness[ix.roll_joint_ids][0]),c_nms_rad=float(m.dof_damping[ix.roll_dof_addresses][0])),
              tip=dict(k_nm_rad=float(m.jnt_stiffness[ix.tip_joint_ids][0]),c_nms_rad=float(m.dof_damping[ix.tip_dof_addresses][0])))
for name,pos in [('body_yaw',ix.body_positions),('body_pitch',ix.pitch_positions),('leg_yaw',ix.leg_positions[:,:,0].reshape(-1)),('leg_pitch',ix.leg_positions[:,:,1].reshape(-1))]:
    physical['active'][name]=dict(kp_nm_rad=float(ix.kp[pos][0]),kd_nms_rad=float(ix.kv[pos][0]))
out=dict(source=r['id'],physical_gains=physical,command_amplitudes_deg=r['case'],
         realized_half_peak_to_peak_deg=dict(leg_yaw=excursion[:,:,0].tolist(),leg_pitch=excursion[:,:,1].tolist()),
         leg_pitch_realized_half_excursion_range_deg=[float(excursion[:,:,1].min()),float(excursion[:,:,1].max())],
         passive_roll_angle_abs_sampled_peak_deg=float(np.abs(body[:,:,2]).max()),
         passive_roll_torque_abs_sampled_peak_nm=float(np.abs(roll_tau).max()),passive_torque_sample_dt_s=.001,
         warning='Sampled passive torque is not an every-physics-step peak. Command amplitude was not changed.',
         foot_sphere_bottom_swing_height_p95_mm=r['metrics']['swing_foot_height_p95_mm']-.45,
         original_center_height_gate_still_in_force=True)
path=args.record.with_name(args.record.stem+'_trace_audit.json');path.write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='realized_half_peak_to_peak_deg'},indent=2))
