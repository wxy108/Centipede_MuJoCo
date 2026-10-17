"""Minimal engineering viability; displacement is the only objective."""
from __future__ import annotations
import hashlib,json,math

SPEC=dict(version='displacement_first_viability_v1',
    finite_warning_free=True,positive_forward=True,
    min_upright=math.cos(math.radians(50)),spine_tortuosity_max=1.2,
    root_roll_abs_peak_deg=35.,center_height_ptp_mm=8.,
    center_vertical_speed_p99_m_s=.25,body_deltaomega_10ms_peak=30.,
    body_pitch_delta10ms_deg=5.,body_roll_delta10ms_deg=5.,
    ground_force_peak_bw=5.,chassis_force_peak_bw=2.,foot_force_peak_bw=2.,
    chassis_contact_fraction=.15,ballistic_gap_s=.08,penetration_peak_mm=.25,
    active_speed_peak=200.,tip_speed_peak=1000.,tip_angle_peak_deg=40.,
    tip_stop_fraction=.10,torque_peak_nm=2e-4,leg_yaw_speed_p99=30.,
    leg_pitch_speed_p99=30.,leg_yaw_reversals_p95=12.,leg_pitch_reversals_p95=12.,
    objective='maximize fixed-axis geometric-center displacement after viability',
    diagnostics_not_gates=['path efficiency','coherent/front roll','slip','work per distance',
      'relative force/impulse','feet/duty symmetry','swing height','waveform residual','tracking error'])
POLICY_SHA=hashlib.sha256(json.dumps(SPEC,sort_keys=True).encode()).hexdigest()


def assess(raw):
    m=raw.get('metrics')
    if not m:return dict(feasible=False,full=False,D=None,metrics=None,gates=None,
        failed=['incomplete_old_prefix_not_ranked'])
    old=raw.get('strict_absolute_gates',raw.get('gates',{}))
    gates=dict(
      finite_warning_free=bool(old.get('finite_warning_free')) and not raw.get('warnings'),
      positive_forward=m['displacement_mm']>0,
      upright=m['min_upright']>=SPEC['min_upright'],spine=m['spine_tortuosity_max']<=1.2,
      root_roll_peak=m['root_roll_abs_peak_deg']<=35.,height=m['center_height_ptp_mm']<=8.,
      vertical_speed=m['center_vertical_speed_p99_m_s']<=.25,
      body_velocity_jump=m['body_deltaomega_10ms_peak']<=30.,
      pitch_10ms=m['body_pitch_delta10ms_deg']<=5.,roll_10ms=m['body_roll_delta10ms_deg']<=5.,
      total_force=m['ground_force_peak_bw']<=5.,chassis_force=m['chassis_force_peak_bw']<=2.,
      foot_force=m['foot_force_peak_bw']<=2.,chassis_contact=m['chassis_contact_fraction']<=.15,
      no_ballistic=m['ballistic_gap_s']<=.08,penetration=m['penetration_peak_mm']<=.25,
      active_speed=m['active_speed_peak']<=200.,tip_speed=m['tip_speed_peak']<=1000.,
      tip_peak=m['tip_angle_peak_deg']<=40.,tip_stops=m['tip_stop_fraction']<=.10,
      torque=m['torque_peak_nm']<=2e-4,leg_yaw_speed=m['leg_yaw_speed_p99']<=30.,
      leg_pitch_speed=m['leg_pitch_speed_p99']<=30.,
      leg_yaw_reversals=m['leg_yaw_reversals_p95']<=12.,
      leg_pitch_reversals=m['leg_pitch_reversals_p95']<=12.)
    return dict(feasible=all(gates.values()),full=True,D=m['displacement_mm'],metrics=m,
      gates=gates,failed=[k for k,v in gates.items() if not v],
      diagnostic_old_failed=[k for k,v in old.items() if not v])
