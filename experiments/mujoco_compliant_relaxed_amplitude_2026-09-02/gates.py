"""Declared engineering gate relaxation; displacement remains the only objective."""
from __future__ import annotations
import hashlib
import json
import math

CHANGED = {
    'body_wave': ('body_wave_residual_p95', .4, .6),
    'leg_yaw_wave': ('leg_yaw_wave_residual_p95', .18, .25),
    'leg_pitch_wave': ('leg_pitch_wave_residual_p95', .4, .55),
    'leg_yaw_speed': ('leg_yaw_speed_p99', 8., 10.),
    'leg_pitch_speed': ('leg_pitch_speed_p99', 7., 9.),
    'leg_yaw_reversals': ('leg_yaw_reversals_p95', 3., 4.),
    'leg_pitch_reversals': ('leg_pitch_reversals_p95', 6.25, 7.5),
    'swing_height': ('swing_foot_height_p95_mm', 5.5, 7.),
    'cycle_consistency': ('cycle_displacement_cv', .15, .2),
    'lr_duty': ('left_right_duty_difference', .1, .15),
}
INTERVALS = {
    'feet_count': ('mean_contacting_feet', (6., 28.), (4., 32.)),
    'duty': ('contact_duty_mean', (.15, .65), (.10, .80)),
}
RELATIVE = {
    'work_per_distance_n': (1.25, 1.50),
    'loaded_foot_slip_p95_mm_s': (1.10, 1.35),
    'left_right_force_imbalance_rms_bw': (1.10, 1.35),
    'normal_impulse_10ms_bw_s': (1.10, 1.25),
    'chassis_impulse_10ms_bw_s': (1.10, 1.25),
}
UNCHANGED_PREFIX_LIMITS = {
    'finite_warning_free': 0., 'torque': 2e-4,
    'body_yaw_speed': 12., 'body_pitch_speed': 8., 'body_roll_speed': 5.,
    'active_speed': 100., 'tip_speed': 1000., 'tip_peak': 40.,
    'root_roll_peak': 20., 'upright': -math.cos(math.radians(50.)),
    'spine': 1.2, 'pitch_10ms': 2., 'roll_10ms': 2.,
    'body_velocity_jump': 10., 'total_force': 2.5,
    'chassis_force': 1., 'foot_force': 1.,
}
SPEC = dict(
    version='relaxed_v1_frozen_before_reassessment',
    upper_limit_changes=CHANGED, interval_changes=INTERVALS,
    relative_limit_changes=RELATIVE,
    supplemental_reference_posture_caps='diagnostic only; absolute roll/height caps unchanged',
    old_tip_axis_reference='diagnostic only, no longer a second hard reference',
    unchanged='all other prior absolute gates, physical parameters, controller and per-wave PD',
    objective='maximize fixed-axis geometric-center displacement among feasible candidates',
    validation='cold nominal 6 s, 12 s, half timestep 6 s; D discrepancy <= 10%',
    amplitude_difference='diagnostic only: any delta >= 5 deg OR >=25% with denominator max(5 deg, old amplitude); never a gate or reward',
)
POLICY_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def absolute(record):
    """Preserve every unchanged original gate; no hidden weighted score."""
    metrics = record['metrics']
    original = record.get('strict_absolute_gates', record['gates'])
    result = dict(original)
    for name, (key, _, upper) in CHANGED.items():
        result[name] = bool(metrics[key] <= upper)
    for name, (key, _, interval) in INTERVALS.items():
        result[name] = bool(interval[0] <= metrics[key] <= interval[1])
    return result


def applicable_prefix_evidence(record):
    """Reuse only an irreversible proof still valid under this policy.

    Old relaxed-away failures must be simulated again. The original screening
    function may stop early on such a failure; that is not a new rejection.
    """
    retained = []
    for evidence in record.get('failure_evidence', []):
        name = evidence['gate']
        if name in UNCHANGED_PREFIX_LIMITS:
            limit = UNCHANGED_PREFIX_LIMITS[name]
            if abs(evidence['limit'] - limit) > 1e-12:
                raise AssertionError((name, evidence['limit'], limit))
            retained.append(dict(evidence))
        elif name == 'posture_center_height_ptp_mm' and evidence['value'] > 2.:
            retained.append(dict(gate='height', value=evidence['value'], limit=2.,
                                 explanation='prefix height range lower-bounds final range'))
        elif name == 'chassis_contact_irreversible':
            proof = record['chassis_contact_proof']
            assert proof['fraction_limit'] == .03
            assert proof['bad_step_lower_bound'] > math.ceil(.03 * proof['max_final_fine_steps']) + 1
            retained.append(dict(evidence))
        # Tighter old quantile budgets are deliberately not reused.
    return retained


def assess(record, reference):
    m = record.get('metrics')
    if not m:
        proof = applicable_prefix_evidence(record)
        assert proof, 'Relaxed-away prefix failure must be re-evaluated'
        return dict(feasible=False, failed=[p['gate'] for p in proof], gates=None,
                    full=False, D=None, metrics=None, retained_prefix_evidence=proof)
    hard = absolute(record)
    for key, (_, factor) in RELATIVE.items():
        hard['reference_' + key] = bool(m[key] <= factor * reference['metrics'][key] + 1e-12)
    return dict(feasible=all(hard.values()), failed=[k for k, v in hard.items() if not v],
                gates=hard, full=True, D=m['displacement_mm'], metrics=m,
                strict_absolute_feasible=all(record.get('strict_absolute_gates', record['gates']).values()))
