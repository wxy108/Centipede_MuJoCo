# Body/leg spatial-phase audit

Scope: Centipede Model only. The Centipede Robot project was inspected read-only
as a controller-definition reference; no Robot file or Step 18/19 output is an
input to, or target of, this experiment.

## Evidence

- Active Robot workspace statement:
  `C:/Users/wxy22/Documents/Centipede_robot_MUJOCO/README.md` says the current
  Step 32 convention samples body yaw at inter-segment joints and leg commands
  at segment centers.
- Active Robot controller:
  `centipede_sim/reliability/cave_grouped_wave_controller.py` defines
  `body_joint_phase = omega*t + j*Delta_phi` and
  `segment_center_phase = omega*t + (i-1/2)*Delta_phi`, and algebraically checks
  that every body-joint phase is the mean of its adjacent segment phases.
- Current Model geometry:
  every leg cluster is attached halfway along the body segment between adjacent
  body-yaw joints. The existing Model controller nevertheless used the same
  spatial sample for `joint_body_(i+1)` and `joint_leg_i_*`.

## Corrected parameterization

The Model uses `Delta = 2*pi*wave/18` and body phase
`theta_body[j] = omega*t - j*Delta`. The segment-center leg phase is therefore

`theta_leg[i] = theta_body[i] + f*Delta`,

where the geometry-derived value is `f=0.5`. Both leg yaw and leg pitch use the
same segment-center sample; their existing within-leg temporal offsets remain
unchanged. Body and leg wave number remain exactly coupled.

The registered optimization may vary only `f` in `[0.35, 0.65]`. This is a
small sensitivity band around the geometric midpoint, not an independent leg
wave number and not a feedback correction.

