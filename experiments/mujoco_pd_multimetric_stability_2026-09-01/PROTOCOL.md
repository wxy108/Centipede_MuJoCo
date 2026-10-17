# Flat-ground PD stability pilot: preregistered engineering envelopes

Scope: Centipede Model only; Robot Step18/19 and historical results untouched.
This is an engineering pilot, not biological calibration or global optimization.

## Frozen contract

Frozen source XML SHA256:
3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520.
Mass/inertia, zero armature, morphology, head/link1 lock, tip spring/damping,
joint limits, contact parameters and solver remain fixed. Only explicitly
listed body joint impedance multipliers may vary. Passive body roll k/c is
the zero-rest equivalent of joint PD, and is included in those multipliers.
CPG frequency, phase, equal body/leg wave numbers, and each wave's existing
three amplitudes stay fixed. No state-feedback stabilization, control gating,
target filtering, or amplitude retuning is added.

Screen on wave2.5, then test promising settings on waves1.5 and3. Parameters
must transfer; a wave2.5-only improvement is not a generally accepted baseline.
Six-second episodes, 1.5–6s steady evaluation, every-step controls at 0.25ms.
Force peaks, actuator peaks and safety are checked every physics step; body
kinematics and foot trajectories at 1kHz. Startup is reported separately.

## Separate gates, diagnostics, objective

All former V2 posture, gait-shape, swing-height, duty, penetration, torque and
support gates are retained, now supplemented by local peak measurements.
The following **provisional engineering** envelopes are declared before
screening. They are not measured biological limits and cannot be relaxed
silently to promote a candidate:

- local body yaw/pitch/roll speed peaks <= 12/8/5 rad/s;
- world segment angular-velocity change over10ms <=10 rad/s;
- local pitch and roll changes over10ms <=2 degrees each;
- peak total normal force <=2.5 bodyweights; peak chassis normal force
  <=1 bodyweight; peak single-foot normal force <=1 bodyweight;
- chassis-ground loaded contact fraction <=3%; no-foot support gap <=80ms;
- tip max angle <=40 degrees, max speed <=1000rad/s and original percentile
  and stop-occupancy gates retained (spikes no longer exempted);
- centroid height excursion <=2mm, root absolute roll peak <=20degrees;
- per-cycle forward-displacement CV <=15% for complete cycles2–3,3–4,4–5,5–6;
- retain at least90% of matching baseline displacement; absolute actuator
  work per forward distance <=125% of matching baseline.
- loaded-foot slip P95 and RMS left/right force imbalance <=110% of matching
  baseline; total/chassis 10ms impulse <=110% of matching baseline. These
  non-regression checks supplement the new absolute force/pose envelopes.

Additionally report: body/leg waveform distortion, all-family tracking,
angular acceleration, 10ms total/chassis force impulse, loaded foot slip,
force imbalance, tip-stop fraction, total positive/absolute work and startup
peaks. Diagnostics are not secretly added as weighted penalties. If an
uncovered failure appears, document it rather than claiming full stability.

Ranking: feasibility first, then forward displacement only among feasible
settings. Show trade-offs/Pareto candidates when none passes every gate;
never rename the least-bad failure as a stable winner. No general guarantee
that every noisy scalar decreases is intended: all essential families must
remain inside stable envelopes while useful motion is preserved.

## Follow-up verification

Any provisional shared setting requires longer replay, half-timestep
sensitivity, and labelled front/side/oblique plus local demos. Random contact
erasure remains a separate robustness test, not a substitute for nominal
feasibility. Existing amplitude optima are conditional on old PD and must
not be carried forward as proven optima after changing impedance.

## Stage 2 amendment, declared before execution

The initial body-only screen shows residual leg-pitch waveform/speed failures
and tip peaks. A six-case gain-only follow-up therefore also permits leg
tracking gains, with every frozen morphology/solver/CPG item unchanged and
no acceptance-threshold changes. This supersedes the body-only restriction
above solely for the explicitly registered Stage2 cases.

Use the diagnostic `pitch_k4_d4_roll_d2` setting as the common reference for
separate body-yaw damping, leg-pitch damping, leg-pitch stiffness, and combined
tracking-damping probes. It was selected to investigate its reduced total
force, not promoted as a stable winner. Leg pitch Kp factors are0.5/1/2;
leg pitch Kd is2; leg yaw Kd is at most2; body yaw Kd is at most4. All factors
are relative to the original baseline, not compounded across runs. Exact
six cases are `p4_*` in the candidate bank. Body-only failures remain visible.

## Stage 3 combined-gain probes, declared before execution

Stage2 `p4_leg_pitch_k2_d2` lowers local pitch speed and total force markedly
while retaining displacement, but still fails other gates. Three explicit
combinations add body yaw damping2, body yaw damping2 plus roll damping4,
or body yaw damping4 plus roll damping4 to that same setting. Names are
`p4_leg_k2_yaw_*`. No threshold, initialization or CPG change is included.
