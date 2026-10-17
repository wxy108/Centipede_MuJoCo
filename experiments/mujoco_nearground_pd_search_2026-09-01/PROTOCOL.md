# Near-ground initialization and fixed-amplitude PD search

User authorization: near-ground resting initialization and continued PD-only
optimization, with amplitudes held fixed while varying spatial wave number.
Centipede Model only; no Robot Step18/19 writes, no source XML overwrite.

## Initialization, before any new screening

Rotate only the free root by the inverse fixed head/link1 rotation, making
link1 and the remaining zero-angle spine horizontal. Translate only root z
until the lowest collision geometry is 25 micrometers above the flat plane.
All relative joint coordinates retain their zero values and qvel is zero.
The original controller holds zero joint targets until t=1s, then uses its
unchanged staggered half-cosine ramp. No added feedback, target filters,
gravity compensation, constraint, force, or velocity resets during settling.
This is a near-ground *resting start*, not an asserted solved equilibrium;
settling dynamics must be measured and disclosed.

## Frozen and varied quantities

Frozen model SHA256:
3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520.
Mass, geometry, inertia, armature=0, joint limits, tip k/c, contact solver and
friction frozen. CPG frequency1Hz, phase, warm-up and coupled body/leg wave
number frozen. Contact erasure OFF in nominal experiments.

Amplitude clarification is pending. Default common-amplitude interpretation:
all waves use the latest demo's existing wave2.5 amplitudes (body yaw,
leg yaw, leg pitch) = (16.918311805232932,49.37746770784939,
22.918311805232932) degrees. If PI chooses previous per-wave amplitudes,
use a separate output partition, never combine objective comparisons.
Both cases freeze amplitudes, not optimize them. Previously unselected
waves cannot silently borrow a per-wave optimum.

Wave coverage: 0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9.
Each wave is allowed its own PD optimum. A shared PD setting is a separate
transfer diagnostic, not forced on per-wave optimization.

Search has ten independent log-scaled multipliers: body yaw P/D, body pitch
P/D, leg yaw P/D, leg pitch P/D, passive body roll k/c. All Kp/k factors
initially span [0.25,16], all Kd/c factors [0.25,32], relative to the actual
frozen baseline. Tip compliance is not part of this domain.

## Metrics, gates and objective

Inherit every absolute and baseline-relative gate from the previous
multi-metric PD protocol, without relaxing thresholds. Baseline-relative
reference must have the same near-ground start, amplitudes, wave, duration
and timestep. Startup torque is still gated; startup force is separately
reported. New initial conditions do not permit comparison against the old
air-drop baseline for acceptance.
Evaluation t=1.5..6s, dt0.25ms, controls every step, contact forces and speed
peaks every physics step. 1kHz pose/shape sampling; the historical explicitly
declared 50ms filter is used only for reversal counts, never videos/peaks.

Objective: maximize forward displacement among all-gates feasible settings.
Infeasible trials are retained. No weighted sum of tracking, force, energy,
slip or displacement, and no 'winner' label for an infeasible point.

Search strategy: baseline/previous-candidate anchors across all waves;
scrambled log-domain Sobol exploration plus transparent anchor/axis probes;
feasibility-directed local refinement from diverse non-dominated violation
vectors and/or feasible high-displacement points; independent validation
of promising points at longer duration and half timestep. A boundary hit
requires disclosure and a new registered range before expansion.
Finite search cannot certify a continuous global optimum. Report evaluated
coverage, feasible counts, repeated-start agreement and remaining gaps.
No optimizer stop condition is proof of global optimality.

## Deliverables

Resumable trial JSONs and exact gains, complete gate tables, initialization
audit, and front/side/oblique demos using unsmoothed recorded trajectories.
No default baseline promotion until robustness checks pass. All amendments
are logged before executing their affected batch.

## Broad batch registration (before execution)

Two independently scrambled Sobol replicates, seeds1901 and2901, 32 points
each in the declared ten-dimensional logarithmic box; 20 individual-axis
endpoint probes around the previous candidate; four uniform-gain probes
(2,4,8,16). Total88 configurations per wave, not an exhaustive 10D grid.
Safe early rejection is permitted on any already-exceeded irreversible
per-step peak gate (torque, body yaw/pitch/roll speed, tip speed/angle).
Its exact failure time/value/threshold is saved. A truncated failure has no
comparable displacement and is never labelled all-metrics evaluated.
A precheck survivor must replay the complete original evaluator before
any feasibility/objective ranking. No threshold is changed by pruning.

## Coordinated-gain probes (declared after 265 checks, before execution)

Twelve explicit parameter-only candidates in build_local.py raise leg
damping in coordinated combinations around the previous candidate, while
also testing body pitch/roll damping and stronger yaw tracking. They stay
inside the previously registered global box. This adds structure that a
sparse 10D space-filling sample may miss; no existing failure is removed.
All14 waves and all previous gates are retained, using the same exact
early-rejection proof. Amplitudes and tip properties are still unchanged.

## Independent-basin refinement (declared after 356 checks, before execution)

Sobol seed1901 point15 at wave1 passes all six precheck peaks, but its full
run fails front roll, body waveform and duty; it also loses displacement
relative to the near-ground baseline. This is NOT a feasible winner.
Probe a27-point3x3x3 grid around this independent basin: body yaw Kp factor
{0.6,1,2}, body pitch Kd factor{1,2,4}, passive roll k factor{0.5,1,2}.
All other gains remain those of point15, including high leg-pitch Kp11.1545.
Evaluate waves1,2,2.5,3 to see whether improved body tracking can recover
forward motion without reopening fast tip/body events. The original broad
search continues; this local subspace is not the global search domain.

## High-displacement basin refinement (declared after444 checks, before execution)

Seed1901 point18 at wave2 yields175.316mm and passes all body/tip peak-speed
and force gates. It still fails mean feet count5.923<6 and swing height
5.932>5.5mm, so is not accepted. Register a4x3x3 local grid: leg pitch Kp
factors{2,4,8,13.134240534506468}, leg pitch Kd{0.25,1,4}, passive roll k
{4,8,14.081530435465202}, keeping all other point18 gains fixed. Evaluate
waves2 and2.5. Baseline-relative constraints also remain required. This
explicitly tests whether support/height can recover through PD alone.

## Second-seed body-wave refinement (declared after833 checks)

Seed2901 point6 passes all original gates except body-wave shape at waves
3.5 and4.5, including the unchanged5.5mm center-height gate. Probe body-yaw
Kp factors{0.6,1,2,4} with every other gain unchanged, at those two waves.
This eight-trial branch needs no height-definition change to be accepted.

## First-feasible boundary expansion (declared after1484/1616 checks)

`axis_leg_yaw_k_0.25` at wave2.5 is the first candidate to pass every
original absolute and baseline-relative gate. It has147.535mm displacement,
but leg-yaw Kp is exactly the registered lower boundary0.25. It is provisional,
not an interior/global optimum. Expand ONLY this coordinate with factors
{0.05,0.1,0.15,0.2,0.3,0.35,0.4,0.5}; all other gains and every original
gate stay unchanged. Evaluate wave2.5, then validate the best feasible point
at12s and half timestep. This changes no command amplitude.

Validation includes the top two V1-feasible factors0.25 and0.2; both must
also pass the new symmetric yaw-tracking assurance gate. Matching baseline,
12s and0.125ms cases are separate from nominal-search counts. Selection is
feasibility/validation first, then nominal forward displacement.

The boundary scan shows0.2 and0.25 feasible,0.3 failing body velocity jump,
and0.35+ failing tip speed. Since nominal displacement peaks inside this
bracket, run factors{0.26,0.27,0.275,0.28,0.29} before final selection.

All five refine points pass nominal V1 gates and the added V2 yaw-tracking
gate. Factor0.29 has the highest tested displacement149.962mm; factor0.3
fails body-velocity jump. Validate the exact0.29 candidate at12s and0.125ms
against the already generated matching baselines. It remains a finite-search
optimum, not a proof over the continuous interval.

## V2 assurance amendment: symmetric yaw tracking

Declared immediately after the first original-gates-feasible point and
before any default promotion. `leg_yaw_tracking_rms_deg` was always reported,
but only leg pitch had a10degree tracking gate. PI requires all indicator
families to remain stable, and fixed command amplitudes must not be satisfied
by making realized yaw arbitrarily weak. Add a symmetric leg-yaw RMS tracking
error gate <=10degrees to every old and new full trial. This is a stricter
assurance gate, not a relaxed threshold. The first candidate's value7.408deg
passes; boundary points do not inherit feasibility until reclassified.

## Diagnostic robustness replay (not a promotion)

Wave2 basin18_lpk2_lpd0.25_rk14.082 passes every original nominal gate except
site-center swing height. Replay this explicitly unaccepted candidate for
12s and at half timestep0.125ms, with matching near-ground baselines. Save
its ordinary6s,12s and half-step traces for three-view/local demos. These
tests investigate whether its other improvements persist; they do not
waive its height failure or establish a global optimum.
