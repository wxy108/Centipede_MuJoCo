# Registered rigid-leg phase + PD optimization

## Fixed

- One-piece rigid leg topology; no knee and no compliant tip DOF.
- Flat ground, near-ground initialization, solver, contact parameters, model
  mass/inertia and fixed common amplitudes from the preceding experiment.
- Body and leg wave number are identical. Only waves 2.5 and 3.0 are in scope.
- Controller remains open-loop and feed-forward. No state feedback, smoothing,
  clipping or special-case stabilization is added.

## Search variables

- Ten impedance multipliers: body yaw/pitch stiffness and damping, passive body
  roll stiffness and damping, leg yaw/pitch stiffness and damping.
- One segment-center phase fraction `f` in `[0.35, 0.65]`; `f=0.5` is the exact
  geometric midpoint.
- Amplitudes are not search variables.

Stage 1 is a phase-only diagnostic at the previously selected PD. Stage 2 is a
scrambled Sobol search in the registered bounds. Stage 3 refines only around a
candidate that passes every gate. This is a finite best-tested result, not a
proof of a continuous global optimum.

## Decision rule

Gates, diagnostics and objective remain separate. A candidate is eligible only
if every applicable absolute stability/tracking/contact gate in the rigid-tip
runner passes, and it also meets the matched compliant-reference non-regression
gates:

- displacement at least 90% of reference;
- work per distance at most 125% of reference;
- loaded-foot slip, left/right force imbalance and 10 ms normal impulse at most
  110% of reference;
- chassis impulse no larger than 110% of reference (zero-reference cases must
  remain effectively zero).

Among eligible candidates only, maximize forward displacement. No weighted
scalar reward may turn a failed gate into a winner. Finalists must pass a 12 s
nominal replay and a 6 s half-timestep replay before a profile is selected.

