# Frozen-plant three-amplitude convergence study

Authorized 2026-09-03: keep the model, controller, existing per-wave PD, and
ALL previous relaxed_v1 gates unchanged. This is Centipede Model, not Robot.
No Robot Step 18/19 or IsaacLab/Clean files are written. All outputs are opt-in
and isolated here; no canonical/default parameter promotion is authorized.

## Frozen interpretation

- Same local-Z compliant tip, mass, inertia, zero armature, passive compliance,
  contacts, solver, near-ground settling, frequency (1 Hz), and phase law.
- Flat only; contact erasure OFF; body and legs use the same wave number.
- Waves: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9.
- Existing PD is fixed WITHIN each wave. It differs BETWEEN waves. Amplitude
  optima remain conditional on that mapping, not a pure wave-number effect.
- Only body yaw / leg yaw / leg pitch command amplitudes vary, respectively
  within [0,36] / [0,90] / [0,60] degrees. These bounds are unchanged.
- Objective: maximize fixed-axis displacement (mm) of the unweighted 21-body
  geometric center over t=1.5..5.999 s, AFTER all gates pass. No scalar penalty
  reward, amplitude-distance reward, or monotonicity/smoothness constraint.
- Same cold nominal 6 s, cold 12 s, and cold half-step 6 s checks; half-step
  displacement discrepancy <= 10%. No lower nominal candidate can be selected
  without checking every higher nominal candidate or using verified old checks.

## Registered finite search (not a continuous global-optimality certificate)

1. Verify hashes and assessments of all 5,036 previous tested conditions.
   Retain them as incumbent evidence, without counting them as new simulations.
2. Transfer the 11 previously validated three-amplitude vectors to every wave,
   always using the DESTINATION wave's frozen PD. Also test the six axial
   neighbors of each own incumbent at 2% of box span. This is a quick challenge
   of the previous winners, not a new gait selection rule.
3. Two independent scrambled Sobol banks of 256 points PER WAVE (512/wave).
   Their seed-level best feasible results and cumulative best curves are kept
   separately. Agreement is evidence of repeatability, not proof of optimality.
4. Profile body yaw on 13 exact nodes: 0,3,...,36 degrees. At each node test eight
   independently scrambled 2-D Sobol leg-amplitude pairs, plus legs from the
   current per-wave best, nearest feasible body-amplitude point, and neighboring
   wave incumbents. Refine the best feasible pair at each node with an 8-direction
   2-D pattern search. No feasible seed is recorded as unresolved, not impossible.
5. Run independent 3-D pattern searches from three separated feasible starts per
   wave: the best available incumbent and best separated points (including the
   independently sampled banks). Initial normalized step .04, halved on a poll
   with no >1e-6 mm improvement. Stop after a non-improving poll at .0025 or 16
   polls. Profile searches use .04, floor .01, and at most 8 polls. A budget stop
   MUST be labelled unresolved, not convergence. On improvement move and poll
   again, rather than running only one shell around an old center.
6. Validate candidates in descending nominal displacement; preserve the old
   qualified solution if none of the faster candidates passes validation.

Every batch and adaptive poll is written before execution and is resumable.
The previous evaluator and exact irreversible-prefix proofs are reused without
changing physics. A truncated trajectory is never ranked by displacement.
Cold replay must reproduce two old cases exactly before search starts.

## Reporting

Report feasible profile points as sampled LOWER estimates of the profile
maximum, never a mathematical upper bound. Show tested near-best ranges (97%
of best NOMINAL feasible displacement, diagnostic only) without pretending
that all parameters between observed points are feasible. Report both command
amplitudes and actual robust half-ranges (P99-P1)/2, excluding zero-command body
joint 0 from mean yaw response. Include individual-body roll, contact/slip,
impulses, tracking, long/half-step checks, unresolved waves and search stops.

Generate unsmoothed front/side/oblique videos for every selected wave, clearly
marking unresolved cases and comparing with the previous validated baseline.
Also provide local compliant-tip comparisons. Interim results and videos must
be labelled INTERIM and cannot be silently promoted to final winners.
