# Proof-only hard-gate early exit

The formal `cert_*` broad search uses the frozen 0.5 ms physics timestep,
stride-1 CPG updates, and the full 6 s / 1.5 s evaluation definition.  A trial
may stop early only after at least one hard-gate failure is mathematically
irreversible over the remaining fixed window.

Examples include a running maximum already over its limit (torque, active
joint speed, spine tortuosity, vertical range), a running minimum already
under its limit (upright cosine), a no-contact run already longer than the
allowed maximum, or a cumulative count already greater than the maximum
number allowed in the complete window (chassis-contact fraction and tip-stop
fraction).  These conditions cannot become feasible by later recovery.

An early-rejected row records `proof_early_exit=true` and the exact
`early_terminated_by` metric.  Its partial displacement is diagnostic only and
is never ranked.  Every natural-feasible candidate necessarily runs the full
window and is ranked only after all hard gates pass.

Validation:

- accepted clean, wave 1.5: no early exit, 132.970 mm, base pass, natural pass;
- known chassis-fail candidate: same fail classification, proof exit on
  `chassis_ground_contact_fraction`, 13.44 s wall time.

Machine-readable evidence is in
`outputs/data/proof_early_exit_equivalence.json`.
