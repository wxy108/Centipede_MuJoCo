# One additional bounded constraint-boundary pass

This pass was specified during `polish`, after wave-2 records showed that some faster full rollouts failed only the existing swing-height gate. It is an adaptive optimization step, not a claim that the entire search was preregistered before any result was observed. The broad, refine, and polish manifests remain unchanged.

After polish completes, the algorithm freezes a separate `boundary_manifest.json`. For each wave with a nominal all-gate feasible point, it takes the highest-displacement feasible point and up to two spatially distinct faster full-run endpoints. Eligible failed endpoints have at most two failed non-catastrophic gate families from the explicit allowlist in `boundary.py`; the absolute body-posture, jump, force, torque, and numerical gates must pass. Duplicate fixed-amplitude and legacy-axis versions of the same metric count as one family for **endpoint selection only**, never for gate evaluation.

It tests joint interpolation of all three amplitudes at fractions 0.25, 0.5, 0.75, 0.875, and 0.9375 toward each endpoint. Known or duplicate conditions are skipped. There are at most 140 added conditions across 14 waves, and no added conditions for waves without a feasible anchor. The interpolation is not assumed to be monotonic, safe, or linear in displacement; every new point is independently screened with the unchanged gates. Only this single bounded pass is added, not an open-ended loop.

Model, PD, frequency, phase, initialization, solver timestep, and contact-erasure setting remain fixed. Failed endpoints are not promoted. Final validation and the tested-set optimum audit include all four banks, including this pass. Continuous global optimality is still not proved.
