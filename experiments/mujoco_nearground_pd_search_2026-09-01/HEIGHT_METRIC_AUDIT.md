# Fixed-amplitude foot-height metric audit

Status: diagnostic finding only. Original gates remain in force pending PI
choice. No candidate has been reclassified by changing this definition.

The inherited `swing_foot_height_p95_mm` is the height of `site_foot_*`, at
the center of the spherical foot collision geom, above the flat z=0 plane.
Every foot sphere has radius0.45mm. Thus sphere-bottom clearance is exactly
site-center height minus0.45mm; these are different physical quantities.
The earlier code is internally consistent with its site-based statistic,
but calling it 'foot clearance' without specifying the reference point is
ambiguous. It is not evidence that the simulated body is jumping.

## Diagnostic kinematic reference

With the fixed amplitudes16.9183/49.3775/22.9183deg, set joint coordinates
exactly to the existing CPG targets, keep body pitch/roll and tip roll zero,
and place root z so the lowest foot sphere just touches the flat ground.
This is a pure kinematic construction, NOT a feasible dynamic rollout,
NOT an optimizer sample, and NOT a proof of dynamic infeasibility.

Across the14 tested waves, its swing-site-height P95 is5.734–5.775mm.
For wave2 it is5.7343mm; for wave2.5 it is5.7563mm. The maximum site height
is5.7896mm. Foot-bottom clearances are0.45mm lower. The all-foot P95 of
bottom clearance is5.273–5.304mm (swing-only P95 is5.284–5.325mm).

Consequently, a5.5mm **site-center** cap is lower than this idealized
fixed-amplitude reference. It may favor tracking error, deformation or
lower stance height during PD optimization. That does not prove no PD
solution exists, but it is a task/gate compatibility issue worth resolving.

Example: seed1901 point18, wave2, has site-height P95=5.9316mm, or spherical
foot-bottom clearance5.4816mm. Its other failed absolute gate is mean foot
count5.923<6. Local refinements recover foot count but still fail the
unchanged site-height gate. All results retain their original failure flags.

No silent subtraction or threshold relaxation is permitted. PI was asked
whether5.5mm should instead be a **foot-bottom clearance** cap; no acceptance
rule changes occur until that choice is confirmed. A changed convention
must be versioned and applied equally to baseline and every candidate.

Evidence: outputs/kinematic_height_reference.json and the original per-trial
JSONs in outputs/common. The reference uses the real compiled model and
the same target-generating controller, not a hand-drawn reconstruction.
