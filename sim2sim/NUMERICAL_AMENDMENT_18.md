# Whole-robot test after isolated De Saxce correction passes

Written before new whole-robot results. Manual zero-correction stepping exactly reproduces mj_step on the prescribed-slide trajectory. The correction reduces normal-speed RMS from 6.458 mm/s to approximately 2.5e-13 mm/s. 20 and 40 outer iterations produce the same trajectory, no unresolved steps, max correction residual 6.1e-9 m/s. Fourfold smaller step also passes. Native TGS measured 3.73e-5 mm/s under the same prescribed boundary.

Now test the unchanged 2.5/B robot at 62.5 µs and 31.25 µs with the held 250 µs clock, physical Mu v8 PD coupling, and the contact correction. This is a diagnostic pair, not full-bank acceptance. Retain all solver-residual and unconverged-step counts; score remains the original 2–6 s. If it supports the mechanism, require independent loaded/impact/friction probes and the complete eight-gait bank with refinement before adopting a new operating profile.

This aligns the normal-versus-tangential contact constitutive relation; it is not a coefficient fit and not an assertion that default MuJoCo and PhysX contact algorithms are identical.
