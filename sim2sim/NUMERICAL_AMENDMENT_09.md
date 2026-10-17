# Full coupled-scheme and native-GPU refinement

The coupled MuJoCo 62.5→31.25 µs bank completed all eight gaits; the maximum displacement change is about 3.42%. Comparing MuJoCo 31.25 µs with the completed native PGS GPU 62.5 µs bank puts every gait below the unchanged 10% symmetric gap limit. MuJoCo 62.5 µs still narrowly exceeds 10% in w=4.5 / B and is not promoted as the operating step.

Freeze the next candidate at MuJoCo 31.25 µs with the coupled effective-inertia update, native GPU PGS 62.5 µs, and the same 62.5 µs command clock. Test MuJoCo at 15.625 µs with held commands, and native GPU at 31.25 µs with the complete-scheme reference refinement already disclosed in amendment 02. Run all eight gaits plus the native calibration row. Keep the 5% per-engine refinement criterion and 10% pairwise performance criterion unchanged.

Native CPU results, including successful CPU timestep and iteration convergence, remain in the report. They differ from GPU in some large leg-yaw gaits. CPU may not be silently substituted for the actual GPU backend used by Isaac Lab RL. A GPU-only qualified operating scope, if it passes, must be labelled accordingly; it is not a claim that CPU/GPU results are interchangeable.

The direct-target transport experiment was not adopted: it changed both batch size and diagnostics, did not reproduce the original full-bank trajectory, and yielded little speed benefit. The full GPU refinement uses the original v6 path. A 1 µs force-to-velocity probe verifies the seven sampled joint families' coupled physical inverse inertia on both CPU and GPU to below 0.0026% in kinetic norm; uniform kg→mg conversion changed the large-gait CPU displacement by only 0.07%. Those precision hypotheses do not currently explain the whole performance gap.
