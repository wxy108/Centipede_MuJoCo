# Runtime alignment receipt

This is a new opt-in branch using the native September rigid-tip plant as the common physical definition. It is not the old MuJoCo mesh-derived plant, and it does not alter historical optimization results.

## Verified setup

| Quantity | Fresh PhysX backend versus compiled MuJoCo |
|---|---:|
| Rigid bodies / revolute joints | 251 / 173 in both |
| Total mass | 2.536051087 g in both |
| Largest per-body relative mass error | 6.681 × 10⁻⁸ |
| Largest full inertia tensor relative error | 1.216 × 10⁻⁷ |
| COM position discrepancy | 0 |
| Largest stiffness / damping relative error | 3.710 × 10⁻⁸ / 5.670 × 10⁻⁸ |
| Initial body-position error | ≤ 0.047 µm |
| Two nonzero joint configurations: body-position error | ≤ 0.066 µm |
| Two nonzero joint configurations: orientation error | ≤ 6.180 × 10⁻⁷ rad |
| All 249 robot material readbacks | static/dynamic friction 0.5/0.5; restitution 0 |
| All 249 robot contact/rest offsets | 20 µm / 0 |
| Physical armature / joint friction | 0 / 0; the MuJoCo numerical inertia increment is separate |
| Torque / joint velocity caps | 1 N·m / 10⁶ rad/s |

Inertia comparison uses tensors about the COM expressed in link-aligned axes. The getter already rotates the principal inertia; applying COM orientation again would be an audit error.

Both systems use one plane at z = 0, self-collision disabled, gravity −9.81 m/s², no rigid-body damping, the same two movable head DOFs, the same SI-radian PD table, the same ramp and wave commands, and the same initial pose and zero velocities. Contact algorithms remain different and require dynamic verification.

Scores use the unweighted centroid of `link_body_0` through `link_body_20`, displacement along negative world X between 2 and 6 seconds. Lateral drift, heading range, height and tracking errors are retained separately.

Raw evidence: `audit/runtime_readback.json`, `runs/native_readback_cpu_000/backend.npz`, `runs/native_calibration_cpu_dt125_fk_000/fk.npz`, `runs/native_calibration_cpu_dt125_fk_000/fk_comparison.json`.

## Current numerical profile

Current status: **validation_pending**. The selected candidate is coupled MuJoCo at 31.25 µs and native PhysX GPU PGS 16/4 at 62.5 µs, with a shared 62.5 µs command clock. The eight nominal displacement gaps have maximum 8.919%. MuJoCo 31.25→15.625 µs refinement has maximum 2.223%.

GPU refinement completion: 8/8 gaits. See AUDIT_AND_STATUS.zh-CN.md and audit/coupled_candidate_summary.json for live acceptance results. The CPU backend retains a separate performance discrepancy and is not interchangeable with the GPU benchmark.

The older split-PD / fixed-soft-contact candidate failed full-bank convergence and was not adopted. In the current method, the contact and velocity solves share M+hD+h²K. The numeric armature slot carries hD+h²K; this is an integration coefficient, not a physical rotor inertia. The physical PD, body masses, COMs and tensors remain fixed. The contact time constant is the numerical rigid approximation 2h. Thus backend numerical fields are not claimed to be identical just because the physical definition is shared.

The complete-matrix comparison also transforms root-origin translation to root-COM velocity coordinates. After that transform, seven 1 µs force-to-velocity probes agree within 0.0026% in kinetic norm on both native CPU and GPU. This excludes a large hidden inverse-inertia loading error in those sampled directions; it does not prove equality of contact algorithms.

MuJoCo's numeric stage forces and native pre-step estimated PD forces are labelled separately. They are used as torque-cap diagnostics, not as interchangeable measurements of mechanical work.
