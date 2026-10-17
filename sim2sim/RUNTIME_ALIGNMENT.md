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

Both systems use one plane at z = 0, self-collision disabled, gravity −9.81 m/s², no rigid-body damping, the same two movable head DOFs, the same SI-radian PD table, the same ramp and wave commands, and the same initial pose and zero velocities. Native contact algorithms differ; the current shared-force branch disables their contact responses. The readable native material attributes above do not define the active compliant contact law.

Scores use the unweighted centroid of `link_body_0` through `link_body_20`, displacement along negative world X between 2 and 6 seconds. Lateral drift, heading range, height and tracking errors are retained separately.

Raw evidence: `audit/runtime_readback.json`, `runs/native_readback_cpu_000/backend.npz`, `runs/native_calibration_cpu_dt125_fk_000/fk.npz`, `runs/native_calibration_cpu_dt125_fk_000/fk_comparison.json`.

## Current numerical profile

Shared compliant-contact benchmark: 已通过本报告限定的物理参数、前进性能、XY 轨迹、方向和半步长验收. Native rigid-contact PGS and TGS candidates remain rejected. Use COMMON_CONTACT_OPERATING_PROFILE.json only when present and validated. Contact forces use COMMON_CONTACT_V2.json in both engines; native ground contact is disabled.

Main physics step is 31.25 microseconds; refinement is 15.625 microseconds. Command period remains 250 microseconds. Native PGS16/4, two CPU workers, TGS-only external iteration flag false. Mu uses the physically derived M+hD+h^2K adapter; numeric armature is not physical rotor inertia.

The current bank places all independent articulations at the same numerical origin. Native collisions remain disabled and contact forces are separate per environment. Additional XY-path and chord-heading gates are required across engines and across time-step refinement. The displaced-origin bank remains preserved as a rejected coordinate layout, despite its excellent forward agreement. GPU and CPU are independently validated. No RL training pipeline has been migrated. See REPORT.md, common_contact_centered_motion_comparison.csv and the coordinate-precision audit.
