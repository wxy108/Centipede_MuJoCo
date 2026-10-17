# MuJoCo armature audit (Centipede Model only)

Date: 2026-08-25

## Scope guard

This report covers only the biologically inspired/abstract **Centipede Model** under
`C:\Users\wxy22\Documents\Centipede_MUJOCO-main`. It does not use or modify the
Centipede Robot workspace, Step 18, or Step 19 data.

## Direct answer

- `1e-5 kg m^2` is **not** a MuJoCo hard minimum. It is the best current working
  regularizer among the values tested with this model and controller.
- Exact zero armature is not currently executable for this MJCF. Model creation at
  zero fails immediately with `FactorizeHessian: rank-deficient sparse Hessian`.
- The observed short-test numerical transition lies in `(1e-9, 1e-8] kg m^2`.
  This is an empirical bracket for this model, timestep, and controller, not a
  general MuJoCo limit.
- Numerical survival is not sufficient for task validity. `1e-8` survives all four
  8 s cases but reverses direction and produces extreme joint speed/acceleration.
  `1e-5` is the cleanest tested operating point.

## Reproducible sweep

Each positive armature value was tested for 8 s in one no-erasure baseline and
three random-contact-erasure runs (`q=0.3`, seeds 0, 1, and 2). The armature
override applies only to internal joint DOFs; the unactuated floating root remains
at zero.

| Internal armature (kg m^2) | Finite cases | Warning cases | Baseline x (m) | Random mean x (m) | Peak speed (rad/s) | Peak acceleration (rad/s^2) | Interpretation |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0/4 | n/a | n/a | n/a | n/a | n/a | immediate rank-deficient Hessian fatal error |
| 1e-12 | 0/4 | 4/4 | n/a | n/a | 132786 | 3.43e9 | unstable |
| 1e-11 | 0/4 | 4/4 | n/a | n/a | 9.33e6 | 2.05e8 | unstable |
| 1e-10 | 0/4 | 4/4 | -0.0409 | -0.1034 | 3.62e6 | 4.83e9 | unstable |
| 1e-9 | 0/4 | 4/4 | -0.00187 | -0.0853 | 4.79e6 | 1.22e9 | unstable |
| 1e-8 | 4/4 | 0/4 | -0.0452 | -0.0444 | 211.7 | 129709 | numerically finite, task behavior invalid |
| 1e-7 | 4/4 | 0/4 | 0.0157 | -0.0193 | 93.8 | 13965 | weak and random cases reverse |
| 1e-6 | 4/4 | 0/4 | 0.0989 | 0.0677 | 33.3 | 2995 | locomotes, but turning/yaw is poor |
| 1e-5 | 4/4 | 0/4 | 0.1377 | 0.1265 | 16.6 | 1129 | best tested current operating point |

At zero armature, the reset mass matrix has minimum eigenvalue
`4.999442678843238e-16` and condition number `5.139920587365315e12`.
Although the computed eigenvalue is positive at the inspected configuration, the
system is too ill-conditioned for MuJoCo's sparse Hessian factorization.

## Structural cause

The source model has 232 bodies, 288 joints, and 293 DOFs, with total mass
`0.00253603109178 kg`. It contains 96 coordinate-carrier bodies with mass exactly
`1e-9 kg` and diagonal inertia `1e-15 kg m^2`; 58 of those bodies carry two joints.
Together they contribute only about `3.785e-5` of total mass. These near-massless
multi-joint carriers create nearly inertia-free generalized directions. Armature
currently regularizes those directions.

This means the limitation belongs to the present inertial tree, not to a MuJoCo
rule that requires armature to be at least `1e-5`.

## Meaning of the present model

The current `1e-5` setup is a **regularized Centipede Model**. It is suitable for
continued locomotion and random-contact experiments when conclusions are stated
as conditional on this calibrated dynamics. It should not yet be treated as a
zero-armature physical reconstruction or as inertia-equivalent to the IsaacLab
articulation.

## Path to exact zero armature

1. Rebuild or collapse the 96 coordinate-carrier bodies while preserving joint
   frames, kinematics, geometry, collision, total segment mass, center of mass,
   and inertia.
2. Do not merely increase the placeholder masses: that would change the animal's
   mass distribution and morphology.
3. Require exact-zero model creation, `mj_forward`, and 8 s baseline/random tests
   to finish with finite state and no MuJoCo warnings.
4. Retune `Kp`/`Kv` after the inertial rebuild because removing reflected inertia
   changes the joint natural frequencies.
5. Re-establish the baseline and rerun the random-contact sweep before comparing
   with earlier results or IsaacLab.

## Evidence files

- `audit_armature_floor.py`: repeatable mass-matrix and dynamic sweep.
- `outputs/armature_audit/armature_floor_summary.csv`: one row per armature value.
- `outputs/armature_audit/armature_floor_cases.csv`: all individual cases.
- `outputs/armature_audit/armature_floor_audit.json`: complete audit payload.
- `outputs/validation/implicit_pd_validation.json`: validated default `1e-5`
  implicit-PD pipeline.
- `outputs/videos/armature_comparison_w3_q000_arm1e-8.mp4`: finite but invalid
  low-armature behavior.
- `outputs/videos/armature_comparison_w3_q000_arm1e-5.mp4`: current working
  baseline behavior.
