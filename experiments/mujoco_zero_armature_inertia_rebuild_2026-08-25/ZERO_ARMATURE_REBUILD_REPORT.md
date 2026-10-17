# Centipede Model zero-armature rebuild report

Date: 2026-08-25

## Answer to the inertia question

Yes: for a homogeneous rigid part, mass and rigid-body inertia should be derived
from density and geometry about the center of mass.  MuJoCo can do this when an
MJCF body has collision geoms and does not supply an explicit `<inertial>`.

The legacy MJCF supplied explicit `<inertial>` elements on every body, so the
compiler did **not** recompute those values from geometry.  More importantly,
the generalized inertia matrix is not a list of per-body inertias.  It is the
configuration-dependent sum of every body's spatial inertia projected through
its joint Jacobian (`M(q) = sum J^T I J`).  Correct-looking individual link
inertias do not rescue a topology with duplicate generalized directions.

In each legacy leg, passive `joint_roll_leg_*` and active `joint_leg_*_1` are at
the same position and on the same axis (opposite sign on the right).  Their
downstream Jacobian columns are therefore duplicate.  Only an intermediate
`1e-15 kg m^2` placeholder inertia and the added armature separated the two
coordinates numerically.  Removing armature exposed the null direction and
caused the rank-deficient Hessian failure.

## Robot evidence used read-only

The clean Robot runtime uses 1 ms physics, PhysX implicit PD, exact
`armature=0`, three projected-gravity channels, and the tip gate
`abs(projected_gravity_z) < 0.259`.  Its composed USD contains 250 joint prims
and 251 MassAPI bodies and does **not** contain the legacy `joint_roll_leg`
name.  This confirms that the duplicate MuJoCo leg axis was not part of the
clean Robot articulation.

The USD audit also shows that the Robot still contains split bodies at
`1e-9 kg / 1e-15 kg m^2`.  Therefore Isaac stability is evidence for its
topology/solver, not evidence that those placeholder inertias are physically
correct.  They were not copied into this rebuild.

## Mass and inertia rebuild

- Inertial source: collision geoms in group 0 only.
- Visual group-2 duplicates: excluded.
- Uniform solved density: `731.5702043225148 kg/m^3`.
- Source total mass: `0.00253603109178 kg`.
- Rebuilt total mass: `0.00253603109178 kg` (zero reported error).
- Minimum positive rebuilt body mass: `2.1508814645819043e-7 kg`.
- Armature: exactly zero on every DOF.
- Structural changes: 20 zero-offset spine carriers collapsed; 38 redundant
  passive leg-roll hinges removed; all 211 active coordinates preserved.
- Result: 212 bodies, 250 joints, 255 velocity DOFs including the free root.
- Reset mass matrix: rank 255/255, minimum eigenvalue
  `2.2609538185692148e-14`, condition number `1.136412978353109e11`.
- `mj_forward`: passes at exact zero armature.

This condition number is still large because the animal is tiny and contains
very slender distal parts, but it is a physical-geometry matrix without an
armature floor or duplicate joint null direction.

## Controller rebuild

The clean Robot gains were not reused unchanged: they make the rebuilt MuJoCo
model fail at 0.026 s.  For each family, the seed effective inertia was measured
as `1 / diag(inv(M))`; seed `Kp` and `Kv` were calculated from an 8-12 Hz target
natural frequency and damping ratio 0.8.  A recorded frequency screen then
selected body multiplier 8 and leg multiplier 64.  The two re-entry-sensitive
families received additional damping only:

| Family | Kp (N m/rad) | Kv (N m s/rad) |
|---|---:|---:|
| body_yaw | 9.230754080420567e-5 | 3.67280035091154e-7 |
| head_pitch | 7.063766073317632e-5 | 2.810583218533322e-7 |
| body_pitch | 7.842522172336795e-5 | 3.120440425087975e-7 |
| body_roll | 3.920227080835262e-5 | 1.559808794894108e-7 |
| legbase_yaw | 2.243270228179637e-4 | 8.92568863764183e-8 |
| legbase_pitch | 2.146879387374597e-4 | 8.542161668069179e-8 |
| legbase_roll | 1.1639923543493583e-5 | 5.0e-8 |
| knee | 6.137455809849142e-5 | 8.140053584546355e-8 |

## Eight-second validation

All cases are wave 3 on flat ground.  Forward is headward `-x`, matching the
clean Robot reward sign.

| Case | Forward displacement (m) | Minimum root abs(gz) | Maximum tortuosity | Peak active speed (rad/s) | MuJoCo warnings |
|---|---:|---:|---:|---:|---:|
| baseline q=0 | 0.068962 | 0.921 | 1.084 | 50.32 | 0 |
| q=0.3 seed 0 | 0.057779 | 0.879 | 1.167 | 88.75 | 0 |
| q=0.3 seed 1 | 0.059607 | 0.916 | 1.101 | 86.65 | 0 |
| q=0.3 seed 2 | 0.060678 | 0.905 | 1.100 | 88.72 | 0 |

Random-contact mean displacement is `0.059355 m`, or 86.1% of the matched
baseline.  No root or chassis segment crosses the Robot tip threshold, and no
case exceeds the 100 rad/s reference boundary.  Frame inspection confirms that
the animal remains elongated rather than folding over itself.

## Remaining limitation

Instantaneously restoring an erased collider produces a one-step constraint
acceleration peak up to `4.99e6 rad/s^2` in the very small leg-roll coordinate.
The state remains finite, speed remains below 100 rad/s, and MuJoCo emits no
warning, but this is a random-contact re-entry artifact.  The present result is
sufficient for a stable zero-armature repair baseline; a full scientific
random-contact sweep should wait for a separate re-entry/impulse sensitivity
audit.

## Evidence and demos

- `models/centipede_geometry_inertia_armature0.xml`: generated Model.
- `outputs/geometry_inertia_build_audit.json`: mass, inertia and matrix audit.
- `outputs/robot_reference_readonly_audit.json`: composed Robot USD read-only audit.
- `outputs/geometry_impedance_seed.json`: mass-matrix gain derivation.
- `outputs/impedance_tuning/`: retained accepted and rejected tuning evidence.
- `outputs/validation/zero_armature_validation.json`: final gates and all cases.
- `outputs/videos/zero_armature_geometry_baseline_w3.mp4`: baseline Demo.
- `outputs/videos/zero_armature_geometry_random_contact_w3_q030_seed0.mp4`:
  random-contact Demo with erased segments highlighted red.

Robot Step 18/19 sentinel hashes were rechecked after this work and are
unchanged.
