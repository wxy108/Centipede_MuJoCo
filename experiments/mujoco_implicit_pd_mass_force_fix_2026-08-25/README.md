# Centipede Model — MuJoCo implicit-PD mass/force fix

## Scope boundary

This directory belongs only to the biological/abstract **Centipede Model** in
`C:\Users\wxy22\Documents\Centipede_MUJOCO-main`.

It does not read from or write to
`C:\Users\wxy22\Documents\Centipede_robot_MUJOCO`.  In particular, Robot
Step 18/19 data are out of scope and must never be used as Model input or as an
output destination.

All new data, figures, and videos are written below this directory's
`outputs\` folder.  The earlier random-contact result directories remain
unchanged.

## Evidence and defect found

The current clean IsaacLab runtime uses one implicit actuator family with
runtime-assigned `Kp/Kv`; its controller emits position and velocity references
only.  Active explicit effort is zero and the controller contains no passive
spring path.

The latest strong MuJoCo random-contact pipeline contains the same eight low
IsaacLab gains in an unused `LockedTravelingWaveController`, but its actual
`run_trial()` calls the older FARMS Python torque controller.  That older path
uses gains as high as `Kp=2.0` for head yaw and roughly `0.083–0.113` for the
main body/leg drives.  This is the mass/force mismatch: the complete Model mass
is only about `0.002536 kg`, yet the active path still uses the historic
high-gain explicit torque architecture.

## Repair

The repaired runtime makes four linked changes:

1. It uses the frozen refined Model MJCF with the smallest previously validated
   stable MuJoCo armature, `1e-5 kg m^2`; the source MJCF is read-only.
2. It keeps passive stiffness and damping at zero on all 211 active joints.
3. It ports the eight live IsaacLab `Kp/Kv` families and makes the gait code a
   target-only reference generator.
4. It converts the existing unit-gear `<general>` actuators at runtime to
   MuJoCo affine PD servos and switches the runtime integrator to
   `implicitfast`.

For each active joint, Python sends

`ctrl = q_ref + (Kv/Kp) * qdot_ref`

and MuJoCo evaluates

`force = Kp*(q_ref-q) + Kv*(qdot_ref-qdot)`.

Thus `data.ctrl` no longer carries torque.  The velocity-feedback term is in
MuJoCo's implicit solve, while the diagnostic torque returned by the controller
is used only for work/peak-force reporting.

The `1e-5` armature is a provisional numerical regularizer, not a physical
motor inertia and not a MuJoCo-defined minimum.  The current implicit-PD audit
found that exact zero still fails immediately with a rank-deficient sparse
Hessian.  Positive values from `1e-8` through `1e-6` can now finish the tested
8 s runs without warnings, overturning the old RK4/explicit-controller claim
that everything through `1e-6` is numerically unstable.  They do not preserve
the intended gait: zero-noise displacement is negative or near zero and peak
joint speed rises from `18.6 rad/s` at `1e-5` to `194.5 rad/s` at `1e-8`.

The source of the zero-armature failure is structural.  The compiled Model has
96 coordinate-carrier bodies with `mass=1e-9 kg` and inertia `1e-15 kg m^2`;
58 of them carry two joints.  At zero armature the smallest mass-matrix
eigenvalue is about `5e-16` and the condition number is `5.14e12`, which is too
ill-conditioned for MuJoCo's sparse factorization.  A physically grounded
zero-armature model therefore requires rebuilding/repartitioning these inertial
bodies, not another controller-gain tweak.

The expanded implicit-PD sweep places the tested numerical transition between
`1e-9` and `1e-8 kg m^2` at a 1 ms step: all four `1e-9` cases developed huge
QACC and failed, while all four `1e-8` cases finished without a MuJoCo warning.
This is only a numerical boundary.  Task behavior remains poor until higher
values: `1e-8` walks backward, `1e-7` is inconsistent under contact erasure,
`1e-6` moves forward but turns strongly, and `1e-5` is the cleanest operating
point for the current controller.  These results do not certify `1e-5` as a
physical parameter.

## Validation and use

Use the project environment documented by the prior experiment:

```powershell
C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe validate_fix.py --full
C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe run_pipeline.py trial --wave 3 --q 0 --seed 0
```

After the numerical gate passes, the independent random-contact sweep can be
run without changing the protocol:

```powershell
C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe run_pipeline.py sweep --waves 1,2,3,4 --q 0,0.05,0.1,0.2,0.3,0.4 --seeds 8 --workers 4
```

Do not compare the resulting displacement/reliability curves directly with
the earlier strong sweep until new zero-noise baselines are generated for every
wave.  The actuator architecture has changed, so those historical baselines
are not valid denominators for this repaired experiment.

To re-audit whether the `1e-5` numerical armature floor is still necessary
after the implicit-PD migration, run:

```powershell
C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe audit_armature_floor.py --workers 4
```

This leaves the unactuated free root at zero and applies each test value to all
internal joint DOFs.  It tests exact zero and four positive orders using the
reset mass-matrix spectrum, an 8 s zero-noise gait, and three 8 s random-contact
seeds per level.
