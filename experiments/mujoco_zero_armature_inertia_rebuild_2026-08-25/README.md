# Zero-armature geometry-inertia Centipede Model

This directory is an isolated **Centipede Model** experiment.  The Centipede
Robot repositories are read-only references; Robot Step 18/19 data are not
inputs and are never written here.

## Current result

The generated MJCF uses exact `armature=0`.  All explicit `<inertial>` elements
from the legacy model are removed, and MuJoCo derives body mass and inertia from
collision geometry (`group=0`) only.  Visual duplicate geoms are excluded.  A
uniform density of `731.5702043225148 kg/m^3` preserves the legacy Model's total
mass exactly at `0.00253603109178 kg`.

The structural rebuild also:

- collapses 20 zero-offset, geometry-free spine carrier bodies into their
  physical child segments;
- removes 38 passive `joint_roll_leg_*` hinges that are coincident and coaxial
  with the active legbase-pitch hinges;
- preserves all 211 active joint coordinates and 38 passive foot joints;
- produces 250 joints, 255 velocity DOFs including the free root, and a
  positive-definite full-rank reset mass matrix.

## Reproduce

Use the Python executable from the `isaaclab_50` environment, which currently
contains MuJoCo 3.10.0.

```powershell
python build_geometry_inertia_model.py
python derive_geometry_impedance.py
python tune_geometry_impedance.py
python validate_zero_armature_model.py
```

The source model remains read-only.  The generated model is
`models/centipede_geometry_inertia_armature0.xml`.

## Validation boundary

The current gate is one 8 s flat-ground wave-3 baseline and three 8 s
random-contact cases (`q=0.3`, seeds 0-2).  All four are finite, warning-free,
headward-moving, below the clean Robot's `100 rad/s` velocity boundary, above
the Robot projected-gravity tip threshold, and below spine tortuosity `1.25`.

This is a repair baseline, not a completed random-contact sweep.  Contact
reactivation still produces very large one-step constraint `qacc` values in the
small axial leg-roll coordinate even though velocity stays bounded.  Do not use
this build for impulse-level contact claims until that protocol artifact has a
separate audit.

See `ZERO_ARMATURE_REBUILD_REPORT.md` and
`outputs/validation/zero_armature_validation.json` for evidence.
