# Fixed head/link1 audit

This experiment belongs only to the biological/template **Centipede Model**. It does not modify Centipede Robot Step 18/19.

## Implemented topology

- Build-time head pitch: **-29.0 deg** (previous controller target -32.66 deg).
- `joint_pitch_body_0` and `joint_roll_body_0` are absent; their actuators and head-pitch sensors are absent.
- `link_body_1` is a fixed child of the head. There is no high-stiffness lock and no armature surrogate.
- The combined STL is visual/export-only. Original link0/link1 collision meshes still determine mass and inertia.
- Regenerate at another angle with: `run_locked_head.py all --head-pitch-deg -29 --duration 6`.

## Structural evidence

- nq/nv/nu: [180, 179, 135] -> [178, 177, 133] (exactly -2/-2/-2).
- Active joints: 114; passive body-roll joints: 19.
- Total mass: 0.00253603109178004 kg; delta 0.000e+00 kg.
- Max per-body inertia delta: 0.000e+00 kg m^2.
- Fixed geometry equivalence max position error: 3.380e-14 m.
- Mass matrix positive definite: True; minimum eigenvalue 1.154e-13.
- Robot sentinels unchanged: True.

## 6 s locomotion evidence

| Terrain | Forward displacement | World roll P95 | Local body roll P95 | Tip roll P95 | Tip-stop fraction | Penetration P95 | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| flat | 143.74 mm | 20.96 deg | 3.26 deg | 27.43 deg | 0.719% | 0.0607 mm | PASS |
| rough | 91.14 mm | 12.93 deg | 2.37 deg | 26.73 deg | 0.988% | 0.0180 mm | PASS |

All six view rows pass both the locked-head stability gates and inherited distal-tip gates: **True**.

The STL is not substituted for collision/inertia because a visual union can change mesh preprocessing and therefore mass properties. This separation keeps the requested CAD asset while preserving evidence-traceable dynamics.
