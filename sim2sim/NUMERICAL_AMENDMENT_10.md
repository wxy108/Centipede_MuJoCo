# Friction constraint shape: independent evidence before full-gait changes

2026-09-11. The GPU 62.5→31.25 µs full-bank run remains active and must finish. Its 2–3 s checkpoint suggests large changes in two B-family gaits, so the nominal eight-gait comparison is insufficient. This amendment does not promote a candidate, alter any acceptance threshold, or replace incomplete scores.

## Independent force probe

A rotation-locked sphere has free X/Y/Z translation, radius 0.45 mm, sphere mass 1 mg, and the same tiny carrier masses in both engines. It starts touching a plane at rest. Static/dynamic friction is 0.5/0.5. A constant horizontal force of magnitude 1.2 times mu times its normal weight is applied for 10 ms. The force was chosen analytically: at 45° it lies outside a circular Coulomb limit but inside the square obtained by bounding each tangent component separately.

At 62.5 µs, the installed PhysX CPU PGS gives approximately 49.406 µm motion for the axis-aligned force, agreeing with MuJoCo's 49.167 µm. For the 45° force, PhysX remains still while the elliptic MuJoCo model moves (34.777, 34.789) µm. This directly demonstrates different tangent-force limits in this installed CPU backend; it is not inferred from whole-robot performance.

MuJoCo's ordinary four-edge pyramid at the same edge coefficient is a diamond, and moves even farther diagonally. Merely switching `cone=pyramidal` is therefore not an equivalent friction model. A four-edge pyramid with edges rotated 45° and edge coefficient sqrt(2)*mu describes the desired square exactly. It reproduces both independent force directions: axis motion 49.167 µm; diagonal drift about 0.000711 µm per axis.

The sqrt(2) factor is an analytic conversion between the axis bound and a pyramid vertex, not a coefficient fitted to locomotion. Both physical tangent-axis bounds remain mu=0.5. The common law is now explicitly described as the native square approximation, whose diagonal maximum is sqrt(2)*mu*N; it must not be advertised as an isotropic circular Coulomb law.

Raw evidence: `audit/friction_cone_comparison.json`, `runs/friction_cone_cpu_angle{0,45}_000`, and preserved Mu trajectories for elliptic, ordinary pyramidal, and derived-square representations. Independent free-rolling tests also agree closely and do not explain the remaining gap: `audit/roll_contact_probe_comparison.json`.

## Source and scope

Public PhysX PGS articulation source clamps each tangent row independently and selects its first tangent from the body's relative linear velocity when its squared magnitude exceeds 0.0001; otherwise it uses a normal-based fallback. This source is a hypothesis for reproducing tangent orientation, not proof of the exact version installed locally. The force probe above is the installed-backend evidence. GPU friction behavior still requires a separate probe after the currently active GPU run.

- https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/lowleveldynamics/src/DySolverConstraints.cpp
- https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/lowleveldynamics/src/DyArticulationContactPrep.cpp

The installed 107.3.18 schema exposes only the `patch` friction type and does not expose the older `improvePatchFriction` / disable-strong-friction fields. No unsupported custom attribute is used as a fix. Contact patch persistence may still differ after matching the force limit.

## Next falsifiable check

Add an opt-in MuJoCo square-friction adapter on top of the independently justified coupled-PD method. Keep the physical masses, geometry, axis friction coefficients, PD, amplitudes, commands and score unchanged. Start with w=2.5 / B, compare both a fixed tangent basis and the source-derived velocity basis, and then validate the full frozen bank if the mechanism remains supported. Do not overwrite v8, the original contract, or previous results. This is a numerical/model-form experiment, not an adopted profile.
