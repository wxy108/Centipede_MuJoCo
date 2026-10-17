# Explicit articulation contact regularization

2026-09-11, before CFM ablation. The friction-limit probe in amendment 10 is real, but the first whole-gait square-friction result did not close the gap: w=2.5/B MuJoCo at 31.25 µs gives 136.248 mm with a fixed basis and 146.845 mm with the public-source velocity basis, versus 160.372 mm with the prior elliptic candidate and 194.838 mm on native CPU. The square adapter therefore remains experimental; it is not selected on the basis of performance.

Installed `omni.usd.schema.physx-107.3.18+107.3.1.wx64.r.cp311.u353/plugins/PhysxSchema/resources/schema.usda` defines `physxRigidBody:cfmScale=0.025`. Its documentation says this weakens articulation constraint response. The field is separate from body mass, physical joint PD and contact material. The current common plant does not explicitly override it. A hidden regularization term can matter in a stiff, low-inertia articulation even when bare mass matrices and no-contact impulses match.

The next isolated native CPU experiment will explicitly set CFM scale to zero on every rigid body before bootstrap. This tests the rigid-contact limit; it is not a physical PD/mass/friction fit. Record all actual authored scene values and keep default-CFM results. No new configuration is adopted until independent mechanics and the frozen full-bank convergence tests support it.

The same installed schema gives `contactSlopCoefficient=0`, so there is no evidence here for a nonzero default contact moment-arm cutoff. Do not claim the public source's offset-slop mechanism caused the observed gap.
