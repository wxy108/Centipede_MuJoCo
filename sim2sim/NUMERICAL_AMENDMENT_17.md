# Non-associated normal / tangential contact correction

Written before the correction probe. The prescribed 30 mm/s sliding test shows MuJoCo mean normal clearance about 4 µm and normal RMS speed 6.46 mm/s, versus essentially zero for TGS CPU. Fourfold Mu timestep refinement, NoSlip 20, and impratio 100/10000 do not remove it. This is consistent with associated cone flow / contact dilatancy, not simply creep in the friction direction. It is not yet proof that the same mechanism explains the whole robot gap.

The Coulomb Signorini formulation uses the De Saxcé modified relative velocity: add mu times tangential speed to the normal component before cone complementarity. Reference: https://jtcam.episciences.org/15960/pdf, equations 8 and 16; original formulation https://doi.org/10.1080/08905459108905146. MuJoCo describes its relaxed complementarity at https://mujoco.readthedocs.io/en/latest/computation.html.

For the current acceleration/reference form, test the corresponding fixed point: alpha = J*qacc - aref_original + R*f; set aref_normal = aref_original_normal - mu*norm(alpha_tangent), and repeat the cone solve. This targets independent normal complementarity while keeping the same circular friction limit, regularizer, mass and PD. It is a contact-law alignment correction, not just another tolerance or a changed friction coefficient. No claim of exact equivalence to PhysX's complete contact algorithm is made.

First verify that the manual MuJoCo pipeline without correction reproduces mj_step. Then compare 20/40 correction iterations and a fourfold finer step on the isolated test, recording residuals. No whole-robot adoption before these tests. Candidate profiles and existing runs remain frozen.
