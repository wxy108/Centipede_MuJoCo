# Align the implicit PD discretization to PGS

The PGS one-DOF probes at 62.5 µs confirm the expected backward-Euler drive law. Without a position-implicit term, MuJoCo-to-PGS response RMS errors are 0.0106%–0.3067%. Including the derived term reduces them to 0.000020%–0.000396%, across all seven physical gain families and their actual effective-inertia scales.

For physical gains K and D, the end-of-step PD law is

`M(v_next − v_old) = h [K(q_target − q_next) + D(v_target − v_next)]`, with `q_next = q_old + h v_next`.

Substitution gives the discrete velocity coefficient **D + h K**. The physical damping remains **D**. In MuJoCo implicitfast, the adapter supplies this derived coefficient to the implicit velocity solve. The velocity-reference feedforward remains `D × v_target`, not `(D + h K) × v_target`. This is not a separately fitted PD table. The physical gains, position targets and velocity targets are shared; the additional term follows from the integration formula, has no fitted constant and vanishes with step size.

This adapter was appropriately rejected for the original TGS comparison and is now independently verified for PGS. Keep the old diagnostic runs. Adopt the adapter for the revised PGS candidate, with MuJoCo contact time constant 2 × internal step. First test the 62.5 / 31.25 µs pair; only use finer steps if the same convergence limits require them. The running unadapted rigid-contact calibration pair is retained as an ablation. The adapter is selected by the mechanical equations/probes, before any of its whole-robot outcomes.

Reports must distinguish the shared physical K/D table from solver coefficients. Do not claim that MuJoCo's numerical affine bias coefficient equals native physical damping. The 1 N·m caps must remain nonbinding, because a force-clamped solve would require separate treatment.
