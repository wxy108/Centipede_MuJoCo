# Align the inertia used by contacts and implicit PD

The position-implicit adapter in amendment 04 aligns isolated drive response but leaves a second splitting difference. In installed MuJoCo 3.10, forward constraints use physical M, then the implicit velocity update uses M+hD+h²K. PhysX PGS solves drive and contact impulses together. Hence contact impulses can be computed against a different inertia from the one that applies them in the split MuJoCo scheme.

Primary evidence: MuJoCo's documented [forward pipeline](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst) orders constraint solution before integration. The newer documented `discrete` method explicitly shares effective inertia with constraints; that enum is absent from this installed runtime. Do not claim this installed build already has it.

For these joint-space affine PD drives, a first-order coupled update can be implemented without changing physical parameters: effective joint inertia increment A=hD+h²K, explicit force K(q_target−q_old)+D v_target−(D+hK)v_old, and semi-implicit Euler using M+A for both constraints and velocity update. Algebraically (M+A)(v_next−v_old)=h[f_other+K(q_target−q_old)+D v_target−(D+hK)v_old+Jᵀf_contact]. This is the same backward-Euler linearized physical PD equation.

Implementation may use MuJoCo's numeric armature slot to supply A, with body masses and physical armature unchanged. The numeric slot must be reported as an integration coefficient, not mislabelled a common physical rotor inertia. Recompute constant effective-weight data after adding A. Do not also apply MuJoCo implicit damping, which would count it twice. MuJoCo contact regularization remains separately verified; this adapter is not a claim of complete equivalence to the newer discrete integrator.

Before whole-robot adoption, compare a 1 mg sphere with free vertical slide and horizontal physical K=2 N/m, D=0.01 N·s/m against native PGS. A 20 µm, 50 Hz position reference plus its analytic velocity drives sliding against the same µ=0.5 plane. Preserve both the split and coupled MuJoCo outcomes. Physical parameters and this factor-free derivation are frozen before the outcome.
