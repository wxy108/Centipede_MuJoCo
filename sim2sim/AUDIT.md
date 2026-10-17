# Sim-to-sim initial audit — before setup changes

2026-09-11. Scope and user authorization are in AUDIT_SCOPE.md. This report was written before creating aligned assets or starting new simulations. Static reads and saved runtime evidence were used; simulation steps during this audit: 0.

## Actual work paths

MuJoCo generic README describes older Blender/FARMS workflows. The September result instead uses common_pd_amplitude_contact_2026-09-09/core.py → rigid_control_amplitude engine → tip_flexion runner → three-amplitude controller / LeanIndex / evaluator. Its frozen plant is centipede_flat_localz_k2em05_d3p2_r38.xml. The common-PD midpoint branch drives flat locomotion at 1 Hz; the completed optimization maximizes 21-body-centroid forward displacement with separate qualification.

Original Isaac Lab research uses centipede_env.py, centipede_env_cfg.py, impedance_controller_torch.py and kinematics.py. The actual September cell is stage_a_vec_sweep_cell_compliance_2026-07-13.py, wrapped by outputs/analysis/common_pd_amplitude_contact_20260909/code/runtime.py. It selects the September rigid-main-leg/passive-tip USD, the head8_ramp025 YAML, a full per-family implicit impedance table and an arena-derived flat surface. The controller produces position and finite-difference velocity targets; the environment sends them to PhysX drives. Policy/reward/observations are additional environment work, not required for isolating the open-loop mechanical gap. The September cell uses 1 ms, not the 4 ms default in the generic environment config.

The historical RL v4 optimizer varied impedance with wave and used near-best/smooth selection. September MuJoCo fixed the full impedance and searched all three amplitudes over a much wider domain. Optimizer outputs alone therefore do not isolate an engine difference. No old optimization/contact queue is resumed here.

## Verified differences

| Item | Current MuJoCo | Actual September Isaac Lab | Interpretation |
|---|---|---|---|
| Total mass | 2.5360310918 g, compiled | 2.5360510870 g, composed; saved backend agrees | Only about 0.000788% apart; this masks redistribution |
| Tail link_body_20 mass | 0.146094614 g | 0.179572322 g | Native tail about 22.915% heavier |
| Tail principal moments | compiled mesh-derived values | explicit converted values | Largest relative principal-moment difference about 23.015% |
| Body link_body_8 | mass 0.114060281 g | mass 0.109677720 g | Its largest sorted moment difference is about 14.765%; COM differs too |
| One complete leg, pair 0 left | 3.434946387 mg | 3.917995336 mg | Native leg about 14.06% heavier; compare entire leg, not only same-named carrier |
| Representation | 173 non-world bodies, 171 hinge DOFs | 251 bodies, 173 revolute DOFs plus 77 fixed joints | Splitting/welding and two extra head DOFs must be handled explicitly |
| Joint position limits | 171 matched hinges | 171 matching backend limits | Prior audit max difference 0.000066 deg; not evidence for matching axes/inertia |
| Head | fixed relative angle −29 deg | head pitch and roll movable | Native actual head pose can approach MuJoCo, but dynamic constraints differ |
| Sliding friction | 1.6 | 0.5 static and dynamic in saved backend | Real material mismatch |
| Internal collisions | on | off | Prior paired MuJoCo ablation reverses gait ranking |
| Contact shape offsets | margin/gap zero, soft contact law | shape-dependent contact offsets about 19.62–111.21 micrometres | Native custom USD fields were not uniformly effective |
| Ground | one flat plane | two enabled surfaces at different heights | Use one matched plane in the new comparison |
| PD discretization | affine force PD; implicitfast integrates velocity derivatives | position and velocity drives solved implicitly in PhysX | Same Kp/Kv does not imply the same discrete response |
| Time grid | actual September dt 0.125 ms | 1 ms with TGS | Source XML's 0.25 ms is scaled by the runner |
| Reference generator | analytic travelling wave + per-segment ramp | stateful CPG + ramp | Saved steady targets approximately match; onset/clock histories need unification |
| Score | 21-body geometric centroid, 1.500–5.999 s | formal 61-spine centroid over 5.952 s after warmup | Raw displacement magnitudes are not a fair sim-to-sim score |
| Reset | fresh MuJoCo model/data | repeated same-process native reset previously differed | First cold native request reproduced exactly; use fresh scene/process for validation |

The static inventory contains full body masses, COMs, principal inertias/quaternions, joint frames, collider transforms and source hashes. It deliberately does not mistake a tiny same-named USD carrier for the mass of a whole welded leg. USD inertia values in this report are authored/composed values; a fresh effective backend inertia readback is still required.

## Drive semantics and unresolved hypotheses

MuJoCo implicitfast is implicit in velocity, not in position stiffness. PhysX solves end-of-step drive constraints. This is an established formulation difference; its contribution to these centipede performance differences is **not yet measured**. A one-DOF response test and timestep convergence must precede attributing optimized PD differences to it. The installed MuJoCo 3.10.0 exposes Euler/RK4/implicit/implicitfast only; the newer `discrete` option visible in current online docs is not available in this installed build and will not be silently assumed.

Primary references: [MuJoCo numerical integration](https://mujoco.readthedocs.io/en/stable/computation/index.html#numerical-integration), [PhysX articulation drives](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html#articulation-joint-drives), [PhysX implicit drive derivation](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf).

The installed Isaac Lab PhysxCfg also defaults to metre-scale friction thresholds (0.04 m offset threshold, 0.025 m correlation distance). These are configuration defaults confirmed in source, not yet a live readback of the old scene. Their impact on sub-millimetre feet is a hypothesis to isolate, not a proven explanation.

## Decision before implementation

Create a dated common physical definition derived explicitly from the composed native rigid-tip asset, including its per-link inertias and joint topology. Generate the matching MuJoCo representation from those values, rather than comparing two separately rebuilt models that merely have the same total mass. Apply one shared material/PD/target/initial-state/score contract to both. Preserve both historical plants and all existing results.

This changes the comparison model from the old MuJoCo plant and must be labelled as an alignment branch. It is not a retroactive correction to the historical optimization results. Backend-specific contact solver tolerances cannot be equated by matching field names; they require mechanical response and convergence checks. Initial setup contract and acceptance limits are frozen in COMMON_CONTRACT.json before dynamic results.
