# Distal tip passive-gliding axis optimization protocol

## Scope and frozen contract

- Centipede **Model** only. The separate `Centipede_robot_MUJOCO` tree and Robot Step 18/19 are out of scope and must not be read, written, or used as outputs.
- Source plant: locked-head `m29p0` flat/rough models.
- Frozen: topology, 38 tip DOFs, all body/leg geometry, total leg length, mass, inertia, contact parameters, zero armature, active controller law, amplitudes, body/leg wave coupling, near-ground initialization, and the validated wave-specific PD maps for wave 2.5 and 3.
- Changed model variables: all 38 distal-tip hinge axes, passive tip stiffness, passive tip damping, and (only in the range refinement stage) symmetric angular limit.
- The historical XML joint suffix `_roll` is retained temporarily as an evaluator compatibility identifier. Its physical meaning in this experiment is distal **flexion/passive gliding**, not axial roll.

## Axis correction

The historical hinge axis was the parent-leg radial direction. It was 30 degrees from the tip capsule centerline, so it was NOT a mechanically inactive axial spin: it produced a cone sweep. It did not directly implement the fore/aft bending component of passive gliding reported in Diaz et al. (2023).

The candidate axis is `0 0 1` in the distal tip body's local frame. It is perpendicular to the local tip centerline (`+/- local Y`). Since the tip body already carries the fixed 30 degree knee rotation, this axis follows the leg pose while allowing fore/aft bending in the inclined leg-tip plane.

## Registered finite bank

Stage A (flat, 6 s, 0.25 ms, wave 2.5 and 3):

- stiffness: `1e-6`, `4.5e-6`, `2e-5` N m/rad
- damping scale relative to the old damping law at the same stiffness: `0.5`, `1`, `2`
- limit fixed at `+/-38 deg`

Stage B refines around all-gates-feasible Stage-A candidates. It may add stiffness/damping neighbors and limits of 30, 38, or 50 degrees, but must preserve the frozen contract above.

Stage B was registered after Stage A and before its simulations as the following nine-point local bank, with the limit still frozen at `+/-38 deg` because Stage-A feasible candidates peaked below 19 degrees:

- (`1.2e-5`, `d=1`), (`1.2e-5`, `d=2`)
- (`1.6e-5`, `d=1`), (`1.6e-5`, `d=2`)
- (`2.0e-5`, `d=1.5`)
- (`2.5e-5`, `d=1`), (`2.5e-5`, `d=2`)
- (`3.2e-5`, `d=1`), (`3.2e-5`, `d=2`)

Stage C was registered after the Stage-B rough failure and a full-mass-matrix audit. The median effective tip inertia changes from `1.15552e-13` kg m2 on the old axis to `3.32650e-13` kg m2 on local Z (2.879 times). Consequently, the old damping scale maps to only about `zeta=0.4715*d`. Stage C therefore tests physically interpretable over-damped points while keeping stiffness in the flat-feasible basin:

- (`1.6e-5`, `d=3.2`, `zeta~1.51`), (`1.6e-5`, `d=4.2`, `zeta~1.98`)
- (`2.0e-5`, `d=3.2`, `zeta~1.51`), (`2.0e-5`, `d=4.2`, `zeta~1.98`)
- (`2.5e-5`, `d=3.2`, `zeta~1.51`), (`2.5e-5`, `d=4.2`, `zeta~1.98`), (`2.5e-5`, `d=6.4`, `zeta~3.02`)

## Selection and validation

Hard gates are evaluated before comparison. A failed trial is diagnostic and cannot be selected by displacement. Among all-gates-feasible trials, selection is lexicographic:

1. maximize the lower of the wave-2.5 and wave-3 displacement-retention ratios against the matching old-axis validated baseline;
2. minimize worst-wave tip-speed peak;
3. minimize worst-wave ground-force peak.

Finalists require nominal flat replay, rough-terrain replay, 12 s flat replay, half-timestep flat replay, and a targeted obstacle-side-contact passive-gliding check. Claims are limited to “best tested in the registered finite bank”; this is not a continuous global-optimum proof.

### Final assurance correction

The pilot runner reports absolute gates only. Final selection also reapplies the inherited near-ground PD assurance thresholds against the old-axis selected-PD record with the SAME wave, amplitudes, PD configuration, duration and timestep: leg-yaw tracking RMS <= 10 degrees; displacement retention >= 90%; work per distance <= 125%; loaded slip P95, left-right force imbalance RMS, total normal impulse in 10 ms and chassis impulse in 10 ms each <= 110% of their matched baseline. These are supplemental feasibility gates, not a new weighted objective. Stage summaries are therefore provisional absolute-gate rankings; `contract_and_selection.json` is authoritative for full assurance.

The provisional `k=2.5e-5, d=4.2` point passed absolute flat gates but FAILED left-right force non-regression (wave 3, and wave 2.5 half-timestep) and chassis-impulse non-regression (wave 2.5 half-timestep). It is rejected as a final certified model. The best remaining nominal point, `k=2e-5, d=3.2`, is advanced to the same unchanged robustness tests. No thresholds are relaxed.

The `k=2e-5` points at `d=2`, `3.2` and `4.2` also failed the wave-2.5 half-timestep left-right force non-regression threshold. The already registered, nominally feasible `k=1.2e-5, d=2` point is therefore advanced to the same tests to check whether additional fore/aft compliance helps. This is a bounded finalist validation, not a new parameter bank. Untested robustness of other nominally feasible points remains unproven.

## Required outputs

- immutable generated candidate XMLs and hashes;
- complete per-trial JSON/NPZ records and gate ledger;
- plant-diff audit proving only registered model fields changed;
- flat and rough three-view videos;
- local obstacle-contact bending video and stills;
- final report with failures as well as successes.
