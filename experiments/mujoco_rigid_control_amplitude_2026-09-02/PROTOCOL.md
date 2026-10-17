# Rigid control and per-wave amplitude optimization

## Scope

Only `Centipede_MUJOCO-main`, biological/template Centipede Model. No access, output or mutation in the separate Robot project; Robot Step 18/19 are excluded. Every output of this experiment is isolated here; prior candidate and baseline files stay unchanged.

## Part 1: genuinely rigid leg control

Reference: the latest `localz_k2em05_d3p2_r38` compliant-tip candidate. Remove all 38 tip joints and merge the distal tip and foot geometry into the corresponding leg-pitch body at neutral tip angle. Keep active leg yaw/pitch at the base. No knee joints, surrogate large stiffness, added armature, shorter reach or changed contact parameters.

Audit equivalent neutral geometry under seven random common-joint poses; equal total mass; equal projected common-DOF mass matrix; equal active PD arrays and common geometry/contact fields. Individual fused-body inertia components necessarily change representation but must equal the aggregate physical inertia of the locked assembly.

Paired trials: flat and rough, waves 2.5 and 3, six seconds, 0.25 ms physics step, same wave-specific historical PD, amplitudes, frequency, phase relationships, near-ground initialization and zero initial velocities. No contact erasure. Save unsmoothed poses and front/side/oblique demos for every pair. Rigid tip angle/speed gates are not applicable, not fictitious successes.

All common absolute gates retain the previous values. Compare raw displacement, body pose, speeds, 10 ms changes, contact forces/impulses, slip, gait tracking, duty and repeatability; keep failed cases visible. Rough is diagnostic, not selected by flat displacement or absolute world-Z swing height. This comparison does not replace the flexible primary model.

## Part 2: amplitude-only flat study

The intended primary plant remains the latest flexible-tip model unless the PI asks otherwise. Rigid remains the paired control. No random contact in this stage.

Wave set: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9. Body and leg wave numbers stay equal. Frequency, phase law, DC offsets, passive tip parameters and all other physical parameters are frozen.

Only three variables: body-yaw amplitude [0,36] degrees, leg-yaw [0,90] degrees and leg-pitch [0,60] degrees, restoring the broad previously declared domain. These are search bounds, not biological envelopes. Realized-motion and stable-pose gates determine admissibility.

PD provenance for every wave must be recorded and frozen before amplitude search. Only waves 2.5 and 3 have previously validated PD maps; other historical best diagnostic PD records are starting points, not validated assumptions. No PD optimization is folded into amplitude search.

Gate-first objective: maximize forward displacement over 1.5 s to the end of the rollout only among all-applicable-gates-feasible records. No weighted reward. Use broad scrambled Sobol coverage plus anchors and boundaries, then refine distinct promising feasible basins. Any early rejection must cite an irreversible existing hard-gate violation and must not be ranked by truncated displacement. Finalists require full cold-instance runs and long-horizon/half-step checks. Do not claim continuous mathematical global optimality; report the finite searched set and unresolved waves explicitly.

The exact sample bank, frozen PD map, comparison references and supplemental gates will be committed to a machine-readable manifest before launching Part 2.

## Reproduction and calculation integrity

`engine.py prepare` constructs and audits the rigid control; `engine.py ablation` runs the eight matched conditions; `render.py` creates paired full and local three-view videos. `search.py freeze`, `reference`, and `broad` register and evaluate the flat bank. `refine.py refine` and `refine.py polish` freeze and evaluate bounded neighborhoods of two distinct seeds, plus the fixed-amplitude reference if absent. Keeping that reference seed prevents slow near-stationary diagnostics from crowding out a high-displacement starting point. `validate.py` performs cold 12-second and half-step checks; `delivery.py` records the finite-set result and demos without overwriting prior selected profiles.

Simulation-only arena reservation is reduced in an in-memory XML copy from 2.18 GB to 64 MiB. No source physical parameters or solver options change. All 426 compiled model arrays and 6-second dynamics checkpoints exactly agree in three checks (flat baseline, flat large-amplitude, rough baseline); zero warnings, peak observed arena usage 2.16 MB. See `outputs/data/runtime_arena_equivalence.json`.

Search-only early rejection checks a 2.5-second prefix. All quantile rejection budgets are calculated against the full six-second horizon: already observed above-limit samples must exceed `ceil((1-q)*maximum_final_sample_count)+1`. Even if every future sample were below the limit, the percentile could not pass. Sampled velocities use the same pre-step timing, and body orientations/contacts the same post-step timing, as the original evaluator. Every survivor is evaluated from a fresh cold instance for the entire six seconds. Finalists are replayed again with saved traces. Four comparisons against existing full trajectories verify no false rejection of the two known feasible references and consistent failure of the diagnostic examples. The one-second zero-command settled-state cache also passes an exact state-copy replay test.

All interrupted compute passes are resumable; their completed records are retained. A resource warning is an evaluation error, never relabelled as a physical gate failure.

Earlier completed prechecks used prefixes up to the full six-second horizon before the prefix-only performance adjustment; these valid failures remain cached with their actual `checked_until_s`. This does not change the measured full-run objective or any safety threshold. Numerical-divergence warnings fail the warning-free gate; memory/contact-capacity warnings invalidate the evaluation and require resource correction instead.

See `OBJECTIVE_CLARIFICATION.md`: the inherited 90%-of-reference displacement floor is retained as a diagnostic, not a primary safety gate. No motion, pose, naturality, force, slip, work or numerical-validation gate is relaxed. The original manifest and initial worker summary are preserved; clarified results are written separately.

Final validation is descending-displacement, in batches of three per wave. If the first batch does not qualify, continue through the remaining nominal feasible candidates. Stop for that wave only after a qualified candidate is found (every higher nominal-displacement candidate has then been checked) or that entire feasible list is exhausted. Each additional batch is recorded before execution. This establishes the highest validated nominal displacement within the evaluated finite candidate bank; it does not prove continuous global optimality outside that bank.
