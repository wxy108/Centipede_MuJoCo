# Amplitude optimization recovery — 2026-09-08

User authorization: review earlier unsatisfactory amplitude optimization, avoid its mistakes, and improve the optimization. Model research only. No random-contact optimization, Robot changes or Clean adoption.

## Frozen question

For each wave 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, maximize fixed-axis forward displacement of the unweighted 21-chassis center on flat terrain with no erasure. Search only body yaw [0,36], leg yaw [0,90], leg pitch [0,60] degrees. Keep the current local-Z compliant-tip plant, 1 Hz, coupled wave law, near-ground initialization and historical per-wave PD fixed. Different waves retain their different established PD profiles.

Use the September 3 displacement-first policy unchanged. Naturalness, smoothness, slip, energy, tracking and visual amplitude difference are diagnostics, not extra rejection gates or rewards. No monotonic amplitude curve or crossover reward. Preserve numerical, physical and viability failures.

Nominal rollout 6 s, timestep 0.25 ms, displacement over evaluator labels 1.5–5.999 s. Each final candidate must pass full 6 s, 12 s and 6 s at half timestep under the unchanged policy, with nominal/half-step displacement disagreement <=10%. Report proximity to this bound and long-horizon speed separately. A validated old incumbent must remain available if new candidates fail. Historical scores are proposal evidence only: all candidates used for current selection receive a fresh current-runtime evaluation. Do not exclude a current candidate merely for drifting from a historical score; diagnose that difference and judge fresh full validation.

## Historical errors and repairs

- The first narrow grid omitted near-zero amplitudes and half-wave values. Include them and the full documented box.
- Later naturalness/reference-relative restrictions dominated displacement. Retain the already authorized displacement-first policy; do not add new aesthetic constraints.
- Latest global queue stopped before high waves, independent global sampling or multistart refinement completed. Use explicit immutable per-stage task banks, individual atomic receipts, bounded in-flight jobs and resumable batches. Completion requires inspecting receipts, not reporting a background launch.
- Old early-stopped records are not complete scores or proof under the current policy. Retain their provenance and rerun when selected.
- Compare old and new configurations in one current runtime. Report actual joint amplitudes as well as commands, and verify full trajectories for final selections.

## Search stages

1. Audit all comparable historical records and re-evaluate diverse top candidates, prior validated selections and the fixed-amplitude reference in the current runtime.
2. Every wave receives the same fresh 32-point scrambled Sobol design over the full box, eight boundary/near-zero anchor challenges, and cross-wave amplitude transfers selected from current qualified candidates. Scores from other waves are never transferred.
3. Four sequential surrogate-assisted batches, eight candidates per wave per batch. Historical complete records can inform proposal models, but only fresh scores enter final ranking. Mix global candidate proposals with neighborhoods of diverse high-performing candidates, retain explicit exploration and deduplicate against evaluated conditions. Save the proposal pool, seed and acquisition rule. This is adaptive finite search, not a proof of global optimality.
4. Two diverse current feasible starts per wave, up to eight coupled 3-D poll rounds (six axial plus eight diagonal directions). Start normalized step .04; halve on no improvement, down to .005. A gain <=0.2 mm counts as no material improvement for step reduction. Log per-start gains and whether the declared resolution or poll budget stopped the search. No-feasible waves remain unresolved and receive the same global/exploration effort.
5. Descending-displacement candidate validation, with at most five candidates per wave initially and fallback to prior validated configurations. Never promote a failed high scorer. If no candidate passes, report unresolved and its specific limitation.
6. Report every wave, before/after matched displacement and amplitudes, search coverage, improvement histories, physical diagnostics, actual motion and front/side/oblique demos of representative changes and unresolved cases.

Prefix rejection may be used only after a necessary unchanged full-horizon gate is irreversibly violated, with exact gate/value/time evidence. Such records have no ranked displacement. No heuristic early stopping based on low prefix speed. Surviving candidates receive all full metrics. Any evaluator instrumentation must match fresh uninstrumented metrics on representative amplitudes before use. All source inputs and amendments are hashed; no source experiment is edited.
