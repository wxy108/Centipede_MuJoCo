# Common finer-timestep amplitude optimization

The user requested that earlier unsuccessful amplitude optimization be reviewed and improved. All work stays in the Model research experiment, with no Robot/Clean changes or random-contact optimization.

Three-level diagnostics found coarse timestep bias in low-wave forward displacement: wave 0.5 gave 107.4595, 77.5551, 74.8761 mm at 0.25, 0.125 and 0.0625 ms; wave 1 gave 154.8647, 125.9202, 120.9870 mm. Accordingly, the coarse search is preparation only and its original remaining search plan is superseded. Coarse and fine scores must never be pooled in a final ranking.

## Fixed experiment

- All 14 waves: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9.
- Same local-Z compliant-tip model, mass/contact parameters, 1 Hz coupled body/leg wave law, near-ground initialization, head configuration and established per-wave PD. Only amplitudes change: body yaw [0,36], leg yaw [0,90], leg pitch [0,60] degrees.
- Nominal 6 s at **0.125 ms** for every wave. Fixed-forward-axis displacement of the unweighted 21-chassis center over labels 1.5–5.999 s is the only final performance objective.
- Preserve September 3 physical/engineering gates. Slip, work, visual amplitude difference and naturalness remain diagnostics. No monotonic curve or crossover reward.
- Every final candidate must additionally pass 12 s at 0.125 ms and 6 s at 0.0625 ms, with nominal/half-step displacement disagreement <=10%. Report long/short speed, actual motion and proximity to this bound. A pass is a numerical/engineering acceptance, not proof of physical fidelity or a global optimum.

## Finite search and checks

1. Reuse only current-session results at the exact finer conditions and unchanged physics/runtime, with individually hashed provenance. Historical and current coarse scores may propose coordinates only.
2. At the finer timestep, compare diverse strong coarse proposals, previous provisional amplitudes and the fixed reference. Include failed coarse validation candidates when their exact fine result is available: judge their current fine condition directly.
3. Every wave receives an identical 16-point scrambled Sobol full-box design, eight boundary/near-zero anchors, and amplitude transfers from current finer-timestep feasible neighboring/extreme waves. Transferred scores are never reused across waves.
4. Three sequential batches of eight new proposals per wave, trained on exact current finer nominal results. Three favor predicted feasible performance, three favor approximate expected improvement, one explores model uncertainty and one fills a coverage gap. Save all pools, training labels, seeds and chosen rules. Ensemble spread is a proposal heuristic, not a calibrated confidence interval.
5. Two diverse current finer feasible starts per wave; up to six coordinate poll rounds, six signed axial directions per active start. Each start also receives one eight-direction coupled diagonal challenge at its first no-material-gain axial poll. Start normalized step .04; halve if the combined poll gain <=0.2 mm, down to .005. Record resolution stops, budget stops and any replacements after failed long/half-step qualification. No-feasible waves retain explicit unresolved status and receive the full global/adaptive effort.
6. Qualify leading contenders before local search and after polls 0, 2 and 4. Do not re-center on known qualification failures. Descending-displacement final validation initially tests up to five additional candidates per wave, preserving qualified fallbacks and the count of faster untested candidates.
7. Report all waves, full-precision selected parameters, matched finer-step old/new performance, search progress, all failures, actual joint amplitudes, and exact recorded-pose front/side/oblique demos. Close owned work and verify the final receipt before describing the optimization as delivered.

The prefix rejection optimization only stops after a necessary unchanged full-horizon gate is irreversibly exceeded. Such a prefix has no ranked displacement. Surviving conditions receive the full metrics. Instrumentation must be checked against uninstrumented finer-step runs. Atomic per-condition results and immutable task banks make a stage resumable. All implementation changes and experimental amendments retain their sources and explanations.
