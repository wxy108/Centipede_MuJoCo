# Common-PD amplitude and random-contact study

Authorized by the PI on 2026-09-09: systematically compare historical results, identify and record causes, optimize the amplitude-versus-wave trend, then test contact erasure and crossover on flat ground. Use MuJoCo first; if adequate MuJoCo investigation does not recover the trend, try original IsaacLab RL. Never modify Clean.

## Locked controller and scientific question

Use the existing common profile `p4_leg_k2_yaw_d2_roll_d4`, already shared by recovery waves 1.5, 2.5, 4, 6 and 7. This is a fixed reference choice, not a new PD search. Every task must carry the identical complete multiplier vector and must verify the actual active Kp/Kv arrays. No wave-specific PD, feedback, frequency search, or noise-specific amplitude tuning. Frequency is 1 Hz. Body and leg wave numbers remain coupled.

Primary plant is the current local-Z passive-tip model, frozen by SHA-256. Keep physics dt=0.125 ms, root-only near-ground initialization, 1 s rest and inherited 0.25 s onset. Primary displacement uses the unweighted 21-chassis centroid, fixed world forward axis, labels 1.5..5.999 s. Finalists require 12 s continuation and 0.0625 ms verification. Preserve raw data and original sealed experiments.

The desired declining body-amplitude trend and low/high-wave crossover are hypotheses, not optimizer rewards or acceptance filters on chosen seeds. Optimize measured displacement among the existing September 3 engineering-feasible gaits. Report individual nonmonotonic points and unresolved waves. No monotonicity constraint, hidden naturality gate, or slope/crossover reward.

## Differences to test

1. Historical source audit already established that recovery used 8 PD vectors across 13 waves and RL v4 used 13. Those maps are not common-PD comparisons. Re-evaluate their amplitudes with the one fixed profile; retain old scores only as historical references.
2. Compare current and August 31 tip plants using the same fixed active PD, initialization, amplitudes, timestep and metric. XML spring/axis differences are explicit plant treatments, never wave-dependent modifications.
3. The present RL code samples the leg-root phase half a segment anterior to the body-joint sample (`+0.5*delta`). MuJoCo currently uses the body-joint phase directly. Audit joint/leg-root geometry, then compare these two specified implementations at identical amplitudes and PD. This is a discrete implementation diagnosis, not an arbitrary phase search. Preserve legacy outcomes.
4. Re-score complete common-PD outcomes under old naturality criteria as a labelled diagnostic only. Determine whether body/leg waveform and tracking gates remove distinct high-performing regions. Do not alter the main engineering-feasible objective to force a trend.
5. Numerical and optimization issues: mandatory historical anchors, broad joint three-amplitude exploration, diverse starts, matched step checks, and full-horizon verification. A prefix rejection proves only violation of a necessary gate and has no displacement score.

## Search and branch decisions

Wave bank: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9. Historical RL range 1.5..4.5 is reported separately from the extended range. Same amplitude domain at each wave: body 0..36 deg, leg yaw 0..90 deg, leg pitch 0..60 deg.

Start with a reproducibility check and matched diagnostic bank. Any physically justified phase correction is declared as a new branch before its optimization, with both source versions retained. Run broad Sobol and historical anchors followed by independent multistart local refinement. Select each wave solely by its own validated forward displacement. No selecting a slower gait simply to obtain a preferred amplitude curve.

Assess trend only after optimization and validation, including local uncertainty and nonmonotonic exceptions. If current MuJoCo results do not support the desired overall relation, first report completed search coverage and mechanism evidence, then prepare the same fixed-PD amplitude-only experiment in a dated directory in original IsaacLab RL. Do not interpret a solver name alone as a root cause.

## Contact erasure, after amplitude selection is frozen

Use the same masks across waves and coupled random numbers across erasure probabilities. The probability axis must be the actual declared Bernoulli erasure probability, with event duration and refresh clock held fixed. Save generated schedules and realized erased-time fractions. Do not silently substitute trigger probability, lost load, or measured contact occupancy for each other.

Primary erasure implementation must avoid restoring collision into deeply penetrating geometry. Diagnose overlap and force impulses explicitly before interpreting a crossing. Keep original collision geometry and controller commands; document any restoration rule and its effect on actual exposure. Full-horizon displacement, upright survival, engineering viability and body-ground support are reported separately. No failure-conditioned means or imputed full-duration scores.

Use all qualified waves for discovery, then new paired seeds for confirmation of any crossing bracket. Require opposite signs of low-minus-high displacement with uncertainty bands at either side, nontrivial forward performance, and step/duration checks. If the crossing only occurs with body dragging, collapse, or restoration artifacts, report it as such and do not claim clean redundancy evidence.

## Records

All new MuJoCo code/data live in this directory. Freeze hashes of source dependencies, selected common PD, model, protocol, and each task bank. Live state and resumable completed records are written atomically. Only owned processes may be stopped. One GPU job at a time for any Isaac fallback. No changes to original/Clean defaults, Robot, previous raw trials or sealed receipts.
