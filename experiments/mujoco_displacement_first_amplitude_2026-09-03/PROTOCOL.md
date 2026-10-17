# Forward-displacement-first amplitude study

Latest PI direction: the previous result is not good enough; use the visibly
wave-conditioned Centipede RL gait work as methodological evidence, maximize
forward displacement, and require only adequate—not over-constrained—stability.

This is a separate Centipede Model experiment. The local-Z compliant-tip
plant, controller law, per-wave PD, phase law, 1 Hz frequency, coupled body/leg
wave number, near-ground initialization, flat terrain and contact-erasure OFF
are frozen. Only body-yaw, leg-yaw and leg-pitch command amplitudes vary within
the existing [0,36] / [0,90] / [0,60] degree bounds. No Robot Step 18/19,
IsaacLab/Clean, canonical configuration, model, controller or PD file is edited.

## Metric and selection

The sole objective is fixed-axis forward displacement (mm) of the unweighted
21-chassis geometric center over t=1.5..5.999 s. There is no weighted reward,
monotonicity/smoothness reward, amplitude-difference reward, slip/work/contact
reward, or reference-relative performance constraint. A visually expected
curve may seed the search but cannot decide the winner.

Minimum viability gates, frozen before reclassification:

| Property | Limit |
|---|---:|
| finite, warning-free; forward displacement | required; > 0 |
| minimum upright; spine tortuosity | >= cos(50 deg); <= 1.2 |
| root world-roll peak | <= 35 deg |
| chassis-center height range; vertical-speed P99 | <= 8 mm; <= 0.25 m/s |
| body angular-velocity jump over 10 ms | <= 30 rad/s |
| body pitch/roll change over 10 ms | <= 5 / 5 deg |
| ground/chassis/foot peak force | <= 5 / 2 / 2 body weights |
| chassis ground-contact fraction | <= 0.15 |
| longest ballistic gap; penetration peak | <= 0.08 s; <= 0.25 mm |
| active/tip velocity peak | <= 200 / 1000 rad/s |
| tip peak angle / stop fraction | <= 40 deg / 0.10 |
| actuator torque peak | <= 2e-4 Nm |
| leg yaw/pitch speed P99 | <= 30 / 30 rad/s |
| leg yaw/pitch reversals P95 per cycle | <= 12 / 12 |

These are catastrophe/contact-discontinuity bounds, not a claim of biological
calibration. Path efficiency, coherent/front roll, slip, work per distance,
force imbalance/impulses relative to the old gait, mean feet, duty symmetry,
swing height, waveform residuals and tracking errors remain reported
diagnostics. They are deliberately not hidden in a scalar penalty.

Every provisional winner receives a cold 12 s and cold half-timestep 6 s run.
Both must pass the same viability gates; half-step displacement must agree
within 10%. An unsmoothed three-view and local-tip comparison is generated.
Visual review may reject an obviously non-centipede-like winner, but the exact
observed failure must be recorded; a reviewer cannot silently prefer a slower
gait because its parameter curve looks smoother.

## Search

1. Recompute viability for every available full 6 s trajectory from the 5,036
   prior conditions and the stopped convergence study. Incomplete prefix
   records are never ranked by partial displacement.
2. Immediately validate/render the best currently available condition per wave
   as a displacement-first pilot. This tests the revised policy before a long
   search. Missing waves remain unresolved.
3. Complete all 256 points/wave in the stopped first Sobol design. Conditions
   rejected early by the old tighter policy receive full cold rollouts.
4. Run a second independent 128-point/wave scrambled Sobol bank.
5. Transfer displacement-first incumbents across waves; add the IsaacLab v5/v4
   command curve as a scale-aware SEED ONLY. Isaac and MuJoCo angle magnitudes
   are not asserted equivalent.
6. Run iterative 3-D pattern refinement from three spatially separated feasible
   starts/wave. Move after improvement; halve a non-improving normalized step
   from .04 to .0025; maximum 16 polls. A budget stop is unresolved, never called
   convergence or a global optimum.
7. Validate candidates in descending nominal displacement. Preserve failures,
   diagnostics and all prior results.

The outcome is a validated finite-tested-set maximum, not a mathematical proof
of a continuous global optimum. Report commanded and realized amplitudes,
per-wave parameter differences, objective gain, viability margins and the
diagnostics removed from the hard gate.
