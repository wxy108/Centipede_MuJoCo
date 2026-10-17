# Wave-dependent contact-loss crossover

2026-09-08. Centipede Model only. The PI asks whether fast low-wave gaits lose performance faster under random contact loss and are overtaken by higher-wave gaits. A crossover is a hypothesis to test, not a selection requirement.

## Frozen plant and gaits

Use the local-Z compliant-tip plant and all ten provisionally qualified profiles from `mujoco_displacement_first_amplitude_2026-09-03/outputs/data/interim_validated_selection.json`. Waves are 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7. Waves 0.5, 1, 8, 9 lack qualified nominal baselines and are excluded with this reason, not because of their noise response.

Freeze each wave's amplitude and PD profile across noise. Keep flat ground, 1 Hz, coupled body/leg waves, near-ground initialization, zero armature, geometry, phase law and the controller unchanged. No noise-specific optimization. These are optimized gait packages with wave-specific PD/amplitude, so differences cannot be attributed to wave number alone. They are not proven global optima.

Use the inherited 6 s evaluator, t=1.5..5.999 s measurement window and 0.25 ms timestep. Primary outcome is unweighted 21-chassis centroid displacement on the frozen forward axis in mm. Preserve path/heading/roll/contact/force/energy diagnostics.

## Noise channel

Use independent bilateral segment-pair collision erasure (19 channels). Disable all currently colliding leg-subtree geoms of a pair; leave chassis collision and controller commands unchanged. This is a physical collision-capability ablation, not sensor dropout or measured actual foot-contact probability.

Noise begins at 1.5 s. Nine external-clock windows of 0.5 s cover the evaluation. Window timing does not depend on gait wave. Every seed supplies the same independent trigger/duration/offset uniforms for all waves and all noise levels. Unlike older wave-seeded/phase-shifted schedules, this guarantees exactly matched realized masks across waves; the new timing is explicit.

The horizontal variable nu is expected erased-time fraction, equivalently the marginal erased probability under the random-time approximation. The event trigger probability q is recorded separately:

- nu <= 0.75: q=nu/0.75; duration/window A~Uniform(0.5,1).
- nu > 0.75: q=1; A~Uniform(2nu-1,1).
- Start offset is uniform within the available part of each window.
- nu=0 and nu=1 are deterministic conditions; run each once per wave, not repeated as independent seeds.
- The same uniforms give nested erasure intervals as nu increases. Save complete event tables, timestep masks, commanded control hashes, collision grouping and restoration diagnostics.

These are the historical strong-burst duration laws with explicitly matched external-clock timing. Results are not pooled with older short-loguniform-burst experiments.

## Stages fixed before inspecting noise results

1. Verify all frozen source hashes. Independently compare instrumented and uninstrumented no-loss wave 2 and wave 6 in the current runtime, requiring exact equality of all inherited metrics between these two implementations. See the pre-noise runtime note below for historical numerical drift. Accept current-runtime no-loss profiles only if displacement is within 1% of the old record and the unchanged speed-first viability gates still pass. Retention uses each newly measured current-runtime baseline. Check every wave before its noise response is interpreted.
2. Pilot: waves 2, 3, 6, 7; nu=0,.25,.5,.75,1; seeds 0,1,2 except deterministic endpoints. Inspect failure modes and runtime.
3. Main descriptive screen: all ten waves, the same five nu values, seeds 0..5 except deterministic endpoints. Include every planned cell and trial, including failures. Pilot records may be reused under identical condition identities.
4. If a crossover bracket appears, select a low/high pair from all eligible wave pairs using a documented rule: positive no-loss low-minus-high displacement and negative high-loss difference, favoring pairs whose two curves still retain positive displacement and upright survival. Add an intermediate nu if useful. Confirm the selected pair around the bracket with new seeds >=100, then check representative matched conditions at half timestep and longer duration. Discovery and confirmation stay separate.
5. If no crossover appears, report that result. Do not retune amplitudes, change gates or select favorable seeds to manufacture one.

## Analysis and failures

Plot absolute displacement versus nu (main), matched-baseline displacement retention, and viability/upright-survival proportions separately. Plot every wave; do not show only a favorable pair. For stochastic cells show seed points or uncertainty. Compare waves through paired seed differences; interpolate a zero only as an estimated bracket, never an exact threshold.

Run full duration while numerically valid, even after an engineering gate fails. Keep those full-horizon displacements in descriptive plots and mark viability failures. This avoids conditioning means on surviving trials. Inherited speed-first gates remain unchanged. Upright survival is separately defined by the inherited min-upright threshold cos(50 degrees); passing it does not imply all engineering gates passed.

On nonfinite state or solver time reset, stop and save the partial trajectory and failure; do not assign its prefix a full-duration displacement. Missing full-horizon outcomes must be counted and reported. Do not automatically zero or impute them.

Contact restoration may introduce penetration and impulses. Log restoration gap, pre/post root velocity jump, force peaks and warnings. A crossing dominated by such artifacts is a diagnostic result, not evidence of biological redundancy. Random-contact gates can fail even when displacement is retained.

## Outputs and boundaries

Write only in this new experiment directory. Preserve all source models, previous experiments, Robot Step 18/19 and IsaacLab/Clean. Keep source hashes, exact profiles, manifests, raw trajectories, failure receipts, plots and a Chinese report. Provide matched front/side/oblique demos with erased legs red, explicit noise/seed labels, and no trajectory smoothing or physics changes for rendering.

## Pre-noise runtime recheck

The initial historical bitwise-reproduction gate failed before any noise run: wave 2 differs by 1.44e-8 mm, wave 6 by 0.08053 mm (0.078%). A separate uninstrumented replay through the preserved engine reproduces the newly instrumented results exactly for every metric in both waves. Thus the added observation/erasure code has no measured no-loss effect. The historical numerical drift's cause is not established; frozen sources match. The current environment and both comparison receipts are saved. Original protocol/script/manifest and failed bitwise-check receipts are retained under `outputs/audit/before_runtime_recheck` and `outputs/trials`. The revised current-runtime baseline contract above was declared before looking at noise outcomes; it changes no physical parameter or viability limit.
